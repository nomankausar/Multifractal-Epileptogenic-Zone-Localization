"""
Batch dominant-eigenvector magnitude extraction for EDF/FIF iEEG files.

OUTPUTS
-------
1. One subject-named row-wise CSV per subject.
2. One merged row-wise CSV containing all subjects.
3. One 300-DPI heatmap PNG per subject.

ROW-WISE CSV FORMAT
-------------------
Subject_ID, Channel, is_soz, Segment_1, Segment_2, ... Segment_N

Only the normalized dominant right-eigenvector magnitude is stored.
Place fracModel.py in the same folder as this script.
"""

from __future__ import annotations

import os
import re
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from fracModel import fracOrdUU


# =============================================================================
# USER SETTINGS
# =============================================================================
DATA_DIR = Path(
    r"F:\Github For Epliepsy Project\Python\data files\child 90\Data"
)
SOZ_CSV = DATA_DIR / "SOZ_Channels_info.csv"

SEGMENT_DURATION_SECONDS = 0.5
USE_MULTIPROCESSING = True
MAX_WORKERS = 8
OVERWRITE_EXISTING = False  # False allows a stopped batch to resume safely.
APPLY_60HZ_NOTCH = True

OUTPUT_DIR = DATA_DIR / "eigenvector_magnitude_rowwise_csvs"
MERGED_CSV_NAME = "ALL_SUBJECTS_eigenvector_magnitude_rowwise.csv"
SUBJECT_CSV_SUFFIX = "_eigenvector_magnitude_rowwise.csv"

SAVE_HEATMAP_PNG = True
HEATMAP_DPI = 300
HEATMAP_DIR = DATA_DIR / "eigenvector_magnitude_heatmaps"
HEATMAP_SUFFIX = "_eigenvector_magnitude_heatmap.png"

SUPPORTED_EXTENSIONS = (".edf", ".fif", ".fif.gz")
CSV_FLOAT_FORMAT = "%.10g"


# Worker-global SOZ mapping. It is populated in main() and in each Windows
# multiprocessing worker through _initialize_worker().
_SUBJECT_SOZ_BASES: Dict[str, Set[str]] = {}


# =============================================================================
# GENERAL HELPERS
# =============================================================================
def safe_print(*args, **kwargs) -> None:
    """Print safely on Windows terminals that use a limited code page."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        encoding = sys.stdout.encoding or "utf-8"
        cleaned = text.encode(encoding, errors="ignore").decode(
            encoding, errors="ignore"
        )
        print(cleaned, **{k: v for k, v in kwargs.items() if k != "file"})


def normalize_subject_id(value: object) -> str:
    """Normalize subject IDs so Detroit005 and DETROIT005 match."""
    return re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()


def channel_base(name: object) -> str:
    """
    Normalize contact names for SOZ matching.

    Examples:
      A13-AV -> A13
      b007   -> B7
      G1     -> G1
    """
    text = re.sub(r"[^A-Za-z0-9]", "", str(name)).upper()
    text = re.sub(r"AV$", "", text)
    match = re.fullmatch(r"([A-Z]+)(\d+)", text)
    if match:
        letters, digits = match.groups()
        return f"{letters}{int(digits)}"
    return text


def natural_sort_key(value: object) -> Tuple[Tuple[int, object], ...]:
    """Natural ordering: A2 comes before A10."""
    parts = re.split(r"(\d+)", str(value).upper())
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in parts
        if part != ""
    )


def channel_sort_key(name: object) -> Tuple:
    """Sort channels naturally using their normalized base name first."""
    return natural_sort_key(channel_base(name)) + natural_sort_key(name)


def extract_subject_id(filename: str) -> str:
    """Extract Detroit005 from names such as sub-Detroit005_ses-01_....fif."""
    match = re.search(r"sub-?([A-Za-z0-9]+)", filename, flags=re.IGNORECASE)
    if match:
        return normalize_subject_id(match.group(1))

    stem = filename
    lower = stem.lower()
    if lower.endswith(".fif.gz"):
        stem = stem[:-7]
    else:
        stem = os.path.splitext(stem)[0]
    return normalize_subject_id(stem.split("_")[0])


def is_supported_file(filename: str) -> bool:
    lower = filename.lower()
    return any(lower.endswith(ext) for ext in SUPPORTED_EXTENSIONS)


def segment_column_number(column: str) -> Optional[int]:
    match = re.fullmatch(r"Segment_(\d+)", str(column), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def atomic_write_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Write a complete CSV and then atomically move it into place."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    df.to_csv(
        temporary_path,
        index=False,
        float_format=CSV_FLOAT_FORMAT,
        na_rep="",
    )
    os.replace(temporary_path, output_path)


def _friendly_tick_step(count: int, max_labels: int) -> int:
    if count <= 0:
        return 1
    raw_step = max(1, int(np.ceil(count / max_labels)))
    friendly_steps = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60]
    for step in friendly_steps:
        if step >= raw_step:
            return step
    return raw_step


# =============================================================================
# SOZ LABEL LOADING
# =============================================================================
def _truthy_soz(series: pd.Series) -> pd.Series:
    """Interpret common SOZ flag encodings as True/False."""
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower()
    return numeric.eq(1) | text.isin({"1", "true", "yes", "y", "soz", "ez"})


def load_soz_mapping(csv_path: Path) -> Dict[str, Set[str]]:
    """
    Load either of these common SOZ CSV layouts:

    Wide:
      Subject, Channel_1, Channel_2, Channel_3, ...

    Long:
      Subject_ID, Channel, is_soz
      (When is_soz is absent, every listed Channel is treated as SOZ.)
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"SOZ CSV not found: {csv_path}")

    soz_df = pd.read_csv(csv_path)
    if soz_df.empty:
        raise ValueError(f"SOZ CSV is empty: {csv_path}")

    subject_candidates = [
        c
        for c in soz_df.columns
        if str(c).strip().lower() in {"subject", "subject_id", "subjectid"}
    ]
    if not subject_candidates:
        raise ValueError(
            "SOZ CSV needs a subject column named Subject or Subject_ID."
        )
    subject_col = subject_candidates[0]

    channel_candidates = [
        c
        for c in soz_df.columns
        if str(c).strip().lower() in {"channel", "channel_name", "channelname"}
    ]
    flag_candidates = [
        c
        for c in soz_df.columns
        if str(c).strip().lower() in {"is_soz", "issoz", "is_ez", "isez"}
    ]

    if channel_candidates:
        channel_col = channel_candidates[0]
        long_df = soz_df[[subject_col, channel_col] + flag_candidates[:1]].copy()
        long_df = long_df.dropna(subset=[subject_col, channel_col])
        if flag_candidates:
            long_df = long_df[_truthy_soz(long_df[flag_candidates[0]])]
        long_df = long_df.rename(
            columns={subject_col: "Subject", channel_col: "Channel"}
        )
    else:
        value_columns = [c for c in soz_df.columns if c != subject_col]
        if not value_columns:
            raise ValueError("No SOZ channel columns were found in the SOZ CSV.")
        long_df = soz_df.melt(
            id_vars=subject_col,
            value_vars=value_columns,
            var_name="Channel_Index",
            value_name="Channel",
        )[[subject_col, "Channel"]]
        long_df = long_df.dropna(subset=[subject_col, "Channel"])
        long_df = long_df.rename(columns={subject_col: "Subject"})

    long_df["Subject_ID"] = long_df["Subject"].map(normalize_subject_id)
    long_df["SOZ_Base"] = long_df["Channel"].map(channel_base)
    long_df = long_df[(long_df["Subject_ID"] != "") & (long_df["SOZ_Base"] != "")]

    mapping = (
        long_df.groupby("Subject_ID")["SOZ_Base"]
        .apply(lambda values: set(values.astype(str)))
        .to_dict()
    )
    if not mapping:
        raise ValueError("No usable subject/SOZ-channel mappings were found.")
    return mapping


def _initialize_worker(soz_mapping: Dict[str, Set[str]]) -> None:
    global _SUBJECT_SOZ_BASES
    _SUBJECT_SOZ_BASES = soz_mapping
    mne.set_log_level("ERROR")


# =============================================================================
# EIGENVECTOR MAGNITUDE
# =============================================================================
def leading_right_eigenvector_magnitude(A_stack: np.ndarray) -> np.ndarray:
    """
    Preserve the original calculation while returning magnitude only.

    For each fracOrdUU A matrix:
      1. Take abs(A).
      2. Divide by max(abs(A)).
      3. Find the eigenvector corresponding to max(abs(eigenvalue)).
      4. Take abs(eigenvector) and normalize its largest channel to 1.
    The final channel value is the mean magnitude across fracOrdUU iterations.
    """
    matrices = np.asarray(A_stack)
    if matrices.ndim == 2:
        matrices = matrices[np.newaxis, ...]
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError(f"Invalid A-matrix stack shape: {matrices.shape}")

    n_channels = matrices.shape[1]
    valid_vectors: List[np.ndarray] = []

    for matrix in matrices:
        matrix = np.asarray(matrix, dtype=float)
        if not np.isfinite(matrix).all():
            continue

        matrix = np.abs(matrix)
        denominator = float(np.max(matrix))
        if denominator > 0:
            matrix = matrix / denominator

        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        if eigenvalues.size == 0:
            continue

        dominant_index = int(np.argmax(np.abs(eigenvalues)))
        magnitude = np.abs(eigenvectors[:, dominant_index]).astype(float)
        maximum = float(np.max(magnitude))
        if maximum > 0:
            magnitude = magnitude / maximum

        if magnitude.shape == (n_channels,) and np.isfinite(magnitude).all():
            valid_vectors.append(magnitude)

    if not valid_vectors:
        raise ValueError("No valid dominant eigenvector was produced.")

    return np.mean(np.vstack(valid_vectors), axis=0)


# =============================================================================
# HEATMAP PLOTTING
# =============================================================================
def save_subject_heatmap(subject_df: pd.DataFrame, output_path: Path, subject_id: str) -> None:
    """Save one 300-DPI heatmap PNG for a subject."""
    if subject_df.empty:
        raise ValueError("Subject dataframe is empty.")

    required_cols = {"Subject_ID", "Channel", "is_soz"}
    if not required_cols.issubset(subject_df.columns):
        raise ValueError("Subject dataframe is missing required columns.")

    segment_columns = [
        col for col in subject_df.columns if segment_column_number(col) is not None
    ]
    segment_columns = sorted(segment_columns, key=lambda col: segment_column_number(col) or 0)
    if not segment_columns:
        raise ValueError("No Segment_* columns were found for heatmap plotting.")

    plot_df = subject_df.copy()
    plot_df["is_soz"] = pd.to_numeric(plot_df["is_soz"], errors="coerce").fillna(0).astype(int)
    plot_df = plot_df.sort_values(
        by=["is_soz", "Channel"],
        ascending=[False, True],
        key=lambda series: series.map(channel_sort_key) if series.name == "Channel" else series,
    ).reset_index(drop=True)

    heatmap_values = plot_df[segment_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if heatmap_values.ndim != 2 or heatmap_values.size == 0:
        raise ValueError("Heatmap matrix is empty.")

    masked = np.ma.masked_invalid(heatmap_values)
    if masked.count() == 0:
        raise ValueError("Heatmap matrix contains only missing values.")

    n_channels, n_segments = masked.shape
    fig_width = float(np.clip(10 + n_segments * 0.08, 10, 30))
    fig_height = float(np.clip(6 + n_channels * 0.22, 6, 40))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(masked, aspect="auto", interpolation="nearest")

    x_step = _friendly_tick_step(n_segments, max_labels=40)
    x_positions = np.arange(n_segments)[::x_step]
    x_labels = [str(segment_column_number(segment_columns[pos])) for pos in x_positions]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=7)

    y_step = _friendly_tick_step(n_channels, max_labels=60)
    y_positions = np.arange(n_channels)[::y_step]
    y_labels = plot_df.loc[y_positions, "Channel"].astype(str).tolist()
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=7)

    for tick_label, row_index in zip(ax.get_yticklabels(), y_positions):
        tick_label.set_color("#d62728" if int(plot_df.iloc[row_index]["is_soz"]) == 1 else "#1f77b4")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Eigenvector Magnitude")

    ax.set_xlabel(f"Segment Number (each segment = {SEGMENT_DURATION_SECONDS:g} s)")
    ax.set_ylabel("Channel")
    ax.set_title(f"{subject_id}: Eigenvector Magnitude Heatmap")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=HEATMAP_DPI, bbox_inches="tight")
    plt.close(fig)


def generate_heatmap_from_existing_csv(csv_path: Path) -> Optional[Path]:
    """Create a heatmap from an already-saved subject CSV if needed."""
    try:
        df = pd.read_csv(csv_path)
        if df.empty or "Subject_ID" not in df.columns:
            return None
        subject_values = df["Subject_ID"].dropna().astype(str).unique().tolist()
        if not subject_values:
            return None
        subject_id = normalize_subject_id(subject_values[0])
        heatmap_path = HEATMAP_DIR / f"{subject_id}{HEATMAP_SUFFIX}"
        save_subject_heatmap(df, heatmap_path, subject_id)
        safe_print(f"[HEATMAP] Saved existing-subject heatmap: {heatmap_path}")
        return heatmap_path
    except Exception as exc:
        safe_print(f"[HEATMAP ERROR] Existing CSV {csv_path.name}: {exc}")
        return None


# =============================================================================
# SUBJECT PROCESSING
# =============================================================================
def _read_raw(file_path: Path):
    lower = file_path.name.lower()
    if lower.endswith(".edf"):
        return mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    return mne.io.read_raw_fif(file_path, preload=True, verbose=False)


def process_subject_file(filename: str) -> Optional[str]:
    """Process one EDF/FIF and save one row-wise subject CSV and one PNG heatmap."""
    if not is_supported_file(filename):
        return None

    subject_id = extract_subject_id(filename)
    if not subject_id:
        safe_print(f"[SKIP] Could not determine subject ID: {filename}")
        return None

    if subject_id not in _SUBJECT_SOZ_BASES:
        safe_print(f"[SKIP] No SOZ mapping for {subject_id}: {filename}")
        return None

    output_name = f"{subject_id}{SUBJECT_CSV_SUFFIX}"
    output_path = OUTPUT_DIR / output_name
    heatmap_path = HEATMAP_DIR / f"{subject_id}{HEATMAP_SUFFIX}"

    if output_path.exists() and not OVERWRITE_EXISTING:
        safe_print(f"[RESUME] Existing CSV kept: {output_name}")
        if SAVE_HEATMAP_PNG and (OVERWRITE_EXISTING or not heatmap_path.exists()):
            generate_heatmap_from_existing_csv(output_path)
        return str(output_path)

    file_path = DATA_DIR / filename
    raw = None
    try:
        safe_print(f"[LOAD] {subject_id}: {filename}")
        raw = _read_raw(file_path)

        if APPLY_60HZ_NOTCH:
            nyquist = float(raw.info["sfreq"]) / 2.0
            if 60.0 < nyquist:
                raw.notch_filter(freqs=[60.0], verbose=False)
            else:
                safe_print(
                    f"[WARN] {subject_id}: 60-Hz notch skipped because Nyquist={nyquist:g} Hz"
                )

        channel_names = list(raw.info["ch_names"])
        channel_bases = [channel_base(name) for name in channel_names]
        soz_set = _SUBJECT_SOZ_BASES[subject_id]
        is_soz = np.asarray([int(base in soz_set) for base in channel_bases], dtype=int)

        sampling_frequency = float(raw.info["sfreq"])
        samples_per_segment = int(round(SEGMENT_DURATION_SECONDS * sampling_frequency))
        if samples_per_segment < 1:
            raise ValueError("Segment duration produces fewer than one sample.")

        n_segments = int(raw.n_times // samples_per_segment)
        if n_segments < 1:
            raise ValueError(
                f"File is shorter than one {SEGMENT_DURATION_SECONDS:g}-second segment."
            )

        n_channels = len(channel_names)
        magnitude_matrix = np.full((n_channels, n_segments), np.nan, dtype=float)
        valid_segment_count = 0

        for segment_index in range(n_segments):
            if segment_index == 0 or (segment_index + 1) % 10 == 0:
                safe_print(
                    f"[RUN] {subject_id}: segment {segment_index + 1}/{n_segments}"
                )

            start_sample = segment_index * samples_per_segment
            stop_sample = start_sample + samples_per_segment

            try:
                X = raw.get_data(start=start_sample, stop=stop_sample)
                if X.ndim != 2 or X.shape[0] != n_channels:
                    raise ValueError(f"Unexpected segment shape: {X.shape}")
                if X.shape[1] < X.shape[0]:
                    raise ValueError(
                        f"Samples ({X.shape[1]}) are fewer than channels ({X.shape[0]})."
                    )
                if not np.isfinite(X).all():
                    raise ValueError("Segment contains NaN or infinite values.")

                model = fracOrdUU(verbose=0)
                model.fit(X)
                magnitudes = leading_right_eigenvector_magnitude(model._AMat)
                if magnitudes.shape != (n_channels,):
                    raise ValueError(
                        f"Eigenvector length {magnitudes.shape} does not match {n_channels} channels."
                    )

                magnitude_matrix[:, segment_index] = magnitudes
                valid_segment_count += 1

            except Exception as exc:
                safe_print(
                    f"[SEGMENT ERROR] {subject_id}, Segment_{segment_index + 1}: {exc}"
                )

        if valid_segment_count == 0:
            raise RuntimeError("Every segment failed; no subject CSV was written.")

        metadata_df = pd.DataFrame(
            {
                "Subject_ID": [subject_id] * n_channels,
                "Channel": channel_names,
                "is_soz": is_soz,
            }
        )
        segment_df = pd.DataFrame(
            magnitude_matrix,
            columns=[f"Segment_{index + 1}" for index in range(n_segments)],
        )
        subject_df = pd.concat([metadata_df, segment_df], axis=1)

        row_order = sorted(
            range(len(subject_df)),
            key=lambda index: (
                -int(pd.to_numeric(subject_df.iloc[index]["is_soz"], errors="coerce") or 0),
                channel_sort_key(subject_df.iloc[index]["Channel"]),
            ),
        )
        subject_df = subject_df.iloc[row_order].reset_index(drop=True)

        atomic_write_csv(subject_df, output_path)
        safe_print(
            f"[SAVED] {subject_id}: {output_path} "
            f"({valid_segment_count}/{n_segments} valid segments)"
        )

        if SAVE_HEATMAP_PNG:
            try:
                save_subject_heatmap(subject_df, heatmap_path, subject_id)
                safe_print(f"[HEATMAP] Saved: {heatmap_path} ({HEATMAP_DPI} dpi)")
            except Exception as exc:
                safe_print(f"[HEATMAP ERROR] {subject_id}: {exc}")

        return str(output_path)

    except Exception as exc:
        safe_print(f"[FILE ERROR] {subject_id}: {exc}")
        return None

    finally:
        if raw is not None:
            try:
                raw.close()
            except Exception:
                pass


# =============================================================================
# MERGING
# =============================================================================
def merge_all_subject_csvs(output_dir: Path) -> Optional[Path]:
    """Merge all subject row-wise CSVs and numerically order Segment columns."""
    subject_files = sorted(
        (
            path
            for path in output_dir.glob(f"*{SUBJECT_CSV_SUFFIX}")
            if path.name != MERGED_CSV_NAME
        ),
        key=lambda path: natural_sort_key(path.name),
    )

    if not subject_files:
        safe_print("[MERGE] No subject CSV files were found.")
        return None

    frames: List[pd.DataFrame] = []
    all_segment_numbers: Set[int] = set()
    required_columns = {"Subject_ID", "Channel", "is_soz"}

    for csv_path in subject_files:
        try:
            frame = pd.read_csv(csv_path)
            if not required_columns.issubset(frame.columns):
                safe_print(f"[MERGE SKIP] Missing required columns: {csv_path.name}")
                continue

            segment_columns = []
            for column in frame.columns:
                number = segment_column_number(column)
                if number is not None:
                    segment_columns.append(column)
                    all_segment_numbers.add(number)

            frame = frame[["Subject_ID", "Channel", "is_soz"] + segment_columns]
            frames.append(frame)
        except Exception as exc:
            safe_print(f"[MERGE SKIP] {csv_path.name}: {exc}")

    if not frames:
        safe_print("[MERGE] No valid subject CSV files could be merged.")
        return None

    ordered_segment_columns = [
        f"Segment_{number}" for number in sorted(all_segment_numbers)
    ]
    ordered_columns = ["Subject_ID", "Channel", "is_soz"] + ordered_segment_columns

    aligned_frames = [frame.reindex(columns=ordered_columns) for frame in frames]
    merged_df = pd.concat(aligned_frames, ignore_index=True)

    merged_order = sorted(
        range(len(merged_df)),
        key=lambda index: (
            natural_sort_key(merged_df.iloc[index]["Subject_ID"]),
            -int(pd.to_numeric(merged_df.iloc[index]["is_soz"], errors="coerce") or 0),
            channel_sort_key(merged_df.iloc[index]["Channel"]),
        ),
    )
    merged_df = merged_df.iloc[merged_order].reset_index(drop=True)
    merged_df["is_soz"] = pd.to_numeric(
        merged_df["is_soz"], errors="coerce"
    ).astype("Int64")

    merged_path = output_dir / MERGED_CSV_NAME
    atomic_write_csv(merged_df, merged_path)
    safe_print(
        f"[MERGED] {len(frames)} subject CSVs, {len(merged_df)} channel rows -> {merged_path}"
    )
    return merged_path


# =============================================================================
# FILE SELECTION
# =============================================================================
def select_one_file_per_subject(filenames: Sequence[str]) -> List[str]:
    """
    Prevent two files for the same subject from racing to the same subject CSV.
    The first naturally sorted file is used and additional files are reported.
    """
    grouped: Dict[str, List[str]] = {}
    for filename in sorted(filenames, key=natural_sort_key):
        grouped.setdefault(extract_subject_id(filename), []).append(filename)

    selected: List[str] = []
    for subject_id in sorted(grouped, key=natural_sort_key):
        files = grouped[subject_id]
        selected.append(files[0])
        for duplicate in files[1:]:
            safe_print(
                f"[DUPLICATE SKIP] {subject_id}: using '{files[0]}', skipping '{duplicate}'"
            )
    return selected


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    global _SUBJECT_SOZ_BASES

    mne.set_log_level("ERROR")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_HEATMAP_PNG:
        HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    if SEGMENT_DURATION_SECONDS <= 0:
        raise ValueError("SEGMENT_DURATION_SECONDS must be greater than zero.")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

    _SUBJECT_SOZ_BASES = load_soz_mapping(SOZ_CSV)
    safe_print(f"[SOZ] Loaded mappings for {len(_SUBJECT_SOZ_BASES)} subjects.")

    all_files = [name for name in os.listdir(DATA_DIR) if is_supported_file(name)]
    selected_files = select_one_file_per_subject(all_files)
    safe_print(
        f"[FILES] Found {len(all_files)} EDF/FIF files; "
        f"processing {len(selected_files)} unique subjects."
    )

    if not selected_files:
        safe_print("[STOP] No EDF/FIF files were found.")
        return

    if USE_MULTIPROCESSING and len(selected_files) > 1:
        worker_count = max(1, min(MAX_WORKERS, cpu_count(), len(selected_files)))
        safe_print(f"[MODE] Multiprocessing with {worker_count} workers.")
        with Pool(
            processes=worker_count,
            initializer=_initialize_worker,
            initargs=(_SUBJECT_SOZ_BASES,),
        ) as pool:
            pool.map(process_subject_file, selected_files)
    else:
        safe_print("[MODE] Sequential processing.")
        _initialize_worker(_SUBJECT_SOZ_BASES)
        for filename in selected_files:
            process_subject_file(filename)

    merge_all_subject_csvs(OUTPUT_DIR)
    safe_print("[DONE] Eigenvector-magnitude row-wise CSV and heatmap batch finished.")


if __name__ == "__main__":
    main()
