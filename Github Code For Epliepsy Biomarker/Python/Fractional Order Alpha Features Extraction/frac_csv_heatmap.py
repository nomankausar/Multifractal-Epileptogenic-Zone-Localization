# === Fractional Order (alpha) only: 500 ms segments, multiprocessing, SOZ-marked plots ===
# Needs: mne, numpy, pandas, matplotlib; your fracModel.py in PYTHONPATH
# Inputs in data_dir: recording files (.edf/.fif) and SOZ_Channels_info.csv

import os
import re
import math
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

from fracModel import fracOrdUU  # your implementation

# ----------------------- CONFIG -----------------------
data_dir            = r"I:\Research After Conference\task 75 cleaned version 5min child data\Cleaned Version\Data"
soz_csv             = os.path.join(data_dir, "SOZ_Channels_info.csv")
segment_duration_s  = 0.5          # 500 ms
notch_hz            = 60.0         # set None to skip
hp_hz               = None         # e.g., 0.5
lp_hz               = None         # e.g., 250
max_workers         = max(12, cpu_count() - 1)
max_segments         = 1000          # save/process Segment_1 ... Segment_300
merged_csv_name      = "ALL_SUBJECTS_alpha_rowwise_500ms.csv"
# ------------------------------------------------------

def extract_subject_id(fname: str) -> str | None:
    m = re.search(r"(?:sub-)?([A-Za-z0-9]+)", fname)
    return m.group(1).upper() if m else None

def clean_channel_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(name)).upper()

def load_soz_mapping(csv_path: str) -> dict[str, list[str]]:
    """Return {SUBJECT_ID: [CLEAN_CHANNEL, ...]} from a wide or long SOZ file."""
    df = pd.read_csv(csv_path)
    if "Subject" in df.columns:
        long_df = (
            df.melt(id_vars="Subject", var_name="col", value_name="Channel")
              .dropna(subset=["Channel"])
        )
        long_df["Subject_ID"] = (
            long_df["Subject"].astype(str)
            .str.replace(r"[^A-Za-z0-9]", "", regex=True)
            .str.upper()
        )
        long_df["Clean_Channel"] = long_df["Channel"].astype(str).apply(clean_channel_name)
        return long_df.groupby("Subject_ID")["Clean_Channel"].apply(list).to_dict()
    else:
        df["Subject_ID"] = (
            df["Subject_ID"].astype(str)
            .str.replace(r"[^A-Za-z0-9]", "", regex=True)
            .str.upper()
        )
        df["Clean_Channel"] = df["Channel"].astype(str).apply(clean_channel_name)
        return df.groupby("Subject_ID")["Clean_Channel"].apply(list).to_dict()

def process_segment(seg_idx: int, raw_fname: str, segment_duration: float, soz_clean: set[str]) -> pd.DataFrame:
    """Worker: compute fractional order for one segment across all channels."""
    raw = mne.io.read_raw(raw_fname, preload=True, verbose=False)
    if notch_hz: raw.notch_filter(notch_hz, verbose=False)
    if hp_hz or lp_hz:
        raw.filter(l_freq=hp_hz, h_freq=lp_hz, verbose=False)

    ch_names = raw.info["ch_names"]
    ch_clean = [clean_channel_name(c) for c in ch_names]
    sf = float(raw.info["sfreq"])
    win = int(segment_duration * sf)

    start = seg_idx * win
    stop  = start + win
    if stop > raw.n_times:
        return pd.DataFrame([])

    X = raw.get_data(start=start, stop=stop)  # (n_ch, n_samp)

    # Guard: if too short or model fails, fill NaNs
    if X.shape[1] < 4:
        fo = np.full((X.shape[0],), np.nan)
    else:
        try:
            model = fracOrdUU(verbose=0)
            model.fit(X)
            fo = np.asarray(model._order, dtype=float)
        except Exception:
            fo = np.full((X.shape[0],), np.nan)

    seg_1b = seg_idx + 1
    rows = []
    for cname, cclean, val in zip(ch_names, ch_clean, fo):
        rows.append({
            "Segment": seg_1b,
            "Channel": cname,
            "Clean_Channel": cclean,
            "is_soz": 1 if cclean in soz_clean else 0,
            "frac_order": float(val) if np.isfinite(val) else np.nan,
        })
    return pd.DataFrame(rows)

def plot_box_and_heatmap(df: pd.DataFrame, sub_id: str, out_dir: str, seg_dur: float):
    # --- Box plot: SOZ vs Non-SOZ (fractional order) ---
    fig, ax = plt.subplots(figsize=(6,6), dpi=150)
    non_soz = df.loc[df["is_soz"]==0, "frac_order"].dropna().values
    soz     = df.loc[df["is_soz"]==1, "frac_order"].dropna().values
    bp = ax.boxplot([non_soz, soz], labels=["Non-SOZ","SOZ"], patch_artist=True)
    bp["boxes"][0].set(facecolor="#9ecae1")   # blue
    bp["boxes"][1].set(facecolor="#fc9272")   # red
    for k in ["whiskers","caps","medians"]:
        for ln in bp[k]: ln.set(linewidth=1.5)
    ax.scatter(np.ones_like(non_soz)*1.0, non_soz, s=4, alpha=0.25, color="k")
    ax.scatter(np.ones_like(soz)*2.0,     soz,     s=4, alpha=0.25, color="k")
    ax.set_title(f"{sub_id}: Fractional Order (α) — SOZ vs Non-SOZ")
    ax.set_ylabel("Fractional Order (α)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{sub_id}_fracorder_boxplot.png"))
    plt.close(fig)

    # --- Heat map: fractional order (channels × segments), SOZ first & colored labels ---
    # order channels: SOZ first (keep within-group order of first appearance)
    ch_order = (df.drop_duplicates(subset=["Channel","is_soz"])
                  .sort_values(by="is_soz", ascending=False)["Channel"].tolist())
    seg_order = sorted(df["Segment"].unique(), key=float)

    mat = (df.pivot_table(index="Channel", columns="Segment", values="frac_order",
                          aggfunc="mean")
              .reindex(index=ch_order, columns=seg_order))

    # dynamic color scale from robust percentiles
    v = mat.values
    vmin = np.nanpercentile(v, 2) if np.isfinite(v).any() else 0.0
    vmax = np.nanpercentile(v, 98) if np.isfinite(v).any() else 1.0
    if not np.isfinite(vmin): vmin = 0.0
    if not np.isfinite(vmax): vmax = 1.0
    if vmax <= vmin: vmax = vmin + 1e-6

    fig2, ax2 = plt.subplots(figsize=(14,8), dpi=150)
    im = ax2.imshow(mat.values, aspect="auto", origin="upper", vmin=vmin, vmax=vmax, cmap="viridis")
    ax2.set_title(f"{sub_id}: Fractional Order (α) — Channels × Segments ")
    ax2.set_ylabel("Channel (SOZ first)")

    # >>> Show time in seconds on x-axis (segment * seg_dur)
    time_vals = np.asarray(seg_order, dtype=float) * seg_dur  # e.g., 0.5 s per segment
    ax2.set_xlabel(f"Time (s) — segment = {seg_dur:.3f}s")

    if len(seg_order) <= 80:
        ax2.set_xticks(np.arange(len(seg_order)))
        ax2.set_xticklabels(np.round(time_vals, 1), rotation=90, fontsize=7)
    else:
        step = max(1, len(seg_order)//50)
        ticks = np.arange(0, len(seg_order), step)
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(np.round(time_vals[ticks], 1), rotation=90, fontsize=7)

    # y ticks colored by SOZ
    ax2.set_yticks(np.arange(len(ch_order)))
    ax2.set_yticklabels(ch_order, fontsize=7)
    soz_set = set(df.loc[df["is_soz"]==1, "Channel"].unique())
    for lab in ax2.get_yticklabels():
        lab.set_color("#d62728" if lab.get_text() in soz_set else "#1f77b4")

    cbar = fig2.colorbar(im, ax=ax2)
    cbar.set_label("Fractional Order (α)")

    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, f"{sub_id}_fracorder_heatmap.png"))
    plt.close(fig2)

def subject_pipeline(file_path: str, subject_soz_map: dict[str, list[str]]):
    fname = os.path.basename(file_path)
    sub_id = extract_subject_id(fname)
    if not sub_id:
        print(f"Skip (no subject id): {fname}")
        return
    sub_id = re.sub(r'[^A-Za-z0-9]', '', sub_id).upper()

    if sub_id not in subject_soz_map:
        print(f"Skip (no SOZ map): {sub_id}")
        return

    # open once to discover sizes
    raw = mne.io.read_raw(file_path, preload=True, verbose=False)
    if notch_hz: raw.notch_filter(notch_hz, verbose=False)
    if hp_hz or lp_hz:
        raw.filter(l_freq=hp_hz, h_freq=lp_hz, verbose=False)

    sf = float(raw.info["sfreq"])
    win = int(segment_duration_s * sf)
    available_segments = math.floor(raw.n_times / win)
    if available_segments < 1:
        print(f"Too short for {segment_duration_s:.3f} s windows: {fname}")
        return None

    # Process at most the first 300 segments. With 500 ms windows, this is 150 seconds.
    n_segments = min(available_segments, max_segments)
    channel_names = list(raw.info["ch_names"])

    soz_clean = set(clean_channel_name(c) for c in subject_soz_map[sub_id])
    print(
        f"[{sub_id}] available_segments={available_segments}, "
        f"processing={n_segments}, channels={len(channel_names)}"
    )

    worker = partial(process_segment,
                     raw_fname=file_path,
                     segment_duration=segment_duration_s,
                     soz_clean=soz_clean)

    with Pool(processes=max_workers) as pool:
        parts = pool.map(worker, list(range(n_segments)))

    df = pd.concat(parts, ignore_index=True)
    if df.empty:
        print(f"No rows for {sub_id}")
        return None

    df.insert(0, "Subject_ID", sub_id)

    # ------------------------------------------------------------------
    # Save one row per channel:
    # Subject_ID, Channel_ID, Channel_Name, Segment_1, ..., Segment_300
    # Missing segments are kept as NaN so every subject has identical headers.
    # ------------------------------------------------------------------
    segment_numbers = list(range(1, max_segments + 1))
    wide_values = (
        df.pivot_table(
            index="Channel",
            columns="Segment",
            values="frac_order",
            aggfunc="first",
            sort=False,
        )
        .reindex(index=channel_names, columns=segment_numbers)
    )
    wide_values.columns = [f"Segment_{int(seg)}" for seg in segment_numbers]
    wide_values.index.name = "Channel_Name"
    wide_values = wide_values.reset_index()

    channel_metadata = pd.DataFrame({
        "Subject_ID": sub_id,
        "Channel_ID": np.arange(1, len(channel_names) + 1, dtype=int),
        "Channel_Name": channel_names,
    })

    rowwise = channel_metadata.merge(
        wide_values, on="Channel_Name", how="left", validate="one_to_one"
    )

    subject_csv = os.path.join(
        data_dir, f"{sub_id}_alpha_rowwise_500ms_segments_1_to_300.csv"
    )
    rowwise.to_csv(subject_csv, index=False)
    print(f"Saved rowwise CSV: {subject_csv}")

    # ---- Plots ----
    plot_box_and_heatmap(df, sub_id, data_dir, segment_duration_s)

    # Return the rowwise table so main can merge all subjects.
    return rowwise

if __name__ == "__main__":
    soz_map = load_soz_mapping(soz_csv)
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
             if f.lower().endswith((".edf", ".fif"))]
    files.sort()

    all_subject_tables = []

    for fp in files:
        try:
            subject_table = subject_pipeline(fp, soz_map)
            if subject_table is not None and not subject_table.empty:
                all_subject_tables.append(subject_table)
        except Exception as e:
            print(f"Error in {os.path.basename(fp)}: {e}")

    # Final merged CSV: all subjects, one row per channel.
    if all_subject_tables:
        merged = pd.concat(all_subject_tables, ignore_index=True, sort=False)
        merged_csv = os.path.join(data_dir, merged_csv_name)
        merged.to_csv(merged_csv, index=False)
        print(f"Saved merged CSV: {merged_csv}")
        print(f"Merged rows={len(merged)}, columns={len(merged.columns)}")
    else:
        print("No subject CSVs were created, so the merged CSV was not saved.")
