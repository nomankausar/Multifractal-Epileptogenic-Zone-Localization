#!/usr/bin/env python3
"""
Batch convert EDF/FIF files to MAT with 60 Hz notch filtering, folder-based, parallel.

– Uses USER SETTINGS at top to configure input/output dirs, workers, resampling, etc.
– Scans INPUT_DIR (non‐recursive) for .edf/.fif
– Applies 60 Hz notch + harmonics, resamples if desired, optionally keeps MISC channels
– Saves .mat files into OUTPUT_DIR with same base names
– Parallel processing via joblib
"""

# ========================= USER SETTINGS ========================= #
INPUT_DIR       = r"C:\Mat file"       # <--- set your input folder
OUTPUT_DIR      = r"C:\Mat file\out"   # <--- set your output folder
WORKERS         = 8                    # number of parallel processes
TARGET_SFREQ    = 1000.0               # set to a number (e.g., 1000) or None to keep file's own rate
INCLUDE_MISC    = False                # True to include MISC channels in output
HARMONICS       = 2                    # number of 60 Hz harmonics to remove (0 = 60 Hz only)
# ================================================================ #

import os
from pathlib import Path

import numpy as np
from scipy.io import savemat
import mne
from joblib import Parallel, delayed


def load_raw(path: Path) -> mne.io.BaseRaw:
    """Load EDF or FIF into an MNE Raw object."""
    ext = path.suffix.lower()
    if ext == ".edf":
        return mne.io.read_raw_edf(str(path), preload=True, stim_channel=None, verbose="ERROR")
    elif ext == ".fif":
        return mne.io.read_raw_fif(str(path), preload=True, verbose="ERROR")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def apply_notch(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Apply a 60 Hz notch filter (plus harmonics) to EEG (and optionally MISC) channels.
    Uses spectrum_fit to minimize ringing.
    """
    picks = mne.pick_types(raw.info, eeg=True, misc=INCLUDE_MISC)
    freqs = [60.0 * (i + 1) for i in range(HARMONICS + 1)]
    nyquist = raw.info["sfreq"] / 2.0
    freqs = [f for f in freqs if f < nyquist]
    if freqs:
        raw.notch_filter(
            freqs=freqs,
            picks=picks,
            method="spectrum_fit",
            filter_length="auto",
            phase="zero",
            verbose="ERROR",
        )
    return raw


def to_mat_dict(raw: mne.io.BaseRaw) -> dict:
    """
    Convert a filtered (and optionally resampled) Raw to a dict for savemat().
    """
    picks = mne.pick_types(raw.info, eeg=True, misc=INCLUDE_MISC)
    raw_sel = raw.copy().pick(picks)
    if TARGET_SFREQ is not None:
        raw_sel.resample(TARGET_SFREQ, verbose="ERROR")

    data = raw_sel.get_data()                      # shape: (n_channels, n_times)
    ch_names = np.array(raw_sel.ch_names, dtype=object)
    sfreq = float(raw_sel.info["sfreq"])

    mat = {
        "data": data,
        "ch_names": ch_names,
        "sfreq": sfreq,
    }

    if len(raw_sel.annotations):
        ann = raw_sel.annotations
        mat["annotations_onset_sec"]     = np.asarray(ann.onset, dtype=float)
        mat["annotations_duration_sec"]  = np.asarray(ann.duration, dtype=float)
        mat["annotations_description"]   = np.asarray(list(ann.description), dtype=object)

    return mat


def process_file(path: Path):
    """Load, filter, convert and save one file to .mat."""
    try:
        raw = load_raw(path)
        apply_notch(raw)
        mat_dict = to_mat_dict(raw)

        out_path = Path(OUTPUT_DIR) / f"{path.stem}.mat"
        savemat(str(out_path), mat_dict, do_compression=True)

        print(f"✓ {path.name} → {out_path.name} "
              f"[{mat_dict['data'].shape[0]}×{mat_dict['data'].shape[1]}, {mat_dict['sfreq']} Hz]")
    except Exception as e:
        print(f"✗ Failed {path.name}: {e}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inp = Path(INPUT_DIR)
    files = [p for p in inp.iterdir() if p.suffix.lower() in {".edf", ".fif"}]

    if not files:
        print(f"No .edf/.fif files found in {INPUT_DIR}")
        return

    Parallel(n_jobs=WORKERS, prefer="processes")(
        delayed(process_file)(f) for f in files
    )


if __name__ == "__main__":
    main()
