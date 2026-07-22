#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IEEG Complete Automation GUI (Customer-ready v3)
================================================

What this app does
------------------
A complete, clinician-friendly pipeline:

1) Batch-load EDF/FIF/FIF.GZ from a folder
2) Optional preprocessing:
     - Keep EEG/SEEG/ECoG (safe fallback if labels are missing)
     - Bandpass (HP/LP)
     - Resample
     - Notch: manual OR auto-use detected 50/60 Hz
3) Clinician-controlled line-noise mode:
     - Auto-detect 50 vs 60 Hz using band-power ratio
     - Or force 50 / 60
4) Auto-detect bad channels from RAW signal:
     - flatline (very low std)
     - noisy (very high std)
     - robust std outlier (z_std via MAD)
     - clipping/outliers fraction
     - line noise ratio around chosen 50/60 band
5) Auto-generate CLEANED file:
     - Always supports FIF
     - EDF export if pyedflib is installed AND user selects "Prefer EDF"
6) Save stacked PNG:
     - BEFORE (post-preprocess, pre-clean)
     - AFTER (cleaned)
     - COMPARE (side-by-side)
7) Review workflow:
     - Per subject: preview images inside the GUI
     - Edit bad channel list (add/remove, search/filter)
     - Re-export cleaned files using edited list
8) Export tables:
     - AUTO tables (CSV + Excel)
     - FINAL tables (CSV + Excel) after clinician edits
     - Metrics sheet included
9) Optional: Export a simple per-subject PDF report (before/after + list)

Dependencies
------------
Required:
  pip install mne numpy pandas matplotlib openpyxl

Optional:
  pip install pyedflib     (EDF export)
  pip install pillow       (in-app PNG preview, recommended)
  pip install reportlab    (PDF report export)

Notes
-----
- Tkinter is NOT thread-safe. This app uses a queue + root.after() polling to keep UI stable.
- Designed to be "customer-ready": safe defaults, robust fallback behaviors, one-click viewing.
"""

from __future__ import annotations

import os
import re
import time
import queue
import threading
import platform
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Optional libs
try:
    from PIL import Image, ImageTk
    PIL_OK = True
except Exception:
    PIL_OK = False

try:
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False


# ---------------------------
# Helpers
# ---------------------------

def is_supported_file(fn: str) -> bool:
    f = fn.lower()
    return f.endswith(".edf") or f.endswith(".fif") or f.endswith(".fif.gz")

def list_data_files(folder: str) -> List[str]:
    out = []
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        if os.path.isfile(p) and is_supported_file(f):
            out.append(p)
    out.sort()
    return out

def strip_ext(name: str) -> str:
    low = name.lower()
    if low.endswith(".fif.gz"):
        return name[:-7]
    if low.endswith(".fif"):
        return name[:-4]
    if low.endswith(".edf"):
        return name[:-4]
    return os.path.splitext(name)[0]

def subject_id_from_filename(path: str) -> str:
    base = os.path.basename(path)
    base = strip_ext(base)
    sid = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")
    return sid or base

def parse_multiline_list(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[,\n\r\t]+", text.strip())
    return [p.strip() for p in parts if p.strip()]

def safe_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None

def safe_int(s: str, default: int) -> int:
    try:
        return int(float(str(s).strip()))
    except Exception:
        return default

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]

def open_path(path: str):
    if not path:
        return
    if not os.path.exists(path):
        return
    try:
        sysname = platform.system()
        if sysname == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sysname == "Darwin":
            subprocess.call(["open", path])
        else:
            subprocess.call(["xdg-open", path])
    except Exception:
        pass


# ---------------------------
# Settings dataclasses
# ---------------------------

@dataclass
class PreprocessSettings:
    tmin: float = 0.0
    tmax: float = 300.0
    notch_hz: Optional[float] = None
    hp_hz: Optional[float] = None
    lp_hz: Optional[float] = None
    resample_hz: Optional[float] = None
    keep_only_ieeg: bool = True
    pre_drop_patterns: List[str] = None

@dataclass
class PlotSettings:
    decim: int = 4
    dpi: int = 300
    fig_w: float = 14.0
    fig_h: float = 9.0
    linewidth: float = 0.4
    label_fontsize: int = 9
    max_labels: int = 220
    title_prefix: str = "iEEG amplitude (stacked)"

@dataclass
class DetectSettings:
    flat_std_uv: float = 1.0
    noisy_std_uv: float = 500.0
    outlier_z_std: float = 8.0

    clip_z: float = 8.0
    clip_fraction_thr: float = 0.02

    line_noise_hz: float = 60.0
    line_noise_band: float = 2.0
    line_noise_ratio: float = 0.35

    max_bad_suggestions: int = 40


# ---------------------------
# Load / preprocess
# ---------------------------

def load_raw_any(path: str, preload: bool = True) -> mne.io.BaseRaw:
    low = path.lower()
    if low.endswith(".edf"):
        return mne.io.read_raw_edf(path, preload=preload, verbose=False)
    if low.endswith(".fif") or low.endswith(".fif.gz"):
        return mne.io.read_raw_fif(path, preload=preload, verbose=False)
    raise ValueError("Unsupported file type.")

def pick_ieeg_types_safe(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw0 = raw
    try:
        raw1 = raw.copy()
        raw1.pick_types(eeg=True, seeg=True, ecog=True, misc=False, stim=False)
        if raw1.info["nchan"] > 0:
            return raw1
        return raw0
    except Exception:
        return raw0

def drop_by_patterns(raw: mne.io.BaseRaw, patterns: List[str]):
    if not patterns:
        return
    pats = [p.lower() for p in patterns]
    drop = []
    for ch in raw.ch_names:
        cl = ch.lower()
        if any(p in cl for p in pats):
            drop.append(ch)
    if drop:
        raw.drop_channels(sorted(set(drop), key=natural_key))

def apply_filters(raw: mne.io.BaseRaw, ps: PreprocessSettings):
    if ps.notch_hz is not None and ps.notch_hz > 0:
        raw.notch_filter(ps.notch_hz, verbose=False)
    if ps.hp_hz is not None or ps.lp_hz is not None:
        raw.filter(l_freq=ps.hp_hz, h_freq=ps.lp_hz, verbose=False)
    if ps.resample_hz is not None and ps.resample_hz > 0:
        raw.resample(ps.resample_hz, npad="auto", verbose=False)

def crop_for_detection(raw: mne.io.BaseRaw, tmin: float, tmax: float) -> Tuple[int, int]:
    sf = float(raw.info["sfreq"])
    max_t = raw.n_times / sf
    tmax_use = min(tmax, max_t)
    start, stop = raw.time_as_index([tmin, tmax_use])
    start = max(0, int(start))
    stop = max(start + 1, int(stop))
    stop = min(stop, raw.n_times)
    return start, stop


# ---------------------------
# Plot stacked
# ---------------------------

def save_stacked_png(raw: mne.io.BaseRaw, out_png: str, ps: PreprocessSettings, plot: PlotSettings, title: Optional[str] = None):
    sf = float(raw.info["sfreq"])
    max_t = raw.n_times / sf
    tmax_use = min(ps.tmax, max_t)
    start, stop = raw.time_as_index([ps.tmin, tmax_use])

    data = raw.get_data(start=start, stop=stop) * 1e6
    times = raw.times[start:stop]

    if plot.decim > 1:
        data = data[:, ::plot.decim]
        times = times[::plot.decim]

    n_ch = data.shape[0]
    if n_ch == 0:
        raise RuntimeError("No channels to plot.")

    global_std = float(np.std(data))
    if global_std <= 0:
        global_std = 1.0
    offset = 4.0 * global_std
    y_positions = np.arange(n_ch) * offset

    plt.ioff()
    fig, ax = plt.subplots(figsize=(plot.fig_w, plot.fig_h))
    for i in range(n_ch):
        ax.plot(times, data[i] + y_positions[i], linewidth=plot.linewidth, color="black")

    labels = list(raw.ch_names)
    if n_ch > plot.max_labels:
        step = int(np.ceil(n_ch / plot.max_labels))
        idx = np.arange(0, n_ch, step)
        ax.set_yticks(y_positions[idx])
        ax.set_yticklabels([labels[i] for i in idx], fontsize=plot.label_fontsize)
    else:
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=plot.label_fontsize)

    max_len = max(len(str(x)) for x in labels) if labels else 10
    left_margin = min(0.48, max(0.20, 0.012 * max_len))
    fig.subplots_adjust(left=left_margin, right=0.985, top=0.90, bottom=0.10)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(title or plot.title_prefix)
    ax.set_xlim(times[0], times[-1])
    ax.grid(False)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=plot.dpi, bbox_inches="tight")
    plt.close(fig)

def save_comparison_png(before_png: str, after_png: str, out_png: str, title: str):
    import matplotlib.image as mpimg
    b = mpimg.imread(before_png)
    a = mpimg.imread(after_png)

    plt.ioff()
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.imshow(b); ax1.axis("off"); ax1.set_title("BEFORE")
    ax2.imshow(a); ax2.axis("off"); ax2.set_title("AFTER")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# Bad-channel detection + line noise auto-detect
# ---------------------------

def bandpower_ratio(x: np.ndarray, sfreq: float, f0: float, band: float) -> float:
    x = x.astype(np.float64)
    x = x - np.mean(x)
    n = len(x)
    if n < 256:
        return 0.0
    w = np.hanning(n)
    X = np.fft.rfft(x * w)
    freqs = np.fft.rfftfreq(n, d=1.0/sfreq)
    psd = (np.abs(X) ** 2)

    valid = freqs >= 1.0
    freqs = freqs[valid]
    psd = psd[valid]
    tot = psd.sum()
    if tot <= 0:
        return 0.0
    band_mask = (freqs >= (f0 - band)) & (freqs <= (f0 + band))
    return float(psd[band_mask].sum() / (tot + 1e-12))

def auto_detect_line_noise_hz(raw: mne.io.BaseRaw, tmin: float, tmax: float) -> Tuple[float, float, float]:
    sf = float(raw.info["sfreq"])
    start, stop = crop_for_detection(raw, tmin, tmax)

    data = raw.get_data(start=start, stop=stop)
    if data.size == 0:
        return 60.0, 0.0, 0.0

    ch_count = data.shape[0]
    idxs = np.linspace(0, ch_count - 1, num=min(7, ch_count), dtype=int)

    r50_list, r60_list = [], []
    for idx in idxs:
        x = data[idx].astype(np.float64) * 1e6
        r50_list.append(bandpower_ratio(x, sf, 50.0, 2.0))
        r60_list.append(bandpower_ratio(x, sf, 60.0, 2.0))

    r50 = float(np.median(r50_list))
    r60 = float(np.median(r60_list))
    chosen = 50.0 if r50 > r60 else 60.0
    return chosen, r50, r60

def detect_bad_channels(raw: mne.io.BaseRaw, ps: PreprocessSettings, ds: DetectSettings) -> Tuple[pd.DataFrame, List[str]]:
    start, stop = crop_for_detection(raw, ps.tmin, ps.tmax)
    data = raw.get_data(start=start, stop=stop) * 1e6
    chs = list(raw.ch_names)
    sf = float(raw.info["sfreq"])

    cols = ["channel","std_uV","z_std","clip_fraction","line_noise_ratio","score","reasons","auto_bad"]
    if data.size == 0:
        return pd.DataFrame(columns=cols), []

    stds = np.std(data, axis=1)
    med = float(np.median(stds))
    mad = float(np.median(np.abs(stds - med)) + 1e-8)

    rows = []
    for i, ch in enumerate(chs):
        x = data[i]
        std_uv = float(stds[i])
        z_std = float((std_uv - med) / mad)

        clip_thr = ds.clip_z * (std_uv + 1e-8)
        clip_frac = float(np.mean(np.abs(x) > clip_thr)) if len(x) else 0.0

        ln_ratio = bandpower_ratio(x, sf, ds.line_noise_hz, ds.line_noise_band)

        reasons = []
        if std_uv < ds.flat_std_uv:
            reasons.append("flatline/low-std")
        if std_uv > ds.noisy_std_uv:
            reasons.append("high-std/noisy")
        if z_std > ds.outlier_z_std:
            reasons.append("std-outlier(z)")
        if clip_frac > ds.clip_fraction_thr:
            reasons.append("clipping/outliers")
        if ln_ratio > ds.line_noise_ratio:
            reasons.append(f"line-noise@{int(ds.line_noise_hz)}Hz")

        score = 0.0
        score += max(0.0, (ds.flat_std_uv - std_uv) / (ds.flat_std_uv + 1e-8)) * 2.0
        score += max(0.0, (std_uv - ds.noisy_std_uv) / (ds.noisy_std_uv + 1e-8)) * 2.0
        score += max(0.0, z_std - ds.outlier_z_std) * 0.7
        score += clip_frac * 20.0
        score += max(0.0, (ln_ratio - ds.line_noise_ratio)) * 10.0

        rows.append({
            "channel": ch,
            "std_uV": std_uv,
            "z_std": z_std,
            "clip_fraction": clip_frac,
            "line_noise_ratio": ln_ratio,
            "reasons": ";".join(reasons),
            "score": score,
            "auto_bad": int(len(reasons) > 0),
        })

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    bad = df[df["auto_bad"] == 1]["channel"].tolist()
    bad = sorted(list(dict.fromkeys(bad)), key=natural_key)
    if len(bad) > ds.max_bad_suggestions:
        bad = bad[:ds.max_bad_suggestions]
    return df, bad


# ---------------------------
# Export cleaned files
# ---------------------------

def export_cleaned(raw_clean: mne.io.BaseRaw, out_base: str, output_format: str = "edf") -> Tuple[str, str]:
    """
    Export cleaned data.

    output_format:
      - "edf": try EDF first (requires pyedflib / MNE EDF export support). If EDF fails, fall back to FIF.
      - "fif": always export FIF.
    """
    raw_clean = raw_clean.copy().load_data()
    os.makedirs(os.path.dirname(out_base), exist_ok=True)

    fmt = (output_format or "edf").strip().lower()

    if fmt == "edf":
        try:
            out_edf = out_base + "_cleaned.edf"
            raw_clean.export(out_edf, fmt="edf", physical_range="auto", overwrite=True)
            return out_edf, "edf"
        except Exception:
            # Fall back to FIF for robustness
            pass

    out_fif = out_base + "_cleaned.fif"
    raw_clean.save(out_fif, overwrite=True)
    return out_fif, "fif"


# ---------------------------
# Optional PDF report
# ---------------------------

def export_subject_pdf_report(out_pdf: str, title: str, before_png: str, after_png: str, bad_channels: List[str]):
    if not REPORTLAB_OK:
        raise RuntimeError("reportlab not installed. Run: pip install reportlab")
    c = rl_canvas.Canvas(out_pdf, pagesize=letter)
    w, h = letter

    c.setFont("Helvetica-Bold", 14)
    c.drawString(36, h - 40, title)

    c.setFont("Helvetica", 10)
    bad_txt = ", ".join(bad_channels) if bad_channels else "(none)"
    c.drawString(36, h - 60, f"Bad channels ({len(bad_channels)}): " + (bad_txt[:160] + ("..." if len(bad_txt) > 160 else "")))

    y_top = h - 90
    img_w = (w - 36*2 - 12) / 2
    img_h = img_w * 0.56

    if os.path.exists(before_png):
        c.drawString(36, y_top, "BEFORE")
        c.drawImage(ImageReader(before_png), 36, y_top - img_h - 14, width=img_w, height=img_h, preserveAspectRatio=True, anchor='c')

    if os.path.exists(after_png):
        c.drawString(36 + img_w + 12, y_top, "AFTER")
        c.drawImage(ImageReader(after_png), 36 + img_w + 12, y_top - img_h - 14, width=img_w, height=img_h, preserveAspectRatio=True, anchor='c')

    c.setFont("Helvetica-Bold", 11)
    c.drawString(36, y_top - img_h - 40, "Bad channel list:")
    c.setFont("Helvetica", 9)

    max_chars = 105
    lines = [bad_txt[i:i+max_chars] for i in range(0, len(bad_txt), max_chars)]
    y = y_top - img_h - 56
    for ln in lines[:18]:
        c.drawString(36, y, ln)
        y -= 12

    c.showPage()
    c.save()


# ---------------------------
# GUI Application
# ---------------------------

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("iEEG Complete Automation v3 (EDF/FIF → Cleaned + Before/After + Tables)")
        root.geometry("1320x900")
        root.minsize(1150, 780)

        self._style()

        self.uiq: "queue.Queue[Tuple[str, object]]" = queue.Queue()

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()

        # Preprocess
        self.v_tmin = tk.StringVar(value="0")
        self.v_tmax = tk.StringVar(value="300")
        self.v_notch = tk.StringVar(value="")
        self.v_auto_notch = tk.BooleanVar(value=True)
        self.v_hp = tk.StringVar(value="")
        self.v_lp = tk.StringVar(value="")
        self.v_resample = tk.StringVar(value="")
        self.v_keep_ieeg = tk.BooleanVar(value=True)
        # Output file format for cleaned data
        # 'edf' is default (best for interoperability); 'fif' is always supported.
        self.v_output_format = tk.StringVar(value="edf")

        # Plot
        self.v_decim = tk.StringVar(value="4")
        self.v_dpi = tk.StringVar(value="300")
        self.v_figw = tk.StringVar(value="14")
        self.v_figh = tk.StringVar(value="9")
        self.v_lw = tk.StringVar(value="0.4")
        self.v_fs = tk.StringVar(value="9")
        self.v_maxlabels = tk.StringVar(value="220")

        # Detect
        self.v_flatstd = tk.StringVar(value="1.0")
        self.v_noisystd = tk.StringVar(value="500")
        self.v_outlierz = tk.StringVar(value="8")
        self.v_clipz = tk.StringVar(value="8")
        self.v_clipfrac = tk.StringVar(value="0.02")
        self.v_lineratio = tk.StringVar(value="0.35")
        self.v_maxbad = tk.StringVar(value="40")

        self.v_line_mode = tk.StringVar(value="auto")

        # Review
        self.v_search = tk.StringVar(value="")
        # Default to a clinician-friendly side-by-side preview
        self.preview_mode = tk.StringVar(value="before_after")

        # State
        self.stop_flag = False
        self.files: List[str] = []
        self.subjects: List[str] = []
        self.bad_by_subj: Dict[str, List[str]] = {}
        self.metrics_by_subj: Dict[str, pd.DataFrame] = {}
        self.paths_by_subj: Dict[str, Dict[str, str]] = {}
        self.sourcefile_by_subj: Dict[str, str] = {}
        self.lineinfo_by_subj: Dict[str, str] = {}
        self.start_time: Optional[float] = None

        # Widgets
        self.pre_drop_box = None
        self.nb = None
        self.tab_run = None
        self.tab_review = None
        self.log = None
        self.pb = None
        self.status = None
        self.btn_run = None
        self.btn_stop = None
        self.cmb = None
        self.tree = None
        self.add_entry = None
        self.info = None
        self.preview_label = None
        self.preview_imgtk = None

        self._build()
        self._poll_ui_queue()

    def _style(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except Exception:
            pass
        s.configure("TButton", padding=8)
        s.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))
        s.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        s.configure("TNotebook.Tab", padding=(14, 8))
        s.configure("Accent.TButton", padding=10)

    def _build(self):
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=12, pady=(10, 0))

        self.status = ttk.Label(top, text="Ready.", anchor="w")
        self.status.pack(side="left", fill="x", expand=True)

        ttk.Button(top, text="Open Output Folder", command=self.open_output_folder).pack(side="right", padx=6)
        ttk.Button(top, text="Help", command=self.show_help).pack(side="right", padx=6)

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=12, pady=12)

        self.tab_run = ttk.Frame(self.nb)
        self.tab_review = ttk.Frame(self.nb)
        self.nb.add(self.tab_run, text="1) Run Automation")
        self.nb.add(self.tab_review, text="2) Review & Export")

        self._build_run()
        self._build_review()

    def show_help(self):
        msg = (
            "Workflow:\n"
            "1) Select Input folder\n"
            "2) Select Output folder\n"
            "3) Run automation\n"
            "4) Review & edit bad channels\n"
            "5) Re-export cleaned + export FINAL tables\n\n"
            f"PNG preview in-app: {'Yes' if PIL_OK else 'No (install pillow)'}\n"
            f"PDF export: {'Yes' if REPORTLAB_OK else 'No (install reportlab)'}\n"
        )
        messagebox.showinfo("Help", msg)

    def _poll_ui_queue(self):
        try:
            while True:
                kind, payload = self.uiq.get_nowait()
                if kind == "log":
                    self.log.insert("end", str(payload) + "\n")
                    self.log.see("end")
                elif kind == "status":
                    self.status.config(text=str(payload))
                elif kind == "progress":
                    i, total = payload
                    self.pb["maximum"] = total
                    self.pb["value"] = i
                elif kind == "done":
                    self._set_running(False)
                    self.subjects = sorted(self.bad_by_subj.keys(), key=natural_key)
                    self.cmb["values"] = self.subjects
                    if self.subjects:
                        self.cmb.set(self.subjects[0])
                        self.refresh_list()
                        self.nb.select(self.tab_review)
        except queue.Empty:
            pass
        self.root.after(120, self._poll_ui_queue)

    def logp(self, s: str):
        self.uiq.put(("log", s))

    def set_status(self, s: str):
        self.uiq.put(("status", s))

    def _set_running(self, running: bool):
        self.btn_run.config(state="disabled" if running else "normal")
        self.btn_stop.config(state="normal" if running else "disabled")

    def pick_input(self):
        p = filedialog.askdirectory()
        if p:
            self.input_dir.set(p)
            if not self.output_dir.get().strip():
                self.output_dir.set(os.path.join(p, "IEEG_Auto_Output"))

    def pick_output(self):
        p = filedialog.askdirectory()
        if p:
            self.output_dir.set(p)

    def open_output_folder(self):
        out = self.output_dir.get().strip()
        if out:
            os.makedirs(out, exist_ok=True)
            open_path(out)

    def stop(self):
        self.stop_flag = True
        self.logp("🛑 Stop requested (will stop after current file).")
        self.set_status("Stopping after current file...")

    def _build_run(self):
        pad = {"padx": 8, "pady": 6}
        lf = ttk.LabelFrame(self.tab_run, text="Folders", style="Section.TLabelframe")
        lf.pack(fill="x", padx=10, pady=10)

        ttk.Label(lf, text="Input folder (EDF/FIF):", style="Header.TLabel").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(lf, textvariable=self.input_dir, width=90).grid(row=0, column=1, **pad)
        ttk.Button(lf, text="Browse", command=self.pick_input).grid(row=0, column=2, **pad)

        ttk.Label(lf, text="Output folder:", style="Header.TLabel").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(lf, textvariable=self.output_dir, width=90).grid(row=1, column=1, **pad)
        ttk.Button(lf, text="Browse", command=self.pick_output).grid(row=1, column=2, **pad)

        pre = ttk.LabelFrame(self.tab_run, text="Preprocessing", style="Section.TLabelframe")
        pre.pack(fill="x", padx=10, pady=10)

        ttk.Label(pre, text="tmin (s)").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(pre, textvariable=self.v_tmin, width=10).grid(row=0, column=1, **pad)
        ttk.Label(pre, text="tmax (s)").grid(row=0, column=2, sticky="w", **pad)
        ttk.Entry(pre, textvariable=self.v_tmax, width=10).grid(row=0, column=3, **pad)

        ttk.Label(pre, text="Manual Notch (Hz)").grid(row=0, column=4, sticky="w", **pad)
        ttk.Entry(pre, textvariable=self.v_notch, width=10).grid(row=0, column=5, **pad)

        ttk.Checkbutton(pre, text="Auto-notch = detected 50/60 (recommended)", variable=self.v_auto_notch).grid(
            row=0, column=6, columnspan=2, sticky="w", **pad
        )

        ttk.Label(pre, text="Bandpass HP (Hz)").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(pre, textvariable=self.v_hp, width=10).grid(row=1, column=1, **pad)
        ttk.Label(pre, text="Bandpass LP (Hz)").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(pre, textvariable=self.v_lp, width=10).grid(row=1, column=3, **pad)

        ttk.Label(pre, text="Resample (Hz)").grid(row=1, column=4, sticky="w", **pad)
        ttk.Entry(pre, textvariable=self.v_resample, width=10).grid(row=1, column=5, **pad)

        ttk.Checkbutton(pre, text="Keep only EEG/SEEG/ECoG (safe fallback)", variable=self.v_keep_ieeg).grid(
            row=1, column=6, sticky="w", **pad
        )
        # Cleaned output format (default EDF, user can switch to FIF)
        ttk.Label(pre, text="Cleaned output format").grid(row=2, column=0, sticky="w", **pad)
        fmtf = ttk.Frame(pre)
        fmtf.grid(row=2, column=1, columnspan=3, sticky="w", **pad)
        ttk.Radiobutton(fmtf, text="EDF (default)", value="edf", variable=self.v_output_format).pack(side="left", padx=6)
        ttk.Radiobutton(fmtf, text="FIF", value="fif", variable=self.v_output_format).pack(side="left", padx=6)
        ttk.Label(pre, text="(EDF needs pyedflib; if EDF export fails, app falls back to FIF)", foreground="#444").grid(
            row=2, column=4, columnspan=4, sticky="w", **pad
        )


        drop = ttk.LabelFrame(self.tab_run, text="Always remove channels whose name contains (optional)",
                              style="Section.TLabelframe")
        drop.pack(fill="x", padx=10, pady=10)
        self.pre_drop_box = tk.Text(drop, height=3, wrap="word")
        self.pre_drop_box.pack(fill="x", padx=10, pady=8)

        de = ttk.LabelFrame(self.tab_run, text="Detection", style="Section.TLabelframe")
        de.pack(fill="x", padx=10, pady=10)

        ttk.Label(de, text="Flat std < (µV)").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(de, textvariable=self.v_flatstd, width=10).grid(row=0, column=1, **pad)
        ttk.Label(de, text="Noisy std > (µV)").grid(row=0, column=2, sticky="w", **pad)
        ttk.Entry(de, textvariable=self.v_noisystd, width=10).grid(row=0, column=3, **pad)
        ttk.Label(de, text="Outlier z_std >").grid(row=0, column=4, sticky="w", **pad)
        ttk.Entry(de, textvariable=self.v_outlierz, width=10).grid(row=0, column=5, **pad)

        ttk.Label(de, text="Clip z").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(de, textvariable=self.v_clipz, width=10).grid(row=1, column=1, **pad)
        ttk.Label(de, text="Clip fraction >").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(de, textvariable=self.v_clipfrac, width=10).grid(row=1, column=3, **pad)
        ttk.Label(de, text="Line ratio >").grid(row=1, column=4, sticky="w", **pad)
        ttk.Entry(de, textvariable=self.v_lineratio, width=10).grid(row=1, column=5, **pad)
        ttk.Label(de, text="Max bad suggestions").grid(row=0, column=6, sticky="w", **pad)
        ttk.Entry(de, textvariable=self.v_maxbad, width=10).grid(row=0, column=7, **pad)

        ttk.Label(de, text="Line noise mode").grid(row=2, column=0, sticky="w", **pad)
        mf = ttk.Frame(de)
        mf.grid(row=2, column=1, columnspan=7, sticky="w", **pad)
        ttk.Radiobutton(mf, text="Auto", value="auto", variable=self.v_line_mode).pack(side="left", padx=6)
        ttk.Radiobutton(mf, text="Force 50 Hz", value="50", variable=self.v_line_mode).pack(side="left", padx=6)
        ttk.Radiobutton(mf, text="Force 60 Hz", value="60", variable=self.v_line_mode).pack(side="left", padx=6)

        ctrl = ttk.Frame(self.tab_run)
        ctrl.pack(fill="x", padx=10, pady=10)

        self.btn_run = ttk.Button(ctrl, text="▶ Run Full Automation", style="Accent.TButton", command=self.run_batch)
        self.btn_run.pack(side="left", padx=6)
        self.btn_stop = ttk.Button(ctrl, text="■ Stop", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=6)

        self.pb = ttk.Progressbar(ctrl, mode="determinate")
        self.pb.pack(side="left", fill="x", expand=True, padx=10)

        logf = ttk.LabelFrame(self.tab_run, text="Log", style="Section.TLabelframe")
        logf.pack(fill="both", expand=True, padx=10, pady=10)
        self.log = tk.Text(logf, height=14, wrap="word")
        self.log.pack(fill="both", expand=True, padx=10, pady=10)

    def _build_review(self):
        top = ttk.Frame(self.tab_review)
        top.pack(fill="x", padx=10, pady=10)

        ttk.Label(top, text="Subject:", style="Header.TLabel").pack(side="left", padx=6)
        self.cmb = ttk.Combobox(top, values=[], state="readonly", width=50)
        self.cmb.pack(side="left", padx=6)
        self.cmb.bind("<<ComboboxSelected>>", lambda e: self.refresh_list())

        ttk.Label(top, text="Search:", style="Header.TLabel").pack(side="left", padx=(18, 6))
        ent_search = ttk.Entry(top, textvariable=self.v_search, width=28)
        ent_search.pack(side="left", padx=6)
        ent_search.bind("<KeyRelease>", lambda e: self.refresh_list())

        ttk.Button(top, text="Re-export CLEANED (edited list)", command=self.reexport_cleaned_for_all).pack(side="right", padx=6)
        ttk.Button(top, text="Export FINAL CSV + Excel", command=self.export_final_tables).pack(side="right", padx=6)
        ttk.Button(top, text="Export PDF (selected)", command=self.export_pdf_selected).pack(side="right", padx=6)

        mid = ttk.PanedWindow(self.tab_review, orient=tk.HORIZONTAL)
        mid.pack(fill="both", expand=True, padx=10, pady=10)

        left = ttk.Labelframe(mid, text="Bad channels (editable)")
        right = ttk.Labelframe(mid, text="Preview & Info")
        mid.add(left, weight=3)
        mid.add(right, weight=4)

        self.tree = ttk.Treeview(left, columns=("ch", "reason"), show="headings", height=18)
        self.tree.heading("ch", text="Channel")
        self.tree.heading("reason", text="Reason(s)")
        self.tree.column("ch", width=180, anchor="w")
        self.tree.column("reason", width=420, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)

        addrow = ttk.Frame(left)
        addrow.pack(fill="x", padx=10, pady=8)

        self.add_entry = ttk.Entry(addrow, width=24)
        self.add_entry.pack(side="left", padx=6)
        ttk.Button(addrow, text="Add", command=self.add_channel).pack(side="left", padx=6)
        ttk.Button(addrow, text="Remove selected", command=self.remove_selected).pack(side="left", padx=6)
        ttk.Button(addrow, text="Open COMPARE PNG", command=lambda: self.open_asset("compare_png")).pack(side="left", padx=6)

        right_top = ttk.Frame(right)
        right_top.pack(fill="x", padx=10, pady=8)

        ttk.Label(right_top, text="Preview:", style="Header.TLabel").pack(side="left")
        ttk.Radiobutton(right_top, text="Before+After", value="before_after", variable=self.preview_mode,
                        command=self.refresh_preview).pack(side="left", padx=8)
        ttk.Radiobutton(right_top, text="Compare", value="compare", variable=self.preview_mode,
                        command=self.refresh_preview).pack(side="left", padx=8)
        ttk.Radiobutton(right_top, text="Before", value="before", variable=self.preview_mode,
                        command=self.refresh_preview).pack(side="left", padx=8)
        ttk.Radiobutton(right_top, text="After", value="after", variable=self.preview_mode,
                        command=self.refresh_preview).pack(side="left", padx=8)

        ttk.Button(right_top, text="Open BEFORE", command=lambda: self.open_asset("before_png")).pack(side="right", padx=6)
        ttk.Button(right_top, text="Open AFTER", command=lambda: self.open_asset("after_png")).pack(side="right", padx=6)
        ttk.Button(right_top, text="Open Output", command=self.open_output_folder).pack(side="right", padx=6)

        self.preview_label = ttk.Label(right, text=("PNG preview requires pillow.\nInstall: pip install pillow") if not PIL_OK else "No subject selected yet.",
                                       anchor="center", justify="center")
        self.preview_label.pack(fill="both", expand=True, padx=10, pady=10)

        self.info = tk.Text(right, height=10, wrap="word")
        self.info.pack(fill="x", padx=10, pady=(0, 10))
        self.info.insert("1.0", "Run automation then select a subject.\n")

    def collect_settings(self) -> Tuple[PreprocessSettings, PlotSettings, DetectSettings]:
        ps = PreprocessSettings(
            tmin=float(self.v_tmin.get().strip() or "0"),
            tmax=float(self.v_tmax.get().strip() or "300"),
            notch_hz=safe_float(self.v_notch.get()),
            hp_hz=safe_float(self.v_hp.get()),
            lp_hz=safe_float(self.v_lp.get()),
            resample_hz=safe_float(self.v_resample.get()),
            keep_only_ieeg=bool(self.v_keep_ieeg.get()),
            pre_drop_patterns=parse_multiline_list(self.pre_drop_box.get("1.0", "end").strip()),
        )
        plot = PlotSettings(
            decim=max(1, safe_int(self.v_decim.get(), 4)),
            dpi=max(72, safe_int(self.v_dpi.get(), 300)),
            fig_w=float(self.v_figw.get().strip() or "14"),
            fig_h=float(self.v_figh.get().strip() or "9"),
            linewidth=float(self.v_lw.get().strip() or "0.4"),
            label_fontsize=max(6, safe_int(self.v_fs.get(), 9)),
            max_labels=max(20, safe_int(self.v_maxlabels.get(), 220)),
        )
        ds = DetectSettings(
            flat_std_uv=float(self.v_flatstd.get().strip() or "1.0"),
            noisy_std_uv=float(self.v_noisystd.get().strip() or "500"),
            outlier_z_std=float(self.v_outlierz.get().strip() or "8"),
            clip_z=float(self.v_clipz.get().strip() or "8"),
            clip_fraction_thr=float(self.v_clipfrac.get().strip() or "0.02"),
            line_noise_hz=60.0,
            line_noise_ratio=float(self.v_lineratio.get().strip() or "0.35"),
            max_bad_suggestions=max(5, safe_int(self.v_maxbad.get(), 40)),
        )
        return ps, plot, ds

    def _choose_line_hz_for_file(self, raw: mne.io.BaseRaw, ps: PreprocessSettings) -> Tuple[float, Optional[Tuple[float, float]]]:
        mode = (self.v_line_mode.get() or "auto").strip().lower()
        if mode == "50":
            return 50.0, None
        if mode == "60":
            return 60.0, None
        chosen_hz, r50, r60 = auto_detect_line_noise_hz(raw, ps.tmin, ps.tmax)
        return chosen_hz, (r50, r60)

    def run_batch(self):
        inp = self.input_dir.get().strip()
        out = self.output_dir.get().strip()

        if not inp or not os.path.isdir(inp):
            messagebox.showerror("Input missing", "Please select a valid input folder.")
            return
        if not out:
            out = os.path.join(inp, "IEEG_Auto_Output")
            self.output_dir.set(out)
        os.makedirs(out, exist_ok=True)

        files = list_data_files(inp)
        if not files:
            messagebox.showinfo("No files", "No EDF/FIF files found in the input folder.")
            return

        ps, plot, ds = self.collect_settings()
        output_format = (self.v_output_format.get() or "edf").strip().lower()
        auto_notch = bool(self.v_auto_notch.get())

        self.stop_flag = False
        self.files = files
        self.bad_by_subj.clear()
        self.metrics_by_subj.clear()
        self.paths_by_subj.clear()
        self.sourcefile_by_subj.clear()
        self.lineinfo_by_subj.clear()
        self.start_time = time.time()

        self.pb["value"] = 0
        self.pb["maximum"] = len(files)
        self._set_running(True)
        self.set_status(f"Running... (0/{len(files)})")
        self.logp(f"📂 Input: {inp}")
        self.logp(f"📁 Output: {out}")
        self.logp(f"✅ Files: {len(files)}")

        def worker():
            for i, fp in enumerate(files, start=1):
                if self.stop_flag:
                    break
                sid = subject_id_from_filename(fp)
                self.sourcefile_by_subj[sid] = fp
                base = strip_ext(os.path.basename(fp))
                out_base = os.path.join(out, base)

                try:
                    self.set_status(f"Processing {sid} ({i}/{len(files)})")
                    raw = load_raw_any(fp, preload=True)
                    if ps.keep_only_ieeg:
                        raw = pick_ieeg_types_safe(raw)
                    drop_by_patterns(raw, ps.pre_drop_patterns)
                    apply_filters(raw, ps)

                    chosen_hz, ratios = self._choose_line_hz_for_file(raw, ps)
                    ds.line_noise_hz = chosen_hz
                    if ratios is not None:
                        r50, r60 = ratios
                        self.lineinfo_by_subj[sid] = f"Line noise AUTO → {int(chosen_hz)} Hz (50Hz={r50:.3f}, 60Hz={r60:.3f})"
                    else:
                        self.lineinfo_by_subj[sid] = f"Line noise MANUAL → {int(chosen_hz)} Hz"
                    self.logp(f"{sid}: {self.lineinfo_by_subj[sid]}")

                    if auto_notch:
                        try:
                            raw.notch_filter(chosen_hz, verbose=False)
                        except Exception:
                            pass

                    before_png = out_base + "__BEFORE_stacked.png"
                    save_stacked_png(raw, before_png, ps, plot, title=f"{sid} (BEFORE)")

                    df_metrics, bad = detect_bad_channels(raw, ps, ds)
                    df_metrics.insert(0, "subject_id", sid)
                    df_metrics.insert(1, "file", os.path.basename(fp))
                    df_metrics["line_noise_used_hz"] = ds.line_noise_hz
                    df_metrics["line_noise_mode"] = self.v_line_mode.get()

                    raw_clean = raw.copy()
                    bad2 = [c for c in bad if c in raw_clean.ch_names]
                    if bad2:
                        raw_clean.drop_channels(bad2)
                        bad = bad2

                    after_png = out_base + "__AFTER_stacked.png"
                    save_stacked_png(raw_clean, after_png, ps, plot, title=f"{sid} (AFTER)")

                    comp_png = out_base + "__COMPARE_before_after.png"
                    save_comparison_png(before_png, after_png, comp_png, title=f"{sid} (Before vs After)")

                    saved_path, fmt = export_cleaned(raw_clean, out_base, output_format=output_format)

                    self.bad_by_subj[sid] = sorted(bad, key=natural_key)
                    self.metrics_by_subj[sid] = df_metrics
                    self.paths_by_subj[sid] = {
                        "before_png": before_png,
                        "after_png": after_png,
                        "compare_png": comp_png,
                        "clean_file": saved_path,
                        "clean_fmt": fmt,
                        "out_base": out_base,
                    }

                    self.logp(f"[{i}/{len(files)}] {sid} ✅ bad={len(bad)} clean={os.path.basename(saved_path)} ({fmt})")
                except Exception as e:
                    self.logp(f"[{i}/{len(files)}] {sid} ❌ {e}")

                self.uiq.put(("progress", (i, len(files))))
                self.set_status(f"Running... ({i}/{len(files)})")

            try:
                self.export_auto_tables()
                self.logp("✅ Exported AUTO tables: bad_channels_AUTO.(csv/xlsx)")
            except Exception as e:
                self.logp(f"⚠️ AUTO table export failed: {e}")

            self.set_status("Done.")
            self.uiq.put(("done", None))

        threading.Thread(target=worker, daemon=True).start()

    def refresh_list(self):
        sid = (self.cmb.get() or "").strip()
        self.tree.delete(*self.tree.get_children())
        self.info.delete("1.0", "end")

        if not sid or sid not in self.bad_by_subj:
            self.refresh_preview()
            return

        bad = self.bad_by_subj.get(sid, [])
        q = (self.v_search.get() or "").strip().lower()

        reasons_map = {}
        dfm = self.metrics_by_subj.get(sid)
        if isinstance(dfm, pd.DataFrame) and not dfm.empty and "channel" in dfm.columns:
            for _, r in dfm.iterrows():
                reasons_map[str(r["channel"])] = str(r.get("reasons", ""))

        for ch in bad:
            if q and q not in ch.lower():
                continue
            self.tree.insert("", "end", values=(ch, reasons_map.get(ch, "")))

        p = self.paths_by_subj.get(sid, {})
        self.info.insert("end", f"Subject: {sid}\n\n")
        self.info.insert("end", f"{self.lineinfo_by_subj.get(sid, '')}\n\n")
        self.info.insert("end", f"CLEAN FILE:\n{p.get('clean_file','')} ({p.get('clean_fmt','')})\n\n")
        self.info.insert("end", f"COMPARE PNG:\n{p.get('compare_png','')}\n\n")
        self.refresh_preview()

    def refresh_preview(self):
        if self.preview_label is None:
            return
        sid = (self.cmb.get() or "").strip()
        p = self.paths_by_subj.get(sid, {})
        mode = self.preview_mode.get()

        if not sid or not p:
            if PIL_OK:
                self.preview_label.config(text="No subject selected yet.", image="")
            return

        if not PIL_OK:
            self.preview_label.config(text="Install pillow for in-app preview.\nUse Open BEFORE/AFTER/COMPARE.", image="")
            return

        # Determine which image to show in-app.
        # - before_after: create a temporary side-by-side preview from BEFORE and AFTER
        # - compare/before/after: show corresponding PNG
        try:
            w = max(520, self.preview_label.winfo_width())
            h = max(360, self.preview_label.winfo_height())

            if mode == "before_after":
                before_path = p.get("before_png", "")
                after_path = p.get("after_png", "")
                if (not before_path) or (not after_path) or (not os.path.exists(before_path)) or (not os.path.exists(after_path)):
                    self.preview_label.config(text="Preview images missing (BEFORE/AFTER).", image="")
                    return

                im_b = Image.open(before_path).convert("RGB")
                im_a = Image.open(after_path).convert("RGB")

                # Resize both to the same height, then concatenate.
                target_h = max(300, int(h * 0.95))
                def _resize_to_h(im):
                    ratio = target_h / float(im.height)
                    return im.resize((max(1, int(im.width * ratio)), target_h))
                im_b = _resize_to_h(im_b)
                im_a = _resize_to_h(im_a)

                gap = 10
                combo = Image.new("RGB", (im_b.width + gap + im_a.width, target_h), (245, 245, 245))
                combo.paste(im_b, (0, 0))
                combo.paste(im_a, (im_b.width + gap, 0))
                combo.thumbnail((w, h))

                self.preview_imgtk = ImageTk.PhotoImage(combo)
                self.preview_label.config(image=self.preview_imgtk, text="")
                return

            key = "compare_png" if mode == "compare" else ("before_png" if mode == "before" else "after_png")
            img_path = p.get(key, "")
            if not img_path or not os.path.exists(img_path):
                self.preview_label.config(text="Preview image missing.", image="")
                return

            im = Image.open(img_path)
            im.thumbnail((w, h))
            self.preview_imgtk = ImageTk.PhotoImage(im)
            self.preview_label.config(image=self.preview_imgtk, text="")
        except Exception:
            self.preview_label.config(text="Preview failed. Use Open BEFORE/AFTER/COMPARE.", image="")

    def add_channel(self):
        sid = (self.cmb.get() or "").strip()
        ch = (self.add_entry.get() or "").strip()
        if not sid or sid not in self.bad_by_subj or not ch:
            return
        if ch not in self.bad_by_subj[sid]:
            self.bad_by_subj[sid].append(ch)
            self.bad_by_subj[sid] = sorted(self.bad_by_subj[sid], key=natural_key)
        self.add_entry.delete(0, "end")
        self.refresh_list()

    def remove_selected(self):
        sid = (self.cmb.get() or "").strip()
        if not sid or sid not in self.bad_by_subj:
            return
        sel = self.tree.selection()
        if not sel:
            return
        rm = set()
        for it in sel:
            vals = self.tree.item(it, "values")
            if vals:
                rm.add(vals[0])
        self.bad_by_subj[sid] = [c for c in self.bad_by_subj[sid] if c not in rm]
        self.refresh_list()

    def open_asset(self, key: str):
        sid = (self.cmb.get() or "").strip()
        p = self.paths_by_subj.get(sid, {})
        open_path(p.get(key, ""))

    def export_auto_tables(self):
        out = self.output_dir.get().strip()
        os.makedirs(out, exist_ok=True)

        rows = []
        for sid, chs in self.bad_by_subj.items():
            for ch in chs:
                rows.append({"subject_id": sid, "bad_channel": ch, "source": "auto"})
        df_bad = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["subject_id", "bad_channel", "source"])
        if not df_bad.empty:
            df_bad = df_bad.sort_values(["subject_id", "bad_channel"], kind="stable")

        df_metrics = pd.concat(list(self.metrics_by_subj.values()), ignore_index=True) if self.metrics_by_subj else pd.DataFrame()

        csv1 = os.path.join(out, "bad_channels_AUTO.csv")
        xls1 = os.path.join(out, "bad_channels_AUTO.xlsx")

        df_bad.to_csv(csv1, index=False)
        with pd.ExcelWriter(xls1, engine="openpyxl") as w:
            df_bad.to_excel(w, sheet_name="auto_bad_channels", index=False)
            if not df_metrics.empty:
                df_metrics.to_excel(w, sheet_name="channel_metrics", index=False)

    def export_final_tables(self):
        out = self.output_dir.get().strip()
        os.makedirs(out, exist_ok=True)

        rows = []
        for sid, chs in self.bad_by_subj.items():
            for ch in chs:
                rows.append({"subject_id": sid, "bad_channel": ch, "source": "final_review"})
        df_final = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["subject_id", "bad_channel", "source"])
        if not df_final.empty:
            df_final = df_final.sort_values(["subject_id", "bad_channel"], kind="stable")

        df_metrics = pd.concat(list(self.metrics_by_subj.values()), ignore_index=True) if self.metrics_by_subj else pd.DataFrame()

        csv1 = os.path.join(out, "bad_channels_FINAL.csv")
        xls1 = os.path.join(out, "bad_channels_FINAL.xlsx")

        df_final.to_csv(csv1, index=False)
        with pd.ExcelWriter(xls1, engine="openpyxl") as w:
            df_final.to_excel(w, sheet_name="final_bad_channels", index=False)
            if not df_metrics.empty:
                df_metrics.to_excel(w, sheet_name="channel_metrics", index=False)

        messagebox.showinfo("Exported", f"Saved:\n{csv1}\n{xls1}")

    def export_pdf_selected(self):
        sid = (self.cmb.get() or "").strip()
        if not sid or sid not in self.paths_by_subj:
            messagebox.showwarning("Select subject", "Please select a subject first.")
            return
        if not REPORTLAB_OK:
            messagebox.showwarning("Missing dependency", "PDF export needs reportlab.\nInstall: pip install reportlab")
            return

        out = self.output_dir.get().strip()
        os.makedirs(out, exist_ok=True)
        p = self.paths_by_subj[sid]
        out_pdf = os.path.join(out, f"{sid}__report.pdf")

        export_subject_pdf_report(out_pdf, f"{sid} – iEEG Cleaning Report",
                                  p.get("before_png",""), p.get("after_png",""),
                                  self.bad_by_subj.get(sid, []))
        messagebox.showinfo("PDF saved", f"Saved:\n{out_pdf}")
        open_path(out_pdf)

    def reexport_cleaned_for_all(self):
        out = self.output_dir.get().strip()
        if not out:
            messagebox.showerror("Missing", "Output folder is missing.")
            return
        if not self.bad_by_subj:
            messagebox.showwarning("Nothing", "Run automation first.")
            return

        ps, plot, ds = self.collect_settings()
        output_format = (self.v_output_format.get() or "edf").strip().lower()
        auto_notch = bool(self.v_auto_notch.get())

        self.stop_flag = False
        self._set_running(True)
        self.set_status("Re-exporting cleaned files (edited lists)...")
        self.logp("🔁 Re-exporting cleaned files using edited bad-channel lists...")

        def worker():
            subs = sorted(self.bad_by_subj.keys(), key=natural_key)
            for i, sid in enumerate(subs, start=1):
                if self.stop_flag:
                    break
                fp = self.sourcefile_by_subj.get(sid)
                if not fp or not os.path.exists(fp):
                    self.logp(f"{sid}: source file missing -> skipped")
                    continue

                base = strip_ext(os.path.basename(fp))
                out_base = os.path.join(out, base)

                try:
                    raw = load_raw_any(fp, preload=True)
                    if ps.keep_only_ieeg:
                        raw = pick_ieeg_types_safe(raw)
                    drop_by_patterns(raw, ps.pre_drop_patterns)
                    apply_filters(raw, ps)

                    chosen_hz, _ = self._choose_line_hz_for_file(raw, ps)
                    ds.line_noise_hz = chosen_hz
                    if auto_notch:
                        try:
                            raw.notch_filter(chosen_hz, verbose=False)
                        except Exception:
                            pass

                    before_png = out_base + "__BEFORE_stacked.png"
                    if not os.path.exists(before_png):
                        save_stacked_png(raw, before_png, ps, plot, title=f"{sid} (BEFORE)")

                    bad = self.bad_by_subj[sid]
                    raw_clean = raw.copy()
                    bad2 = [c for c in bad if c in raw_clean.ch_names]
                    if bad2:
                        raw_clean.drop_channels(bad2)

                    after_png = out_base + "__AFTER_stacked.png"
                    save_stacked_png(raw_clean, after_png, ps, plot, title=f"{sid} (AFTER)")

                    comp_png = out_base + "__COMPARE_before_after.png"
                    save_comparison_png(before_png, after_png, comp_png, title=f"{sid} (Before vs After)")

                    saved_path, fmt = export_cleaned(raw_clean, out_base, output_format=output_format)
                    self.paths_by_subj[sid] = {
                        "before_png": before_png,
                        "after_png": after_png,
                        "compare_png": comp_png,
                        "clean_file": saved_path,
                        "clean_fmt": fmt,
                        "out_base": out_base,
                    }
                    self.logp(f"{sid}: ✅ re-exported clean={os.path.basename(saved_path)} ({fmt})")
                except Exception as e:
                    self.logp(f"{sid}: ❌ {e}")

                self.uiq.put(("progress", (i, len(subs))))

            self.set_status("Re-export completed.")
            self.uiq.put(("done", None))

        threading.Thread(target=worker, daemon=True).start()


def main():
    mne.set_log_level("WARNING")
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
