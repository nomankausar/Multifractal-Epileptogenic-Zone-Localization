import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, freeze_support
from scipy.stats import ks_2samp

# === PARAMETERS ===
data_dir           = r"D:\Github For Epliepsy Project\Python\data files\child 90\Data"
soz_csv            = os.path.join(data_dir, "SOZ_Channels_info.csv")
seconds_to_analyze = None
q_values           = np.linspace(-20.0, 20.0,401)
scale_min, scale_max = 16, 4096

# [ADDED] Keep the old code and comments, but allow a renamed SOZ CSV such as
# SOZ_Channels_info(1).csv to be found automatically when the exact name is absent.
if not os.path.isfile(soz_csv):
    soz_candidates = sorted(
        glob.glob(os.path.join(data_dir, "SOZ_Channels_info*.csv"))
    )
    script_candidate = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "SOZ_Channels_info.csv"
    )
    if os.path.isfile(script_candidate):
        soz_candidates.insert(0, script_candidate)
    if soz_candidates:
        soz_csv = soz_candidates[0]
    else:
        raise FileNotFoundError(
            "SOZ CSV was not found. Put SOZ_Channels_info.csv inside:\n"
            f"{data_dir}"
        )

# [ADDED] The FIF filename-convention message is harmless, so hide only that warning.
warnings.filterwarnings(
    "ignore",
    message=r"This filename .* does not conform to MNE naming conventions.*",
    category=RuntimeWarning,
)

# === LOAD SOZ INFO AND CLEAN ===
soz_df = pd.read_csv(soz_csv)
soz_long = (
    soz_df
    .melt(id_vars="Subject", var_name="Channel_Index", value_name="Channel")
    .dropna(subset=["Channel"])
)
soz_long["Subject_ID"] = soz_long["Subject"].astype(str)\
    .apply(lambda s: re.sub(r'[^A-Za-z0-9]', '', s).upper())
soz_long["Clean_Channel"] = soz_long["Channel"].astype(str)\
    .apply(lambda s: re.sub(r'[^A-Za-z0-9]', '', s).upper())

subject_soz_map = (
    soz_long
    .groupby("Subject_ID")["Clean_Channel"]
    .apply(list)
    .to_dict()
)

# === HELPERS ===
def extract_subject_id(fname):
    m = re.search(r"sub-([A-Za-z0-9]+)_", fname)
    return m.group(1).upper() if m else None

def clean_channel_name(name):
    return re.sub(r'[^A-Za-z0-9]', '', str(name)).upper()

# [ADDED] Convert a contact such as A01 to A1 while keeping electrode letters.
def normalize_contact(token):
    token = re.sub(r'[^A-Za-z0-9]', '', str(token)).upper()
    match = re.fullmatch(r'([A-Z]+)0*(\d+)', token)
    if not match:
        return None
    letters, number = match.groups()
    return f"{letters}{int(number)}"

# [ADDED] Extract exact electrode contacts. This avoids A1 matching A10.
def extract_contact_tokens(name):
    text = str(name).upper()
    raw_tokens = re.findall(r'[A-Z]+\s*0*\d+', text)
    contacts = set()

    for raw_token in raw_tokens:
        token = normalize_contact(raw_token)
        if not token:
            continue
        contacts.add(token)

        # [ADDED] Handle cleaned labels such as POLG14 or EEGLAH01.
        for prefix in ('POL', 'EEG'):
            if token.startswith(prefix):
                stripped = normalize_contact(token[len(prefix):])
                if stripped:
                    contacts.add(stripped)

    return contacts

def fuzzy_match(ch, soz_list):
    # [ADDED] Exact contact comparison replaces unsafe substring comparison.
    ch_contacts = extract_contact_tokens(ch)
    soz_contacts = set()
    for soz in soz_list:
        soz_contacts.update(extract_contact_tokens(soz))
    return bool(ch_contacts & soz_contacts)

# [ADDED] Allow UCLA02/UCLA2 or Detroit006/Detroit6 fallback matching.
def canonical_subject_id(subject):
    subject = clean_channel_name(subject)
    return re.sub(r'\d+', lambda m: str(int(m.group(0))), subject)

# [ADDED] Find the correct SOZ subject key without changing the original map.
def find_subject_soz_list(subject):
    if subject in subject_soz_map:
        return subject_soz_map[subject]

    canonical = canonical_subject_id(subject)
    matches = [
        values
        for key, values in subject_soz_map.items()
        if canonical_subject_id(key) == canonical
    ]
    return matches[0] if len(matches) == 1 else []

def MFDFA(signal, scale_min, scale_max, q_vals):
    x = np.cumsum(signal - np.mean(signal))

    # [ADDED] Do not use a scale larger than half of the available signal.
    usable_scale_max = min(scale_max, max(scale_min, len(x)//2))
    scales = 2 ** np.arange(int(np.log2(scale_min)),
                           int(np.log2(usable_scale_max))+1)
    # instantiation variables
    fluct = np.zeros((len(scales), len(q_vals)))
    Hq    = np.zeros(len(q_vals))

    # find fluctuations across all scales
    for si, s in enumerate(scales):
        segs = len(x)//s # length of the segment
        F_s  = [] # fluctuations
        for i in range(segs): # iterate over all segments
            seg = x[i*s:(i+1)*s] # extract each data segment
            t   = np.arange(s) # array from 0 to s-1
            cfs = np.polyfit(t, seg, 1)
            trend = np.polyval(cfs, t)
            F_s.append(np.sqrt(np.mean((seg - trend)**2))) # compute fluctuation and append
        F_s = np.array(F_s) # convert to a numpy array
        F_s = F_s[F_s>1e-8] # check if fluctuation is greater thn 1e-8 and replace with 0

        # [ADDED] Mark an unusable scale instead of allowing invalid math.
        if len(F_s) == 0:
            fluct[si, :] = np.nan
            continue

        for qi, q in enumerate(q_vals):
            if q==0:
                fluct[si,qi] = np.exp(0.5*np.mean(np.log(F_s**2)))
            else:
                fluct[si,qi] = np.mean(F_s**q)**(1.0/q)

    log_sc = np.log2(scales)
    for qi in range(len(q_vals)):
        log_F = np.log2(fluct[:,qi])

        # [ADDED] Fit only finite scale values.
        valid = np.isfinite(log_sc) & np.isfinite(log_F)
        Hq[qi] = np.polyfit(log_sc[valid], log_F[valid],1)[0] \
            if np.count_nonzero(valid) >= 2 else np.nan

    return Hq

def process_channel(args):
    data, ch_name, soz_list = args
    # [MODIFIED] Use the raw channel label so POL G14 can match G14 exactly.
    label = 'EZ' if fuzzy_match(ch_name, soz_list) else 'Non-EZ'
    hq    = MFDFA(data, scale_min, scale_max,q_values)
    return label, hq

def process_file(filepath, soz_list, max_dur=None):
    ext = filepath.lower().split('.')[-1]
    reader = mne.io.read_raw_edf if ext=='edf' else mne.io.read_raw_fif
    raw = reader(filepath, preload=True, verbose=False)
    if max_dur:
        raw.crop(0, max_dur)
    raw.load_data()
    # raw.notch_filter(60)
    eeg, _ = raw.get_data(return_times=True)
    chs    = raw.info['ch_names']
    args   = [(eeg[i], chs[i], soz_list) for i in range(len(chs))]

    # [ADDED] Limit worker count to reduce memory use with large FIF files.
    worker_count = min(8, cpu_count(), len(args))
    with Pool(worker_count) as p:
        return p.map(process_channel, args)

def main():
    all_results = []  # New: to store all H(q) values across subjects

    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(('.edf', '.fif')):
            continue

        sub = extract_subject_id(fname)
        if not sub:
            continue

        # [MODIFIED] Use exact matching first and zero-insensitive matching second.
        soz_list = find_subject_soz_list(sub)
        if not soz_list:
            print(f"WARNING: No SOZ information found for {sub}; skipped.")
            continue

        fp      = os.path.join(data_dir, fname)
        results = process_file(fp, soz_list, seconds_to_analyze)
        base    = os.path.splitext(fname)[0]

        # split EZ vs Non‑EZ
        ez_hqs  = [hq for label, hq in results if label=='EZ']
        non_hqs = [hq for label, hq in results if label=='Non-EZ']

        # [ADDED] Never call np.vstack on an empty list.
        EZ      = np.vstack(ez_hqs) if ez_hqs else None
        NON_EZ  = np.vstack(non_hqs) if non_hqs else None

        if EZ is None:
            print(
                f"WARNING: {sub} has zero matched EZ channels. "
                "Check the SOZ CSV and recording channel names."
            )
        if NON_EZ is None:
            print(f"WARNING: {sub} has zero Non-EZ channels.")

        # compute mean curves
        mean_ez   = np.nanmean(EZ, axis=0) if EZ is not None else None
        mean_non  = np.nanmean(NON_EZ, axis=0) if NON_EZ is not None else None

        # [ADDED] Standard deviations are used for the new shaded plot.
        std_ez    = np.nanstd(EZ, axis=0) if EZ is not None else None
        std_non   = np.nanstd(NON_EZ, axis=0) if NON_EZ is not None else None

        # save the mean-H(q) CSV
        mean_data = []
        mean_index = []
        if mean_non is not None:
            mean_data.append(mean_non)
            mean_index.append('Non-EZ')
        if mean_ez is not None:
            mean_data.append(mean_ez)
            mean_index.append('EZ')

        if mean_data:
            df_mean = pd.DataFrame(
                mean_data,
                index=mean_index,
                columns=[f"{q:.1f}" for q in q_values]
            )
            df_mean.index.name = 'Group'
            df_mean.to_csv(os.path.join(data_dir, f"{base}_mean_hurst.csv"))
            print(f"→ Saved mean-H(q) to {base}_mean_hurst.csv")

        # === PLOT ALL CHANNELS INDIVIDUALLY ===
        plt.figure(figsize=(10, 6))

        # Plot Non-EZ channels in blue
        for hq in non_hqs:
            plt.plot(q_values, hq, color='blue', alpha=0.3, linewidth=1.0)

        # Plot EZ channels in red
        for hq in ez_hqs:
            plt.plot(q_values, hq, color='red', alpha=0.6, linewidth=1.5)

        # [ADDED] Keep the previous plot and add its mean curves when available.
        if mean_non is not None:
            plt.plot(q_values, mean_non, color='blue', linewidth=2.5,
                     label=f'Non-EZ mean (n={len(non_hqs)})')
        if mean_ez is not None:
            plt.plot(q_values, mean_ez, color='red', linewidth=2.5,
                     label=f'EZ mean (n={len(ez_hqs)})')

        # Labels and Title
        plt.xlabel('q‑order')
        plt.ylabel('H(q)')
        plt.title(f'MFDFA H(q) Curves — {sub}')
        plt.grid('--', lw=0.5)
        if mean_non is not None or mean_ez is not None:
            plt.legend(frameon=False)
        plt.tight_layout()

        # Save Plot
        plt.savefig(os.path.join(data_dir, f"{base}_all_channels_hurst.png"), dpi=300)
        plt.close()

        # [ADDED] NEW SHADED PLOT: mean ± standard deviation, saved at 300 dpi.
        with plt.rc_context({
            'font.family': 'serif',
            'axes.titlesize': 34,
            'axes.labelsize': 38,
            'xtick.labelsize': 29,
            'ytick.labelsize': 29,
            'legend.fontsize': 32,
        }):
            fig, ax = plt.subplots(figsize=(11, 8))

            if mean_non is not None:
                ax.fill_between(
                    q_values,
                    mean_non - std_non,
                    mean_non + std_non,
                    color='royalblue',
                    alpha=0.22
                )
                ax.plot(
                    q_values,
                    mean_non,
                    color='royalblue',
                    linewidth=4.5,
                    label='Non-EZ'
                )

            if mean_ez is not None:
                ax.fill_between(
                    q_values,
                    mean_ez - std_ez,
                    mean_ez + std_ez,
                    color='firebrick',
                    alpha=0.22
                )
                ax.plot(
                    q_values,
                    mean_ez,
                    color='firebrick',
                    linewidth=4.5,
                    label='EZ'
                )

            ax.set_xlabel('q-order', labelpad=15)
            ax.set_ylabel(r'$H(q)$', labelpad=12)
            ax.set_title(f'Subject: {sub}', pad=15)
            ax.set_xticks([-20, 0, 20])
            ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.35)
            ax.legend(loc='upper right', frameon=True, fancybox=True,
                      borderpad=0.7, handlelength=2.4)
            fig.tight_layout()
            fig.savefig(
                os.path.join(data_dir, f"{base}_shaded_mean_std_hurst.png"),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close(fig)

        # === NEW: Collect data for Excel output ===
        chs = mne.io.read_raw_fif(fp, verbose=False).info['ch_names'] if fp.endswith('.fif') \
              else mne.io.read_raw_edf(fp, verbose=False).info['ch_names']
        for (label, hq), ch in zip(results, chs):
            clean_ch = clean_channel_name(ch)
            all_results.append({
                'Subject_ID': sub,
                'Channel': ch,
                # [MODIFIED] Use the same exact contact matcher used during analysis.
                'is_soz': 1 if fuzzy_match(ch, soz_list) else 0,
                **{f"{q:.1f}": val for q, val in zip(q_values, hq)}
            })

    # === NEW: Write all results to one Excel file ===
    if all_results:
        df_all = pd.DataFrame(all_results)
        excel_path = os.path.join(data_dir, "all_subjects_q_values.xlsx")
        df_all.to_excel(excel_path, index=False)
        print(f"→ Saved all H(q) values to {excel_path}")




if __name__ == "__main__":
    freeze_support()
    main()
