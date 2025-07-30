import matplotlib.pyplot as plt
import numpy as np;
import pandas as pd;
import seaborn as sns

def ecg_info(r_wave_locs, fs):
    
    forward_rri= np.concatenate((r_wave_locs,[0]))-np.concatenate(([0],r_wave_locs)) ;
    forward_rri = forward_rri[1:];
    forward_rri = np.append(forward_rri,0);
    backward_rri = np.concatenate((r_wave_locs,[0]))-np.concatenate(([0],r_wave_locs)) ;
    # print(np.diff(r_wave_locs))
    #instant_hr = 60 * fs / forward_rri;
    instant_hr = np.where(forward_rri == 0, 0, 60 * fs / forward_rri) # instant_hr = 0 if forward_rri = 0

    
    for i in range(np.size(r_wave_locs)):
        if instant_hr[i]<0: instant_hr[i]=0
        if forward_rri[i]<0: forward_rri[i]=0
        if backward_rri[i]<0: forward_rri[i]=0
        if forward_rri[i]>2*fs: forward_rri[i]=0;
        if backward_rri[i]>2*fs: backward_rri[i]=0;
        if instant_hr[i]> fs : instant_hr[i]=0;
    forward_rri=forward_rri[0:forward_rri.size-1].copy()
    backward_rri= backward_rri[0:backward_rri.size-1].copy()
    instant_hr = instant_hr[0:instant_hr.size-1].copy()
    return forward_rri, backward_rri, instant_hr;


def compare_r_peaks(annotation_df, manual_PT, tolerance=30):
    """
    Compare R-wave locations from manual_PT['r_wave_locs'] with annotated reference values.

    Parameters:
    - annotation_df: pandas DataFrame with 'Sample#' and 'Class' columns.
    - manual_PT: pandas DataFrame with 'r_wave_locs', 'instant_hr', 'frri', 'brri' columns.
    - tolerance: int, maximum allowed difference to consider a match (in samples).

    Returns:
    - result_df: DataFrame with columns 'Sample#', 'Nearest_R_PT', 'Missed_PT', 'Diff', 'frri', 'Class'.
    - total_missed_PT: int, total number of missed R-peaks.
    - total_error_PT: float, sum of differences for non-missed peaks.
    - avg_deviation_PT: float, average deviation for non-missed peaks.
    """
    # Step 1: Create result_df from annotation_df
    result_df = annotation_df.copy()
    r_wave_locs = manual_PT['r_wave_locs'].to_numpy()

    # Step 2: Compute Nearest_R_PT, Diff, and Missed_PT
    result_df['Nearest_R_PT'] = result_df['Sample#'].apply(
        lambda x: r_wave_locs[np.argmin(np.abs(r_wave_locs - x))]
    )
    result_df['Diff'] = np.abs(result_df['Sample#'] - result_df['Nearest_R_PT'])
    result_df['Missed_PT'] = result_df['Diff'].gt(tolerance).astype(int)

    # Step 3: Add frri from manual_PT for non-missed peaks
    # Create a mapping from r_wave_locs to frri
    frri_map = dict(zip(manual_PT['r_wave_locs'], manual_PT['frri']))
    brri_map = dict(zip(manual_PT['r_wave_locs'], manual_PT['brri']))
    instant_hr = dict(zip(manual_PT['r_wave_locs'], manual_PT['instant_hr']))
    # Assign frri based on Nearest_R_PT, set to NaN for Missed_PT == 1
    result_df['frri'] = result_df['Nearest_R_PT'].map(frri_map)
    result_df['brri'] = result_df['Nearest_R_PT'].map(brri_map);
    result_df['instant_hr'] = result_df['Nearest_R_PT'].map(instant_hr);
    result_df.loc[result_df['Missed_PT'] == 1, 'frri'] = np.nan

    # Step 4: Calculate metrics
    total_missed_PT = result_df['Missed_PT'].sum()
    total_error_PT = result_df.loc[result_df['Missed_PT'] == 0, 'Diff'].sum()
    avg_deviation_PT = (
        total_error_PT / len(result_df[result_df['Missed_PT'] == 0])
        if len(result_df[result_df['Missed_PT'] == 0]) > 0
        else 0
    )

    # Ensure columns are in the correct order
    result_df = result_df[['Sample#', 'Nearest_R_PT', 'Missed_PT', 'Diff', 'frri','brri','instant_hr' ,'Type']]

    return result_df, total_missed_PT, total_error_PT, avg_deviation_PT


import numpy as np
import pandas as pd

def segment_ecg(signal, sample_locs, window=180):
    """
    Extract fixed-length segments of an ECG signal centered around given sample locations,
    returning the result as a pandas DataFrame of raw ECG amplitude values.

    Parameters
    ----------
    signal : array-like
        1D ECG signal (list, numpy array, or pandas Series).
    sample_locs : array-like of int
        Indices (0-based) around which to center each segment.
    window : int, optional
        Half-window size (number of samples on each side of the center).
        Total segment length will be 2 * window. Default is 180 → 360 samples.

    Returns
    -------
    segments_df : pandas.DataFrame, shape (len(sample_locs), 2*window)
        Each row is a segment of raw ECG amplitudes; out-of-bounds positions are NaN.
        Columns are labeled 0 .. (2*window-1).
    """
    # Flatten the signal to 1D numpy array
    sig = np.asarray(signal).flatten()
    # sig = signal
    # print(sig)
    n_samples = sig.size
    seg_len   = 2*window
    n_beats   = len(sample_locs)

    # Preallocate segments array with NaN
    segments = np.full((n_beats, seg_len), np.nan, dtype=float)

    # Fill in each beat-window
    for i, loc in enumerate(sample_locs):
        start = loc - window
        end   = loc + window

        # Clamp to valid signal indices
        valid_start = max(start, 0)
        valid_end   = min(end, n_samples)

        # Compute insert positions within the segment
        insert_start = valid_start - start
        insert_end   = insert_start + (valid_end - valid_start)

        segments[i, insert_start:insert_end] = sig[valid_start:valid_end]
        # print(segments[i, insert_start:insert_end])

    # Build DataFrame: rows = beats, cols = sample offsets 0..seg_len-1
    segments_df = pd.DataFrame(segments)
    # segments_df = signal[segments_df]
    return segments_df


def output_plot(filtered_ecg, input_output_df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    sns.set(style="whitegrid", context="notebook")

    r_wave_locs = input_output_df["Sample#"]
    r_wave_locs = r_wave_locs.astype(int)
    type_labels = input_output_df['Type'].reset_index(drop=True)

    plt.figure(figsize=(18, 5))

    # Plot full ECG signal
    sns.lineplot(x=np.arange(len(filtered_ecg)), y=filtered_ecg, linewidth=1.5,
                 color="#3375b7", label='Filtered ECG')

    # Plot R-peaks
    r_peak_amplitudes = filtered_ecg[r_wave_locs]
    sns.scatterplot(x=r_wave_locs, y=r_peak_amplitudes, s=70, color='crimson',
                    edgecolor='black', linewidth=0.5, label='R-Peaks', zorder=3)

    # Annotate each R-peak with sample index and arrhythmia type
    for i, loc in enumerate(r_wave_locs):
        if i < len(type_labels):
            type_str = str(type_labels[i])[:5]  # truncate long types
            label_str = f"{loc} ({type_str})"
            plt.text(loc + 50, filtered_ecg[loc], label_str,
                     fontsize=9, weight='bold', color='black',
                     ha='left', va='bottom',
                     bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.8))

    # Labels and formatting
    plt.title("Filtered ECG with R-Peaks and Arrhythmia Types", fontsize=16, fontweight='bold')
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.legend(loc='upper right', fontsize=10, frameon=True)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
