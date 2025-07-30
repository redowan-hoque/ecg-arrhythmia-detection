import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.io import loadmat

def dwt(ecg, fs=360, gr=True):
    # Time vector
    ecg = ecg/ abs(ecg).max()
    t = np.arange(len(ecg)) / fs

    # ---------- Baseline Wander Removal ----------
    win = int(0.2* fs)
    half = win // 2
    baseline_removed = np.array([
        ecg[i] - np.mean(ecg[max(0, i - half):min(len(ecg), i + half + 1)])
        for i in range(len(ecg))
    ])

    # ---------- Low-pass Filtering ----------
    nyq = 0.5 * fs
    cutoff = 40
    b, a = butter(4, cutoff / nyq, btype="low")
    lowpass_filtered = filtfilt(b, a, baseline_removed)

    # ---------- Wavelet NeighBlock Denoising ----------
    coeffs = pywt.wavedec(lowpass_filtered, 'db6', level=6)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    block = 2
    for i in range(1, len(coeffs)):
        cD = coeffs[i]
        for j in range(0, len(cD), block):
            blk = cD[j:j + block]
            T = np.sqrt(len(blk)) * sigma * np.sqrt(2 * np.log(len(blk)))
            norm = np.linalg.norm(blk)
            if norm < T:
                cD[j:j + block] = 0
            else:
                cD[j:j + block] *= max(0, 1 - T / norm)
    denoised = pywt.waverec(coeffs, 'db6')[:len(ecg)]

    # ---------- R-peak Detection ----------
    thr_coeff=0.2; thr = thr_coeff * np.max(denoised)
    cand = np.where(denoised > thr)[0]
    rlocs = []
    last = -np.inf
    for c in cand:
        if c - last > int(0.25 * fs):
            peak = max(0, c-40) + np.argmax(denoised[max(0, c-40):min(len(denoised), c+40)])
            rlocs.append(peak)
            last = peak
    rlocs = np.array(rlocs, dtype=int)

    # ---------- Plot ----------
    if gr:
        plt.figure(figsize=(12, 9))
        plt.subplot(4,1,1); plt.plot(t, ecg); plt.title("Original ECG")
        plt.subplot(4,1,2); plt.plot(t, baseline_removed); plt.title("Baseline Wander Removed")
        plt.subplot(4,1,3); plt.plot(t, lowpass_filtered); plt.title("Low-pass Filtered")
        plt.subplot(4,1,4); plt.plot(t, denoised); plt.plot(rlocs/fs, denoised[rlocs], 'ro')
        plt.title("Final Denoised Signal with R-peaks"); plt.xlabel("Time (s)")
        plt.tight_layout(); plt.show()

    return denoised, rlocs, denoised[rlocs]
