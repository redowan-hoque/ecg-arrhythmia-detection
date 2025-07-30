from scipy.signal import butter, filtfilt, find_peaks
import numpy as np
import matplotlib.pyplot as plt


def pt(ecg, fs, gr=True):
    if len(ecg.shape) != 1:
        raise ValueError("ECG must be a 1D vector")
    
    ecg = ecg.flatten()
    ecg = ecg/abs(ecg).max()

    ecg = ecg - np.mean(ecg)  

    delay = 0
    skip = 0
    m_selected_RR = 0
    mean_RR = 0
    ser_back = 0

    if fs == 200: 
        # Low-pass filter with cutoff at 12 Hz
        b, a = butter(3, 12 * 2 / fs, btype='low')
        ecg_l = filtfilt(b, a, ecg)
        ecg_l /= np.max(np.abs(ecg_l))  # Normalize

        # High-pass filter with cutoff at 5 Hz
        b, a = butter(3, 5 * 2 / fs, btype='high')
        ecg_h = filtfilt(b, a, ecg_l)
        ecg_h /= np.max(np.abs(ecg_h)) 
    else: 
        # Butterworth bandpass filter, order=3, passband = 5 to 15 Hz
        b, a = butter(3, [5 * 2 / fs, 15 * 2 / fs], btype='band')
        ecg_h = filtfilt(b, a, ecg)
        ecg_h /= np.max(np.abs(ecg_h))

    # Derivative filter
    # Generating a 5-point FIR filter with fixed coefficients
    if fs != 200:
        int_c = (5 - 1) / (fs * 1 / 40)
        x = np.linspace(1, 5, int(np.ceil((5 - 1) / (1 / (fs / 40)))) )
        b_vals = np.array([1, 2, 0, -2, -1]) * (1 / 8) * fs
        b = np.interp(x, np.arange(1, 6), b_vals)
    else:
        b = np.array([1, 2, 0, -2, -1]) * (1 / 8) * fs

    ecg_d = filtfilt(b, [1], ecg_h)
    ecg_d /= np.max(np.abs(ecg_d))

    # Squaring
    ecg_s = ecg_d ** 2

    # Moving average integration
    ma_window = int(round(0.15 * fs))
    ecg_m = np.convolve(ecg_s, np.ones(ma_window) / ma_window, mode='same')
    delay += int(ma_window / 2)

    # Fiducial marks
    min_peak_distance = int(round(0.2 * fs))
    locs, _ = find_peaks(ecg_m, distance=min_peak_distance)
    pks = ecg_m[locs]
    LLp = len(pks)

    qrs_c = np.zeros(LLp)
    qrs_i = np.zeros(LLp, dtype=int)
    qrs_amp_raw = np.zeros(LLp)
    qrs_i_raw = np.zeros(LLp, dtype=int)
    nois_c = np.zeros(LLp)
    nois_i = np.zeros(LLp, dtype=int)

    SIGL_buf = np.zeros(LLp)
    NOISL_buf = np.zeros(LLp)
    SIGL_buf1 = np.zeros(LLp)
    NOISL_buf1 = np.zeros(LLp)
    THRS_buf = np.zeros(LLp)
    THRS_buf1 = np.zeros(LLp)

    # Main QRS Detection
    THR_SIG = np.max(ecg_m[:2 * fs]) / 3
    THR_NOISE = np.mean(ecg_m[:2 * fs]) / 2
    SIG_LEV = THR_SIG
    NOISE_LEV = THR_NOISE

    # False(T-Wave) Detection Check
    THR_SIG1 = np.max(ecg_h[:2 * fs]) / 3
    THR_NOISE1 = np.mean(ecg_h[:2 * fs]) / 2
    SIG_LEV1 = THR_SIG1
    NOISE_LEV1 = THR_NOISE1

    Beat_C = 0
    Beat_C1 = 0
    Noise_Count = 0

    # Debug print to verify variables
    # print("Initial Values:")
    # print("pks:", pks[:10])
    # print("locs:", locs[:10])

    # QRS detection loop logic
    for i in range(LLp):
        if pks[i] >= THR_SIG:
            Beat_C += 1
            qrs_c[Beat_C] = pks[i]
            qrs_i[Beat_C] = locs[i]
            refractory_period_sec = 0.3
            if Beat_C >= 3 and (locs[i] - qrs_i[Beat_C - 1]) <= round(refractory_period_sec * fs):
                Slope1 = np.mean(np.diff(ecg_m[locs[i] - round(0.075 * fs):locs[i]]))
                Slope2 = np.mean(np.diff(ecg_m[qrs_i[Beat_C - 1] - round(0.075 * fs):qrs_i[Beat_C - 1]]))
                if abs(Slope1) <= 0.5 * abs(Slope2):
                    Noise_Count += 1
                    nois_c[Noise_Count] = pks[i]
                    nois_i[Noise_Count] = locs[i]
                    skip = 1  # T wave detected
                    NOISE_LEV1 = 0.125 * qrs_amp_raw[Beat_C] + 0.875 * NOISE_LEV1
                    NOISE_LEV = 0.125 * pks[i] + 0.875 * NOISE_LEV
                else:
                    skip = 0
            if skip == 0:
                Beat_C1 += 1
                qrs_i_raw[Beat_C1] = locs[i]
                qrs_amp_raw[Beat_C1] = pks[i]
    qrs_i_raw = qrs_i_raw[1:Beat_C1]
    qrs_amp_raw = qrs_amp_raw[1:Beat_C1]
    # # Debug print to verify final values
    # print("Final Values:")
    # print("qrs_i_raw:", qrs_i_raw[:50])
    # print("qrs_amp_raw:", qrs_amp_raw[:50])
    # Plotting results
    if gr:
        plt.figure(figsize=(12, 10))

        plt.subplot(3, 2, 1)
        plt.plot(ecg)
        plt.title('Raw ECG Signal')
        plt.grid(True)

        plt.subplot(3, 2, 2)
        plt.plot(ecg_h)
        plt.title('Bandpass Filtered ECG')
        plt.grid(True)

        plt.subplot(3, 2, 3)
        plt.plot(ecg_d)
        plt.title('Derivative Filtered')
        plt.grid(True)

        plt.subplot(3, 2, 4)
        plt.plot(ecg_s)
        plt.title('Squared Signal')
        plt.grid(True)

        plt.subplot(3, 2, 5)
        plt.plot(ecg_m)
        plt.title('Moving Average (Integrated Signal)')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(12, 4))
        plt.plot(ecg_h, label="Filtered ECG")
        plt.scatter(qrs_i_raw, qrs_amp_raw, color='red', label="Detected QRS")
        plt.title("QRS Detection on Filtered Signal")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return ecg_h,qrs_i_raw,qrs_amp_raw



# import os
# from scipy.io import loadmat
# # Load your ECG data here
# data_path = os.path.join("database/RAW_ADC/201m.mat")
# data = loadmat(data_path)
# EKG1 = data['val'][0].flatten()
# EKG1 = EKG1[0:1000]  # Use only a portion for demo
# fs = 360

# # Call the function
# filtered_ecg, r_wave_locs, qrs_amp_raw = pt(EKG1, fs, gr=0)

# print("Detected R-peak indices:", r_wave_locs)
# print("Detected R-peak iam;s:", qrs_amp_raw)