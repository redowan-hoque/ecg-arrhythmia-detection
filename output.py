
import pickle
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pan_tompkins import pt
from discrete_wavelet_transform import dwt
from utils import ecg_info, segment_ecg, output_plot


test_signal_folder = "database/RAW_ADC";
test_signal_path = os.path.join(test_signal_folder,"201m.mat");

test_signal = loadmat(test_signal_path)

test_signal = test_signal['val'][0].flatten()


test_signal = test_signal[0:2000]
fs = 360

filtered_ecg, r_wave_locs, qrs_amp_raw = pt(test_signal, fs, gr=1)
r_wave_locs = r_wave_locs

with open('model_pt.pkl', 'rb') as f:
    clf = pickle.load(f)

forward_rri, backward_rri, instant_hr = ecg_info(r_wave_locs, fs)
segmented = segment_ecg(filtered_ecg, r_wave_locs, window=180)
ecg_info_df = pd.DataFrame({
    'Sample#': r_wave_locs,
    'Nearest_R_PT': r_wave_locs,
    'Missed_PT': np.ones_like(r_wave_locs),
    'Diff': np.zeros_like(r_wave_locs),
    'frri': forward_rri,
    'brri': backward_rri,
    'instant_hr': instant_hr
})




test_dataset = pd.concat([ecg_info_df, segmented], axis=1 , ignore_index=0)
test_dataset.columns = test_dataset.columns.astype(str)
# print(test_dataset)
test_dataset = test_dataset.loc[
    ~(
        (test_dataset['frri'] == 0) |
        (test_dataset['brri'] == 0) |
        (test_dataset['frri'].isna())|
        (test_dataset['brri'].isna())|
        (test_dataset['instant_hr'] == 0)|
        (test_dataset['0'].isna())|
        (test_dataset['359'].isna())
    )
]

# print(test_dataset)

X_test = test_dataset.drop(columns=["Sample#","Diff","Missed_PT","Nearest_R_PT"]);



# Diff
# - Missed_PT
# - Nearest_R_PT
# - Sample#
#X_test = test_dataset;
y_test_predicted = clf.predict(X_test);

# print(X_test)
# print(y_test_predicted)

y_test_predicted_df = pd.DataFrame({
    "Type":y_test_predicted
})

input_output_df = pd.concat([test_dataset.reset_index(drop=True), y_test_predicted_df.reset_index(drop=True)], axis=1 )
# print(test_dataset.shape)
# print(y_test_predicted_df.shape)
print(input_output_df.columns)



output_plot(filtered_ecg, input_output_df)

