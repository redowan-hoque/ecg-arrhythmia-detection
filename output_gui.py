import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pickle
import numpy as np
import pandas as pd
from scipy.io import loadmat
from pan_tompkins import pt
from discrete_wavelet_transform import dwt
from utils import ecg_info, segment_ecg, output_plot

class ECGApp:
    def __init__(self, master):
        self.master = master
        master.title("ECG Classification GUI")
        master.geometry('500x450')

        self.method = tk.StringVar()
        self.method.set("Pan-Tompkins")
        self.gr = tk.BooleanVar()
        self.gr.set(False)
        self.signal_length = tk.StringVar()
        self.signal_length.set("0:2000")

        ttk.Label(master, text="ECG Arrhythmia Classification", font=('Helvetica', 16)).pack(pady=10)

        # Method Selection
        ttk.Label(master, text="Select Method:").pack(pady=5)
        method_select = ttk.Combobox(master, textvariable=self.method, state="readonly",
                                     values=["Pan-Tompkins", "DWT"])
        method_select.pack(pady=5)

        # Graph Toggle
        ttk.Checkbutton(master, text="Show Graph", variable=self.gr).pack(pady=5)

        # Signal Length Input
        ttk.Label(master, text="Select Signal Length (start:end):").pack(pady=5)
        signal_entry = ttk.Entry(master, textvariable=self.signal_length)
        signal_entry.pack(pady=5)

        # Signal file selection
        self.signal_file = tk.StringVar()
        ttk.Button(master, text="Select ECG Signal (.mat)", command=self.select_signal).pack(pady=5)
        self.signal_label = ttk.Label(master, text="No file selected")
        self.signal_label.pack()

        # Model file selection
        self.model_file = tk.StringVar()
        ttk.Button(master, text="Select Model (.pkl)", command=self.select_model).pack(pady=5)
        self.model_label = ttk.Label(master, text="No file selected")
        self.model_label.pack()

        # Run button
        ttk.Button(master, text="Run", command=self.run_classification).pack(pady=20)

    def select_signal(self):
        filename = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
        if filename:
            self.signal_file.set(filename)
            self.signal_label.config(text=filename.split('/')[-1])

    def select_model(self):
        filename = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if filename:
            self.model_file.set(filename)
            self.model_label.config(text=filename.split('/')[-1])

    def run_classification(self):
        if not self.signal_file.get() or not self.model_file.get():
            messagebox.showwarning("Missing Files", "Please select both the signal and model files.")
            return

        test_signal = loadmat(self.signal_file.get())['val'][0].flatten()
        try:
            start, end = map(int, self.signal_length.get().split(':'))
            test_signal = test_signal[start:end]
        except Exception as e:
            messagebox.showerror("Error", "Invalid signal length format. Use start:end")
            return

        fs = 360

        if self.method.get() == "Pan-Tompkins":
            filtered_ecg, r_wave_locs, _ = pt(test_signal, fs, gr=int(self.gr.get()))
        else:
            filtered_ecg, r_wave_locs, _ = dwt(test_signal, fs, gr=int(self.gr.get()))

        with open(self.model_file.get(), 'rb') as f:
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

        test_dataset = pd.concat([ecg_info_df, segmented], axis=1)
        test_dataset.columns = test_dataset.columns.astype(str)

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

        X_test = test_dataset.drop(columns=["Sample#", "Diff", "Missed_PT", "Nearest_R_PT"])

        y_test_predicted = clf.predict(X_test)

        y_test_predicted_df = pd.DataFrame({"Type": y_test_predicted})

        input_output_df = pd.concat([test_dataset.reset_index(drop=True), y_test_predicted_df], axis=1)

        output_plot(filtered_ecg, input_output_df)

if __name__ == "__main__":
    root = tk.Tk()
    app = ECGApp(root)
    root.mainloop()