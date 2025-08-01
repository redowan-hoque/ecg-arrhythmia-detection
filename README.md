# ECG Arrhythmia Detection & Classification

A full-featured pipeline for ECG arrhythmia detection and classification using the MIT-BIH Arrhythmia Database. This repository implements both the Pan-Tompkins and Discrete Wavelet Transform (DWT) methods for R-peak detection, followed by SVM-based classification of arrhythmia types. The code supports both Jupyter Notebook and script-based workflows, and includes a GUI for easy model and file selection.

---

## 📁 Directory Structure

```
├── database/
│   ├── RAW_ADC/          # Raw ECG signals in .mat format
│   └── Text_files/       # Annotation files in .txt format
├── discrete_wavelet_transform.py    # DWT-based R-peak detection
├── pan_tompkins.py                  # Pan-Tompkins algorithm for R-peak detection
├── utils.py                         # Helper utilities for segmentation, info, plotting, etc.
├── final_PT.ipynb                   # Pan-Tompkins pipeline notebook
├── final_DWT.ipynb                  # DWT pipeline notebook
├── model_pt.pkl                     # Trained SVM model (Pan-Tompkins)
├── model_dwt.pkl                    # Trained SVM model (DWT)
├── output.py                        # Inference script for predictions
├── output_gui.py                    # (Optional) GUI for method and file selection
└── requirements.txt                 # Python dependencies
```

---

## 📝 Project Overview

- **Goal:** Detect and classify arrhythmias from ECG signals using two preprocessing pipelines (Pan-Tompkins and DWT) and SVM classification.
- **Data:** MIT-BIH Arrhythmia Database (.mat and .txt files)
- **Key Steps:**
  1. R-peak detection (Pan-Tompkins or DWT)
  2. Feature extraction and segmentation
  3. Model training (SVM)
  4. Model evaluation and inference
  5. (Optional) GUI for interactive processing

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ecg-arrhythmia-detection.git
cd ecg-arrhythmia-detection
```

### 2. Prepare the Data

- Place raw ECG `.mat` files in `database/RAW_ADC/`.
- Place annotation `.txt` files in `database/Text_files/`.

### 3. Install Dependencies

All required packages are listed in `requirements.txt`.  
Install them via pip:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
numpy
scipy
pywt
matplotlib
pandas
seaborn
scikit-learn
jupyter        # Only if using notebooks
# tk            # tkinter is standard in Python but may need 'python3-tk' on Linux
```

> **Note for Linux:**  
> If you encounter errors with `tkinter`, install it via:  
> `sudo apt-get install python3-tk`

---

## 💻 Usage

### A. **Jupyter Notebook Pipeline**

- **Pan-Tompkins:**  
  Open and run `final_PT.ipynb` to:
  - Preprocess data
  - Detect R-peaks
  - Extract features
  - Train and validate the SVM model
  - Save the trained model as `model_pt.pkl`

- **DWT:**  
  Open and run `final_DWT.ipynb` for the DWT-based approach.

### B. **Script-Based Inference**

Use the provided `output.py` to predict arrhythmia types for a new ECG signal:

```bash
python output.py
```

This script will:
- Load the chosen ECG `.mat` file
- Apply the selected R-peak detection method
- Use the corresponding trained model to predict arrhythmia annotations

### C. **(Optional) GUI Usage**

If you wish to use the GUI (`output_gui.py`), run:

```bash
python output_gui.py
```

The GUI allows:
- Selection between Pan-Tompkins and DWT methods
- Choosing signal length and `gr` flag
- Importing `.mat` files and SVM `.pkl` files
- Visualizing results interactively

---

## 📦 Main Python Packages Used

- `numpy` – Numeric arrays and operations
- `scipy` – Signal processing, loading `.mat` files, filtering
- `pywt` – Discrete Wavelet Transform
- `matplotlib` – Signal and result plotting
- `pandas` – Data management and DataFrame operations
- `seaborn` – Enhanced statistical plots
- `scikit-learn` – SVM classification, evaluation
- `tkinter` – GUI support (standard with Python)
- `pickle` – Model serialization

---

## 🏆 Results

- **Two pipelines:** Compare Pan-Tompkins and DWT approaches head-to-head.
- **Customizable models:** Easy retraining and inference for new signals.
- **Rich visualizations:** Plots for R-peaks, signal segments, and classification results.

---



## 🙏 Credits

- **MIT-BIH Arrhythmia Database** – [PhysioNet](https://physionet.org/content/mitdb/)
- **Pan-Tompkins Algorithm:** Pan, J., & Tompkins, W. J. (1985), *IEEE Transactions on Biomedical Engineering*
- **DWT for ECG:** Standard references in biomedical signal processing literature
- **MI ECG Signal Filtering** -Farhatul Fityah - (https://www.kaggle.com/code/farhatulfityah/mi-ecg-signal-filtering)

--

## Developed by:  
-Md. Abu Saleh Akib (ID 2106007)  
-Md. Redowanul Hoque (ID 2106008)  
-Fariha Anjum Oshin (ID 2106009)  
-Sudipta Mondal (ID 2106010)  

Department of EEE  
Bangladesh University of Engineering and Technology, Dhaka, Bangladesh


---

## 📜 License

This project is open-sourced under the [MIT License](LICENSE).

---

**For any questions, feel free to open an issue or contact the maintainer!**

