# EEG Seizure Detection Using Leakage-Aware and Imbalance-Robust Deep Learning

A deep learning research framework designed to evaluate automated epileptic seizure detection from multi-channel EEG recordings under **patient-independent validation** (`GroupKFold`) and **extreme class imbalance**.

---

## Research Focus
1. **Data Leakage vs. Patient Generalization**: Comparing patient-independent validation against conventional random segment splitting.
2. **Imbalance-Robust Learning**: Systematically comparing Standard Cross-Entropy, Class-Weighted Loss, Focal Loss, and Balanced Sampling.
3. **Multi-Metric & Calibration Evaluation**: Benchmarking models using Macro-F1, Sensitivity, Specificity, PR-AUC, ROC-AUC, MCC, and Expected Calibration Error (ECE).

---

## Project Structure

```
├── configs/                  # Experiment configuration files (YAML)
├── data/
│   ├── raw/chbmit/           # Raw CHB-MIT EEG recordings (.edf) & summary annotations
│   ├── interim/              # Filtered EEG signals & parsed annotations
│   └── processed/            # Segmented arrays & dataset metadata manifests
├── experiments/              # Experiment runner scripts & checkpoints
├── notebooks/                # Exploratory signal analysis and visualization notebooks
├── reports/
│   ├── figures/              # Publication-ready plots and spectrograms
│   ├── tables/               # Formatted metric tables across patient folds
│   └── research_notes.md     # Hypotheses, experiment matrix, and methodology logs
├── scripts/
│   ├── inspect_eeg.py        # Inspect EDF headers, sampling rate, and channels
│   ├── visualize_seizure.py  # Plot multi-channel EEG during seizure events
│   └── test_real_segmentation.py # Validate window segmentation against ground truth
├── src/
│   └── eeg_seizure/
│       ├── __init__.py       # Package entrypoint
│       ├── calibration.py    # ECE, Brier score, and reliability curves
│       ├── data_loader.py    # PyTorch Datasets and balanced samplers
│       ├── evaluate.py       # Cross-fold evaluation loop
│       ├── losses.py         # Standard BCE, Class-Weighted BCE, and Focal Loss
│       ├── metrics.py        # Macro-F1, PR-AUC, ROC-AUC, Sensitivity, Specificity, MCC
│       ├── models.py         # 1D Baseline CNN and 1D ResNet architectures
│       ├── preprocessing.py  # Bandpass (0.5–45Hz), Notch (60Hz), Montage alignment
│       ├── segmentation.py   # Window slicing with stride and annotation labeling
│       ├── splitting.py      # Patient-level GroupKFold vs Random segment splitting
│       ├── train.py          # Model training with early stopping & lr scheduling
│       └── utils.py          # Seed setting, checkpointing, and IO helpers
├── tests/                    # Unit tests for preprocessing, segmentation, and metrics
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Quickstart

### 1. Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest tests/
```

### 3. Verify Segmentation on Real EEG
```bash
python3 scripts/test_real_segmentation.py
```
