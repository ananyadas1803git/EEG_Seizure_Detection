# Research Notes: EEG Seizure Detection

## Proposed Problem Statement
**EEG Seizure Detection Using Leakage-Aware and Imbalance-Robust Deep Learning**

Continuous electroencephalography (EEG) monitoring produces vast amounts of multi-channel time-series data. Automated seizure detection faces two primary pitfalls in conventional literature:
1. **Data Leakage**: Conventional random segment-level train/test splits place segments from the same patient in both training and test sets, artificially inflating reported performance due to patient-specific EEG signatures.
2. **Extreme Class Imbalance**: Seizure events account for <1% of long-term continuous EEG recordings, causing models trained on naive cross-entropy to achieve high accuracy simply by predicting the majority non-seizure class.

---

## Core Research Questions & Hypotheses

### Primary Question
> *Can patient-independent validation and imbalance-aware deep learning improve the reliability and generalization of automated EEG seizure detection compared with conventional training and evaluation approaches?*

### Sub-Questions
1. **Leakage Analysis**: Does patient-level splitting (`GroupKFold` / `StratifiedGroupKFold`) produce significantly lower yet clinically realistic performance estimates than random segment-level splitting?
2. **Imbalance Handling**: Which imbalance-handling strategy—class weighting, focal loss ($\gamma=2.0$), or balanced sampling—best improves minority-class seizure detection (Sensitivity & PR-AUC)?
3. **Metric & Fold Consistency**: Do observed improvements remain consistent across multiple evaluation metrics (Macro-F1, Sensitivity, Specificity, PR-AUC, ROC-AUC, MCC, Calibration ECE) and patient folds?

---

## Experimental Design Matrix

| Experiment ID | Model Architecture | Splitting Strategy | Imbalance Strategy | Target Metrics |
|---|---|---|---|---|
| **EXP-01 (Leaked Baseline)** | 1D CNN | Random Segment Split (5-Fold) | Standard Cross-Entropy | Accuracy, Macro-F1, ROC-AUC, PR-AUC |
| **EXP-02 (Leakage-Free Baseline)** | 1D CNN | Patient GroupKFold (5-Fold) | Standard Cross-Entropy | Macro-F1, Sensitivity, PR-AUC, MCC |
| **EXP-03 (Class Weighted)** | 1D CNN | Patient GroupKFold (5-Fold) | Class-Weighted BCE ($N_{neg}/N_{pos}$) | Macro-F1, Sensitivity, Specificity, PR-AUC |
| **EXP-04 (Focal Loss)** | 1D CNN | Patient GroupKFold (5-Fold) | Focal Loss ($\gamma=2.0, \alpha=0.25$) | PR-AUC, Sensitivity, Calibration (ECE) |
| **EXP-05 (Balanced Sampling)** | 1D CNN | Patient GroupKFold (5-Fold) | `WeightedRandomSampler` | PR-AUC, Sensitivity, Specificity |
| **EXP-06 (ResNet-1D Leakage-Free)** | 1D ResNet | Patient GroupKFold (5-Fold) | Best Imbalance Loss | Full Metric Suite + Calibration Curves |

---

## Roadmap Tracker (30-Day Research Plan)

- [x] **Day 1–3**: Problem formulation, dataset inspection (CHB-MIT), project structure setup.
- [ ] **Day 4–7**: EEG Preprocessing pipeline (0.5–45 Hz bandpass, 60 Hz notch filter, channel montage standardization, sliding window segmentation with metadata logging).
- [ ] **Day 8–12**: Baseline 1D CNN architecture + DataLoaders + leakage experiment (Patient GroupKFold vs Random Split).
- [ ] **Day 13–16**: 1D ResNet architecture with residual skip connections.
- [ ] **Day 17–21**: Systematic imbalance experiments (Standard CE, Weighted Loss, Focal Loss, Balanced Sampler).
- [ ] **Day 22–25**: Ablation study & cross-fold statistical significance tests (Wilcoxon signed-rank / paired t-test).
- [ ] **Day 26–27**: Probability calibration (ECE, Brier score, reliability diagrams) & error analysis.
- [ ] **Day 28–30**: Final report, summary tables, publication-quality figures, and GitHub release.
