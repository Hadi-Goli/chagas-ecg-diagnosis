# chagas-ecg-diagnosis

[🇮🇷 برای مطالعه توضیحات به زبان فارسی اینجا کلیک کنید](README_FA.md)
Automated detection of Chagas disease from 12-lead ECGs using Ensemble Learning and Deep Neural Networks (ResNet/Transformers). Developed for the PhysioNet Challenge 2025.

## 🧠 Approaches

### Classic Machine Learning (`team_code_classic_machine_learning.py`)
This approach relies on hand-crafted feature engineering and an ensemble of traditional machine learning models:
- **Feature Extraction:** Extracts Morphological (statistics, RMS, zero crossings), Frequency domain (PSD in various bands), and HRV (Heart Rate Variability / R-R intervals) features across the 12 ECG leads.
- **Data Balancing:** Addresses class imbalance through source-based sample weighting (down-weighting CODE-15% weak labels) and SMOTE + ENN resampling.
- **Ensemble Classifier:** A soft-voting combination of Random Forest (40%), Gradient Boosting (40%), and Logistic Regression (20%), using an optimized decision threshold.

### 1D Convolutional Neural Network (`team_code_1D_CNN.py`)
This approach utilizes a custom 1D ResNet architecture trained directly on raw ECG signals combined with multimodal demographic data:
- **Preprocessing:** Reorders leads to a standard 12-lead order, resamples to 500 Hz, normalizes using Z-score per lead, and pads/crops the signal to a fixed 10-second window (5000 samples). Also normalizes Age and Sex from header files.
- **Architecture:** A 1D adaptation of ResNet-18. It features an initial wide receptive field (`kernel_size=51`) to capture the macro-waveform of the ECG. Age and Sex features are concatenated before the final fully connected layer. Employs `Dropout(p=0.5)` for regularization.
- **Training Strategy:** Trained with `BCEWithLogitsLoss`. Crucially, dynamically calculates class imbalances and applies a `pos_weight` constraint to heavily penalize missing the minority positive class. Uses the AdamW optimizer (learning rate 1e-3).
- **Validation Engine:** Built-in Early Stopping evaluating against a dedicated `validation_data` holdout directory (created via `split_data.py`). Includes a dynamic Decision Threshold Optimization (grid-search) mapped to maximize the F-measure on the validation probabilities.

### Spectrogram + EfficientNet-B0 (`team_code_spectrogram.py`)
This approach transforms 1D waveforms into 2D Time-Frequency representations to capture both immediate morphology and long-term variations simultaneously:
- **Preprocessing & STFT:** Resamples the ECG to 400Hz and isolates a fixed 10-second window. Computes Short-Time Fourier Transform (STFT) per lead, filters for the medially relevant 0-40Hz spectrum, and converts magnitude to decibels.
- **Image Grid:** Arranges the 12 resulting spectrograms into an optimized 4x3 grid. This prevents destructive aspect-ratio stretching when the final composition is resized to the target 224x224 RGB image footprint required by the model.
- **Architecture & Training:** Employs a pretrained `EfficientNet-B0` backbone, coupled with PyTorch Automatic Mixed Precision (AMP) to maintain a batch size of 32 within a strict 4GB VRAM constraint. The final classifier concatenates extracted image features with normalized Age and Sex parameters.
- **Validation Engine:** Built-in Early Stopping against `validation_data`. Includes dynamic Decision Threshold Optimization, finding an optimal threshold of 0.70 to maximize the F-measure.

## 📊 Benchmarks

| Model | Challenge Score | AUROC | AUPRC | Accuracy | F-measure |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Classic Machine Learning | 0.629 | 0.913 | 0.611 | 0.984 | 0.611 |
| 1D CNN | 0.442 | 0.856 | 0.316 | 0.966 | 0.327 |
| Spectrogram + EfficientNet-B0 | 0.379 | 0.815 | 0.202 | 0.959 | 0.265 |
