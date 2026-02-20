# chagas-ecg-diagnosis

Automated detection of Chagas disease from 12-lead ECGs using Ensemble Learning and Deep Neural Networks (ResNet/Transformers). Developed for the PhysioNet Challenge 2025.

## 🧠 Approaches

### Classic Machine Learning (`team_code_classic_machine_learning.py`)
This approach relies on hand-crafted feature engineering and an ensemble of traditional machine learning models:
- **Feature Extraction:** Extracts Morphological (statistics, RMS, zero crossings), Frequency domain (PSD in various bands), and HRV (Heart Rate Variability / R-R intervals) features across the 12 ECG leads.
- **Data Balancing:** Addresses class imbalance through source-based sample weighting (down-weighting CODE-15% weak labels) and SMOTE + ENN resampling.
- **Ensemble Classifier:** A soft-voting combination of Random Forest (40%), Gradient Boosting (40%), and Logistic Regression (20%), using an optimized decision threshold.

## 📊 Benchmarks

| Model | Challenge Score | AUROC | AUPRC | Accuracy | F-measure |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Classic Machine Learning | 0.629 | 0.913 | 0.611 | 0.984 | 0.611 |
