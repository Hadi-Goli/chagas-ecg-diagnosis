# Project Context: Chagas ECG Diagnosis

> [!IMPORTANT]
> **SYSTEM INSTRUCTIONS FOR ALL AI AGENTS (Cursor, Copilot, Claude, etc.):**
> This `PROJECT_CONTEXT.md` is the single source of truth for the Chagas ECG Diagnosis project. 
> 
> **Your Responsibilities:**
> 1. **Update Automatically:** When you modify architecture, add new features, change workflows, or alter the data pipeline, you **MUST** update this file immediately to reflect the changes.
> 2. **Prune and Overwrite (Living Document):** Do NOT just append new information at the bottom. If a feature or logic is changed, you must LOCATE the old documentation in this file and OVERWRITE it. Delete dead code references and outdated context entirely.
> 3. **Strict Formatting (AI-Optimized Style):**
>    - Be highly concise, declarative, and to-the-point (Cheat-sheet style).
>    - NO conversational fluff. NO long paragraphs.
>    - Use Bullet points (`-`) and Bold text (`**`) for key-value mappings (e.g., `- **file_name.py**: Entry point for X`).
>    - Map out pipelines step-by-step using arrows (e.g., `Step A -> Step B`).
>    - Keep the document structured with clear Markdown headers (`##`).

## 🚀 Key Features

### 1. Dynamic Team Module Selection (`-tm` / `--team_module`)
The pipeline supports multiple `team_code.py` implementations. You can switch between them without modifying the training/running scripts.

- **Default:** `team_code`
- **Usage:**
  ```bash
  python train_model.py -d training_data -m model -tm team_code_classic_machine_learning
  python run_model.py -d holdout_data -m model -o holdout_outputs -tm team_code_classic_machine_learning
  ```
- **Modified Scripts:** `train_model.py`, `run_model.py`.

### 2. Classic Machine Learning Pipeline
A traditional ML implementation (`team_code_classic_machine_learning.py`) relying on feature engineering.
- **Features Extracted:** Morphological (RMS, kurtosis, etc.), Frequency (power bands), HRV (R-R intervals), cross-lead correlation, and demographics.
- **Data Balancing:** Source-aware sample weighting (sub-sampling CODE-15%) combined with SMOTE-ENN resampling.
- **Ensemble Model:** Weighted voting of Random Forest (40%), Gradient Boosting (40%), and Logistic Regression (20%) with an optimized threshold of 0.45.

### 3. 1D CNN Pipeline
A deep learning implementation (`team_code_1D_CNN.py`) using raw ECG signals alongside demographic data.
- **Preprocessing:** 12-lead reordering, 500 Hz resampling, Z-score normalization, and 10-second fixed length constraint. Age/Sex are extracted and normalized.
- **Architecture:** 1D ResNet-18. Enhanced with a wide convolutional kernel (`size=51`) for initial ECG macro-structure capture. Age and Sex are concatenated prior to the final fully connected layer.
- **Training:** AdamW optimizer, batch size 128. Combats severe data imbalance dynamically via a `pos_weight` in `BCEWithLogitsLoss`. Includes a validation loop looking for `validation_data/` mapped to Early Stopping (patience=5), finishing with a dynamic Threshold Optimization (grid-search) based on validation probabilities to maximize the F-measure.
- **Limitations:** Performance saturated around `~0.44` Challenge Score. Although 1D convolutions extract morphological features well, they structurally fail to adequately capture long-term Heart Rate Variability (HRV) patterns over a fixed 10-second window, justifying a shift towards 2D Spectrogram approaches.

### 4. Data Splitting Pipeline (`split_data.py`)
An automated local script to properly arrange the datasets prior to training.
- Uses `scikit-learn` to perform a **Stratified Split**: 80% Train, 10% Validation, 10% Holdout.
- Instead of physically moving large files, it generates lightweight **Symlinks** to the raw data files (`.hea`, `.dat`, `.mat`).
- Exports metadata CSVs per folder for data tracking.

### 5. Local Virtual Environment Reference
A local `venv` symlink simplifies environment activation.

- **Path:** `./venv` -> `/home/hadi/Coding/ML/ptorch_env`
- **Activation:**
  ```bash
  source .venv/bin/activate
  ```

##  Data Sources (PhysioNet 2025)
The challenge uses datasets from Central/South America and Europe:
- **CODE-15%:** ~300,000 records (Brazil). Length: 7.3s or 10.2s. `FS`: 400Hz. **Labels:** Weak (self-reported), mostly negative.
- **SaMi-Trop:** 1,631 records (Brazil). Length: 7.3s or 10.2s. `FS`: 400Hz. **Labels:** Strong (serological tests), all positive.
- **PTB-XL:** 21,799 records (Europe). Length: 10s. `FS`: 500Hz. **Labels:** Strong (geography-based), all negative.

## 📂 Project Structure

- `split_data.py`: Prepares the dataset directories (`training_data`, `validation_data`, `holdout_data`) using stratified symlinking.
- `train_model.py`: Entry point for training. Supports dynamic module loading.
- `run_model.py`: Entry point for inference. Supports dynamic module loading.
- `helper_code.py`: Core utility functions (PhysioNet standard).
- `team_code.py`: Default implementation.
- `team_code_*.py`: Alternative implementations (e.g., `classic_machine_learning`, `1D_CNN`).

## 💻 Hardware & Execution Environment
- **Primary Execution Environment (Laptop):**
  - **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU
  - **VRAM:** 4096 MB
  - **CPU:** Intel Core i5-12450H
  - **RAM:** 7.38 GB
  - **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU
  - **VRAM:** 4096 MB
  - **CPU:** Intel Core i5-12450H
  - **RAM:** 7.38 GB
- **Secondary Execution Environments:** Google Colab, Kaggle Notebook
*Code is prioritized to run on the primary laptop environment, then adapted for Colab/Kaggle as needed.*

## 📄 Documentation
- **Main README:** `README.md` (English)
- **Persian README:** `README_FA.md` (MUST be kept in sync / updated whenever `README.md` is updated).

## ⚖️ Evaluation
Use `evaluate_model.py` to calculate scores (AUROC, AUPRC, etc.) from model outputs.
```bash
python evaluate_model.py -d holdout_data -o holdout_outputs -s scores.csv
```
