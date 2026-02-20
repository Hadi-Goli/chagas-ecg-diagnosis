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

### 3. Local Virtual Environment Reference
A local `venv` symlink simplifies environment activation.

- **Path:** `./venv` -> `/home/hadi/Coding/ML/ptorch_env`
- **Activation:**
  ```bash
  source venv/bin/activate
  ```

## 📂 Project Structure

- `train_model.py`: Entry point for training. Supports dynamic module loading.
- `run_model.py`: Entry point for inference. Supports dynamic module loading.
- `helper_code.py`: Core utility functions (PhysioNet standard).
- `team_code.py`: Default implementation.
- `team_code_*.py`: Alternative implementations (e.g., `classic_machine_learning`, `spectrogram`).

## ⚖️ Evaluation
Use `evaluate_model.py` to calculate scores (AUROC, AUPRC, etc.) from model outputs.
```bash
python evaluate_model.py -d holdout_data -o holdout_outputs -s scores.csv
```
