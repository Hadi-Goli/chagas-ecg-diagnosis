# Project Context: Chagas ECG Diagnosis

## 🚀 Key Features

### 1. Dynamic Team Module Selection (`-tm` / `--team_module`)
The pipeline supports multiple `team_code.py` implementations. You can switch between them without modifying the training/running scripts.

- **Default:** `team_code`
- **Usage:**
  ```bash
  python train_model.py -d training_data -m model -tm team_code_spectrogram
  python run_model.py -d holdout_data -m model -o holdout_outputs -tm team_code_spectrogram
  ```
- **Modified Scripts:** `train_model.py`, `run_model.py`.

### 2. Local Virtual Environment Reference
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
- `team_code_*.py`: Alternative implementations (e.g., `random_forest`, `spectrogram`).

## ⚖️ Evaluation
Use `evaluate_model.py` to calculate scores (AUROC, AUPRC, etc.) from model outputs.
```bash
python evaluate_model.py -d holdout_data -o holdout_outputs -s scores.csv
```
