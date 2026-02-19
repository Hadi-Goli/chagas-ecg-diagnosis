def train_model(data_folder, model_folder, verbose):
    print("--- Running Spectrogram Trainer ---")
    print(f"Data: {data_folder}, Model: {model_folder}")

def load_model(model_folder, verbose):
    print("--- Loading Spectrogram Model ---")
    return "spec_model"

def run_model(record, model, verbose):
    print(f"--- Running Spectrogram on {record} ---")
    return 0, 0.1
