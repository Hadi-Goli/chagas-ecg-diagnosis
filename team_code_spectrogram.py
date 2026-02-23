#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

import os
import numpy as np
import scipy.signal
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms.functional as TF
from torch.amp import autocast, GradScaler
from helper_code import *

# Constants
TARGET_FS = 400
TARGET_LENGTH = 4000  # 10 seconds at 400 Hz
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
BATCH_SIZE = 64
EPOCHS = 10#50
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_BACKBONE = 1e-4

# Preprocessing
def preprocess_signal(signal, fs, current_leads, augment=False):
    # 1. Reorder
    signal = reorder_signal(signal, current_leads, LEADS)
    # 2. Resample
    if fs != TARGET_FS:
        num_samples = int(signal.shape[0] * TARGET_FS / fs)
        signal = scipy.signal.resample(signal, num_samples, axis=0)
    
    # Optional primitive augmentation
    if augment:
        # Time shift
        shift = np.random.randint(-200, 200)
        signal = np.roll(signal, shift, axis=0)
        # Amplitude scaling
        scale = np.random.uniform(0.9, 1.1)
        signal = signal * scale
        # Minimal noise
        noise = np.random.normal(0, 0.01, signal.shape)
        signal = signal + noise

    # 3. Z-score
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0)
    std = np.where(std == 0, 1, std)
    signal = (signal - mean) / std

    # 4. Pad or Crop
    current_length = signal.shape[0]
    if current_length < TARGET_LENGTH:
        pad_len = TARGET_LENGTH - current_length
        signal = np.pad(signal, ((0, pad_len), (0, 0)), 'constant')
    elif current_length > TARGET_LENGTH:
        start = (current_length - TARGET_LENGTH) // 2
        signal = signal[start:start+TARGET_LENGTH, :]

    # Return shape (Channels, Length)
    return signal.transpose()

# Spectrogram conversion
def signal_to_spectrogram_image(signal_tensor):
    """
    Transforms [12, 4000] signal to a [3, 224, 224] RGB image grid of spectrograms.
    """
    device = signal_tensor.device
    # Compute STFT
    # n_fft=512, hop_length=64, win_length=512
    window = torch.hann_window(512).to(device)
    # Output shape: [12, Freq=257, Time=(4000//64+1)=63] depending on padding
    # return_complex=True is now standard in PyTorch
    stft = torch.stft(signal_tensor, n_fft=512, hop_length=64, win_length=512, window=window, return_complex=True)
    
    # Magnitude
    mag = torch.abs(stft)
    
    # Crop to 0-40 Hz
    # freq resolution = fs / n_fft = 400 / 512 = 0.78125 Hz per bin
    # 40 Hz = 40 / 0.78125 = 51.2 -> 52 bins
    mag = mag[:, :52, :]
    
    # dB conversion with clipping [-80, 0]
    ref = mag.max() + 1e-8
    db = 20 * torch.log10(mag / ref + 1e-8)
    db = torch.clamp(db, min=-80.0, max=0.0)
    
    # Z-score normalize the image conceptually
    # (per sample, across all leads)
    mean = db.mean()
    std = db.std() + 1e-8
    db = (db - mean) / std
    
    # db shape is [12, 52, Time~63]
    num_leads, freq_bins, time_bins = db.shape
    
    # Create 4x3 grid
    # Grid shape: [4*freq_bins, 3*time_bins]
    grid = torch.zeros((4 * freq_bins, 3 * time_bins), dtype=torch.float32, device=device)
    for i in range(12):
        row = i // 3
        col = i % 3
        grid[row*freq_bins:(row+1)*freq_bins, col*time_bins:(col+1)*time_bins] = db[i]
        
    # Add channel dim to make it [1, H, W]
    grid = grid.unsqueeze(0)
    
    # Resize to 224x224 (requires float tensor)
    # Use bilinear interpolation
    grid = TF.resize(grid, [224, 224], antialias=True)
    
    # Repeat to 3 channels for ImageNet models
    grid = grid.repeat(3, 1, 1)
    
    return grid

# Dataset
class SpectrogramDataset(Dataset):
    def __init__(self, records, data_folder, augment=False):
        self.records = records
        self.data_folder = data_folder
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_path = os.path.join(self.data_folder, self.records[idx])
        signal, fields = load_signals(record_path)
        header = load_header(record_path)
        
        fs = get_sampling_frequency(header)
        current_leads = get_signal_names(header)
        label = load_label(record_path)
        
        age = get_age(header)
        age = age if age is not None else 50.0
        
        sex = get_sex(header)
        sex_encoded = 0.0
        if sex:
            if sex.casefold().startswith('f'):
                sex_encoded = 1.0 # Female
            elif sex.casefold().startswith('m'):
                sex_encoded = -1.0 # Male
                
        demographics = np.array([age / 100.0, sex_encoded], dtype=np.float32)
        
        processed_signal = preprocess_signal(signal, fs, current_leads, self.augment)
        
        return (torch.tensor(processed_signal, dtype=torch.float32), 
                torch.tensor(demographics, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32))

# Model Architecture
class SpectrogramEfficientNet(nn.Module):
    def __init__(self, num_classes=1, extra_features=2):
        super(SpectrogramEfficientNet, self).__init__()
        # Load pretrained EfficientNet-B0
        # By default, num_classes is 1000 for ImageNet, so we drop the classifier
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = self.backbone.classifier[1].in_features
        
        # Replace the classifier with Identity so it outputs the raw pool vector
        self.backbone.classifier = nn.Identity()
        
        # New Head combining image features with demographics
        self.head = nn.Sequential(
            nn.Linear(num_ftrs + extra_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
            # No Sigmoid here! BCEWithLogitsLoss expects raw logits
        )

    def forward(self, img, demographics):
        # img shape: [B, 3, 224, 224]
        features = self.backbone(img) # [B, 1280]
        
        combined = torch.cat((features, demographics), dim=1) # [B, 1280 + 2]
        output = self.head(combined) # [B, 1]
        
        return output

# Training Function
def train_model(data_folder, model_folder, verbose):
    if verbose:
        print('Finding data records...')
    records = find_records(data_folder)
    num_records = len(records)
    if num_records == 0:
        raise Exception('No data was provided.')
        
    base_dir = os.path.dirname(data_folder) if os.path.dirname(data_folder) else '.'
    val_folder = os.path.join(base_dir, 'validation_data')
    
    if os.path.exists(val_folder):
        if verbose:
            print(f'Using dedicated validation data folder: {val_folder}')
        train_records = records
        val_records = find_records(val_folder)
        val_data_folder = val_folder
    else:
        if verbose:
            print(f'Dedicated validation folder not found. Splitting (80/20)...')
        np.random.seed(42)
        np.random.shuffle(records)
        split_idx = int(0.8 * num_records)
        train_records = records[:split_idx]
        val_records = records[split_idx:]
        val_data_folder = data_folder

    if verbose:
        print(f'Creating datasets... Train: {len(train_records)}, Val: {len(val_records)}')
    
    train_dataset = SpectrogramDataset(train_records, data_folder, augment=True)
    val_dataset = SpectrogramDataset(val_records, val_data_folder, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f'Using device: {device}')
    
    model = SpectrogramEfficientNet().to(device)
    
    # Calculate pos_weight
    if verbose:
        print('Estimating class weights...')
    num_pos, num_neg = 0.0, 0.0
    subset_size = min(5000, len(train_records))
    for i in range(subset_size):
        lbl = load_label(os.path.join(data_folder, train_records[i]))
        if lbl == 1: num_pos += 1
        else: num_neg += 1
            
    num_pos = max(1.0, num_pos)
    num_neg = max(1.0, num_neg)
    
    pos_weight_val = min((num_neg / num_pos) * 0.5, 15.0) 
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    if verbose:
        print(f'Estimated pos_weight: {pos_weight_val:.2f}')

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer with differential learning rates
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': LEARNING_RATE_BACKBONE},
        {'params': model.head.parameters(), 'lr': LEARNING_RATE_HEAD}
    ], weight_decay=1e-4)

    # AMP Scaler
    scaler = GradScaler('cuda')

    best_val_loss = float('inf')
    best_val_probs, best_val_labels = [], []
    
    patience, patience_counter = 5, 0
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, 'model_spectrogram.pth')

    if verbose:
        print('Starting training...')
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
        
        for inputs, demos, labels in pbar:
            inputs = inputs.to(device)
            demos = demos.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            # Convert raw 1D signals to 2D image grids on GPU
            images = torch.stack([signal_to_spectrogram_image(sig) for sig in inputs])

            optimizer.zero_grad()
            
            # Using AMP for Mixed Precision
            with autocast('cuda'):
                outputs = model(images, demos)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss = running_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        epoch_val_probs, epoch_val_labels = [], []
        pbar_val = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]')
        
        with torch.no_grad():
            for inputs, demos, labels in pbar_val:
                inputs = inputs.to(device)
                demos = demos.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                images = torch.stack([signal_to_spectrogram_image(sig) for sig in inputs])
                
                with autocast('cuda'):
                    outputs = model(images, demos)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                epoch_val_probs.extend(probs.cpu().numpy())
                epoch_val_labels.extend(labels.cpu().numpy())
                
                preds = (probs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        if verbose:
            print(f'Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_val_probs, best_val_labels = epoch_val_probs, epoch_val_labels
            
            if verbose:
                print('Validation loss improved. Saving model checkpoint...')
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f'Early stopping triggered after {patience} epochs without improvement!')
                break

    if verbose:
        print('Calculating final dynamic decision threshold...')
    
    from sklearn.metrics import f1_score
    best_threshold, best_f1 = 0.5, 0.0
    val_probs_arr, val_labels_arr = np.array(best_val_probs), np.array(best_val_labels)
    
    for th in np.arange(0.1, 0.95, 0.05):
        preds = (val_probs_arr >= th).astype(int)
        score = f1_score(val_labels_arr, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = th

    if verbose:
        print(f'Optimal Validation Threshold found: {best_threshold:.2f} (F1: {best_f1:.4f})')

    config = {
        'fs': TARGET_FS,
        'length': TARGET_LENGTH,
        'leads': LEADS,
        'threshold': float(best_threshold)
    }
    np.save(os.path.join(model_folder, 'config_spectrogram.npy'), config)

# Load Model
def load_model(model_folder, verbose):
    if verbose:
        print('Loading Spectrogram model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpectrogramEfficientNet().to(device)
    
    model_path = os.path.join(model_folder, 'model_spectrogram.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    config_path = os.path.join(model_folder, 'config_spectrogram.npy')
    threshold = 0.5
    if os.path.exists(config_path):
        config = np.load(config_path, allow_pickle=True).item()
        threshold = config.get('threshold', 0.5)
        
    return {'model': model, 'threshold': threshold}

# Run Model
def run_model(record, model_dict, verbose):
    model = model_dict['model']
    threshold = model_dict['threshold']
    device = next(model.parameters()).device
    
    signal, fields = load_signals(record)
    header = load_header(record)
    
    fs = get_sampling_frequency(header)
    current_leads = get_signal_names(header)
    
    age = get_age(header)
    age = age if age is not None else 50.0
    
    sex = get_sex(header)
    sex_encoded = 0.0
    if sex:
        if sex.casefold().startswith('f'): sex_encoded = 1.0
        elif sex.casefold().startswith('m'): sex_encoded = -1.0
            
    demographics = np.array([age / 100.0, sex_encoded], dtype=np.float32)
    processed_signal = preprocess_signal(signal, fs, current_leads, augment=False)
    
    input_tensor = torch.tensor(processed_signal, dtype=torch.float32).to(device)
    demo_tensor = torch.tensor(demographics, dtype=torch.float32).unsqueeze(0).to(device)
    
    image = signal_to_spectrogram_image(input_tensor).unsqueeze(0)
    
    with torch.no_grad():
        with autocast('cuda'):
            output = model(image, demo_tensor)
        probability = torch.sigmoid(output).item()
    
    binary_prediction = probability >= threshold
    return binary_prediction, probability
