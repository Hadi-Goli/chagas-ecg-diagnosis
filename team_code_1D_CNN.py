#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions.
#
################################################################################

import os
import numpy as np
import scipy.signal
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from helper_code import *

################################################################################
#
# Global constants and configuration
#
################################################################################

TARGET_FS = 500
TARGET_LENGTH = 5000  # 10 seconds at 500 Hz
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3

################################################################################
#
# Preprocessing functions
#
################################################################################

def preprocess_signal(signal, fs, current_leads):
    """
    Preprocess the ECG signal:
    1. Reorder leads to standard 12-lead order.
    2. Resample to TARGET_FS.
    3. Normalize (Z-score).
    4. Pad or crop to TARGET_LENGTH.
    """
    # 1. Reorder leads
    signal = reorder_signal(signal, current_leads, LEADS)

    # 2. Resample
    if fs != TARGET_FS:
        num_samples = int(signal.shape[0] * TARGET_FS / fs)
        signal = scipy.signal.resample(signal, num_samples, axis=0)

    # 3. Normalize (Z-score per lead)
    # Handle potential division by zero if std is 0
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0)
    std = np.where(std == 0, 1, std)
    signal = (signal - mean) / std

    # 4. Pad or Crop
    current_length = signal.shape[0]
    if current_length < TARGET_LENGTH:
        pad_len = TARGET_LENGTH - current_length
        # Pad with zeros at the end
        signal = np.pad(signal, ((0, pad_len), (0, 0)), 'constant')
    elif current_length > TARGET_LENGTH:
        # Crop from the center
        start = (current_length - TARGET_LENGTH) // 2
        signal = signal[start:start+TARGET_LENGTH, :]

    # Transpose to (Channels, Length) for PyTorch
    signal = signal.transpose()
    
    return signal

################################################################################
#
# Dataset Class
#
################################################################################

class ECGDataset(Dataset):
    def __init__(self, records, data_folder):
        self.records = records
        self.data_folder = data_folder

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_name = self.records[idx]
        record_path = os.path.join(self.data_folder, record_name)
        
        # Load signal and header
        signal, fields = load_signals(record_path)
        header = load_header(record_path)
        
        # Get metadata
        fs = get_sampling_frequency(header)
        current_leads = get_signal_names(header)
        label = load_label(record_path)
        
        # Extract demographic features (Age & Sex)
        age = get_age(header)
        age = age if age is not None else 50.0 # Default age
        
        sex = get_sex(header)
        sex_encoded = 0.0 # Unknown
        if sex:
            if sex.casefold().startswith('f'):
                sex_encoded = 1.0 # Female
            elif sex.casefold().startswith('m'):
                sex_encoded = -1.0 # Male
                
        demographics = np.array([age / 100.0, sex_encoded], dtype=np.float32) # Normalize age roughly
        
        # Preprocess
        processed_signal = preprocess_signal(signal, fs, current_leads)
        
        return (torch.tensor(processed_signal, dtype=torch.float32), 
                torch.tensor(demographics, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32))

################################################################################
#
# Model Architecture (ResNet-1D)
#
################################################################################

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1, extra_features=2):
        super(ResNet1D, self).__init__()
        self.in_planes = 64

        # Changed kernel_size to 51 to capture wider temporal context (approx 100ms at 500Hz)
        self.conv1 = nn.Conv1d(12, 64, kernel_size=51, stride=2, padding=25, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        # Wider maxpool
        self.maxpool = nn.MaxPool1d(kernel_size=7, stride=2, padding=3)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.5)
        
        # Connect extra features (age, sex) alongside CNN features
        self.fc = nn.Linear(512 * block.expansion + extra_features, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, demographics):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        # Concatenate demographic features
        x = torch.cat((x, demographics), dim=1)
        x = self.fc(x)
        return x

def ResNet18_1D():
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2])

################################################################################
#
# Training function
#
################################################################################

def train_model(data_folder, model_folder, verbose):
    # Find data records
    if verbose:
        print('Finding data records...')
    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise Exception('No data was provided.')
        
    # Check for a dedicated validation folder (assuming it's a sibling to data_folder)
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
            print(f'Dedicated validation folder not found. Splitting {data_folder} (80/20)...')
        np.random.seed(42)
        np.random.shuffle(records)
        split_idx = int(0.8 * num_records)
        train_records = records[:split_idx]
        val_records = records[split_idx:]
        val_data_folder = data_folder

    # Create datasets
    if verbose:
        print(f'Creating datasets... Train: {len(train_records)}, Val: {len(val_records)}')
    train_dataset = ECGDataset(train_records, data_folder)
    val_dataset = ECGDataset(val_records, val_data_folder)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f'Using device: {device}')
    
    model = ResNet18_1D().to(device)
    
    # Calculate pos_weight for Imbalanced Dataset
    # Reading a subset to estimate ratio, as reading all labels is slow
    if verbose:
        print('Estimating class weights from a subset of train data...')
    num_pos = 0.0
    num_neg = 0.0
    subset_size = min(5000, len(train_records))
    for i in range(subset_size):
        lbl = load_label(os.path.join(data_folder, train_records[i]))
        if lbl == 1:
            num_pos += 1
        else:
            num_neg += 1
            
    # Avoid division by zero
    num_pos = max(1.0, num_pos)
    num_neg = max(1.0, num_neg)
    
    # Up-weight the minority class (usually Positive for Chagas in CODE-15%)
    # Using a cautious scale to prevent extreme gradients
    pos_weight_val = min((num_neg / num_pos) * 0.5, 15.0) 
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    
    if verbose:
        print(f'Estimated pos_weight: {pos_weight_val:.2f}')

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Early Stopping tracking
    best_val_loss = float('inf')
    best_val_probs = []
    best_val_labels = []
    
    patience = 5
    patience_counter = 0
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, 'model_1D_CNN.pth')

    # Training Loop
    if verbose:
        print('Starting training...')
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
        
        for inputs, demos, labels in pbar:
            inputs = inputs.to(device)
            demos = demos.to(device)
            labels = labels.to(device).unsqueeze(1) # Match shape (B, 1)

            optimizer.zero_grad()
            outputs = model(inputs, demos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss = running_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        epoch_val_probs = []
        epoch_val_labels = []
        
        pbar_val = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]')
        
        with torch.no_grad():
            for inputs, demos, labels in pbar_val:
                inputs = inputs.to(device)
                demos = demos.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                outputs = model(inputs, demos)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                epoch_val_probs.extend(probs.cpu().numpy())
                epoch_val_labels.extend(labels.cpu().numpy())
                
                preds = (probs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        if verbose:
            print(f'Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
            
        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save probabilities for threshold calculation later
            best_val_probs = []
            best_val_labels = []
            
            # We need to collect raw probabilities for the *best* model to calculate threshold
            # Since we just ran validation, we could theoretically reuse those tensors if we stored them,
            # but standard practice is to recalculate them precisely or just store them per epoch.
            # For efficiency, we will store them during the epoch evaluation loop above.
            
            if verbose:
                print('Validation loss improved. Saving model checkpoint...')
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f'Early stopping triggered after {patience} epochs without improvement!')
                break

    # Optimal Threshold Search
    if verbose:
        print('Calculating final dynamic decision threshold...')
    
    from sklearn.metrics import f1_score
    best_threshold = 0.5
    best_f1 = 0.0
    
    val_probs_arr = np.array(best_val_probs)
    val_labels_arr = np.array(best_val_labels)
    
    for th in np.arange(0.1, 0.95, 0.05):
        preds = (val_probs_arr >= th).astype(int)
        score = f1_score(val_labels_arr, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = th

    if verbose:
        print(f'Optimal Validation Threshold found: {best_threshold:.2f} (F1: {best_f1:.4f})')

    # Save config
    config = {
        'fs': TARGET_FS,
        'length': TARGET_LENGTH,
        'leads': LEADS,
        'threshold': float(best_threshold)
    }
    np.save(os.path.join(model_folder, 'config_1D_CNN.npy'), config)

################################################################################
#
# File I/O functions
#
################################################################################

def load_model(model_folder, verbose):
    if verbose:
        print('Loading model...')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18_1D().to(device)
    
    model_path = os.path.join(model_folder, 'model_1D_CNN.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load optimal threshold
    config_path = os.path.join(model_folder, 'config_1D_CNN.npy')
    threshold = 0.5
    if os.path.exists(config_path):
        config = np.load(config_path, allow_pickle=True).item()
        threshold = config.get('threshold', 0.5)
        
    return {'model': model, 'threshold': threshold}

################################################################################
#
# Running function
#
################################################################################

def run_model(record, model_dict, verbose):
    model = model_dict['model']
    threshold = model_dict['threshold']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load signal
    signal, fields = load_signals(record)
    header = load_header(record)
    
    fs = get_sampling_frequency(header)
    current_leads = get_signal_names(header)
    
    # Extract demographic features (Age & Sex)
    age = get_age(header)
    age = age if age is not None else 50.0 # Default age
    
    sex = get_sex(header)
    sex_encoded = 0.0 # Unknown
    if sex:
        if sex.casefold().startswith('f'):
            sex_encoded = 1.0 # Female
        elif sex.casefold().startswith('m'):
            sex_encoded = -1.0 # Male
            
    demographics = np.array([age / 100.0, sex_encoded], dtype=np.float32)

    # Preprocess
    processed_signal = preprocess_signal(signal, fs, current_leads)
    
    # Prepare for inference
    input_tensor = torch.tensor(processed_signal, dtype=torch.float32).unsqueeze(0).to(device) # Add batch dim
    demo_tensor = torch.tensor(demographics, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor, demo_tensor)
        probability = torch.sigmoid(output).item()
    
    # Thresholding with optimal threshold
    binary_prediction = probability >= threshold
    
    return binary_prediction, probability