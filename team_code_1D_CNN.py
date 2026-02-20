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
        
        # Preprocess
        processed_signal = preprocess_signal(signal, fs, current_leads)
        
        return torch.tensor(processed_signal, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

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
    def __init__(self, block, num_blocks, num_classes=1):
        super(ResNet1D, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
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

    # Create dataset and dataloader
    if verbose:
        print('Creating dataset...')
    dataset = ECGDataset(records, data_folder)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f'Using device: {device}')
    
    model = ResNet18_1D().to(device)
    
    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    if verbose:
        print('Starting training...')
    
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1) # Match shape (B, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if verbose:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(dataloader):.4f}')

    # Save model
    if verbose:
        print('Saving model...')
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, 'model_1D_CNN.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save config (optional but good practice)
    config = {
        'fs': TARGET_FS,
        'length': TARGET_LENGTH,
        'leads': LEADS
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
    return model

################################################################################
#
# Running function
#
################################################################################

def run_model(record, model, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load signal
    signal, fields = load_signals(record)
    header = load_header(record)
    
    fs = get_sampling_frequency(header)
    current_leads = get_signal_names(header)
    
    # Preprocess
    processed_signal = preprocess_signal(signal, fs, current_leads)
    
    # Prepare for inference
    input_tensor = torch.tensor(processed_signal, dtype=torch.float32).unsqueeze(0).to(device) # Add batch dim
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()
    
    # Thresholding
    binary_prediction = probability > 0.5
    
    return binary_prediction, probability