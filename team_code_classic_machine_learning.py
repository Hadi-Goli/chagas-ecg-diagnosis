#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import signal
from scipy.stats import skew, kurtosis
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
import sys
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting advanced ECG features and labels from the data...')

    # Iterate over the records to extract the features and labels.
    features = list()
    labels = list()
    sample_weights = list()
    
    for i in tqdm(range(num_records), desc="Processing records", unit="record", mininterval=1.0):
        if verbose:
            width = len(str(num_records))
            # print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        
        try:
            # Extract comprehensive features
            feature_vector, dataset_source = extract_comprehensive_features(record)
            label = load_label(record)
            
            # Handle different datasets with appropriate sampling and weighting
            include_sample = True
            base_weight = 1.0
            
            if dataset_source == 'CODE-15%':
                # Skip most CODE-15% samples due to weak labels and large volume, but keep some
                if (i % 8) != 0:  # Keep 1/8 of CODE-15% samples
                    include_sample = False
                else:
                    base_weight = 0.7  # Reduce weight for weak labels
            elif dataset_source in ['SaMi-Trop', 'PTB-XL']:
                base_weight = 1.0  # Strong labels get full weight
            
            if include_sample:
                features.append(feature_vector)
                labels.append(label)
                
                # Additional weighting for class balance
                if label == 1:  # Positive class (Chagas)
                    sample_weights.append(base_weight * 2.5)  # Upweight positive cases
                else:
                    sample_weights.append(base_weight)
                    
        except Exception as e:
            if verbose:
                print(f'Error processing {records[i]}: {e}')
            continue

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=bool)
    sample_weights = np.asarray(sample_weights, dtype=np.float32)

    if verbose:
        print(f'Training data shape: {features.shape}')
        print(f'Class distribution: {np.bincount(labels.astype(int))}')

    # Handle severe class imbalance with SMOTE + ENN
    if len(np.unique(labels)) > 1 and np.sum(labels) > 1:
        if verbose:
            print('Applying SMOTE + ENN resampling for class balance...')
        try:
            smote_enn = SMOTEENN(
                smote=SMOTE(random_state=42, k_neighbors=min(5, np.sum(labels) - 1)),
                enn=EditedNearestNeighbours(n_neighbors=3),
                random_state=42
            )
            features_resampled, labels_resampled = smote_enn.fit_resample(features, labels)
            if verbose:
                print(f'After resampling: {features_resampled.shape}')
                print(f'New class distribution: {np.bincount(labels_resampled.astype(int))}')
            
            # Use resampled data and reset weights
            features = features_resampled
            labels = labels_resampled
            sample_weights = np.ones(len(labels))
            
        except Exception as e:
            if verbose:
                print(f'Resampling failed: {e}, proceeding without resampling')

    # Train ensemble models
    if verbose:
        print('Training ensemble models...')

    models = {}
    scalers = {}
    
    # Model 1: Random Forest with balanced class weights
    if verbose:
        print('Training Random Forest with advanced features...')
    
    scaler_rf = RobustScaler()
    features_scaled_rf = scaler_rf.fit_transform(features)
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(features_scaled_rf, labels, sample_weight=sample_weights)
    
    models['rf'] = rf_model
    scalers['rf'] = scaler_rf
    
    # Model 2: Gradient Boosting
    if verbose:
        print('Training Gradient Boosting...')
    
    scaler_gb = StandardScaler()
    features_scaled_gb = scaler_gb.fit_transform(features)
    
    gb_model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(features_scaled_gb, labels, sample_weight=sample_weights)
    
    models['gb'] = gb_model
    scalers['gb'] = scaler_gb
    
    # Model 3: Logistic Regression
    if verbose:
        print('Training Logistic Regression...')
    
    scaler_lr = StandardScaler()
    features_scaled_lr = scaler_lr.fit_transform(features)
    
    lr_model = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
        C=0.1
    )
    lr_model.fit(features_scaled_lr, labels, sample_weight=sample_weights)
    
    models['lr'] = lr_model
    scalers['lr'] = scaler_lr

    # Create ensemble model
    ensemble_model = {
        'models': models,
        'scalers': scalers,
        'ensemble_weights': {'rf': 0.4, 'gb': 0.4, 'lr': 0.2}
    }

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_model(model_folder, ensemble_model)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model_classic_machine_learning.sav')
    model_data = joblib.load(model_filename)
    return model_data

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model_data, verbose):
    # Extract the comprehensive features.
    feature_vector, dataset_source = extract_comprehensive_features(record)
    features = feature_vector.reshape(1, -1)

    # Get ensemble models
    models = model_data['model']['models']
    scalers = model_data['model']['scalers']
    ensemble_weights = model_data['model']['ensemble_weights']

    # Get predictions from ensemble
    ensemble_prob = 0
    for model_name, weight in ensemble_weights.items():
        scaler = scalers[model_name]
        model = models[model_name]
        
        features_scaled = scaler.transform(features)
        prob = model.predict_proba(features_scaled)[0, 1]
        ensemble_prob += weight * prob

    # Binary prediction with optimized threshold
    threshold = 0.45  # Slightly lower threshold to catch more positive cases
    binary_output = bool(ensemble_prob >= threshold)
    probability_output = float(ensemble_prob)

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

def normalize_sampling_rate(ecg_data, current_rate, target_rate=400):
    """Normalize ECG data to target sampling rate"""
    if current_rate == target_rate:
        return ecg_data
    
    # Calculate resampling factor
    factor = target_rate / current_rate
    new_length = int(ecg_data.shape[0] * factor)
    
    # Resample each lead
    resampled_data = np.zeros((new_length, ecg_data.shape[1]))
    for lead in range(ecg_data.shape[1]):
        resampled_data[:, lead] = signal.resample(ecg_data[:, lead], new_length)
    
    return resampled_data

def extract_morphological_features(ecg_lead):
    """Extract morphological features from ECG lead"""
    features = []
    
    # Basic statistical features
    features.extend([
        np.mean(ecg_lead),
        np.std(ecg_lead),
        np.var(ecg_lead),
        skew(ecg_lead),
        kurtosis(ecg_lead),
        np.median(ecg_lead),
        np.ptp(ecg_lead),  # peak-to-peak
        np.percentile(ecg_lead, 25),
        np.percentile(ecg_lead, 75)
    ])
    
    # RMS and energy features
    rms = np.sqrt(np.mean(ecg_lead**2))
    features.append(rms)
    features.append(np.sum(ecg_lead**2))  # Energy
    
    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(ecg_lead)) != 0)
    features.append(zero_crossings / len(ecg_lead))
    
    return features

def extract_frequency_features(ecg_lead, sampling_rate=400):
    """Extract frequency domain features"""
    features = []
    
    try:
        # Power spectral density
        freqs, psd = signal.welch(ecg_lead, fs=sampling_rate, nperseg=min(256, len(ecg_lead)//4))
        
        # Frequency bands (typical for ECG analysis)
        bands = {
            'very_low': (0.01, 0.1),
            'low': (0.1, 0.5),
            'mid': (0.5, 2.0),
            'high': (2.0, 10.0),
            'very_high': (10.0, 40.0)
        }
        
        total_power = np.sum(psd)
        
        for band_name, (low_freq, high_freq) in bands.items():
            band_indices = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
            if len(band_indices) > 0:
                band_power = np.sum(psd[band_indices])
                features.append(band_power / total_power if total_power > 0 else 0)
            else:
                features.append(0)
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(psd)
        features.append(freqs[dominant_freq_idx])
        
        # Spectral centroid
        spectral_centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
        features.append(spectral_centroid)
        
    except:
        # Fallback if frequency analysis fails
        features = [0] * 7
    
    return features

def extract_rr_interval_features(ecg_lead, sampling_rate=400):
    """Extract R-R interval features (HRV analysis)"""
    try:
        # Simple R-peak detection
        peaks, _ = signal.find_peaks(ecg_lead, height=np.std(ecg_lead), 
                                   distance=int(0.6 * sampling_rate))
        
        if len(peaks) < 3:
            return [60, 0, 0, 0, 0, 0, 0, 0]  # Default values if not enough peaks
        
        # Calculate R-R intervals
        rr_intervals = np.diff(peaks) / sampling_rate * 1000  # in ms
        
        features = []
        # Time domain HRV features
        features.extend([
            np.mean(rr_intervals),  # Mean RR
            np.std(rr_intervals),   # SDNN
            np.std(np.diff(rr_intervals)),  # RMSSD approximation
            len(peaks) / (len(ecg_lead) / sampling_rate) * 60,  # Heart rate
            np.ptp(rr_intervals),   # Range of RR intervals
            skew(rr_intervals),
            kurtosis(rr_intervals),
            np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) if len(rr_intervals) > 1 else 0
        ])
        
        return features
    except:
        return [60, 0, 0, 0, 0, 0, 0, 0]

def extract_comprehensive_features(record):
    """Extract comprehensive features from ECG record"""
    
    header = load_header(record)

    # Extract basic demographic features
    age = get_age(header)
    age = np.array([age if age is not None else 50])  # Default age if missing

    # Extract sex and encode
    sex = get_sex(header)
    sex_features = np.zeros(3)
    if sex and sex.casefold().startswith('f'):
        sex_features[0] = 1
    elif sex and sex.casefold().startswith('m'):
        sex_features[1] = 1
    else:
        sex_features[2] = 1

    # Extract dataset source
    source = get_source(header)
    
    # Load signal data
    signal_data, fields = load_signals(record)
    channels = fields['sig_name']
    sampling_rate = fields.get('fs', 400)

    # Reorder channels to standard 12-lead format
    reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    signal_data = reorder_signal(signal_data, channels, reference_channels)
    
    # Normalize sampling rate
    signal_data = normalize_sampling_rate(signal_data, sampling_rate, target_rate=400)
    
    # Extract advanced ECG features
    all_features = []
    
    # Add demographic features
    all_features.extend(age)
    all_features.extend(sex_features)
    
    # Extract features from each lead
    num_leads = min(12, signal_data.shape[1])
    
    for lead_idx in range(num_leads):
        ecg_lead = signal_data[:, lead_idx]
        
        # Handle missing or invalid data
        if not np.any(np.isfinite(ecg_lead)) or len(ecg_lead) == 0:
            # Add zeros for this lead
            all_features.extend([0] * 19)  # 12 morph + 7 freq features
            continue
        
        # Clean the signal (remove NaN/inf values)
        ecg_lead = ecg_lead[np.isfinite(ecg_lead)]
        if len(ecg_lead) == 0:
            all_features.extend([0] * 19)
            continue
        
        # Morphological features (12 features)
        morph_features = extract_morphological_features(ecg_lead)
        all_features.extend(morph_features)
        
        # Frequency features (7 features)
        freq_features = extract_frequency_features(ecg_lead, 400)
        all_features.extend(freq_features)
    
    # Pad with zeros if we have fewer than 12 leads
    while len(all_features) < 4 + 12 * 19:  # 4 demographic + 12 leads * 19 features
        all_features.extend([0] * 19)
    
    # Extract R-R interval features from lead II (index 1)
    if num_leads >= 2:
        rr_features = extract_rr_interval_features(signal_data[:, 1], 400)
        all_features.extend(rr_features)
    else:
        all_features.extend([60, 0, 0, 0, 0, 0, 0, 0])  # Default RR features
    
    # Cross-lead correlation features
    if num_leads >= 6:
        # Calculate correlations between important lead pairs
        lead_pairs = [(0, 1), (1, 2), (4, 5)]  # I-II, II-III, AVL-AVF
        for i, j in lead_pairs:
            if i < num_leads and j < num_leads:
                lead_i = signal_data[:, i]
                lead_j = signal_data[:, j]
                
                # Clean data
                valid_idx = np.isfinite(lead_i) & np.isfinite(lead_j)
                if np.sum(valid_idx) > 10:
                    corr = np.corrcoef(lead_i[valid_idx], lead_j[valid_idx])[0, 1]
                    all_features.append(corr if not np.isnan(corr) else 0)
                else:
                    all_features.append(0)
            else:
                all_features.append(0)
    else:
        all_features.extend([0, 0, 0])
    
    # Dataset source indicators (helps with domain adaptation)
    source_features = [0, 0, 0]  # CODE-15%, SaMi-Trop, PTB-XL
    if source:
        if 'CODE' in source.upper():
            source_features[0] = 1
        elif 'SAMI' in source.upper():
            source_features[1] = 1
        elif 'PTB' in source.upper():
            source_features[2] = 1
    
    all_features.extend(source_features)
    
    # Convert to numpy array and handle any remaining NaN/inf values
    feature_vector = np.array(all_features, dtype=np.float32)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return feature_vector, source

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model_classic_machine_learning.sav')
    joblib.dump(d, filename, protocol=0)