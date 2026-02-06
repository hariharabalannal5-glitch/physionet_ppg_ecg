"""
PhysioNet Challenge 2015 - Signal Extraction Script
====================================================
This script extracts PPG and ECG II signals from the training dataset.

Author: Your Team
Date: 2024
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import warnings
warnings.filterwarnings('ignore')

class SignalExtractor:
    """Extract PPG and ECG II signals from PhysioNet Challenge 2015 dataset"""
    
    def __init__(self, training_dir='training', output_dir='extracted_data'):
        """
        Initialize the extractor
        
        Args:
            training_dir: Path to the training directory with .mat and .hea files
            output_dir: Path to save extracted signals
        """
        self.training_dir = training_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📁 Training directory: {training_dir}")
        print(f"📁 Output directory: {output_dir}")
    
    def read_header(self, header_file):
        """
        Read signal names and metadata from .hea file
        
        Args:
            header_file: Path to .hea file
            
        Returns:
            signal_names: List of signal names
            sampling_rate: Sampling rate in Hz
            num_samples: Number of samples
        """
        with open(header_file, 'r') as f:
            lines = f.readlines()
        
        # First line contains: record_name num_signals sampling_rate num_samples
        first_line = lines[0].strip().split()
        num_signals = int(first_line[1])
        sampling_rate = int(first_line[2])
        num_samples = int(first_line[3])
        
        # Subsequent lines contain signal information
        signal_names = []
        for i in range(1, num_signals + 1):
            parts = lines[i].strip().split()
            # Signal name is usually the last part or second-to-last part
            signal_name = parts[-1] if parts[-1] else parts[-2]
            signal_names.append(signal_name)
        
        return signal_names, sampling_rate, num_samples
    
    def read_record_label(self, record_name):
        """
        Read the label for a record (True alarm = 1, False alarm = 0)
        
        Args:
            record_name: Name of the record (e.g., 'a00')
            
        Returns:
            label: 1 for True alarm, 0 for False alarm
        """
        # Labels are typically stored in RECORDS-ANNOTATIONS or similar file
        # For this dataset, we'll read from the RECORDS file which has format:
        # record_name alarm_type label
        
        # Try to find the label from RECORDS file or annotations
        records_file = os.path.join(self.training_dir, 'RECORDS')
        
        if os.path.exists(records_file):
            with open(records_file, 'r') as f:
                for line in f:
                    if line.startswith(record_name):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            # The label might be in the file name or separate annotation
                            # For PhysioNet 2015, check the reference.txt or similar
                            pass
        
        # Alternative: Check for ANNOTATORS or reference files
        ref_file = os.path.join(self.training_dir, 'ANNOTATORS.txt')
        if not os.path.exists(ref_file):
            ref_file = os.path.join(self.training_dir, 'REFERENCE.csv')
        
        # Default implementation - you may need to adjust based on actual file structure
        # The training set should have labels in a CSV or text file
        return None  # Will be filled from reference file
    
    def extract_signals(self, record_name):
        """
        Extract PPG and ECG II signals from a single record
        
        Args:
            record_name: Name of the record (e.g., 'a00')
            
        Returns:
            ppg_signal: PPG signal array (num_samples,) or None
            ecg_signal: ECG II signal array (num_samples,) or None
        """
        # File paths
        mat_file = os.path.join(self.training_dir, f'{record_name}.mat')
        hea_file = os.path.join(self.training_dir, f'{record_name}.hea')
        
        if not os.path.exists(mat_file) or not os.path.exists(hea_file):
            print(f"⚠️  Files not found for {record_name}")
            return None, None
        
        # Read header to get signal names
        signal_names, sampling_rate, num_samples = self.read_header(hea_file)
        
        # Load MATLAB file
        mat_data = loadmat(mat_file)
        
        # The signal data is typically stored in 'val' key
        if 'val' in mat_data:
            signals = mat_data['val']
        else:
            print(f"⚠️  'val' key not found in {record_name}.mat")
            return None, None
        
        # Find PPG and ECG II indices
        ppg_signal = None
        ecg_signal = None
        
        # Common signal name variations
        ppg_names = ['PLETH', 'PPG', 'pleth', 'ppg', 'SpO2']
        ecg_names = ['II', 'ECG_II', 'ECGII', 'ECG II', 'Lead II']
        
        for i, name in enumerate(signal_names):
            # Check for PPG
            if any(ppg_name in name for ppg_name in ppg_names):
                ppg_signal = signals[i, :]
                print(f"  ✓ Found PPG at channel {i}: {name}")
            
            # Check for ECG II
            if any(ecg_name in name for ecg_name in ecg_names):
                ecg_signal = signals[i, :]
                print(f"  ✓ Found ECG II at channel {i}: {name}")
        
        return ppg_signal, ecg_signal
    
    def load_labels(self):
        """
        Load labels for all records from RECORDS file
        
        In PhysioNet Challenge 2015, the RECORDS file format is:
        record_name alarm_type
        
        The alarm type can be extracted from the record suffix:
        - Records ending in 's' or 'l' indicate short (5 min) or long (5.5 min) recordings
        - Labels must be created manually or obtained from annotations
        
        Returns:
            labels_dict: Dictionary mapping record_name to label (0 or 1)
        """
        labels_dict = {}
        
        # First, try to find a pre-made REFERENCE file (if user created one)
        reference_files = ['REFERENCE.csv', 'REFERENCE.txt', 'answers.txt']
        for ref_file in reference_files:
            ref_path = os.path.join(self.training_dir, ref_file)
            if os.path.exists(ref_path):
                print(f"📄 Found reference file: {ref_file}")
                with open(ref_path, 'r') as f:
                    for line in f:
                        # Skip header line
                        if line.startswith('record') or line.startswith('#'):
                            continue
                        
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            record_name = parts[0].strip()
                            label_str = parts[1].strip().lower()
                            
                            if label_str in ['1', 'true', 't']:
                                labels_dict[record_name] = 1
                            elif label_str in ['0', 'false', 'f']:
                                labels_dict[record_name] = 0
                
                if labels_dict:
                    print(f"✓ Loaded {len(labels_dict)} labels from {ref_file}")
                    return labels_dict
        
        # If no REFERENCE file exists, try RECORDS file with annotations
        records_file = os.path.join(self.training_dir, 'RECORDS')
        if os.path.exists(records_file):
            print(f"📄 Found RECORDS file")
            with open(records_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        record_name = parts[0].strip()
                        # Sometimes label is in the second column
                        if len(parts) > 1:
                            label_str = parts[1].strip().lower()
                            if label_str in ['1', 'true', 't']:
                                labels_dict[record_name] = 1
                            elif label_str in ['0', 'false', 'f']:
                                labels_dict[record_name] = 0
        
        # If still no labels, check for .arousal or .alarm annotation files
        if not labels_dict:
            print("\n⚠️  No label file found!")
            print("="*60)
            print("IMPORTANT: You need to create a label file manually.")
            print("="*60)
            print("\nThe PhysioNet Challenge 2015 training data doesn't include")
            print("a simple label file. You need to either:")
            print()
            print("1. Download the sample entry from PhysioNet which includes")
            print("   the answers.txt file with labels for all training records")
            print()
            print("2. Create a REFERENCE.csv file with this format:")
            print("   record_name,label")
            print("   a103l,1")
            print("   a104s,0")
            print("   ...")
            print()
            print("Where label is:")
            print("   1 = True alarm (genuine arrhythmia)")
            print("   0 = False alarm")
            print("="*60)
        
        return labels_dict
    
    def process_all_records(self):
        """
        Process all records in the training directory
        
        Returns:
            ppg_signals: List of PPG signals
            ecg_signals: List of ECG II signals
            labels: List of labels
            record_names: List of record names
        """
        # Get list of all .mat files
        mat_files = [f[:-4] for f in os.listdir(self.training_dir) 
                    if f.endswith('.mat')]
        
        print(f"\n🔍 Found {len(mat_files)} records")
        
        # Load labels
        labels_dict = self.load_labels()
        print(f"📊 Loaded labels for {len(labels_dict)} records\n")
        
        ppg_signals = []
        ecg_signals = []
        labels = []
        record_names = []
        
        successful = 0
        failed = 0
        
        for record_name in sorted(mat_files):
            print(f"Processing {record_name}...")
            
            # Extract signals
            ppg, ecg = self.extract_signals(record_name)
            
            # Check if both signals were extracted
            if ppg is not None and ecg is not None:
                # Get label
                label = labels_dict.get(record_name)
                
                if label is not None:
                    ppg_signals.append(ppg)
                    ecg_signals.append(ecg)
                    labels.append(label)
                    record_names.append(record_name)
                    successful += 1
                    print(f"  ✅ Success - Label: {label}\n")
                else:
                    print(f"  ⚠️  No label found\n")
                    failed += 1
            else:
                print(f"  ❌ Failed to extract both signals\n")
                failed += 1
        
        print(f"\n{'='*60}")
        print(f"✅ Successfully processed: {successful} records")
        print(f"❌ Failed: {failed} records")
        print(f"{'='*60}\n")
        
        return ppg_signals, ecg_signals, labels, record_names
    
    def save_data(self, ppg_signals, ecg_signals, labels, record_names):
        """
        Save extracted data to files
        
        Args:
            ppg_signals: List of PPG signals
            ecg_signals: List of ECG II signals
            labels: List of labels
            record_names: List of record names
        """
        
        print("\n🔧 Standardizing signal lengths...")
        
        # Check signal lengths
        ppg_lengths = [len(ppg) for ppg in ppg_signals]
        ecg_lengths = [len(ecg) for ecg in ecg_signals]
        
        print(f"   PPG length range: {min(ppg_lengths)} - {max(ppg_lengths)} samples")
        print(f"   ECG length range: {min(ecg_lengths)} - {max(ecg_lengths)} samples")
        
        # Use the most common length or 75000 (5 minutes at 250 Hz)
        # Standard length: 5 minutes at 250 Hz = 75,000 samples
        target_length = 75000
        
        # Alternatively, use the minimum length to avoid padding
        # target_length = min(ppg_lengths)
        
        print(f"   Standardizing to: {target_length} samples")
        
        # Standardize all signals to target_length
        ppg_standardized = []
        ecg_standardized = []
        
        for i, (ppg, ecg) in enumerate(zip(ppg_signals, ecg_signals)):
            # Truncate or pad PPG
            if len(ppg) > target_length:
                ppg_std = ppg[:target_length]  # Truncate
            elif len(ppg) < target_length:
                # Pad with zeros
                ppg_std = np.pad(ppg, (0, target_length - len(ppg)), mode='constant')
            else:
                ppg_std = ppg
            
            # Truncate or pad ECG
            if len(ecg) > target_length:
                ecg_std = ecg[:target_length]  # Truncate
            elif len(ecg) < target_length:
                # Pad with zeros
                ecg_std = np.pad(ecg, (0, target_length - len(ecg)), mode='constant')
            else:
                ecg_std = ecg
            
            ppg_standardized.append(ppg_std)
            ecg_standardized.append(ecg_std)
        
        # Convert to numpy arrays
        ppg_array = np.array(ppg_standardized)
        ecg_array = np.array(ecg_standardized)
        labels_array = np.array(labels)
        
        print(f"   ✓ Standardization complete!")
        
        # Save as .npy files
        np.save(os.path.join(self.output_dir, 'ppg_signals.npy'), ppg_array)
        np.save(os.path.join(self.output_dir, 'ecg_signals.npy'), ecg_array)
        np.save(os.path.join(self.output_dir, 'labels.npy'), labels_array)
        
        # Save metadata as CSV
        metadata = pd.DataFrame({
            'record_name': record_names,
            'label': labels,
            'original_ppg_length': ppg_lengths,
            'original_ecg_length': ecg_lengths,
            'standardized_length': [target_length] * len(record_names)
        })
        metadata.to_csv(os.path.join(self.output_dir, 'metadata.csv'), index=False)
        
        print(f"\n💾 Saved data:")
        print(f"   - PPG signals: {ppg_array.shape}")
        print(f"   - ECG signals: {ecg_array.shape}")
        print(f"   - Labels: {labels_array.shape}")
        print(f"   - Metadata: {len(record_names)} records")
        print(f"\n📊 Label distribution:")
        print(f"   - True alarms (1): {np.sum(labels_array == 1)}")
        print(f"   - False alarms (0): {np.sum(labels_array == 0)}")
        
        # Save a summary report
        summary_file = os.path.join(self.output_dir, 'extraction_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("PhysioNet Challenge 2015 - Data Extraction Summary\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total records processed: {len(record_names)}\n")
            f.write(f"Standardized signal length: {target_length} samples\n")
            f.write(f"Sampling rate: 250 Hz\n")
            f.write(f"Duration: {target_length/250} seconds ({target_length/250/60:.1f} minutes)\n\n")
            f.write(f"Label distribution:\n")
            f.write(f"  - True alarms (1): {np.sum(labels_array == 1)} ({np.sum(labels_array == 1)/len(labels_array)*100:.1f}%)\n")
            f.write(f"  - False alarms (0): {np.sum(labels_array == 0)} ({np.sum(labels_array == 0)/len(labels_array)*100:.1f}%)\n\n")
            f.write(f"Original signal length statistics:\n")
            f.write(f"  PPG: min={min(ppg_lengths)}, max={max(ppg_lengths)}, mean={np.mean(ppg_lengths):.0f}\n")
            f.write(f"  ECG: min={min(ecg_lengths)}, max={max(ecg_lengths)}, mean={np.mean(ecg_lengths):.0f}\n\n")
            f.write("="*60 + "\n")
        
        print(f"\n📄 Summary saved to: {summary_file}")


def main():
    """Main execution function"""
    
    print("\n" + "="*60)
    print("PhysioNet Challenge 2015 - Signal Extraction")
    print("="*60 + "\n")
    
    # Initialize extractor
    # IMPORTANT: Update the path to your training directory
    extractor = SignalExtractor(
        training_dir='training',  # Change this to your actual path
        output_dir='extracted_data'
    )
    
    # Process all records
    ppg_signals, ecg_signals, labels, record_names = extractor.process_all_records()
    
    # Save data
    if len(ppg_signals) > 0:
        extractor.save_data(ppg_signals, ecg_signals, labels, record_names)
        print("\n✅ Extraction completed successfully!")
    else:
        print("\n❌ No data was extracted. Please check your file paths and data format.")


if __name__ == "__main__":
    main()