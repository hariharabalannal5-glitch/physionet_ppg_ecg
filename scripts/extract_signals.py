"""
PhysioNet Challenge 2015 - Signal Extraction Script
====================================================
This script downloads the training dataset, extracts PPG and ECG-II signals,
and saves them in organized files for further analysis.

Author: Your Team
Date: February 2026
"""

import os
import requests
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
import wfdb
from tqdm import tqdm
import pickle

class PhysioNetExtractor:
    def __init__(self, base_dir='physionet_data'):
        """
        Initialize the extractor with directory structure
        
        Parameters:
        -----------
        base_dir : str
            Base directory to store all data
        """
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / 'raw'
        self.extracted_dir = self.base_dir / 'extracted'
        self.training_zip = self.raw_dir / 'training.zip'
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset URL
        self.dataset_url = "https://physionet.org/files/challenge-2015/1.0.0/training.zip"
        
    def download_dataset(self):
        """Download the training dataset if not already present"""
        if self.training_zip.exists():
            print(f"✓ Dataset already downloaded at {self.training_zip}")
            return
        
        print(f"Downloading dataset from {self.dataset_url}...")
        print("This is a large file (322.9 MB), please be patient...")
        
        try:
            response = requests.get(self.dataset_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.training_zip, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            
            print(f"✓ Download complete: {self.training_zip}")
        except Exception as e:
            print(f"✗ Error downloading dataset: {e}")
            raise
    
    def extract_dataset(self):
        """Extract the downloaded zip file"""
        training_dir = self.raw_dir / 'training'
        
        if training_dir.exists() and len(list(training_dir.glob('*.mat'))) > 0:
            print(f"✓ Dataset already extracted at {training_dir}")
            return training_dir
        
        print(f"Extracting dataset...")
        try:
            with zipfile.ZipFile(self.training_zip, 'r') as zip_ref:
                zip_ref.extractall(self.raw_dir)
            print(f"✓ Extraction complete")
            return training_dir
        except Exception as e:
            print(f"✗ Error extracting dataset: {e}")
            raise
    
    def get_signal_info(self, record_path):
        """
        Get information about available signals in a record
        
        Parameters:
        -----------
        record_path : str
            Path to the record (without extension)
        
        Returns:
        --------
        dict : Signal information
        """
        try:
            record = wfdb.rdrecord(str(record_path))
            return {
                'sig_name': record.sig_name,
                'n_sig': record.n_sig,
                'fs': record.fs,
                'sig_len': record.sig_len
            }
        except Exception as e:
            print(f"Error reading {record_path}: {e}")
            return None
    
    def extract_ppg_ecg2(self, record_path):
        """
        Extract PPG and ECG-II signals from a record
        
        Parameters:
        -----------
        record_path : str or Path
            Path to the record (without extension)
        
        Returns:
        --------
        dict : Dictionary containing PPG, ECG-II signals and metadata
        """
        try:
            # Read the record
            record = wfdb.rdrecord(str(record_path))
            
            # Get signal names
            sig_names = [name.strip().upper() for name in record.sig_name]
            
            # Initialize output
            result = {
                'record_name': record_path.name if isinstance(record_path, Path) else os.path.basename(record_path),
                'fs': record.fs,
                'sig_len': record.sig_len,
                'ppg': None,
                'ecg2': None,
                'ppg_available': False,
                'ecg2_available': False,
                'all_signals': sig_names
            }
            
            # Find PPG signal (may be labeled as 'PLETH', 'PPG', or similar)
            ppg_keywords = ['PLETH', 'PPG', 'PHOTO']
            ppg_idx = None
            for i, name in enumerate(sig_names):
                if any(keyword in name for keyword in ppg_keywords):
                    ppg_idx = i
                    result['ppg_signal_name'] = record.sig_name[i]
                    break
            
            # Find ECG-II signal (may be labeled as 'II', 'ECG', 'ECG2', or similar)
            ecg2_keywords = ['II', 'ECG2', 'ECGII']
            ecg2_idx = None
            for i, name in enumerate(sig_names):
                # Check for exact match or 'II' in the name
                if name == 'II' or any(keyword in name for keyword in ecg2_keywords):
                    ecg2_idx = i
                    result['ecg2_signal_name'] = record.sig_name[i]
                    break
            
            # If exact 'II' not found, look for any ECG signal
            if ecg2_idx is None:
                for i, name in enumerate(sig_names):
                    if 'ECG' in name:
                        ecg2_idx = i
                        result['ecg2_signal_name'] = record.sig_name[i]
                        print(f"  Note: Using {record.sig_name[i]} instead of ECG-II for {result['record_name']}")
                        break
            
            # Extract signals
            if ppg_idx is not None:
                result['ppg'] = record.p_signal[:, ppg_idx]
                result['ppg_available'] = True
            
            if ecg2_idx is not None:
                result['ecg2'] = record.p_signal[:, ecg2_idx]
                result['ecg2_available'] = True
            
            return result
            
        except Exception as e:
            print(f"✗ Error processing {record_path}: {e}")
            return None
    
    def load_annotations(self, training_dir):
        """
        Load the RECORDS file to get list of all records and their annotations
        
        Parameters:
        -----------
        training_dir : Path
            Directory containing training data
        
        Returns:
        --------
        pd.DataFrame : DataFrame with record names and alarm types
        """
        records_file = training_dir / 'RECORDS'
        
        if not records_file.exists():
            print(f"✗ RECORDS file not found at {records_file}")
            return None
        
        # Read RECORDS file
        with open(records_file, 'r') as f:
            record_names = [line.strip() for line in f.readlines() if line.strip()]
        
        # Extract alarm type and label from filenames
        data = []
        for record in record_names:
            # Format: aXXXX (a=alarm type, XXXX=number)
            # Alarm types: a=Asystole, b=Bradycardia, t=Tachycardia, v=Ventricular_Tachycardia, f=Ventricular_Flutter_Fib
            
            # Read the header file to get alarm type
            header_file = training_dir / f"{record}.hea"
            alarm_type = "Unknown"
            is_true_alarm = None
            
            if header_file.exists():
                with open(header_file, 'r') as f:
                    for line in f:
                        if 'Alarm' in line or 'alarm' in line:
                            if 'Asystole' in line:
                                alarm_type = 'Asystole'
                            elif 'Bradycardia' in line:
                                alarm_type = 'Bradycardia'
                            elif 'Tachycardia' in line and 'Ventricular' not in line:
                                alarm_type = 'Tachycardia'
                            elif 'Ventricular_Tachycardia' in line or 'VT' in line:
                                alarm_type = 'Ventricular_Tachycardia'
                            elif 'Ventricular_Flutter' in line or 'VFib' in line or 'Fibrillation' in line:
                                alarm_type = 'Ventricular_Flutter_Fib'
                        
                        # Check for true/false label
                        if 'True' in line or 'true' in line:
                            is_true_alarm = True
                        elif 'False' in line or 'false' in line:
                            is_true_alarm = False
            
            data.append({
                'record': record,
                'alarm_type': alarm_type,
                'is_true_alarm': is_true_alarm
            })
        
        return pd.DataFrame(data)
    
    def process_all_records(self):
        """
        Main processing function to extract PPG and ECG-II from all records
        """
        print("\n" + "="*60)
        print("PhysioNet Challenge 2015 - Signal Extraction")
        print("="*60 + "\n")
        
        # Step 1: Download dataset
        self.download_dataset()
        
        # Step 2: Extract dataset
        training_dir = self.extract_dataset()
        
        # Step 3: Load annotations
        print("\nLoading record annotations...")
        annotations = self.load_annotations(training_dir)
        
        if annotations is not None:
            print(f"✓ Found {len(annotations)} records")
            print(f"\nAlarm type distribution:")
            print(annotations['alarm_type'].value_counts())
            
            # Save annotations
            annotations.to_csv(self.extracted_dir / 'annotations.csv', index=False)
            print(f"✓ Saved annotations to {self.extracted_dir / 'annotations.csv'}")
        
        # Step 4: Extract signals from all records
        print(f"\nExtracting PPG and ECG-II signals from all records...")
        
        all_data = []
        stats = {
            'total': 0,
            'ppg_available': 0,
            'ecg2_available': 0,
            'both_available': 0,
            'errors': 0
        }
        
        for idx, row in tqdm(annotations.iterrows(), total=len(annotations), desc="Processing records"):
            record_path = training_dir / row['record']
            
            # Extract signals
            result = self.extract_ppg_ecg2(record_path)
            
            if result is not None:
                # Add annotation info
                result['alarm_type'] = row['alarm_type']
                result['is_true_alarm'] = row['is_true_alarm']
                
                all_data.append(result)
                
                # Update statistics
                stats['total'] += 1
                if result['ppg_available']:
                    stats['ppg_available'] += 1
                if result['ecg2_available']:
                    stats['ecg2_available'] += 1
                if result['ppg_available'] and result['ecg2_available']:
                    stats['both_available'] += 1
            else:
                stats['errors'] += 1
        
        # Step 5: Save extracted data
        print(f"\nSaving extracted signals...")
        
        # Save as pickle (for easy loading with numpy arrays)
        with open(self.extracted_dir / 'extracted_signals.pkl', 'wb') as f:
            pickle.dump(all_data, f)
        print(f"✓ Saved to {self.extracted_dir / 'extracted_signals.pkl'}")
        
        # Save summary statistics
        summary_df = pd.DataFrame([
            {'record': d['record_name'], 
             'alarm_type': d['alarm_type'],
             'is_true_alarm': d['is_true_alarm'],
             'ppg_available': d['ppg_available'],
             'ecg2_available': d['ecg2_available'],
             'fs': d['fs'],
             'duration_sec': d['sig_len'] / d['fs'],
             'all_signals': ', '.join(d['all_signals'])
            } for d in all_data
        ])
        summary_df.to_csv(self.extracted_dir / 'extraction_summary.csv', index=False)
        print(f"✓ Saved summary to {self.extracted_dir / 'extraction_summary.csv'}")
        
        # Print statistics
        print("\n" + "="*60)
        print("EXTRACTION STATISTICS")
        print("="*60)
        print(f"Total records processed: {stats['total']}")
        print(f"Records with PPG: {stats['ppg_available']} ({stats['ppg_available']/stats['total']*100:.1f}%)")
        print(f"Records with ECG-II: {stats['ecg2_available']} ({stats['ecg2_available']/stats['total']*100:.1f}%)")
        print(f"Records with BOTH: {stats['both_available']} ({stats['both_available']/stats['total']*100:.1f}%)")
        print(f"Errors: {stats['errors']}")
        print("="*60 + "\n")
        
        return all_data, summary_df


def main():
    """Main execution function"""
    
    # Create extractor instance
    extractor = PhysioNetExtractor(base_dir='physionet_data')
    
    # Process all records
    extracted_data, summary = extractor.process_all_records()
    
    print("\n✓ Signal extraction complete!")
    print(f"\nExtracted data location: {extractor.extracted_dir}")
    print("\nFiles created:")
    print(f"  - extracted_signals.pkl (all signal data)")
    print(f"  - extraction_summary.csv (summary statistics)")
    print(f"  - annotations.csv (alarm annotations)")
    
    return extracted_data, summary


if __name__ == "__main__":
    main()