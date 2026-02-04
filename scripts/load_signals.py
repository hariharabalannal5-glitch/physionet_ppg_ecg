"""
PhysioNet Challenge 2015 - Data Loading and Visualization Script
==================================================================
This script loads the extracted PPG and ECG-II signals and provides
utilities for visualization and analysis.

Author: Your Team
Date: February 2026
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

class SignalLoader:
    def __init__(self, extracted_dir='physionet_data/extracted'):
        """
        Initialize the signal loader
        
        Parameters:
        -----------
        extracted_dir : str
            Directory containing extracted signals
        """
        self.extracted_dir = Path(extracted_dir)
        self.data = None
        self.summary = None
        
    def load_data(self):
        """Load the extracted signals"""
        pkl_file = self.extracted_dir / 'extracted_signals.pkl'
        summary_file = self.extracted_dir / 'extraction_summary.csv'
        
        if not pkl_file.exists():
            raise FileNotFoundError(f"Extracted signals not found at {pkl_file}")
        
        print("Loading extracted signals...")
        with open(pkl_file, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"✓ Loaded {len(self.data)} records")
        
        if summary_file.exists():
            self.summary = pd.read_csv(summary_file)
            print(f"✓ Loaded summary data")
        
        return self.data
    
    def get_record_by_index(self, idx):
        """Get a specific record by index"""
        if self.data is None:
            self.load_data()
        return self.data[idx]
    
    def get_record_by_name(self, record_name):
        """Get a specific record by name"""
        if self.data is None:
            self.load_data()
        
        for record in self.data:
            if record['record_name'] == record_name:
                return record
        return None
    
    def get_records_with_both_signals(self):
        """Get all records that have both PPG and ECG-II"""
        if self.data is None:
            self.load_data()
        
        return [r for r in self.data if r['ppg_available'] and r['ecg2_available']]
    
    def get_statistics(self):
        """Get statistics about the dataset"""
        if self.data is None:
            self.load_data()
        
        stats = {
            'total_records': len(self.data),
            'with_ppg': sum(1 for r in self.data if r['ppg_available']),
            'with_ecg2': sum(1 for r in self.data if r['ecg2_available']),
            'with_both': sum(1 for r in self.data if r['ppg_available'] and r['ecg2_available']),
            'true_alarms': sum(1 for r in self.data if r.get('is_true_alarm') == True),
            'false_alarms': sum(1 for r in self.data if r.get('is_true_alarm') == False),
        }
        
        # Alarm type distribution
        alarm_types = {}
        for r in self.data:
            alarm_type = r.get('alarm_type', 'Unknown')
            alarm_types[alarm_type] = alarm_types.get(alarm_type, 0) + 1
        stats['alarm_types'] = alarm_types
        
        return stats
    
    def plot_signals(self, record_idx=0, duration=None, save_path=None):
        """
        Plot PPG and ECG-II signals for a record
        
        Parameters:
        -----------
        record_idx : int
            Index of record to plot
        duration : float
            Duration in seconds to plot (None = plot all)
        save_path : str
            Path to save figure (None = display only)
        """
        if self.data is None:
            self.load_data()
        
        record = self.data[record_idx]
        
        # Prepare figure
        n_plots = sum([record['ppg_available'], record['ecg2_available']])
        
        if n_plots == 0:
            print(f"No signals available for record {record['record_name']}")
            return
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4*n_plots))
        if n_plots == 1:
            axes = [axes]
        
        # Time axis
        fs = record['fs']
        if duration is not None:
            samples = int(duration * fs)
        else:
            samples = record['sig_len']
        
        time = np.arange(samples) / fs
        
        plot_idx = 0
        
        # Plot PPG
        if record['ppg_available']:
            ppg = record['ppg'][:samples]
            axes[plot_idx].plot(time, ppg, 'b-', linewidth=0.8)
            axes[plot_idx].set_ylabel('PPG Amplitude', fontsize=12)
            axes[plot_idx].set_title(f"PPG Signal - {record['record_name']} ({record.get('ppg_signal_name', 'PPG')})", 
                                     fontsize=14, fontweight='bold')
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot ECG-II
        if record['ecg2_available']:
            ecg2 = record['ecg2'][:samples]
            axes[plot_idx].plot(time, ecg2, 'r-', linewidth=0.8)
            axes[plot_idx].set_ylabel('ECG-II Amplitude (mV)', fontsize=12)
            axes[plot_idx].set_title(f"ECG-II Signal - {record['record_name']} ({record.get('ecg2_signal_name', 'ECG-II')})", 
                                     fontsize=14, fontweight='bold')
            axes[plot_idx].set_xlabel('Time (seconds)', fontsize=12)
            axes[plot_idx].grid(True, alpha=0.3)
        
        # Add metadata
        alarm_type = record.get('alarm_type', 'Unknown')
        is_true = record.get('is_true_alarm', None)
        label = 'TRUE ALARM' if is_true else 'FALSE ALARM' if is_true is not None else 'UNKNOWN'
        color = 'red' if is_true else 'green' if is_true is not None else 'gray'
        
        fig.suptitle(f"Alarm Type: {alarm_type} | Label: {label}", 
                     fontsize=16, fontweight='bold', color=color, y=0.995)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved figure to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_multiple_examples(self, n_examples=4, alarm_type=None, save_dir=None):
        """
        Plot multiple example records
        
        Parameters:
        -----------
        n_examples : int
            Number of examples to plot
        alarm_type : str
            Specific alarm type to plot (None = random)
        save_dir : str
            Directory to save figures
        """
        if self.data is None:
            self.load_data()
        
        # Filter by alarm type if specified
        if alarm_type:
            records = [r for r in self.data if r.get('alarm_type') == alarm_type and 
                      r['ppg_available'] and r['ecg2_available']]
        else:
            records = self.get_records_with_both_signals()
        
        # Sample random records
        indices = np.random.choice(len(records), min(n_examples, len(records)), replace=False)
        
        for i, idx in enumerate(indices):
            record_idx = self.data.index(records[idx])
            
            if save_dir:
                save_path = Path(save_dir) / f"example_{i+1}_{records[idx]['record_name']}.png"
            else:
                save_path = None
            
            self.plot_signals(record_idx, duration=30, save_path=save_path)
    
    def create_dataset_summary_plot(self, save_path=None):
        """Create a comprehensive summary plot of the dataset"""
        if self.data is None:
            self.load_data()
        
        stats = self.get_statistics()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Signal availability
        ax = axes[0, 0]
        categories = ['PPG Only', 'ECG-II Only', 'Both Signals', 'Neither']
        ppg_only = stats['with_ppg'] - stats['with_both']
        ecg_only = stats['with_ecg2'] - stats['with_both']
        neither = stats['total_records'] - stats['with_ppg'] - stats['with_ecg2'] + stats['with_both']
        values = [ppg_only, ecg_only, stats['with_both'], neither]
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightgray']
        ax.bar(categories, values, color=colors, edgecolor='black')
        ax.set_ylabel('Number of Records', fontsize=12)
        ax.set_title('Signal Availability', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Alarm type distribution
        ax = axes[0, 1]
        alarm_types = stats['alarm_types']
        ax.bar(alarm_types.keys(), alarm_types.values(), color='orange', edgecolor='black')
        ax.set_ylabel('Number of Records', fontsize=12)
        ax.set_title('Alarm Type Distribution', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # 3. True vs False alarms
        ax = axes[1, 0]
        labels = ['True Alarms', 'False Alarms']
        values = [stats['true_alarms'], stats['false_alarms']]
        colors = ['red', 'green']
        ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Alarm Classification', fontsize=14, fontweight='bold')
        
        # 4. Summary statistics text
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = f"""
        DATASET SUMMARY
        ═══════════════════════════════
        
        Total Records: {stats['total_records']}
        
        Signal Availability:
          • PPG Available: {stats['with_ppg']} ({stats['with_ppg']/stats['total_records']*100:.1f}%)
          • ECG-II Available: {stats['with_ecg2']} ({stats['with_ecg2']/stats['total_records']*100:.1f}%)
          • Both Signals: {stats['with_both']} ({stats['with_both']/stats['total_records']*100:.1f}%)
        
        Alarm Labels:
          • True Alarms: {stats['true_alarms']} ({stats['true_alarms']/stats['total_records']*100:.1f}%)
          • False Alarms: {stats['false_alarms']} ({stats['false_alarms']/stats['total_records']*100:.1f}%)
        
        False Alarm Rate: {stats['false_alarms']/(stats['true_alarms']+stats['false_alarms'])*100:.1f}%
        """
        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved summary plot to {save_path}")
        else:
            plt.show()
        
        return fig


def main():
    """Main execution function demonstrating usage"""
    
    print("\n" + "="*60)
    print("PhysioNet Challenge 2015 - Data Loading Demo")
    print("="*60 + "\n")
    
    # Create loader
    loader = SignalLoader()
    
    # Load data
    data = loader.load_data()
    
    # Print statistics
    stats = loader.get_statistics()
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total records: {stats['total_records']}")
    print(f"Records with PPG: {stats['with_ppg']}")
    print(f"Records with ECG-II: {stats['with_ecg2']}")
    print(f"Records with both: {stats['with_both']}")
    print(f"\nTrue alarms: {stats['true_alarms']}")
    print(f"False alarms: {stats['false_alarms']}")
    print(f"\nAlarm type distribution:")
    for alarm_type, count in stats['alarm_types'].items():
        print(f"  {alarm_type}: {count}")
    print("="*60 + "\n")
    
    # Create summary visualization
    print("Creating dataset summary visualization...")
    loader.create_dataset_summary_plot(save_path='dataset_summary.png')
    
    # Plot a few examples
    print("\nPlotting example signals...")
    loader.plot_signals(record_idx=0, duration=30, save_path='example_signal.png')
    
    print("\n✓ Demo complete!")
    print("\nTo use this in your analysis:")
    print("  from load_signals import SignalLoader")
    print("  loader = SignalLoader()")
    print("  data = loader.load_data()")
    print("  record = loader.get_record_by_index(0)")


if __name__ == "__main__":
    main()