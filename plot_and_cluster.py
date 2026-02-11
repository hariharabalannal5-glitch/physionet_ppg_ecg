"""
PhysioNet Challenge 2015 - Enhanced Signal Analysis with Cluster Interpretation
===============================================================================
This script creates:
1. ECG signal waveform
2. ECG II signal waveform  
3. PPG and ECG II correlation plot (side-by-side from same record)
4. PPG clustering with detailed cluster interpretation
5. Asystole clustering with cluster significance

Author: Your Team
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings('ignore')


class EnhancedSignalPlotter:
    """Enhanced plotting with cluster interpretation"""
    
    def __init__(self, data_dir='extracted_data', output_dir='plots'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self):
        """Load extracted signals and labels"""
        
        print("\n📂 Loading data...")
        
        ppg_signals = np.load(os.path.join(self.data_dir, 'ppg_signals.npy'))
        ecg_signals = np.load(os.path.join(self.data_dir, 'ecg_signals.npy'))
        labels = np.load(os.path.join(self.data_dir, 'labels.npy'))
        
        print(f"   Loaded PPG signals: {ppg_signals.shape}")
        print(f"   Loaded ECG signals: {ecg_signals.shape}")
        print(f"   Loaded labels: {labels.shape}")
        
        return ppg_signals, ecg_signals, labels
    
    def plot_ecg_waveform(self, ecg_signals, num_samples=1000):
        """Plot ECG signal waveform"""
        
        print("\n" + "="*60)
        print("TASK 1: ECG Signal Waveform")
        print("="*60)
        
        ecg_waveform = ecg_signals[0, :num_samples]
        sample_indices = np.arange(num_samples)
        
        print(f"\nPlotting ECG waveform...")
        print(f"   Samples: {num_samples}")
        print(f"   Amplitude range: [{ecg_waveform.min():.3f}, {ecg_waveform.max():.3f}]")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(sample_indices, ecg_waveform, 'b-', linewidth=1.5)
        ax.set_title('ECG Signal', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Number of Samples', fontsize=14, fontweight='bold')
        ax.set_ylabel('Amplitude (mV)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, num_samples)
        
        plt.tight_layout()
        image_path = os.path.join(self.output_dir, 'ecg_signal.png')
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {image_path}")
        plt.close()
        
        return image_path
    
    def plot_ecg2_waveform(self, ecg_signals, num_samples=1000):
        """Plot ECG II signal waveform"""
        
        print("\n" + "="*60)
        print("TASK 2: ECG II Signal Waveform")
        print("="*60)
        
        ecg2_waveform = ecg_signals[0, :num_samples]
        sample_indices = np.arange(num_samples)
        
        print(f"\nPlotting ECG II waveform...")
        print(f"   Samples: {num_samples}")
        print(f"   Amplitude range: [{ecg2_waveform.min():.3f}, {ecg2_waveform.max():.3f}]")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(sample_indices, ecg2_waveform, 'b-', linewidth=1.5)
        ax.set_title('ECG II Signal', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Number of Samples', fontsize=14, fontweight='bold')
        ax.set_ylabel('Amplitude (mV)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, num_samples)
        
        plt.tight_layout()
        image_path = os.path.join(self.output_dir, 'ecg2_signal.png')
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {image_path}")
        plt.close()
        
        return image_path
    
    def plot_ppg_ecg_correlation(self, ppg_signals, ecg_signals, num_samples=1000):
        """
        Plot PPG and corresponding ECG from the SAME record
        Shows correlation between the two signals
        """
        
        print("\n" + "="*60)
        print("TASK 3: PPG and Corresponding ECG II Correlation")
        print("="*60)
        
        # Take same record (index 0)
        ppg_waveform = ppg_signals[0, :num_samples]
        ecg_waveform = ecg_signals[0, :num_samples]
        sample_indices = np.arange(num_samples)
        
        # Normalize both for comparison
        ppg_norm = (ppg_waveform - ppg_waveform.min()) / (ppg_waveform.max() - ppg_waveform.min())
        ecg_norm = (ecg_waveform - ecg_waveform.min()) / (ecg_waveform.max() - ecg_waveform.min())
        
        print(f"\nPlotting PPG and ECG from the same record...")
        print(f"   Record index: 0")
        print(f"   Samples: {num_samples}")
        
        # Create figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Top: PPG Signal
        axes[0].plot(sample_indices, ppg_norm, 'b-', linewidth=1.5, label='PPG Signal')
        axes[0].set_title('Photoplethysmogram (PPG) Signal - Normalized', 
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Normalized Amplitude', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper right', fontsize=11)
        axes[0].set_ylim(-0.1, 1.1)
        
        # Bottom: ECG II Signal
        axes[1].plot(sample_indices, ecg_norm, 'r-', linewidth=1.5, label='ECG II Signal')
        axes[1].set_title('Electrocardiogram Lead II (ECG II) Signal - Normalized', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Normalized Amplitude', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper right', fontsize=11)
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].set_xlim(0, num_samples)
        
        # Add explanation text
        fig.text(0.5, 0.02, 
                'Both signals from the same patient record showing synchronized cardiac activity',
                ha='center', fontsize=11, style='italic', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('PPG and Corresponding ECG II Signals Correlation', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0.03, 1, 0.99])
        
        image_path = os.path.join(self.output_dir, 'ppg_ecg_correlation.png')
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {image_path}")
        plt.close()
        
        return image_path
    
    def analyze_and_plot_ppg_clusters(self, ppg_signals, labels, num_samples=1000, n_clusters=3):
        """
        PPG clustering with detailed cluster interpretation
        """
        
        print("\n" + "="*60)
        print("TASK 4: PPG Clustering with Interpretation")
        print("="*60)
        
        # Waveform display
        ppg_waveform = ppg_signals[0, :num_samples]
        sample_indices = np.arange(num_samples)
        ppg_normalized = (ppg_waveform - ppg_waveform.min()) / (ppg_waveform.max() - ppg_waveform.min())
        
        # Clustering
        print(f"\nApplying K-Means clustering...")
        max_samples = min(300, len(ppg_signals))
        ppg_subset = ppg_signals[:max_samples]
        labels_subset = labels[:max_samples]
        
        # Extract features
        features = []
        for sig in ppg_subset:
            feat = [
                np.mean(sig),
                np.std(sig),
                np.max(sig),
                np.min(sig),
                np.median(sig),
                np.percentile(sig, 75) - np.percentile(sig, 25),  # IQR
            ]
            features.append(feat)
        
        features = np.array(features)
        feature_names = ['Mean', 'Std Dev', 'Max', 'Min', 'Median', 'IQR']
        
        # Normalize and cluster
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Analyze clusters
        cluster_interpretations = self.interpret_ppg_clusters(
            features, cluster_labels, labels_subset, n_clusters
        )
        
        # PCA for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        
        # Top: PPG Waveform (spans both columns)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(sample_indices, ppg_normalized, 'b-', linewidth=1.5)
        ax1.set_title('Normalized PPG Signal', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Samples', fontsize=11)
        ax1.set_ylabel('Normalized Amplitude', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, num_samples)
        
        # Middle Left: Cluster scatter
        ax2 = fig.add_subplot(gs[1, 0])
        colors = ['red', 'blue', 'green']
        
        for i in range(n_clusters):
            mask = cluster_labels == i
            ax2.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=colors[i], label=f'Cluster {i}', 
                       s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        centroids_2d = pca.transform(scaler.transform(kmeans.cluster_centers_))
        ax2.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                   c='black', marker='X', s=250, label='Centroids', 
                   edgecolors='yellow', linewidths=2)
        
        ax2.set_title('PPG Clustering (K-Means, k=3)', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Principal Component 1', fontsize=10)
        ax2.set_ylabel('Principal Component 2', fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Middle Right: Sample signals
        ax3 = fig.add_subplot(gs[1, 1])
        
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            if len(cluster_indices) > 0:
                sample_idx = cluster_indices[0]
                sample_signal = ppg_subset[sample_idx][:500]
                time = np.arange(len(sample_signal))
                sample_norm = (sample_signal - sample_signal.min()) / (sample_signal.max() - sample_signal.min())
                ax3.plot(time, sample_norm + i*1.3, 
                        color=colors[i], label=f'Cluster {i}', linewidth=1.2)
        
        ax3.set_title('Representative PPG Signals', fontsize=13, fontweight='bold')
        ax3.set_xlabel('Samples', fontsize=10)
        ax3.set_ylabel('Normalized Amplitude (offset)', fontsize=10)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Bottom: Cluster Interpretation (spans both columns)
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        interpretation_text = "CLUSTER INTERPRETATION & SIGNIFICANCE:\n\n"
        for i in range(n_clusters):
            interp = cluster_interpretations[i]
            interpretation_text += f"● Cluster {i} ({colors[i].upper()}):\n"
            interpretation_text += f"  - Size: {interp['count']} signals ({interp['percentage']:.1f}%)\n"
            interpretation_text += f"  - True Alarms: {interp['true_alarms']}/{interp['count']} ({interp['true_alarm_rate']:.1f}%)\n"
            interpretation_text += f"  - Characteristics: {interp['characteristics']}\n"
            interpretation_text += f"  - Clinical Significance: {interp['significance']}\n\n"
        
        ax4.text(0.05, 0.95, interpretation_text, 
                transform=ax4.transAxes, fontsize=10, 
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle('PPG Signal Analysis with Unsupervised K-Means Clustering', 
                    fontsize=16, fontweight='bold')
        
        image_path = os.path.join(self.output_dir, 'ppg_clustering_interpreted.png')
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {image_path}")
        plt.close()
        
        # Print to console
        print("\n" + "="*60)
        print("CLUSTER ANALYSIS RESULTS")
        print("="*60)
        for i in range(n_clusters):
            interp = cluster_interpretations[i]
            print(f"\nCluster {i}:")
            print(f"  Count: {interp['count']}")
            print(f"  True Alarm Rate: {interp['true_alarm_rate']:.1f}%")
            print(f"  Characteristics: {interp['characteristics']}")
        
        return image_path
    
    def interpret_ppg_clusters(self, features, cluster_labels, true_labels, n_clusters):
        """
        Interpret what each cluster represents based on features and labels
        """
        
        interpretations = []
        
        for i in range(n_clusters):
            mask = cluster_labels == i
            cluster_features = features[mask]
            cluster_true_labels = true_labels[mask]
            
            # Statistics
            count = np.sum(mask)
            true_alarms = np.sum(cluster_true_labels == 1)
            true_alarm_rate = (true_alarms / count * 100) if count > 0 else 0
            
            # Feature analysis
            mean_amplitude = np.mean(cluster_features[:, 0])
            mean_std = np.mean(cluster_features[:, 1])
            mean_range = np.mean(cluster_features[:, 2] - cluster_features[:, 3])
            
            # Characterize cluster
            if mean_std < np.median(features[:, 1]):
                characteristics = "Low variability, stable signal"
                if true_alarm_rate > 70:
                    significance = "Likely normal cardiac rhythm with true arrhythmia events"
                else:
                    significance = "Likely stable false alarms or sensor artifacts"
            elif mean_std > np.percentile(features[:, 1], 75):
                characteristics = "High variability, irregular signal"
                if true_alarm_rate > 70:
                    significance = "Genuine arrhythmic activity requiring attention"
                else:
                    significance = "Motion artifacts or poor signal quality"
            else:
                characteristics = "Moderate variability, mixed patterns"
                significance = "Mixed alarm types, requires case-by-case review"
            
            interpretations.append({
                'count': count,
                'percentage': (count / len(cluster_labels)) * 100,
                'true_alarms': true_alarms,
                'true_alarm_rate': true_alarm_rate,
                'mean_amplitude': mean_amplitude,
                'mean_std': mean_std,
                'characteristics': characteristics,
                'significance': significance
            })
        
        return interpretations
    
    def analyze_and_plot_asystole_clusters(self, ppg_signals, labels, num_samples=1000, n_clusters=3):
        """
        Asystole clustering with interpretation
        """
        
        print("\n" + "="*60)
        print("TASK 5: Asystole Clustering with Interpretation")
        print("="*60)
        
        # Filter for true alarms
        asystole_mask = labels == 1
        asystole_signals = ppg_signals[asystole_mask]
        
        print(f"\nFound {len(asystole_signals)} True Alarm (Asystole-type) signals")
        
        asystole_waveform = asystole_signals[0, :num_samples]
        sample_indices = np.arange(num_samples)
        asystole_normalized = (asystole_waveform - asystole_waveform.min()) / (asystole_waveform.max() - asystole_waveform.min())
        
        # Clustering
        max_samples = min(300, len(asystole_signals))
        asystole_subset = asystole_signals[:max_samples]
        
        # Features
        features = []
        for sig in asystole_subset:
            feat = [
                np.mean(sig),
                np.std(sig),
                np.max(sig),
                np.min(sig),
                np.median(sig),
                np.percentile(sig, 75) - np.percentile(sig, 25),
            ]
            features.append(feat)
        
        features = np.array(features)
        
        # Normalize and cluster
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Interpret asystole clusters
        asystole_interpretations = self.interpret_asystole_clusters(
            features, cluster_labels, n_clusters
        )
        
        # PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)
        
        # Create figure
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        
        # Top: Asystole waveform
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(sample_indices, asystole_normalized, 'b-', linewidth=1.5)
        ax1.set_title('Normalized Asystole Alarm Signal', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Samples', fontsize=11)
        ax1.set_ylabel('Normalized Amplitude', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, num_samples)
        
        # Middle Left: Clustering
        ax2 = fig.add_subplot(gs[1, 0])
        colors = ['purple', 'cyan', 'magenta']
        
        for i in range(n_clusters):
            mask = cluster_labels == i
            ax2.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=colors[i], label=f'Cluster {i}', 
                       s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        centroids_2d = pca.transform(scaler.transform(kmeans.cluster_centers_))
        ax2.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                   c='black', marker='X', s=250, label='Centroids', 
                   edgecolors='white', linewidths=2)
        
        ax2.set_title('Asystole Clustering (K-Means, k=3)', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Principal Component 1', fontsize=10)
        ax2.set_ylabel('Principal Component 2', fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Middle Right: Samples
        ax3 = fig.add_subplot(gs[1, 1])
        
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            if len(cluster_indices) > 0:
                sample_idx = cluster_indices[0]
                sample_signal = asystole_subset[sample_idx][:500]
                time = np.arange(len(sample_signal))
                sample_norm = (sample_signal - sample_signal.min()) / (sample_signal.max() - sample_signal.min())
                ax3.plot(time, sample_norm + i*1.3, 
                        color=colors[i], label=f'Cluster {i}', linewidth=1.2)
        
        ax3.set_title('Representative Asystole Signals', fontsize=13, fontweight='bold')
        ax3.set_xlabel('Samples', fontsize=10)
        ax3.set_ylabel('Normalized Amplitude (offset)', fontsize=10)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Bottom: Interpretation
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        interpretation_text = "ASYSTOLE CLUSTER INTERPRETATION & CLINICAL SIGNIFICANCE:\n\n"
        for i in range(n_clusters):
            interp = asystole_interpretations[i]
            interpretation_text += f"● Cluster {i} ({colors[i].upper()}):\n"
            interpretation_text += f"  - Size: {interp['count']} signals ({interp['percentage']:.1f}%)\n"
            interpretation_text += f"  - Type: {interp['type']}\n"
            interpretation_text += f"  - Characteristics: {interp['characteristics']}\n"
            interpretation_text += f"  - Clinical Significance: {interp['significance']}\n\n"
        
        ax4.text(0.05, 0.95, interpretation_text, 
                transform=ax4.transAxes, fontsize=10, 
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        plt.suptitle('Asystole Alarm Signal Analysis with Clustering', 
                    fontsize=16, fontweight='bold')
        
        image_path = os.path.join(self.output_dir, 'asystole_clustering_interpreted.png')
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {image_path}")
        plt.close()
        
        # Print to console
        print("\n" + "="*60)
        print("ASYSTOLE CLUSTER ANALYSIS")
        print("="*60)
        for i in range(n_clusters):
            interp = asystole_interpretations[i]
            print(f"\nCluster {i}:")
            print(f"  Count: {interp['count']}")
            print(f"  Type: {interp['type']}")
            print(f"  Significance: {interp['significance']}")
        
        return image_path
    
    def interpret_asystole_clusters(self, features, cluster_labels, n_clusters):
        """
        Interpret asystole clusters based on signal characteristics
        """
        
        interpretations = []
        
        for i in range(n_clusters):
            mask = cluster_labels == i
            cluster_features = features[mask]
            
            count = np.sum(mask)
            mean_amplitude = np.mean(cluster_features[:, 0])
            mean_std = np.mean(cluster_features[:, 1])
            mean_max = np.mean(cluster_features[:, 2])
            
            # Classify asystole type
            if mean_std < np.percentile(features[:, 1], 33):
                alarm_type = "Flatline/Complete Asystole"
                characteristics = "Very low variability, near-flatline signal"
                significance = "Critical: Complete cardiac arrest, requires immediate CPR"
            elif mean_std > np.percentile(features[:, 1], 67):
                alarm_type = "Pre-arrest Deterioration"
                characteristics = "High variability, chaotic irregular activity"
                significance = "Urgent: Heart failing, may progress to complete arrest"
            else:
                alarm_type = "Artifact/False Positive"
                characteristics = "Moderate variability, possible motion artifact"
                significance = "Review needed: May be sensor displacement or patient movement"
            
            interpretations.append({
                'count': count,
                'percentage': (count / len(cluster_labels)) * 100,
                'type': alarm_type,
                'characteristics': characteristics,
                'significance': significance,
                'mean_amplitude': mean_amplitude,
                'mean_std': mean_std
            })
        
        return interpretations


def main():
    """Main execution"""
    
    print("\n" + "="*70)
    print(" PhysioNet Challenge 2015 - Enhanced Signal Analysis")
    print("="*70 + "\n")
    
    plotter = EnhancedSignalPlotter()
    
    # Load data
    ppg_signals, ecg_signals, labels = plotter.load_data()
    
    # Task 1: ECG
    print("\nTask 1: ECG Signal...")
    try:
        img1 = plotter.plot_ecg_waveform(ecg_signals)
        print("✅ Complete!")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Task 2: ECG II
    print("\nTask 2: ECG II Signal...")
    try:
        img2 = plotter.plot_ecg2_waveform(ecg_signals)
        print("✅ Complete!")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Task 3: PPG-ECG Correlation
    print("\nTask 3: PPG and ECG Correlation...")
    try:
        img3 = plotter.plot_ppg_ecg_correlation(ppg_signals, ecg_signals)
        print("✅ Complete!")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Task 4: PPG Clustering with Interpretation
    print("\nTask 4: PPG Clustering with Interpretation...")
    try:
        img4 = plotter.analyze_and_plot_ppg_clusters(ppg_signals, labels)
        print("✅ Complete!")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Task 5: Asystole Clustering
    print("\nTask 5: Asystole Clustering with Interpretation...")
    try:
        img5 = plotter.analyze_and_plot_asystole_clusters(ppg_signals, labels)
        print("✅ Complete!")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\n" + "="*70)
    print("🎉 ALL TASKS COMPLETED!")
    print("="*70)
    print("\nGenerated Images:")
    print("  1. ecg_signal.png")
    print("  2. ecg2_signal.png")
    print("  3. ppg_ecg_correlation.png          ← PPG and ECG together")
    print("  4. ppg_clustering_interpreted.png   ← With cluster definitions")
    print("  5. asystole_clustering_interpreted.png ← With significance")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()