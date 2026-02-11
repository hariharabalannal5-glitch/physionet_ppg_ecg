"""
PhysioNet Challenge 2015 - Memory-Efficient Model Training Script
==================================================================
This script trains a lightweight deep learning model optimized for limited RAM.

Key optimizations:
- Downsamples signals to reduce memory usage
- Lighter CNN-LSTM architecture
- Smaller batch sizes

Author: Your Team
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import signal as scipy_signal
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class MemoryEfficientAlarmClassifier:
    """Memory-efficient binary classifier for arrhythmia alarm classification"""
    
    def __init__(self, data_dir='extracted_data', model_dir='models', downsample_factor=10):
        """
        Initialize the classifier
        
        Args:
            data_dir: Directory containing extracted signals
            model_dir: Directory to save trained models
            downsample_factor: Factor to downsample signals (10 = 250Hz -> 25Hz)
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.downsample_factor = downsample_factor
        os.makedirs(model_dir, exist_ok=True)
        
        self.model = None
        self.scaler_ppg = StandardScaler()
        self.scaler_ecg = StandardScaler()
        
        print(f"⚡ Memory optimization enabled!")
        print(f"   Downsample factor: {downsample_factor}x")
        print(f"   Original: 250 Hz → Downsampled: {250//downsample_factor} Hz")
    
    def load_data(self):
        """Load extracted signals and labels"""
        
        print("\n📂 Loading data...")
        
        ppg_signals = np.load(os.path.join(self.data_dir, 'ppg_signals.npy'))
        ecg_signals = np.load(os.path.join(self.data_dir, 'ecg_signals.npy'))
        labels = np.load(os.path.join(self.data_dir, 'labels.npy'))
        
        print(f"   Original PPG shape: {ppg_signals.shape}")
        print(f"   Original ECG shape: {ecg_signals.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Class distribution: {np.bincount(labels.astype(int))}")
        
        return ppg_signals, ecg_signals, labels
    
    def downsample_signals(self, ppg_signals, ecg_signals):
        """
        Downsample signals to reduce memory usage
        
        Args:
            ppg_signals: Original PPG signals
            ecg_signals: Original ECG signals
            
        Returns:
            Downsampled signals
        """
        
        print(f"\n⬇️  Downsampling signals by factor of {self.downsample_factor}...")
        
        # Downsample using decimation (includes anti-aliasing filter)
        ppg_downsampled = []
        ecg_downsampled = []
        
        for i in range(len(ppg_signals)):
            ppg_down = scipy_signal.decimate(ppg_signals[i], self.downsample_factor, zero_phase=True)
            ecg_down = scipy_signal.decimate(ecg_signals[i], self.downsample_factor, zero_phase=True)
            
            ppg_downsampled.append(ppg_down)
            ecg_downsampled.append(ecg_down)
            
            if (i + 1) % 100 == 0:
                print(f"   Processed {i+1}/{len(ppg_signals)} records...")
        
        ppg_downsampled = np.array(ppg_downsampled)
        ecg_downsampled = np.array(ecg_downsampled)
        
        print(f"   ✓ Downsampled PPG shape: {ppg_downsampled.shape}")
        print(f"   ✓ Downsampled ECG shape: {ecg_downsampled.shape}")
        print(f"   Memory reduction: {100 * (1 - ppg_downsampled.nbytes / ppg_signals.nbytes):.1f}%")
        
        return ppg_downsampled, ecg_downsampled
    
    def preprocess_signals(self, ppg_signals, ecg_signals, fit_scaler=True):
        """
        Preprocess signals (normalization, reshaping)
        
        Args:
            ppg_signals: PPG signals array
            ecg_signals: ECG II signals array
            fit_scaler: Whether to fit the scaler
            
        Returns:
            Preprocessed signals
        """
        
        print("🔧 Preprocessing signals...")
        
        # Reshape for scaling
        ppg_reshaped = ppg_signals.reshape(ppg_signals.shape[0], -1)
        ecg_reshaped = ecg_signals.reshape(ecg_signals.shape[0], -1)
        
        # Normalize
        if fit_scaler:
            ppg_normalized = self.scaler_ppg.fit_transform(ppg_reshaped)
            ecg_normalized = self.scaler_ecg.fit_transform(ecg_reshaped)
        else:
            ppg_normalized = self.scaler_ppg.transform(ppg_reshaped)
            ecg_normalized = self.scaler_ecg.transform(ecg_reshaped)
        
        # Reshape to (samples, timesteps, 1)
        ppg_normalized = ppg_normalized.reshape(ppg_signals.shape[0], -1, 1)
        ecg_normalized = ecg_normalized.reshape(ecg_signals.shape[0], -1, 1)
        
        return ppg_normalized, ecg_normalized
    
    def build_lightweight_model(self, input_shape):
        """
        Build a lightweight CNN-LSTM model
        
        Args:
            input_shape: Shape of input signals
            
        Returns:
            Compiled Keras model
        """
        
        print("🏗️  Building lightweight model architecture...")
        
        # Inputs
        ppg_input = layers.Input(shape=input_shape, name='ppg_input')
        ecg_input = layers.Input(shape=input_shape, name='ecg_input')
        
        # Shared CNN architecture (lightweight)
        def create_cnn_branch(x, name_prefix):
            # First conv block
            x = layers.Conv1D(32, kernel_size=7, activation='relu', 
                            padding='same', name=f'{name_prefix}_conv1')(x)
            x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
            x = layers.MaxPooling1D(pool_size=4, name=f'{name_prefix}_pool1')(x)
            
            # Second conv block
            x = layers.Conv1D(64, kernel_size=5, activation='relu', 
                            padding='same', name=f'{name_prefix}_conv2')(x)
            x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
            x = layers.MaxPooling1D(pool_size=4, name=f'{name_prefix}_pool2')(x)
            
            # Third conv block
            x = layers.Conv1D(128, kernel_size=3, activation='relu', 
                            padding='same', name=f'{name_prefix}_conv3')(x)
            x = layers.BatchNormalization(name=f'{name_prefix}_bn3')(x)
            x = layers.MaxPooling1D(pool_size=2, name=f'{name_prefix}_pool3')(x)
            
            # Single LSTM layer (not bidirectional to save memory)
            x = layers.LSTM(64, return_sequences=False, dropout=0.3,
                          name=f'{name_prefix}_lstm')(x)
            
            return x
        
        # Process both signals
        ppg_features = create_cnn_branch(ppg_input, 'ppg')
        ecg_features = create_cnn_branch(ecg_input, 'ecg')
        
        # Concatenate
        concatenated = layers.Concatenate(name='concat')([ppg_features, ecg_features])
        
        # Classification head
        x = layers.Dense(128, activation='relu', name='dense1')(concatenated)
        x = layers.Dropout(0.5, name='dropout1')(x)
        x = layers.Dense(64, activation='relu', name='dense2')(x)
        x = layers.Dropout(0.4, name='dropout2')(x)
        
        # Output
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        model = models.Model(inputs=[ppg_input, ecg_input], outputs=output, 
                           name='LightweightAlarmClassifier')
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        return model
    
    def train(self, ppg_train, ecg_train, y_train, 
              ppg_val, ecg_val, y_val,
              epochs=100, batch_size=16):
        """Train the model with memory-efficient settings"""
        
        print(f"\n🚀 Training model...")
        print(f"   Training samples: {len(y_train)}")
        print(f"   Validation samples: {len(y_val)}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size} (small for memory efficiency)")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                os.path.join(self.model_dir, 'best_model.h5'),
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train
        history = self.model.fit(
            [ppg_train, ecg_train], y_train,
            validation_data=([ppg_val, ecg_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, ppg_test, ecg_test, y_test):
        """Evaluate model on test data"""
        
        print("\n📊 Evaluating model...")
        
        results = self.model.evaluate([ppg_test, ecg_test], y_test, verbose=0)
        
        # Get predictions
        y_pred_prob = self.model.predict([ppg_test, ecg_test], verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Loss: {results[0]:.4f}")
        print(f"Accuracy: {results[1]:.4f}")
        print(f"Precision: {results[2]:.4f}")
        print(f"Recall: {results[3]:.4f}")
        print(f"AUC: {results[4]:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['False Alarm', 'True Alarm']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return {
            'loss': results[0],
            'accuracy': results[1],
            'precision': results[2],
            'recall': results[3],
            'auc': results[4]
        }
    
    def plot_training_history(self, history):
        """Plot training history"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train')
        axes[0, 1].plot(history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC
        axes[1, 0].plot(history.history['auc'], label='Train')
        axes[1, 0].plot(history.history['val_auc'], label='Validation')
        axes[1, 0].set_title('Model AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Precision & Recall
        axes[1, 1].plot(history.history['precision'], label='Train Precision')
        axes[1, 1].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'training_history.png'), dpi=300)
        print(f"\n📈 Training history saved to {self.model_dir}/training_history.png")
        plt.close()
    
    def save_model(self, filename='alarm_classifier_lite.h5'):
        """Save the trained model"""
        filepath = os.path.join(self.model_dir, filename)
        self.model.save(filepath)
        print(f"\n💾 Model saved to {filepath}")


def main():
    """Main training pipeline"""
    
    print("\n" + "="*60)
    print("PhysioNet Challenge 2015 - Memory-Efficient Training")
    print("="*60 + "\n")
    
    # Initialize classifier with downsampling
    classifier = MemoryEfficientAlarmClassifier(downsample_factor=10)
    
    # Load data
    ppg_signals, ecg_signals, labels = classifier.load_data()
    
    # Downsample to reduce memory
    ppg_signals, ecg_signals = classifier.downsample_signals(ppg_signals, ecg_signals)
    
    # Split data
    ppg_temp, ppg_test, ecg_temp, ecg_test, y_temp, y_test = train_test_split(
        ppg_signals, ecg_signals, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    ppg_train, ppg_val, ecg_train, ecg_val, y_train, y_val = train_test_split(
        ppg_temp, ecg_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )
    
    print(f"\n📊 Data split:")
    print(f"   Training: {len(y_train)} samples")
    print(f"   Validation: {len(y_val)} samples")
    print(f"   Test: {len(y_test)} samples")
    
    # Preprocess
    ppg_train, ecg_train = classifier.preprocess_signals(ppg_train, ecg_train, fit_scaler=True)
    ppg_val, ecg_val = classifier.preprocess_signals(ppg_val, ecg_val, fit_scaler=False)
    ppg_test, ecg_test = classifier.preprocess_signals(ppg_test, ecg_test, fit_scaler=False)
    
    # Build model
    input_shape = (ppg_train.shape[1], 1)
    classifier.model = classifier.build_lightweight_model(input_shape)
    
    # Print model summary
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    classifier.model.summary()
    
    # Train model
    history = classifier.train(
        ppg_train, ecg_train, y_train,
        ppg_val, ecg_val, y_val,
        epochs=100,
        batch_size=16  # Small batch size for memory efficiency
    )
    
    # Plot history
    classifier.plot_training_history(history)
    
    # Evaluate
    results = classifier.evaluate(ppg_test, ecg_test, y_test)
    
    # Save model
    classifier.save_model()
    
    print("\n✅ Training completed successfully!")
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"Final Test AUC: {results['auc']:.4f}")
    print("\n💡 Tip: Use predict_lite.py for predictions with this model")


if __name__ == "__main__":
    main()