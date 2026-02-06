"""
PhysioNet Challenge 2015 - Prediction Script (Lightweight Model)
=================================================================
This script makes predictions using the lightweight downsampled model.

Author: Your Team
Date: 2024
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy import signal as scipy_signal
import os


class LightweightAlarmPredictor:
    """Predict true/false alarms using the lightweight model"""
    
    def __init__(self, model_path='models/alarm_classifier_lite.h5', downsample_factor=10):
        """
        Initialize predictor
        
        Args:
            model_path: Path to the trained model file
            downsample_factor: Must match training (default: 10)
        """
        self.model_path = model_path
        self.downsample_factor = downsample_factor
        self.model = None
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        print(f"📥 Loading model from {self.model_path}...")
        self.model = keras.models.load_model(self.model_path)
        print("✅ Model loaded successfully!")
    
    def downsample_signal(self, signal_data):
        """Downsample a signal to match training"""
        return scipy_signal.decimate(signal_data, self.downsample_factor, zero_phase=True)
    
    def preprocess_signal(self, ppg_signal, ecg_signal):
        """
        Preprocess input signals
        
        Args:
            ppg_signal: PPG signal array
            ecg_signal: ECG II signal array
            
        Returns:
            Preprocessed signals
        """
        
        # Ensure numpy arrays
        ppg_signal = np.array(ppg_signal)
        ecg_signal = np.array(ecg_signal)
        
        # Downsample
        ppg_down = self.downsample_signal(ppg_signal)
        ecg_down = self.downsample_signal(ecg_signal)
        
        # Normalize
        ppg_normalized = (ppg_down - np.mean(ppg_down)) / (np.std(ppg_down) + 1e-8)
        ecg_normalized = (ecg_down - np.mean(ecg_down)) / (np.std(ecg_down) + 1e-8)
        
        # Reshape
        ppg_processed = ppg_normalized.reshape(1, -1, 1)
        ecg_processed = ecg_normalized.reshape(1, -1, 1)
        
        return ppg_processed, ecg_processed
    
    def predict(self, ppg_signal, ecg_signal, return_probability=False):
        """
        Predict whether an alarm is true or false
        
        Args:
            ppg_signal: PPG signal array (75,000 samples at 250 Hz)
            ecg_signal: ECG II signal array (75,000 samples at 250 Hz)
            return_probability: If True, return probability
            
        Returns:
            prediction: Binary prediction or probability
        """
        
        # Preprocess
        ppg_processed, ecg_processed = self.preprocess_signal(ppg_signal, ecg_signal)
        
        # Predict
        probability = self.model.predict([ppg_processed, ecg_processed], verbose=0)[0][0]
        
        if return_probability:
            return probability
        else:
            return int(probability > 0.5)
    
    def interpret_prediction(self, prediction, probability=None):
        """Interpret the prediction"""
        
        if prediction == 0:
            result = "FALSE ALARM ❌"
            meaning = "This alarm is likely a false positive and can be suppressed."
        else:
            result = "TRUE ALARM ✅"
            meaning = "This alarm is genuine and requires attention."
        
        interpretation = f"\n{'='*60}\n"
        interpretation += f"PREDICTION: {result}\n"
        interpretation += f"{'='*60}\n"
        
        if probability is not None:
            interpretation += f"Confidence: {probability:.2%}\n"
            interpretation += f"Interpretation: {meaning}\n"
        
        interpretation += f"{'='*60}\n"
        
        return interpretation


def demo_prediction():
    """Demo function"""
    
    print("\n" + "="*60)
    print("PhysioNet Challenge 2015 - Lightweight Prediction Demo")
    print("="*60 + "\n")
    
    # Initialize predictor
    try:
        predictor = LightweightAlarmPredictor()
    except FileNotFoundError:
        print("❌ Model not found. Please train the model first using train_model_lite.py")
        return
    
    # Test on real data if available
    if os.path.exists('extracted_data/ppg_signals.npy'):
        ppg_signals = np.load('extracted_data/ppg_signals.npy')
        ecg_signals = np.load('extracted_data/ecg_signals.npy')
        labels = np.load('extracted_data/labels.npy')
        
        print(f"Testing on extracted data...\n")
        
        # Test on first 5 records
        for i in range(min(5, len(ppg_signals))):
            prediction = predictor.predict(ppg_signals[i], ecg_signals[i])
            probability = predictor.predict(ppg_signals[i], ecg_signals[i], return_probability=True)
            
            print(f"Record {i+1}:")
            print(f"  Predicted: {'True Alarm' if prediction == 1 else 'False Alarm'}")
            print(f"  Actual: {'True Alarm' if labels[i] == 1 else 'False Alarm'}")
            print(f"  Confidence: {probability:.2%}")
            print(f"  {'✅ Correct' if prediction == labels[i] else '❌ Incorrect'}\n")
    else:
        print("No extracted data found. Run extract_signals.py first.\n")
    
    print("="*60)


if __name__ == "__main__":
    demo_prediction()