import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy import signal as scipy_signal
import os

class ManualInputPredictor:
    """Predict alarms from manually entered signal values"""
    
    def __init__(self, model_path='models/alarm_classifier_lite.h5', downsample_factor=10):
        self.model_path = model_path
        self.downsample_factor = downsample_factor
        # The model expects this specific length after preprocessing
        self.REQUIRED_LENGTH = 7500 
        self.model = None
        self.load_model()
    
    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Model not found at {self.model_path}")
        
        print(f"📥 Loading model...")
        self.model = keras.models.load_model(self.model_path)
        print("✅ Model loaded!\n")
    
    def get_signal_from_user(self, signal_name):
        print(f"\n{'='*60}\n📊 Enter {signal_name} Signal Values\n{'='*60}")
        print("1. Space-separated  2. Comma-separated  3. Multi-line ('DONE')")
        
        method = input("Choose input method (1/2/3): ").strip()
        values = []
        
        if method == '1':
            user_input = input(f"Enter {signal_name} (space-separated): > ")
            values = [float(x) for x in user_input.replace(',', ' ').split() if x.strip()]
        elif method == '2':
            user_input = input(f"Enter {signal_name} (comma-separated): > ")
            values = [float(x) for x in user_input.split(',') if x.strip()]
        elif method == '3':
            while True:
                line = input("> ").strip()
                if line.upper() == 'DONE': break
                try: values.append(float(line))
                except: print("Skipped invalid value")
        
        if not values: raise ValueError("No values entered")
        return np.array(values, dtype=np.float32)

    def preprocess_signals(self, ppg_signal, ecg_signal):
        """Preprocess and fix shape to match model requirements"""
        print(f"\n🔧 PREPROCESSING SIGNALS")
        
        # 1. Handle length mismatch between PPG and ECG
        min_len = min(len(ppg_signal), len(ecg_signal))
        ppg_signal, ecg_signal = ppg_signal[:min_len], ecg_signal[:min_len]

        # 2. Downsampling (only if user provided enough data)
        if len(ppg_signal) >= self.downsample_factor * 2:
            ppg_proc = scipy_signal.decimate(ppg_signal, self.downsample_factor, zero_phase=True)
            ecg_proc = scipy_signal.decimate(ecg_signal, self.downsample_factor, zero_phase=True)
        else:
            ppg_proc, ecg_proc = ppg_signal, ecg_signal

        # 3. Normalization
        ppg_proc = (ppg_proc - np.mean(ppg_proc)) / (np.std(ppg_proc) + 1e-8)
        ecg_proc = (ecg_proc - np.mean(ecg_proc)) / (np.std(ecg_proc) + 1e-8)

        # 4. FIX: Padding or Truncating to exactly 7500
        # This solves the "expected shape=(None, 7500, 1)" error
        def adjust_length(sig, target):
            if len(sig) > target:
                return sig[:target]
            else:
                return np.pad(sig, (0, target - len(sig)), 'constant')

        ppg_final = adjust_length(ppg_proc, self.REQUIRED_LENGTH)
        ecg_final = adjust_length(ecg_proc, self.REQUIRED_LENGTH)

        # 5. Reshape for CNN input: (batch, timesteps, channels)
        ppg_final = ppg_final.reshape(1, self.REQUIRED_LENGTH, 1)
        ecg_final = ecg_final.reshape(1, self.REQUIRED_LENGTH, 1)
        
        return ppg_final, ecg_final

    def predict(self, ppg_signal, ecg_signal):
        ppg_proc, ecg_proc = self.preprocess_signals(ppg_signal, ecg_signal)
        print(f"\n🤖 RUNNING PREDICTION (Input shape: {ppg_proc.shape})")
        
        probability = self.model.predict([ppg_proc, ecg_proc], verbose=0)[0][0]
        prediction = int(probability > 0.5)
        
        self.display_result(prediction, probability)
        return {'prediction': prediction, 'probability': float(probability)}

    def display_result(self, prediction, probability):
        print(f"\n{'='*60}")
        if prediction == 0:
            print("🟢 PREDICTION: FALSE ALARM (Confidence: {:.2%})".format(1-probability))
        else:
            print("🔴 PREDICTION: TRUE ALARM (Confidence: {:.2%})".format(probability))
        print(f"{'='*60}\n")

def main():
    try:
        predictor = ManualInputPredictor()
    except Exception as e:
        print(e); return

    while True:
        print("1. Manual Input  2. Random Test  3. Exit")
        choice = input("Select: ")
        
        if choice == '1':
            try:
                ppg = predictor.get_signal_from_user("PPG")
                ecg = predictor.get_signal_from_user("ECG")
                predictor.predict(ppg, ecg)
            except Exception as e: print(f"Error: {e}")
        elif choice == '2':
            # Generate dummy data of the correct size
            p, e = np.random.randn(75000), np.random.randn(75000)
            predictor.predict(p, e)
        elif choice == '3': break

if __name__ == "__main__":
    main()