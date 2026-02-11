# 1. Install dependencies
pip install numpy pandas scipy scikit-learn tensorflow matplotlib

# 2. Download training data (from PhysioNet website)

# 3. Create labels
python create_labels.py

# 4. Extract signals
python extract_signals.py

# 5. Train model (lightweight - RECOMMENDED)
python train_model_lite.py

# 6. Make predictions
python predict_manual_input.py  # Type values directly!
