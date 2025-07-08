import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from .model import build_model, compile_model
from .preprocess import process_dataset

def create_dataset(sequences, seq_length=32):
    """Create training sequences and targets"""
    xs, pitch_ys, step_ys, duration_ys = [], [], [], []
    
    for seq in sequences:
        for i in range(len(seq) - seq_length):
            # Input sequence
            x = seq[i:i+seq_length]
            xs.append(x)
            
            # Target (next event after sequence)
            next_event = seq[i+seq_length]
            pitch_ys.append(next_event[0])
            step_ys.append(next_event[1])
            duration_ys.append(next_event[2])
    
    return (
        np.array(xs),
        {
            'pitch': np.array(pitch_ys),
            'step': np.array(step_ys),
            'duration': np.array(duration_ys)
        }
    )

def train():
    # Load data
    sequences = np.load("../data/processed/sequences.npy", allow_pickle=True)
    X, y = create_dataset(sequences)
    
    # Build model
    model = build_model()
    model = compile_model(model)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint("output/model_weights.h5", save_best_only=True),
        EarlyStopping(patience=5, restore_best_weights=True)
    ]
    
    # Train
    history = model.fit(
        X, y,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks
    )
    
    print("Training complete! Model saved to output/model_weights.h5")

if __name__ == "__main__":
    train()