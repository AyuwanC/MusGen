import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# Assuming these are defined in your project
from .model import build_model, compile_model 
from .preprocess import process_dataset

# Paths for model saving
weights_path = "../output/model_weights.h5"
full_model_path = "../output/model_full.h5"
data_path = "../data/processed/sequences.npy"

def create_dataset(sequences, seq_length=32):
    """Create training sequences and targets"""
    xs, pitch_ys, step_ys, duration_ys = [], [], [], []
    
    for seq in sequences:
        for i in range(len(seq) - seq_length):
            x = seq[i:i+seq_length]
            xs.append(x)
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
    # Load dataset
    sequences = np.load(data_path, allow_pickle=True)
    X, y = create_dataset(sequences)

    # Load full model if available
    if os.path.exists(full_model_path):
        print(f"Loading full model from {full_model_path}")
        model = tf.keras.models.load_model(full_model_path)
    else:
        print("No full model found. Building from scratch.")
        model = build_model()
        model = compile_model(model)
        # Optionally load just weights if they exist
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            print(f"Loaded weights from {weights_path}")

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=weights_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            patience=5,
            restore_best_weights=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=full_model_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_weights_only=False,
            verbose=1
        )
    ]

    # Train the model
    history = model.fit(
        X, y,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks
    )

    print("Training complete! Model and weights saved.")

if __name__ == "__main__":
    train()
