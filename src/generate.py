import numpy as np
import pretty_midi
from .model import build_model

def generate_music(seed_sequence, model_path="output/model_weights.h5", length=100):
    """
    Generate new music from seed sequence
    - seed_sequence: initial input sequence (shape: [seq_length, 3])
    - length: number of events to generate
    """
    # Load model
    model = build_model()
    model.load_weights(model_path)
    
    generated = []
    current_seq = seed_sequence.copy()
    
    for _ in range(length):
        # Predict next event
        pred = model.predict(np.expand_dims(current_seq, 0))
        pitch = np.argmax(pred[0][0])
        step = pred[1][0][0]
        duration = pred[2][0][0]
        
        # Apply music theory constraints
        step = max(0.1, min(step, 2.0))  # Limit step size
        duration = max(0.1, min(duration, 4.0))  # Limit note length
        
        # Update sequence
        new_event = [pitch, step, duration]
        generated.append(new_event)
        current_seq = np.vstack([current_seq[1:], new_event])
    
    return np.array(generated)

def events_to_midi(events, output_path, resolution=0.25):
    """Convert events to MIDI file"""
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(0)  # Piano
    
    current_time = 0
    for pitch, step, duration in events:
        current_time += step
        note = pretty_midi.Note(
            velocity=100,
            pitch=int(pitch),
            start=current_time,
            end=current_time + duration
        )
        instrument.notes.append(note)
    
    midi.instruments.append(instrument)
    midi.write(output_path)

if __name__ == "__main__":
    # Start with random seed
    seed = np.random.randint(0, 128, size=(32, 3))
    generated = generate_music(seed, length=150)
    events_to_midi(generated, "output/generated/sample.mid")
    print("MIDI generated at output/generated/sample.mid")