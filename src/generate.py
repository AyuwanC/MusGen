import numpy as np
import pretty_midi
import os
try:
    from model import build_model
    from preprocess import midi_to_events
except ImportError:
    from .model import build_model
    from .preprocess import midi_to_events

def get_seed(seed_type="random", sequence_length=32):
    """Get different types of seed sequences"""
    if seed_type == "random":
        return np.random.randint(0, 128, size=(sequence_length, 3))
    
    elif seed_type == "ascending":
        # Ascending scale pattern
        pitches = np.arange(60, 60+sequence_length) % 128
        steps = np.full(sequence_length, 0.25)
        durations = np.full(sequence_length, 0.5)
        return np.column_stack((pitches, steps, durations))
    
    elif seed_type == "chordal":
        # Chord progression pattern
        chords = [60, 64, 67]  # C Major chord
        pitches = np.array([chords[i % 3] for i in range(sequence_length)])
        steps = np.full(sequence_length, 0.5)
        durations = np.full(sequence_length, 1.0)
        return np.column_stack((pitches, steps, durations))
    
    elif seed_type == "rhythmic":
        # Strong rhythmic pattern
        pitches = np.random.randint(60, 72, size=sequence_length)
        steps = [0.25 if i % 2 == 0 else 0.5 for i in range(sequence_length)]
        durations = [0.25 if i % 4 == 0 else 0.5 for i in range(sequence_length)]
        return np.column_stack((pitches, steps, durations))
    
    else:
        raise ValueError(f"Unknown seed type: {seed_type}")
def constrain_to_scale(pitch, key, scale):
    """Force pitch to fit in selected scale"""
    # Key mapping (C=0, C#=1, ... B=11)
    key_offset = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
        "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
        "A#": 10, "Bb": 10, "B": 11
    }[key]
    
    # Scale patterns
    scale_patterns = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],
        "pentatonic": [0, 2, 4, 7, 9],
        "blues": [0, 3, 5, 6, 7, 10]
    }
    
    pattern = scale_patterns[scale]
    note_class = (pitch - key_offset) % 12
    
    # Find nearest note in scale
    if note_class not in pattern:
        distances = [min(abs(nc - note_class), 12 - abs(nc - note_class)) for nc in pattern]
        closest = pattern[np.argmin(distances)]
        pitch = pitch - note_class + closest
    
    return pitch

def constrain_to_range(pitch, pitch_range):
    """Force pitch within specified range"""
    min_pitch, max_pitch = pitch_range
    return max(min_pitch, min(pitch, max_pitch))

def apply_rhythm_density(step, duration, density):
    """Adjust rhythm based on density parameter"""
    # Higher density = shorter steps and durations
    adjusted_step = step * (1.5 - density)  # Range: 0.5-1.5x
    adjusted_duration = duration * (0.5 + density/2)  # Range: 0.5-1.0x
    return max(0.1, adjusted_step), max(0.1, adjusted_duration)

def apply_temperature(pitch_probs, temperature):
    """Apply temperature to pitch probabilities"""
    logits = np.log(pitch_probs) / temperature
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    return np.random.choice(len(probs), p=probs)
def generate_music(seed_sequence, model_path="output/model_weights.h5", length=100,
                  temperature=0.7, key="C", scale="major", rhythm_density=0.5,
                  pitch_range=(48, 84)):
    """
    Generate music with style controls:
    - temperature: Creativity level (0.1-1.5)
    - key: Musical key (C, D, Eb, etc.)
    - scale: Scale type (major, minor, pentatonic)
    - rhythm_density: Note density (0.0-1.0)
    - pitch_range: Min/max MIDI pitches
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
        step = max(0.1, min(float(pred[1][0][0]), 2.0))
        duration = max(0.1, min(float(pred[2][0][0]), 4.0))
        
        # Apply style constraints
        pitch = constrain_to_scale(pitch, key, scale)
        pitch = constrain_to_range(pitch, pitch_range)
        step, duration = apply_rhythm_density(step, duration, rhythm_density)
        
        # Apply temperature sampling
        if temperature != 1.0:
            pitch = apply_temperature(pred[0][0], temperature)
        
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
    events_to_midi(generated, "../output/generated/sample.mid")
    print("MIDI generated at output/generated/sample.mid")