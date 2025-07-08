import librosa
import numpy as np
import pretty_midi
import os

def rhythmic_regularity(midi_path):
    """Calculate variance of beat intervals"""
    midi = pretty_midi.PrettyMIDI(midi_path)
    beats = midi.get_beats()
    intervals = np.diff(beats)
    return np.var(intervals)  # Lower = more regular

def melodic_smoothness(midi_path):
    """Calculate pitch transition smoothness"""
    midi = pretty_midi.PrettyMIDI(midi_path)
    pitches = [note.pitch for inst in midi.instruments for note in inst.notes]
    deltas = np.abs(np.diff(pitches))
    return np.mean(deltas), np.var(deltas)  # Lower = smoother

def phrase_structure(midi_path, phrase_length=4):
    """Analyze repetition of musical phrases"""
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = sorted([note for inst in midi.instruments for note in inst.notes], key=lambda x: x.start)
    
    # Extract phrases
    phrases = []
    for i in range(0, len(notes) - phrase_length, phrase_length):
        phrase = [n.pitch for n in notes[i:i+phrase_length]]
        phrases.append(phrase)
    
    # Calculate phrase similarity
    similarities = []
    for i in range(len(phrases) - 1):
        # Simple cosine similarity
        v1 = np.array(phrases[i])
        v2 = np.array(phrases[i+1])
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        similarities.append(cos_sim)
    
    return np.mean(similarities)  # Higher = better structure

def evaluate_generation(output_dir="output/generated"):
    """Run evaluation on all generated files"""
    for file in os.listdir(output_dir):
        if file.endswith(".mid"):
            path = os.path.join(output_dir, file)
            print(f"\nEvaluating {file}:")
            print(f"Rhythmic Regularity: {rhythmic_regularity(path):.4f}")
            mean_delta, var_delta = melodic_smoothness(path)
            print(f"Melodic Smoothness: mean={mean_delta:.2f}, var={var_delta:.2f}")
            print(f"Phrase Structure: {phrase_structure(path):.2f}")