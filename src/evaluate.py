import pretty_midi
import numpy as np
import librosa
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import os

def rhythmic_regularity(midi_path):
    """
    Calculate rhythmic regularity (variance of beat intervals)
    Lower values indicate steadier tempo
    """
    midi = pretty_midi.PrettyMIDI(midi_path)
    
    # Get beat times (automatic beat tracking)
    beats = midi.get_beats()
    
    if len(beats) < 2:
        return float('inf')  # Invalid for short sequences
    
    intervals = np.diff(beats)
    variance = np.var(intervals)
    mean = np.mean(intervals)
    
    # Plot rhythm analysis
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.hist(intervals, bins=20, color='skyblue')
    plt.title(f"Beat Intervals (Mean: {mean:.3f}s)")
    plt.xlabel("Time (seconds)")
    
    plt.subplot(122)
    plt.plot(beats[:-1], intervals, 'o-', color='salmon')
    plt.title(f"Variance: {variance:.6f}")
    plt.ylabel("Interval Duration")
    
    plt.tight_layout()
    plt.savefig("../outputs/evaluation/rhythm_analysis.png")
    plt.close()
    
    return variance

def melodic_smoothness(midi_path):
    """
    Calculate melodic smoothness (mean pitch change)
    Lower values indicate smoother melodies
    """
    midi = pretty_midi.PrettyMIDI(midi_path)
    pitches = []
    
    # Collect all pitches in order
    for instrument in midi.instruments:
        for note in sorted(instrument.notes, key=lambda x: x.start):
            pitches.append(note.pitch)
    
    if len(pitches) < 2:
        return float('inf'), float('inf')
    
    deltas = np.abs(np.diff(pitches))
    mean_delta = np.mean(deltas)
    var_delta = np.var(deltas)
    
    # Plot melodic contour
    plt.figure(figsize=(10, 4))
    plt.plot(pitches, 'o-', color='purple')
    plt.title(f"Melodic Contour (Avg Δ: {mean_delta:.2f} semitones)")
    plt.ylabel("Pitch (MIDI note)")
    plt.xlabel("Note index")
    plt.grid(alpha=0.3)
    plt.savefig("../outputs/evaluation/melodic_contour.png")
    plt.close()
    
    return mean_delta, var_delta

def phrase_structure(midi_path, phrase_length=8):
    """
    Analyze phrase structure using cosine similarity
    Higher values indicate better phrase repetition
    """
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = sorted([n for inst in midi.instruments for n in inst.notes], 
                  key=lambda x: x.start)
    
    if len(notes) < 2 * phrase_length:
        return 0.0  # Not enough notes for phrases
    
    # Extract phrases as pitch sequences
    phrases = []
    for i in range(0, len(notes) - phrase_length, phrase_length):
        phrase = [notes[i+j].pitch for j in range(phrase_length)]
        phrases.append(phrase)
    
    # Calculate cosine similarities between consecutive phrases
    similarities = []
    for i in range(len(phrases) - 1):
        v1 = np.array(phrases[i])
        v2 = np.array(phrases[i+1])
        
        # Normalize vectors
        v1 = (v1 - np.mean(v1)) / (np.std(v1) + 1e-8)
        v2 = (v2 - np.mean(v2)) / (np.std(v2) + 1e-8)
        
        # Cosine similarity (1 - cosine distance)
        sim = 1 - cosine(v1, v2)
        similarities.append(max(0, sim))  # Clip negative values
    
    avg_similarity = np.mean(similarities) if similarities else 0
    
    # Plot phrase similarity
    plt.figure(figsize=(8, 4))
    plt.plot(similarities, 'o-', color='green')
    plt.axhline(avg_similarity, color='red', linestyle='--')
    plt.title(f"Phrase Similarity (Avg: {avg_similarity:.2f})")
    plt.xlabel("Phrase pair index")
    plt.ylabel("Cosine similarity")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.savefig("../outputs/evaluation/phrase_similarity.png")
    plt.close()
    
    return avg_similarity

def evaluate_generation(output_dir="../outputs/generated"):
    """Run full evaluation on generated MIDI files"""
    os.makedirs("../outputs/evaluation", exist_ok=True)
    results = []
    
    for file in os.listdir(output_dir):
        if file.endswith(".mid"):
            path = os.path.join(output_dir, file)
            print(f"\nEvaluating {file}:")
            
            # Run all metrics
            rhythm_var = rhythmic_regularity(path)
            mean_delta, var_delta = melodic_smoothness(path)
            phrase_sim = phrase_structure(path)
            
            # Save results
            results.append({
                "file": file,
                "rhythmic_variance": rhythm_var,
                "melodic_mean_delta": mean_delta,
                "melodic_variance_delta": var_delta,
                "phrase_similarity": phrase_sim
            })
            
            print(f"  - Rhythmic Variance: {rhythm_var:.6f}")
            print(f"  - Melodic Smoothness: mean={mean_delta:.2f}, var={var_delta:.2f}")
            print(f"  - Phrase Similarity: {phrase_sim:.2f}")
    
    # Save comprehensive report
    with open("../outputs/evaluation/summary.txt", "w") as f:
        f.write("Music Generation Evaluation Report\n")
        f.write("=================================\n\n")
        for res in results:
            f.write(f"File: {res['file']}\n")
            f.write(f"Rhythmic Variance: {res['rhythmic_variance']:.6f}\n")
            f.write(f"Melodic Smoothness: mean={res['melodic_mean_delta']:.2f}, "
                    f"variance={res['melodic_variance_delta']:.2f}\n")
            f.write(f"Phrase Similarity: {res['phrase_similarity']:.2f}\n")
            f.write("\n" + "-"*50 + "\n")
    
    print("\nEvaluation complete! See output/evaluation/ for results")

if __name__ == "__main__":
    evaluate_generation()