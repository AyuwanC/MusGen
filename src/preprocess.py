import pretty_midi
import numpy as np
import os
from tqdm import tqdm

def midi_to_events(midi_path, resolution=0.25):
    """
    Convert MIDI file to sequence of events [pitch, step, duration]
    - resolution: quantize time to this fraction (0.25 = 16th notes)
    """
    midi = pretty_midi.PrettyMIDI(midi_path)
    events = []
    current_time = 0
    
    for instrument in midi.instruments:
        for note in instrument.notes:
            # Quantize to grid
            start = np.round(note.start / resolution) * resolution
            end = np.round(note.end / resolution) * resolution
            duration = end - start
            step = start - current_time  # Time since last event
            
            events.append([note.pitch, step, duration])
            current_time = start
    
    return np.array(events)

def process_dataset(input_dir="../data/maestro/maestro-v3.0.0/2004", output_dir="../data/processed"):
    """
    Process all MIDI files in directory
    """
    os.makedirs(output_dir, exist_ok=True)
    sequences = []
    
    for file in tqdm(os.listdir(input_dir)):
        if file.endswith(".midi") or file.endswith(".mid"):
            path = os.path.join(input_dir, file)
            try:
                events = midi_to_events(path)
                sequences.append(events)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
    
    # Save sequences
    np.save(os.path.join(output_dir, "sequences.npy"), np.array(sequences, dtype=object))
    print(f"Processed {len(sequences)} files. Data saved to {output_dir}")

if __name__ == "__main__":
    process_dataset()