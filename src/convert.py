import os
import subprocess
import pretty_midi
import soundfile as sf
from tqdm import tqdm

def midi_to_wav(midi_path, wav_path):
    """
    Convert MIDI to WAV with robust fallback
    """
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    
    # First try: PrettyMIDI's built-in synthesis
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        audio = midi.synthesize()
        sf.write(wav_path, audio, 44100)
        print(f"Converted using PrettyMIDI synthesis: {wav_path}")
        return True
    except Exception as e:
        print(f"PrettyMIDI synthesis failed: {e}")
    
    # Second try: FluidSynth if installed
    try:
        soundfont = "../soundfonts/FluidR3_GM.sf2"
        if not os.path.exists(soundfont):
            # Try to download if missing
            subprocess.run([
                "wget", "https://github.com/FluidSynth/fluidsynth/raw/master/sf2/FluidR3_GM.sf2",
                "-O", soundfont
            ], check=True)
        
        subprocess.run([
            "fluidsynth", "-ni", soundfont, 
            midi_path, "-F", wav_path, "-r", "44100"
        ], check=True)
        print(f"Converted using FluidSynth: {wav_path}")
        return True
    except Exception as e:
        print(f"FluidSynth conversion failed: {e}")
    
    # Final fallback: Simple sine wave synthesis
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        # Create simple sine wave instrument
        for instrument in midi.instruments:
            instrument.program = 0  # Acoustic Grand Piano
            
        audio = midi.synthesize()
        sf.write(wav_path, audio, 44100)
        print(f"Converted using simple sine synthesis: {wav_path}")
        return True
    except Exception as e:
        print(f"All conversion methods failed: {e}")
        return False

def convert_all_generated(output_dir="../outputs/generated"):
    """Convert all MIDI files in directory to WAV"""
    converted = 0
    for file in tqdm(os.listdir(output_dir)):
        if file.endswith((".mid", ".midi")):
            midi_path = os.path.join(output_dir, file)
            wav_path = os.path.join(output_dir, file.replace(".mid", ".wav").replace(".midi", ".wav"))
            
            if midi_to_wav(midi_path, wav_path):
                converted += 1
    
    print(f"\nSuccessfully converted {converted} files to audio")
    return converted

if __name__ == "__main__":
    convert_all_generated()