import argparse
from generate import generate_music, events_to_midi, get_seed
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Generate music with style controls")
    parser.add_argument("--seed", type=str, default="random", 
                        choices=["random", "ascending", "chordal", "rhythmic"],
                        help="Type of seed sequence")
    parser.add_argument("--key", type=str, default="C", 
                        help="Musical key (e.g., C, G, Dm, Eb)")
    parser.add_argument("--scale", type=str, default="major", 
                        choices=["major", "minor", "pentatonic", "blues"],
                        help="Musical scale type")
    parser.add_argument("--temp", type=float, default=0.7, 
                        help="Creativity temperature (0.1-1.5)")
    parser.add_argument("--density", type=float, default=0.5, 
                        help="Rhythm density (0.0-1.0)")
    parser.add_argument("--low", type=int, default=48, 
                        help="Lowest MIDI pitch (0-127)")
    parser.add_argument("--high", type=int, default=84, 
                        help="Highest MIDI pitch (0-127)")
    parser.add_argument("--length", type=int, default=100, 
                        help="Number of notes to generate")
    parser.add_argument("--output", type=str, default="output/generated/custom.mid",
                        help="Output MIDI file path")
    
    args = parser.parse_args()
    
    seed = get_seed(args.seed)
    melody = generate_music(
        seed,
        temperature=args.temp,
        key=args.key[:-1] if args.key.endswith('m') else args.key,
        scale="minor" if args.key.endswith('m') else args.scale,
        rhythm_density=args.density,
        pitch_range=(args.low, args.high),
        length=args.length
    )
    events_to_midi(melody, args.output)
    print(f"Generated music saved to {args.output}")

if __name__ == "__main__":
    main()