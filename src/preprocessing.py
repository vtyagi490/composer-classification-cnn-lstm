import os
import numpy as np

try:
    import pretty_midi
except Exception:
    pretty_midi = None


def list_midi_files(data_dir):
    """Recursively list MIDI files in `data_dir`.

    Returns a sorted list of absolute paths.
    """
    mids = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith((".mid", ".midi")):
                mids.append(os.path.join(root, f))
    return sorted(mids)


def load_midi(path):
    """Load a MIDI file and return a PrettyMIDI instance.

    Requires `pretty_midi` to be installed.
    """
    if pretty_midi is None:
        raise ImportError("pretty_midi is required: pip install pretty_midi")
    return pretty_midi.PrettyMIDI(path)


def midi_to_pianoroll(pm, fs=100):
    """Convert a PrettyMIDI object to a piano-roll (128 x T) numpy array.

    Values are velocities (0-127)."""
    # pretty_midi.PrettyMIDI.get_piano_roll returns (128, T)
    pr = pm.get_piano_roll(fs=fs)
    return pr.astype(np.uint8)


def extract_note_sequence(pm):
    """Extract a list of notes (pitch, start, end, velocity) sorted by start time."""
    notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append((n.pitch, n.start, n.end, n.velocity))
    notes.sort(key=lambda x: x[1])
    return notes


def save_pianoroll(pr, out_path):
    """Save piano-roll numpy array to `out_path` (npz).

    Example: save_pianoroll(pr, 'features/song1.npz')
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, piano_roll=pr)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <data_dir>")
        sys.exit(1)
    data_dir = sys.argv[1]
    mids = list_midi_files(data_dir)
    print(f"Found {len(mids)} MIDI files")
    # print first 10
    for m in mids[:10]:
        print(m)
