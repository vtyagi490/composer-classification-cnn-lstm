import os
import random
import numpy as np
from src.preprocessing import load_midi, midi_to_pianoroll, list_midi_files


def infer_label_from_path(path, data_dir):
    """Infer composer label from file path.

    - If file is inside a subdirectory of data_dir, use that directory name.
    - Otherwise, use the first token of filename as a heuristic.
    """
    parent = os.path.basename(os.path.dirname(path))
    if parent and parent != os.path.basename(os.path.abspath(data_dir)):
        return parent
    fname = os.path.basename(path)
    # heuristic: take first word before space or dash
    label = fname.split()[0].split('-')[0]
    return label


def build_file_label_list(data_dir):
    files = list_midi_files(data_dir)
    file_label = []
    for f in files:
        lbl = infer_label_from_path(f, data_dir)
        file_label.append((f, lbl))
    return file_label


def make_label_map(file_label):
    labels = sorted({lbl for _, lbl in file_label})
    label2idx = {l: i for i, l in enumerate(labels)}
    return label2idx


def midi_path_to_segments(path, time_steps=500, fs=100):
    """Convert a MIDI file to an array of segments with shape (N, time_steps, 128)."""
    pm = load_midi(path)
    pr = midi_to_pianoroll(pm, fs=fs)  # shape (128, T)
    pr = pr.T  # (T, 128)
    T = pr.shape[0]
    if T < time_steps:
        pad = np.zeros((time_steps - T, pr.shape[1]), dtype=np.uint8)
        seg = np.vstack([pr, pad])
        return np.expand_dims(seg.astype(np.float32) / 127.0, axis=0)
    # split into non-overlapping segments
    n_segments = T // time_steps
    segments = []
    for i in range(n_segments):
        s = pr[i * time_steps:(i + 1) * time_steps]
        segments.append(s)
    return np.stack(segments).astype(np.float32) / 127.0


def generator(file_label, label2idx, batch_size=8, time_steps=500, shuffle=True):
    """Yield batches of (X, y).

    X shape: (batch, time_steps, 128)
    y shape: (batch,)
    """
    entries = list(file_label)
    while True:
        if shuffle:
            random.shuffle(entries)
        X_batch = []
        y_batch = []
        for path, lbl in entries:
            try:
                segs = midi_path_to_segments(path, time_steps=time_steps)
            except Exception:
                continue
            idx = label2idx[lbl]
            for s in segs:
                X_batch.append(s)
                y_batch.append(idx)
                if len(X_batch) >= batch_size:
                    X = np.stack(X_batch)
                    y = np.array(y_batch, dtype=np.int32)
                    yield X, y
                    X_batch = []
                    y_batch = []
        # yield remaining
        if X_batch:
            X = np.stack(X_batch)
            y = np.array(y_batch, dtype=np.int32)
            yield X, y
            X_batch = []
            y_batch = []


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    args = parser.parse_args()
    fl = build_file_label_list(args.data_dir)
    lm = make_label_map(fl)
    print('Found composers:', lm)
    # print sample
    for i, (p, l) in enumerate(fl[:10]):
        print(i, l, p)
