import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from src.data_generator import build_file_label_list, make_label_map, midi_path_to_segments


def evaluate(data_dir, model_path, time_steps=500, out_dir='models'):
    files = build_file_label_list(data_dir)
    label_map = make_label_map(files)
    inv_map = {v: k for k, v in label_map.items()}
    num_labels = len(label_map)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print('Loading model from', model_path)
    model = load_model(model_path)

    split = int(0.8 * len(files))
    val_files = files[split:]

    y_true = []
    y_pred = []

    for path, lbl in val_files:
        try:
            segs = midi_path_to_segments(path, time_steps=time_steps)
        except Exception as e:
            print('Skipping', path, 'error', e)
            continue
        if segs.size == 0:
            continue
        preds = model.predict(segs, verbose=0)
        pidx = preds.argmax(axis=1)
        y_true.extend([label_map[lbl]] * len(pidx))
        y_pred.extend(pidx.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print('\nClassification report:')
    print(classification_report(y_true, y_pred, target_names=[inv_map[i] for i in range(num_labels)]))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_labels)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, 'confusion_matrix.png')

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_labels)
    plt.xticks(tick_marks, [inv_map[i] for i in range(num_labels)], rotation=90)
    plt.yticks(tick_marks, [inv_map[i] for i in range(num_labels)])

    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, format(cm_norm[i, j], '.2f'),
                     horizontalalignment='center',
                     color='white' if cm_norm[i, j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight')
    print('Saved confusion matrix to', out_png)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='../data')
    p.add_argument('--model', default='../models/final_model.h5')
    p.add_argument('--time-steps', type=int, default=500)
    p.add_argument('--out-dir', default='models')
    args = p.parse_args()
    evaluate(args.data_dir, args.model, time_steps=args.time_steps, out_dir=args.out_dir)
