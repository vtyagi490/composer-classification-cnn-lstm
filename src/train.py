import os
import math
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from src.data_generator import build_file_label_list, make_label_map, generator
from src.model import build_cnn_lstm


def train(data_dir, epochs=10, batch_size=8, time_steps=500, out_dir='models'):
    files = build_file_label_list(data_dir)
    label_map = make_label_map(files)
    num_classes = len(label_map)
    os.makedirs(out_dir, exist_ok=True)

    # simple train/val split
    split = int(0.8 * len(files))
    train_files = files[:split]
    val_files = files[split:]

    train_gen = generator(train_files, label_map, batch_size=batch_size, time_steps=time_steps, shuffle=True)
    val_gen = generator(val_files, label_map, batch_size=batch_size, time_steps=time_steps, shuffle=False)

    model = build_cnn_lstm((time_steps, 128), num_classes)
    steps_per_epoch = max(1, math.ceil(sum(max(1, os.path.getsize(p) and 1) for p, _ in train_files) * 1.0 / batch_size))
    val_steps = max(1, math.ceil(len(val_files) / batch_size))

    ckpt = ModelCheckpoint(os.path.join(out_dir, 'best_weights.h5'), monitor='val_accuracy', save_best_only=True, verbose=1)
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    model.fit(train_gen, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=val_gen, validation_steps=val_steps, callbacks=[ckpt, reduce])
    model.save(os.path.join(out_dir, 'final_model.h5'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--time-steps', type=int, default=500)
    parser.add_argument('--out-dir', default='models')
    args = parser.parse_args()
    train(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, time_steps=args.time_steps, out_dir=args.out_dir)
