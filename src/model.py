import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_lstm(input_shape, num_classes, conv_filters=32, lstm_units=128):
    """Build and compile a simple CNN+LSTM model.

    - `input_shape` should be (time_steps, n_pitches) e.g. (T, 128)
    - returns a compiled `tf.keras.Model`.
    """
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(conv_filters, kernel_size=3, padding='same', activation='relu')(inp)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(conv_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units))(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # quick smoke test
    model = build_cnn_lstm((500, 128), num_classes=10)
    model.summary()
