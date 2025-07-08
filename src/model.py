import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, Concatenate
from tensorflow.keras.models import Model

def build_model(vocab_size=128, seq_length=32):
    inputs = Input(shape=(seq_length, 3))

    # ✅ CuDNN-compatible LSTM
    x = LSTM(
        units=128,
        return_sequences=True,
        activation='tanh',
        recurrent_activation='sigmoid',
        use_bias=True,
        unroll=False,
        dropout=0.0,
        recurrent_dropout=0.0
    )(inputs)

    x = Dropout(0.3)(x)

    # ✅ CuDNN-compatible LSTM
    x = LSTM(
        512,
        return_sequences=True,
        activation='tanh',
        recurrent_activation='sigmoid',
        use_bias=True,
        unroll=False,
        dropout=0.0,
        recurrent_dropout=0.0
    )(x)

    # ✅ Attention
    attention = Attention()([x, x])
    x = Concatenate(axis=-1)([x, attention])

    # ✅ Final CuDNN-compatible LSTM
    x = LSTM(
        512,
        activation='tanh',
        recurrent_activation='sigmoid',
        use_bias=True,
        unroll=False,
        dropout=0.0,
        recurrent_dropout=0.0
    )(x)

    pitch_out = Dense(vocab_size, activation='softmax', name='pitch')(x)
    step_out = Dense(1, activation='relu', name='step')(x)
    duration_out = Dense(1, activation='relu', name='duration')(x)

    return Model(inputs=inputs, outputs=[pitch_out, step_out, duration_out])

def compile_model(model):
    model.compile(
        optimizer='adam',
        loss={
            'pitch': 'sparse_categorical_crossentropy',
            'step': 'mse',
            'duration': 'mse'
        },
        metrics={
            'pitch': 'accuracy'
        }
    )
    return model