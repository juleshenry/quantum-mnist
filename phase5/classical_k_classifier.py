import tensorflow as tf

def create_fair_classical_k_model(k, hidden_units=1, learning_rate=0.01, input_shape=(16,)):
    # 16-H-k architecture.
    # Total parameters: 16*H (weights) + H (bias) + H*k (weights) + k (bias)
    # H=1: 17 + 2k
    # H=2: 34 + 3k
    # QNN with 1 layer (16 data qubits): 32 params (16 XX, 16 ZZ).
    # QNN with 2 layers (16 data qubits): 64 params.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(hidden_units, activation='relu'),
        tf.keras.layers.Dense(k, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_cnn_k_model(k, input_shape=(28, 28, 1)):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(k, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
