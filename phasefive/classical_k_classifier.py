import tensorflow as tf

def create_fair_classical_k_model(k, input_shape=(4, 4, 1)):
    # QNN (1 layer) has 32 parameters.
    # Classical MLP with 1 hidden unit:
    # 16*1 (weights) + 1 (bias) + 1*k (weights) + k (bias)
    # For k=5: 16 + 1 + 5 + 5 = 27 parameters.
    # For k=8: 16 + 1 + 8 + 8 = 33 parameters.
    # This is very close to the QNN's 32.
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(1, activation='relu'),
        tf.keras.layers.Dense(k, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
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
