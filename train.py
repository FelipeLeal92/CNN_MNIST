import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Carrega e normaliza o dataset MNIST
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)

# 2. Define a arquitetura da CNN
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Treina e salva o modelo
def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()
    model.summary()

    # Treinamento
    model.fit(x_train, y_train, epochs=5, batch_size=64,
              validation_split=0.1)

    # Avaliação
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    # Salva o modelo
    os.makedirs('saved_model', exist_ok=True)
    model.save('saved_model/cnn_mnist.keras')  # Salva no formato nativo Keras (.keras)
    print("Modelo salvo em saved_model/cnn_mnist")

if __name__ == '__main__':
    main()