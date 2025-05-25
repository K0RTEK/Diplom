import tensorflow as tf
from tensorflow.keras import layers, Model


def build_autoencoder(input_dim):
    class AnomalyDetector(Model):
        def __init__(self, input_dim, **kwargs):
            super(AnomalyDetector, self).__init__(**kwargs)
            self.encoder = tf.keras.Sequential([
                layers.Dense(16, activation='relu'),
                layers.Dense(8, activation='relu'),
                layers.Dense(4, activation='relu')
            ])
            self.decoder = tf.keras.Sequential([
                layers.Dense(8, activation='relu'),
                layers.Dense(16, activation='relu'),
                layers.Dense(input_dim, activation='linear')
            ])

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    return AnomalyDetector(input_dim)


def save_model(model, path):
    model.save(path)

