import tensorflow as tf
import data

def get_classifier(n_classes, input_dim = 768):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation = 'relu', input_shape = (input_dim, )),
        tf.keras.layers.Dense(n_classes)
    ])
