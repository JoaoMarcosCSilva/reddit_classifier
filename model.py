import tensorflow as tf
import data

def get_classifier(n_classes):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation = 'relu', input_shape = (768, )),
        tf.keras.layers.Dense(n_classes)
    ])
