from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import tensorflow as tf
import numpy as np

def plot_confusion_matrix(model, datagen, repeat, columns):

    matrix = None
    for i in range(repeat):
        for x, y in datagen:
            pred = model(x)

            if matrix is not None:
              matrix += metrics.confusion_matrix(tf.argmax(y, axis = 1), tf.argmax(pred, axis = 1), list(range(len(labels))))
            else:
              matrix = metrics.confusion_matrix(tf.argmax(y, axis = 1), tf.argmax(pred, axis = 1), list(range(len(labels))))
   
    df_matrix = pd.DataFrame((matrix) / (1e-6 + np.sum(matrix, axis = 0)), index = columns, columns = columns)
    sn.heatmap(df_matrix, vmin = 0, vmax = 1, annot = True)

def plot_predictions(model, text, datagen, columns):

    inputs = datagen.tokenizer([text], padding = True, return_tensors = 'tf', truncation = True)
    x = datagen.transformation(datagen.bert(inputs)[0])
    y = tf.nn.softmax(model(x)).numpy()

    df = pd.DataFrame(y, columns = columns)
    plt.title(text)
    sn.heatmap(df)        
