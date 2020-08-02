from sklearn import metrics
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
              matrix += metrics.confusion_matrix(tf.argmax(y, axis = 1), tf.argmax(pred, axis = 1), list(range(19)))
            else:
              matrix = metrics.confusion_matrix(tf.argmax(y, axis = 1), tf.argmax(pred, axis = 1), list(range(19)))
   
    df_matrix = pd.DataFrame((matrix) / (1e-6 + np.sum(matrix, axis = 0)), index = columns, columns = columns)
    sn.heatmap(df_matrix)
