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
              matrix += metrics.confusion_matrix(tf.argmax(y, axis = 1), tf.argmax(pred, axis = 1), list(range(19)))
            else:
              matrix = metrics.confusion_matrix(tf.argmax(y, axis = 1), tf.argmax(pred, axis = 1), list(range(19)))
   
    df_matrix = pd.DataFrame((matrix) / (1e-6 + np.sum(matrix, axis = 0)), index = columns, columns = columns)
    sn.heatmap(df_matrix)

def plot_predictions(classifier, text, train_gen):
    if type(text) == str:
        text = [text]

    inputs = train_gen.tokenizer(text, padding = True, return_tensors = 'tf', truncation = True)
    x = train_gen.transformation(train_gen.bert(inputs)[0])
    y = tf.nn.softmax(classifier(x)).numpy()

    df = pd.DataFrame(y, columns = labels)
    plt.title(text)
    sn.heatmap(df)        
