import pandas as pd
import tensorflow as tf
import json, os
import numpy as np
import transformations

def download_from_kaggle(kaggle_json_filepath):
    kaggle_token = json.load(open('kaggle.json'))
    os.environ['KAGGLE_USERNAME'] = kaggle_token['username'] # username from the json file 
    os.environ['KAGGLE_KEY'] = kaggle_token['key'] # key from the json file
    os.system('kaggle datasets download -d joaopedromattos/reddit-dataset')
    os.system('unzip reddit-dataset.zip')

class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, x_df, y_df, tokenizer, batch_size = 32, steps = 300):
    self.tokenizer = tokenizer 
    self.x_df = x_df
    self.y_df = y_df
    self.batch_size = batch_size
    self.steps = steps
    self.length = len(self.x_df)

  def __len__(self):
    return self.steps

  def __data_generation(self, mask):
    x = self.tokenizer(self.x_df[mask].values.tolist(), padding = True, return_tensors = 'tf', truncation = True)
    y = tf.convert_to_tensor(self.y_df[mask].values, dtype = np.float32)
    return x, y

  def __getitem__(self, index):
    mask = np.concatenate([np.zeros(self.length - self.batch_size), np.ones(self.batch_size)]) == 1
    np.random.shuffle(mask)
    return self.__data_generation(mask)

def load_data(filepath):
    print('Reading csv file...')

    df = pd.read_csv(filepath, sep="|", lineterminator='\n')

    print('Preprocessing data...')

    df = df[df['Text'].apply(lambda x: isinstance(x, str))]
    df = df.drop_duplicates()
    return df

def split_data(df):
    labels = pd.get_dummies(df.Label)

    msk = np.random.rand(len(df)) < 0.6
    x_train_df = df.Text[msk]
    y_train_df = labels[msk]

    msk2 = np.random.rand(len(df) - len(x_train_df)) < 0.5

    x_test_df = df.Text[~msk][msk2]
    y_test_df = labels[~msk][msk2]

    x_val_df = df.Text[~msk][~msk2]
    y_val_df = labels[~msk][~msk2]

    return (x_train_df, y_train_df), (x_val_df, y_val_df), (x_test_df, y_test_df)
