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
  def __init__(self, x_df, y_df, tokenizer, bert, transformation = 'cls', batch_size = 32, steps = 300, return_mask = True):
    self.bert = bert 
    self.tokenizer = tokenizer 
    self.x_df = x_df
    self.y_df = y_df
    self.batch_size = batch_size
    self.steps = steps
    self.length = len(self.x_df)
    self.baked = False
    self.return_mask = return_mask

    self.position = 0
    self.total_batches = self.length // self.batch_size

    if type(transformation) == str:
        if transformation in transformations.str_to_func:
            self.transformation = transformations.str_to_func[transformation]
        else:
            print("Please pass a valid transformation")
    else:
        self.transformation = transformation

  def __len__(self):
    if self.baked:
        return len(self.baked_tensors)
    return self.steps

  def __data_generation(self, mask):
    inputs = self.tokenizer(self.x_df[mask].values.tolist(), padding = True, return_tensors = 'tf', truncation = True)
    y = tf.convert_to_tensor(self.y_df[mask].values, dtype = np.float32)

    x = self.bert(inputs)[0]    
    if self.transformation is not None:
        x = self.transformation(x)

    return ((x, inputs['attention_mask']), y) if self.return_mask else (x, y)

  def __getitem__(self, index):
    if self.baked:
        return self.baked_tensors[index]

    self.position = self.position % self.total_batches

    mask = np.concatenate([
            np.zeros(self.position * self.batch_size),
            np.ones(self.batch_size),
            np.zeros(self.length - (1 + self.position)*self.batch_size)]) == 1
    
    self.position += 1
    return self.__data_generation(mask)

  def bake(self):
    self.baked_tensors = []
    for i in self:
        self.baked_tensors.append(i)

    self.baked = True

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
