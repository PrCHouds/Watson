import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import keras_nlp
import seaborn as sns
import matplotlib.pyplot as plt
import os
from googletrans import Translator

def read_data(dir_path):
    df_train = pd.read_csv(dir_path+ "train.csv")
    df_test = pd.read_csv(dir_path + "test.csv")
    return df_train, df_test
def replace_after_translation(train_data, test_data):
    translator = Translator()
    for i in range(len(train_data['premise'])):
        if train_data['lang_abv'][i] == 'en':
            continue
        else:
            train_data.at[i, 'premise'] = translator.translate(train_data['premise'][i], dest='en').text
            train_data.at[i, 'hypothesis'] = translator.translate(train_data['hypothesis'][i], dest='en').text
    for i in range(len(test_data['premise'])):
        if test_data['lang_abv'][i] == 'en':
            continue
        else:
            test_data.at[i, 'premise'] = translator.translate(test_data['premise'][i], dest='en').text
            test_data.at[i, 'hypothesis'] = translator.translate(test_data['hypothesis'][i], dest='en').text
    return train_data, test_data

def split_labels(x, y):
    return (x[0], x[1]), y

def preparing_for_model(data_train, batch_size, train_size):
    training_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                data_train[['premise', 'hypothesis']].values,
                keras.utils.to_categorical(data_train['label'], num_classes=3)
            )
        )
    )

    train_dataset = training_dataset.take(train_size)
    val_dataset = training_dataset.skip(train_size)

    train_preprocessed = train_dataset.map(split_labels, tf.data.AUTOTUNE).batch(batch_size,  drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)
    val_preprocessed = val_dataset.map(split_labels, tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)

    return train_preprocessed, val_preprocessed