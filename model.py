import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import keras_nlp
import seaborn as sns
import matplotlib.pyplot as plt
import os

from Config import *
#tpu initializing
def tpu_init():
    try:
      tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
      print(f'Running on a TPU w/{tpu.num_accelerators()["TPU"]} cores')
    except ValueError:
      raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)

    return strategy

def model_init(strategy):
    with strategy.scope():
        classifier = keras_nlp.models.BertClassifier.from_preset("bert_base_multi", num_classes=3)
        classifier.compile(optimizer=keras.optimizers.Adam(1e-5 * strategy.num_replicas_in_sync),
                           loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        classifier.summary()
    return classifier

def train_model(train_preprocessed, val_preprocessed, classifier):
    history = classifier.fit(train_preprocessed,
                             epochs=EPOCHS,
                             validation_data=val_preprocessed)
def classifier_predict(classifier, data_test, batch_size):
    predictions = classifier.predict((data_test['premise'], data_test['hypothesis']), batch_size=batch_size)
    submission = data_test.id.copy().to_frame()
    submission["prediction"] = np.argmax(predictions[:-5], axis=1)
    return submission


if __name__ == '__main__':
    print("TensorFlow version:", tf.__version__)
    print("KerasNLP version:", keras_nlp.__version__)