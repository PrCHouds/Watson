import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import keras_nlp
import seaborn as sns
import matplotlib.pyplot as plt
import os

from Config import *
from model import *
from prep import *

if __name__ == '__main__':
    # model
    strategy = tpu_init()
    BATCH_SIZE = 16 * strategy.num_replicas_in_sync
    classifier = model_init(strategy)
    # data
    df_train, df_test = read_data(DATA_DIR)
    df_train, df_test = replace_after_translation(df_train, df_test)
    TRAIN_SIZE = int(df_train.shape[0] * (1 - VALIDATION_SPLIT))
    train_preprocessed, val_preprocessed = preparing_for_model(df_train, BATCH_SIZE, TRAIN_SIZE)

    train_model(classifier, train_preprocessed, val_preprocessed)

    submission = classifier_predict(classifier, df_test, BATCH_SIZE)

    answers = submission['prediction'].tolist()
    print(answers)
    right_sub = pd.read_csv(DATA_DIR + "submission.csv")
    right_answers = right_sub['prediction'].tolist()
    print(right_answers)
    k = 0
    for i in range(len(answers)):
        if answers[i] == right_answers[i]:
            k += 1
    print(f'Right answers: {k} from {len(answers)}')
    print(f'Accuracy: {k / len(answers) * 100}')

    submission.to_csv("my_submission.csv", index=False)


