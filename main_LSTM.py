import jieba
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt
import sys
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
import time
import os
import preprocess
import LSTM
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# file = open('./figures/output1.txt', 'w')
# sys.stdout = file


if __name__ == '__main__':
    preprocessor = preprocess.Preprocessor()
    ratio = 0.7
    # preprocessor.visualize_data()
    preprocessor.tokenize_data()
    training_data_x = preprocessor.sequenced_summaries[:int(ratio * len(preprocessor.sequenced_summaries))]
    training_data_y = preprocessor.rewards[:int(ratio * len(preprocessor.sequenced_summaries))]
    data_x = preprocessor.sequenced_summaries[int(ratio * len(preprocessor.sequenced_summaries)):]
    data_y = preprocessor.rewards[int(ratio * len(preprocessor.sequenced_summaries)):]
    # print("hey there")
    # for i in range(7):
    hidden = 60

    model1 = LSTM.LSTM(hidden_size=hidden, num_layer=20)
    print('hidden size : %d, num_layer : 20' % hidden)
    print('training using summaries')
    # print(int(ratio * len(preprocessor.sequenced_summaries)))
    # training_data_x = preprocessor.sequenced_summaries[:int(ratio * len(preprocessor.sequenced_summaries))]
    # training_data_y = preprocessor.rewards[:int(ratio * len(preprocessor.sequenced_summaries))]
    LSTM.train_model(training_data_x, training_data_y, model=model1)
    # data_x = preprocessor.sequenced_summaries[int(ratio * len(preprocessor.sequenced_summaries)):]
    # data_y = preprocessor.rewards[int(ratio * len(preprocessor.sequenced_summaries)):]
    LSTM.test_model(data_x, data_y, model1)

    model1 = LSTM.LSTM(hidden_size=hidden, num_layer=20)
    print('training using titles')
    training_data_x = preprocessor.sequenced_titles[:int(ratio * len(preprocessor.sequenced_titles))]
    data_x = preprocessor.sequenced_titles[int(ratio * len(preprocessor.sequenced_summaries)):]
    # print(int(ratio * len(preprocessor.sequenced_summaries)))
    # training_data_x = preprocessor.sequenced_summaries[:int(ratio * len(preprocessor.sequenced_summaries))]
    # training_data_y = preprocessor.rewards[:int(ratio * len(preprocessor.sequenced_summaries))]
    LSTM.train_model(training_data_x, training_data_y, model=model1)
    # data_x = preprocessor.sequenced_summaries[int(ratio * len(preprocessor.sequenced_summaries)):]
    # data_y = preprocessor.rewards[int(ratio * len(preprocessor.sequenced_summaries)):]
    LSTM.test_model(data_x, data_y, model1)

    model2 = LSTM.LSTM(hidden_size=hidden, num_layer=30)
    print('hidden size : %d, num_layer : 30' % hidden)
    print('training using summaries')
    training_data_x = preprocessor.sequenced_summaries[:int(ratio * len(preprocessor.sequenced_summaries))]
    data_x = preprocessor.sequenced_summaries[int(ratio * len(preprocessor.sequenced_summaries)):]
    # print(int(ratio * len(preprocessor.sequenced_summaries)))
    # training_data_x = preprocessor.sequenced_summaries[:int(ratio * len(preprocessor.sequenced_summaries))]
    # training_data_y = preprocessor.rewards[:int(ratio * len(preprocessor.sequenced_summaries))]
    LSTM.train_model(training_data_x, training_data_y, model=model2)
    # data_x = preprocessor.sequenced_summaries[int(ratio * len(preprocessor.sequenced_summaries)):]
    # data_y = preprocessor.rewards[int(ratio * len(preprocessor.sequenced_summaries)):]
    LSTM.test_model(data_x, data_y, model2)

    model2 = LSTM.LSTM(hidden_size=hidden, num_layer=30)
    print('hidden size : %d, num_layer : 30' % hidden)
    print('training using summaries')
    training_data_x = preprocessor.sequenced_titles[:int(ratio * len(preprocessor.sequenced_summaries))]
    data_x = preprocessor.sequenced_titles[int(ratio * len(preprocessor.sequenced_summaries)):]
    # print(int(ratio * len(preprocessor.sequenced_summaries)))
    # training_data_x = preprocessor.sequenced_summaries[:int(ratio * len(preprocessor.sequenced_summaries))]
    # training_data_y = preprocessor.rewards[:int(ratio * len(preprocessor.sequenced_summaries))]
    LSTM.train_model(training_data_x, training_data_y, model=model2)
    # data_x = preprocessor.sequenced_summaries[int(ratio * len(preprocessor.sequenced_summaries)):]
    # data_y = preprocessor.rewards[int(ratio * len(preprocessor.sequenced_summaries)):]
    LSTM.test_model(data_x, data_y, model2)
    # file.close()
