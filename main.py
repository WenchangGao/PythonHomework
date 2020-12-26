import jieba
import torch
import numpy as np
import torch.nn as nn
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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Preprocessor:
    """
    完成分词, 排序, 整合数据等工作, 处理后数据按时间排序
    self.time_list: 该条数据的时间
    self.rewards: 是否为谣言
    self.titles: 数据中title项
    self.summaries: 数据中mainSummary项
    """
    def __init__(self, path='./dataset/covid19_rumors.csv'):
        raw_data = pd.read_csv(path).sort_values(['crawlTime'])

        crawl_times = list(raw_data['crawlTime'])
        self.time_list = list(map(lambda x: int(time.mktime(time.strptime(x, '%Y-%m-%d'))), crawl_times))
        # print(self.time_list)

        mapping = {'fake': -1, 'true': 1, 'doubt': 0}
        self.rewards = list(map(lambda x: mapping.get(x), list(raw_data['rumorType'])))
        # print(self.rewards)

        titles = list(raw_data['title'])
        summaries = list(raw_data['mainSummary'])
        self.titles = list(map(lambda x: list(jieba.cut(x)), titles))
        self.summaries = list(map(lambda x: list(jieba.cut(x)), summaries))
        # print(self.titles)
        # print(self.summaries)

    def visualize_data(self):
        plt.xlabel('type')
        plt.ylabel('number')
        x = ('fake', 'doubt', 'true')
        y = [self.rewards.count(-1), self.rewards.count(0), self.rewards.count(1)]
        plt.bar(x, y)
        plt.title('rumor statics')
        plt.savefig('./datafigure/rumor_statics.png')
        plt.show()


class LSTM(nn.Module):
    """
    implemented using LSTM module based on pytorch
    """
    def __init__(self, input_size=20, output_size=20):
        super(LSTM, self).__init__()


if __name__ == '__main__':
    preprocessor = Preprocessor()
    # preprocessor.visualize_data()
