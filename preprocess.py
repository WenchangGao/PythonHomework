import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import time


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

        self.sequenced_titles = []
        self.sequenced_summaries = []

    def visualize_data(self):
        plt.xlabel('type')
        plt.ylabel('number')
        x = ('fake', 'doubt', 'true')
        y = [self.rewards.count(-1), self.rewards.count(0), self.rewards.count(1)]
        plt.bar(x, y)
        plt.title('rumor statistics')
        plt.savefig('./figures/rumor_statistics.png')
        plt.show()

    def tokenize_data(self):
        tokenizer = Tokenizer(num_words=5000)
        processed_data = []
        processed_data.extend(self.titles)
        processed_data.extend(self.summaries)
        # processed_data.extend(self.time_list)
        # processed_data.extend(self.rewards)
        tokenizer.fit_on_texts(processed_data)

        title_ids = tokenizer.texts_to_sequences(self.titles)
        summaries_ids = tokenizer.texts_to_sequences(self.summaries)
        self.sequenced_titles = sequence.pad_sequences(title_ids, 32, dtype=np.float32)
        self.sequenced_summaries = sequence.pad_sequences(summaries_ids, 32, dtype=np.float32)
        # print(title_ids)
        # print(summaries_ids)
        # print('sequenced summaries: ', self.sequenced_summaries)
        # print('shape: ', self.sequenced_summaries.shape)
        # print('dtype: ', self.sequenced_summaries.dtype)
