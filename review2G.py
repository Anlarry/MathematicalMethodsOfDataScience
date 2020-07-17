import pandas as pd
import sys, os
from  nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import re

def tsv2csv():
    if os.path.exists('data/hair_dryer.csv'):
        print('exists')
        return 
    tsv = open('data/hair_dryer.tsv', encoding='utf-8')
    csv = tsv.read().replace('\t', ',')
    with open('data/hair_dryer.csv', 'w', encoding='utf-8') as F:
        print(csv, file=F)

class Graph:
    def __init__(self, csv:str, clear_num = False):
        self.word_Graph = {}
        self.word_cnt = {}
        df = pd.read_csv('data/hair_dryer.csv')['review_body']
        sents = self.sent_token(df)
        stop_word = stopwords.words('english')
        for w in [',', '.', '!', '?', ')', '(', '#', '*', '$', "'s", "n't", '/', '>', '<', '-', '...', '..', ':','--', "'m", ';', '&']:
            stop_word.append(w)
        for sent in sents:
            words = [word for word in [x.lower() for x in word_tokenize(sent)] if word not in stop_word]
            if clear_num:
                words = [w for w in words if not w.isdigit()]
            for w in words:
                if w not in self.word_Graph:
                    self.word_Graph[w] = {}
            for i in range(1, len(words)):
                self.add_edge(words[i-1], words[i])
                self.add_edge(words[i], words[i-1])
            for w in words :
                if w not in self.word_cnt:
                    self.word_cnt[w] = 1
                else: self.word_cnt[w] += 1

    def sent_token(self, sent_df):
        res = []
        for sent in sent_df:
            res += sent_tokenize(sent)
        return res
    def add_edge(self, w1, w2):
        if w2 in self.word_Graph[w1] :
            self.word_Graph[w1][w2] += 1
        else:
            self.word_Graph[w1][w2] = 1
    def freq(self, N):
        fig, ax = plt.subplots()
        xy = sorted(list(self.word_cnt.items()), key=lambda x: x[1], reverse=True)
        x, y = zip(*xy)
        plt.plot(x[:N], y[:N])
        plt.xticks(rotation=270)
        print(x[:N])
        plt.show()
    def __getitem__(self, idx):
        return self.word_Graph[idx]
    def __iter__(self):
        return self.word_Graph.__iter__()
    def __len__(self):
        return len(self.word_Graph)
    def remove_freq(self, freq):
        word = {w for w in self.word_cnt if self.word_cnt[w] < freq}
        for w in word:
            del self.word_Graph[w]
        for w in self.word_Graph:
            self.word_Graph[w] = {x: w for x, w in self.word_Graph[w].items() if x not in word}

tsv2csv()

# df = pd.read_csv('data/hair_dryer.csv')

# for x in df['review_body'][:3]:
#     print(x)

# G = Graph("data/hair_dryer.csv", True)
# G.remove_freq(100)
# print(len(G))