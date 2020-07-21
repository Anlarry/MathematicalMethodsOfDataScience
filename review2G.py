import pandas as pd
import sys, os
from  nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import numpy as np
import re

def tsv2csv(File):
    if os.path.exists('data/{}.csv'.format(File)):
        return 
    tsv = open('data/{}.tsv'.format(File), encoding='utf-8')
    csv = tsv.read().replace('\t', ',')
    with open('data/{}.csv'.format(File), 'w', encoding='utf-8') as F:
        print(csv, file=F)

class Graph:
    '''
        index     : if index not None, return review[index] Graph
    '''
    @classmethod
    def lemmatize(cls, word_tag:tuple):
        def get_wordnet_pos(tag):
            if tag.startswith('J'):
                return wordnet.ADJ
            elif tag.startswith('V'):
                return wordnet.VERB
            elif tag.startswith('N'):
                return wordnet.NOUN
            elif tag.startswith('R'):
                return wordnet.ADV
            else:
                return None
        wnl = WordNetLemmatizer()
        wordnet_pos = get_wordnet_pos(word_tag[1]) or wordnet.NOUN
        return wnl.lemmatize(word_tag[0], pos=wordnet_pos)
    def __init__(self, csv:str, clear_num = True, index = None):
        self.word_Graph = {}
        self.word_cnt = {}
        self.sent_token=[]
        df = pd.read_csv(csv)['review_body']
        sents = self.__sent_token(df, index)
        stop_word = stopwords.words('english')
        for w in ["7/19/2015", ',', '.', '!', '?', ')', '(', '#', '*', '$', "'s", "n't", '/', '>', '<', '-', '...', '..', ':','--', "'m", ';', '&', "'ve", 'br']:
            stop_word.append(w)
        for sent in sents:
            tag = [tag for tag in pos_tag(word_tokenize(sent))]
            tag = {t[0].lower():Graph.lemmatize((t[0].lower(), t[1])) for t in tag}
            words = [word for word in [x.lower() for x in word_tokenize(sent)] if word not in stop_word]
            if clear_num:
                words = [w for w in words if not w.isdigit()]
                words = [tag[w] for w in words]
            self.sent_token.append(words)
            for w in words:
                if w not in self.word_Graph:
                    self.word_Graph[w] = {}
            for i in range(1, len(words)):
                self.__add_edge(words[i-1], words[i])
                self.__add_edge(words[i], words[i-1])
            for w in words :
                if w not in self.word_cnt:
                    self.word_cnt[w] = 1
                else: self.word_cnt[w] += 1
    
    def __sent_token(self, sent_df, index):
        res = []
        if index is not None:
            sent_df = sent_df[index]
        for sent in sent_df:
            try:
                res += sent_tokenize(sent)
            except TypeError:
                pass
        return res
    def __add_edge(self, w1, w2):
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

def key_word(G,freq,n):
    G.remove_freq(freq)
    idx = {}
    i = 0
    for w in G:
        idx[w]  = i
        i += 1
    P = [[0 for i in range(len(G))] for i in range(len(G))]
    for word in G:
        row = idx[word]
        tot = 0
        for x in G[word]:
            col = idx[x]
            P[row][col] = G[word][x]
        if len(G[word]) == 0:
            P[row] = [1/len(G) for i in range(len(G))]            
        else :
            tot = sum(P[row])
            P[row] = [x/tot for x in P[row]]
    A = np.mat(P)
    B = [[1/len(G) for j in range(len(G))] for k in range(len(G))] 
    C = np.mat(B)
    A = 0.9*A+0.1*C
    x = np.mat([1 for j in range(len(G))])
    while True:
        O = x*A
        if abs(np.array(O)[0][0]/np.array(x)[0][0] -1) <0.01:
            break
        x = O
    x = list(np.array(x)[0])
    x = sorted([(x[i], i) for i in range(len(x))], reverse = True, key=lambda w: w[0])
    word = {idx[x]:x for x in idx}
    x = [(word[i],pg) for pg, i in x]
    return x[:n]

def key_phrase(G:Graph,freq,n):
    keyword = set(key_word(G,freq,n))
    res = set()
    for sent in G.sent_token:
        i = 0
        cur_phase = []
        while i < len(sent):
            if sent[i] in keyword: 
                cur_phase.append(sent[i])
            # elif cur_phase != []:
            elif len(cur_phase) > 1:
                res.add(' '.join(cur_phase))
                cur_phase = []
            i += 1
        if len(cur_phase) > 1:
            res.add(' '.join(cur_phase))
    return [x for x in res]

tsv2csv('hair_dryer')
tsv2csv('microwave')
tsv2csv('pacifier')

# df = pd.read_csv('data/hair_dryer.csv')

# for x in df['review_body'][:3]:
#     print(x)

# G = Graph("data/hair_dryer.csv", True)
# keys = key_word(G, 10, 5000)
# print(keys)
# G.remove_freq(100)
# print(len(G))