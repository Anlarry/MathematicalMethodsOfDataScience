from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
import numpy as np
from nltk.text import TextCollection
from pandas import DataFrame

from tfidf import tf_idf_sort, read_comments
# from keyword import key_word

import os, sys, json

from review2G import Graph, key_word


# doc = read_comments('data/hair_dryer.csv')
# print(tf_idf_sort(doc, 11469))

def has_vec_set(doc:str):
    def docator(func):
        def f(*args, **kargs):
            vec_set_file = doc + '_vec_set.json'
            if os.path.exists(vec_set_file):
                with open(vec_set_file) as F:
                    return json.load(F)
                # return json.load(F)
            else :
                res = func(*args, **kargs)
                with open(vec_set_file, 'w') as F:
                    json.dump(res, F)
                return res
        return f
    return docator


class DocVec:
    def __init__(self, doc: str, vec_size: int, alpha=0.06):
        @has_vec_set(doc)
        def get_vec_set(doc_vec):
            res = {}
            cur_line = 0
            # tc = TextCollection(self.doc)
            while 1:
                try :
                    cur_words = tf_idf_sort(doc_vec.doc, doc_vec.tc, cur_line)
                    for w, v in cur_words :
                        if w in res:
                            res[w] = max(res[w], v)
                        else:
                            res[w] = v
                except IndexError:
                    break
                cur_line += 1
                print("{} \r".format(cur_line), end='')
            return res
        self.doc = read_comments(doc)
        self.tc = TextCollection(self.doc)
        self.vec_set = get_vec_set(self)
        self.vec_set = [(w, self.vec_set[w]) for w in self.vec_set]
        self.vec_set = DataFrame(self.vec_set)
        Max = self.vec_set[1].max()
        Min = self.vec_set[1].min()
        self.vec_set[1] = self.vec_set[1].apply(lambda x: (x-Min)/(Max-Min))
        self.vec_set[1] = self.vec_set[1].apply(lambda x: x * (1-alpha))
        self.vec_set = zip(self.vec_set[0], self.vec_set[1])
        self.vec_set = {w:v for w, v in self.vec_set}
        G = Graph(doc, True)
        tex_rank_key_word = DataFrame(key_word(G, 10, 5000))
        Min = tex_rank_key_word[1].min()
        Max = tex_rank_key_word[1].max()
        tex_rank_key_word[1] = tex_rank_key_word[1].apply(lambda x : alpha * (x-Min)/(Max-Min))
        tex_rank_key_word = list(zip(tex_rank_key_word[0], tex_rank_key_word[1]))
        self.vec_set = [(w,self.vec_set[w]) for w, v in tex_rank_key_word if self.vec_set[w] >= alpha]
        # for w, v in tex_rank_key_word:
        #     if w in self.vec_set:
        #         self.vec_set[w] += v
        #     else:
        #         self.vec_set[w] = v        
        # self.vec_set = sorted([(w, self.vec_set[w]) for w in self.vec_set], key=lambda x: x[1], reverse=True)
        self.vec_set = sorted(self.vec_set, key=lambda x:x[1], reverse=True)
        print(len(self.vec_set))
        self.vec_size = vec_size
    # def average(self):

    def vec(self, index: int):
        cur_words = tf_idf_sort(self.doc, self.tc,index)
        res = [0 for i in range(self.vec_size)]
        word_idx = {self.vec_set[i][0]:i for i in range(len(self.vec_set))}
        for w, v in cur_words:
            if w in word_idx:
                res[word_idx[w]] = v
        return np.array(res)


# doc_vec = DocVec('data/hair_dryer.csv', 5000)
# for i in range(2):
#     print(sum(doc_vec.vec(i)))