from typing import List, Set

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.text import TextCollection


def read_comments(path: str):
    doc = pd.read_csv(path)['review_body']
    words = []
    stop_word: Set[str] = set(stopwords.words('english'))
    stop_word = stop_word.union(
        ["7/19/2015", ',', '.', '!', '?', ')', '(', '#', '*', '$', "'s", "n't", '/', '>', '<', '-', '...', '..', ':',
         '--', "'m", ';', '&', "'ve", 'br'])
    for a in doc:
        this_comment = word_tokenize(a)
        this_comment = [token for token in this_comment if token not in stop_word]
        words.append(this_comment)
    return words


def tf_idf_sort(docs: List[List[str]], index: int):
    words = set(docs[index])
    tc = TextCollection(docs)
    res = [(word, tc.tf_idf(word, docs[index])) for word in words]
    res.sort(key=lambda p: p[1], reverse=True)
    return res


# %%
# docs = read_comments('data/hair_dryer.csv')
# a = tf_idf_sort(docs, 6)
