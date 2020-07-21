from typing import List, Set

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.text import TextCollection
from nltk import pos_tag

from review2G import Graph


def read_comments(path: str):
    doc = pd.read_csv(path)['review_body']
    words = []
    stop_word: Set[str] = set(stopwords.words('english'))
    stop_word = stop_word.union(
        ["7/19/2015", ',', '.', '!', '?', ')', '(', '#', '*', '$', "'s", "n't", '/', '>', '<', '-', '...', '..', ':',
         '--', "'m", ';', '&', "'ve", 'br'])
    for a in doc:
        try:
            this_comment = word_tokenize(a)
            tag = [tag for tag in pos_tag(this_comment)]
            tag = {t[0].lower():Graph.lemmatize((t[0].lower(), t[1])) for t in tag}
            this_comment =  [x.lower() for x in this_comment]
            this_comment = [token for token in this_comment if token not in stop_word]
            this_comment = [w for w in this_comment if not w.isdigit()]
            this_comment = [tag[w] for w in this_comment]
            words.append(this_comment)
        except TypeError:
            pass
    return words


def tf_idf_sort(docs: List[List[str]], tc: TextCollection, index: int):
    words = set(docs[index])
    # tc = TextCollection(docs)
    res = [(word, tc.tf_idf(word, docs[index])) for word in words]
    res.sort(key=lambda p: p[1], reverse=True)
    return res



# docs = read_comments('data/hair_dryer.csv')
# print(len((docs)))
# res = {}
# cur_line = 0
# # tc = TextCollection(self.doc)
# while 1:
#     try :
#         cur_words = tf_idf_sort(docs, TextCollection(docs), cur_line)
#         for w, v in cur_words :
#             if w in res:
#                 res[w] = max(res[w], v)
#             else:
#                 res[w] = v
#     except IndexError:
#         break
#     cur_line += 1
#     print("{} \r".format(cur_line), end='')
# print(res)
