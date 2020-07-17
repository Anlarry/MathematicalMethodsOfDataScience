import numpy as np
from review2G import Graph

import math
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
    x = [word[i] for pg, i in x]
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

# G = Graph('data/hair_dryer.csv', True)
# print(key_phrase(G, 100, 100))




    
    

            
    

    

    