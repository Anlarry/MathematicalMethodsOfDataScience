from doc_vec import  DocVec
from sklearn.decomposition import PCA
import pandas as pd
import os, sys, json
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN

from decorate import has_file

def pca_vec(doc_path:str, vec_size):
    @has_file(doc_path+'_pac_vec.json')
    def __pca_vec(doc_path:str, vec_size):
        doc_vec = DocVec(doc_path, vec_size)
        cur_line = 0
        vecs = []
        while 1 : 
            try:
                vecs.append(doc_vec.vec(cur_line))
            except IndexError:
                break
            cur_line += 1
            print("{} \r".format(cur_line), end='')
        pca = PCA(n_components=10)
        vecs = pca.fit_transform(vecs)
        # print(vecs)
        return vecs
    return __pca_vec(doc_path, vec_size)

if __name__ == "__main__":
    vecs = pca_vec('data/hair_dryer.csv', 5000)
    df = pd.DataFrame(vecs)
    rating_df = pd.read_csv('data/hair_dryer.csv')['star_rating']
    for i in range(10):
        Min = df[i].min()
        Max = df[i].max()
        df[i] = df[i].apply(lambda x: (x-Min)/(Max-Min))
    print(df)

    kmeans = KMeans(n_clusters=10)
    y = kmeans.fit(df).predict(df)
    # fig, ax = plt.subplots()
    # plt.scatter(df[0], df[1], c=y)
    fig = plt.figure(121)
    ax = Axes3D(fig)
    ax.scatter(df[0], df[1], df[2], c=y)
    fig = plt.figure(122)
    ax = Axes3D(fig)
    color = []
    for x in rating_df:
        if x == 5:
            color.append('red')
        elif x == 4:
            color.append('red')
        elif x == 3:
            color.append('black')
        elif x == 2:
            color.append('black')
        else :
            color.append('black')
    ax.scatter(df[0], df[1], df[2], c=color)
    plt.show()

    # doc_vec = DocVec('data/hair_dryer.csv', 5000)
    # cur_line = 0
    # vecs = []
    # while 1 : 
    #     try:
    #         vecs.append(doc_vec.vec(cur_line))
    #     except IndexError:
    #         break
    #     cur_line += 1
    #     print("{} \r".format(cur_line), end='')
    # pca = PCA(n_components=2)
    # vecs = pca.fit_transform(vecs)
    # print(vecs)

            