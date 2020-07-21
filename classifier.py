import pandas as pd
from sklearn.neural_network import MLPClassifier
import os, json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import svm
import  matplotlib.pyplot as plt
from sklearn.decomposition import PCA

DOC_PATH = './data/hair_dryer.csv'

if __name__ == "__main__":
    with open("{}_pac_vec.json".format(DOC_PATH)) as F:
        X = pd.DataFrame(json.load(F))
    df = pd.read_csv(DOC_PATH)
    sid = SentimentIntensityAnalyzer()
    # X[X.shape[1]] = df['review_body'].apply(lambda x : sid.polarity_scores(x)['pos'])
    Y = df['star_rating'] = df['star_rating']
    for i in range(X.shape[1]):
        Min = X[i].min()
        Max = X[i].max()
        X[i] = X[i].apply(lambda x: (x-Min)/(Max-Min))
    xx = X[:7000]
    yy = Y[:7000]
    clf = MLPClassifier()
    clf = svm.SVC()
    clf.fit(xx, yy)
    pre_Y = list(clf.predict(X[7000:]))
    a,b = 0, 0
    for y1, y2 in zip(pre_Y, list(Y[7000:])):
        b += 1
        a += 1 if y1 == y2 else 0
    print(a/b)
    pca = PCA(n_components=2)
    X = pd.DataFrame(pca.fit_transform(X))
    color = []
    for cls_ in Y[7000:]:
        if cls_ == 1:
            color.append('black')
        elif cls_ == 2:
            color.append('cyan')
        elif cls_ == 3:
            color.append('green')
        elif cls_ == 4:
            color.append('orange')
        else :
            color.append('red')
    fig, ax = plt.subplots()
    ax.scatter(X[7000:][0], X[7000:][1],c=color)
    plt.show() 

    