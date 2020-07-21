from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt

def rating_sentiment_corr(doc:str):
    sid = SentimentIntensityAnalyzer()
    df = pd.read_csv(doc)
    df = pd.DataFrame({'review_body':df['review_body'], 'star_rating':df['star_rating']})
    Min = df['star_rating'].min()
    Max = df['star_rating'].max()
    df['star_rating'] = df['star_rating'].apply(lambda x : (x-Min) / (Max-Min))
    df['review_body'] = df['review_body'].apply(lambda x : sid.polarity_scores(x)['pos'])
    return df.corr()
    # fig, ax = plt.subplots()
    # ax.scatter(np.arange(0, len(df['review_body']), 1), df['review_body'], c='blue', s=3)
    # ax.scatter(np.arange(0, len(df['star_rating']), 1), df['star_rating'], c='red', s=3)
    # plt.show()
    # print(df.corr())
    # print(sid.polarity_scores(df[0]))
print(rating_sentiment_corr('data/hair_dryer.csv'))
# rating_sentiment_corr('data/hair_dryer.csv')
