import pandas as pd
from datetime import datetime, date
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np

class SalesVolume(list):
    def __init__(self, doc:str):
        super().__init__([])
        df = pd.read_csv(doc)['review_date']
        sale_volume = {}
        for t in df:
            t = datetime.strptime(t, '%m/%d/%Y')
            t = date(t.year, t.month, 1)
            if t in sale_volume:
                sale_volume[t] += 1
            else :
                sale_volume[t] = 1
        self += sorted([(x, v) for x, v in sale_volume.items()], key=lambda x: (x[0].year, x[0].month))
        self = [("{}/{}".format(x.year, x.month), v) for x, v in self]
    def volume(self):
        return [v for _, v in self]
    def plot(self, diff_n=1):
        plt.subplot(121)
        x, y = zip(*self)
        plt.plot(x, y)
        plt.xticks(rotation=270)
        plt.subplot(122)
        print(adfuller(np.diff(sv.volume(),n=diff_n)))
        plt.plot(np.diff(y, n=diff_n))
        plt.show()
    

sv = SalesVolume('./data/hair_dryer.csv')
sv.plot(diff_n=2)
# plt.plot(np.diff(sv.volume()))
# plt.show()
# print(sv)
# sv.plot()