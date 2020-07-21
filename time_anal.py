import pandas as pd
from datetime import datetime, date
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA 

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
    def acf_pacf(self, diff_n=1):
        # dta = np.diff(self.volume(), n=diff_n)
        data = pd.Series(self.volume())
        D_data = data.diff().dropna()
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        plot_acf(D_data, ax=ax1)    
        ax2 = fig.add_subplot(212)
        plot_pacf(D_data, ax=ax2)   
        plt.show()
    def calcu(self):
        # data = pd.DataFrame(self.volume())
        data = pd.Series(self.volume()[:120])
        model = ARIMA(data, order=(12, 2, 0)).fit()
        y = model.forecast(20)[0]
        # plt.figure(figsize=(6,1))
        plt.ylim(0,1000)
        plt.plot(self.volume()[120:], c='red')
        plt.plot(y, c='blue')
        plt.show()

sv = SalesVolume('./data/hair_dryer.csv')
# sv.plot(diff_n=2)
# print(len(sv.volume()))
# sv.acf_pacf(2)
sv.calcu()
# sv.acf_pacf(diff_n=2)
# plt.plot(np.diff(sv.volume()))
# plt.show()
# print(sv)
# sv.plot()