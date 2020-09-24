import math
import matplotlib.pyplot as plt
import numpy as np

class DataHandle:
    diz_historical_stocks = {}
    diz_historical_returns = {}
    def __init__(self):
        f = open("./data/diz_historical_6_may_2016.dat", 'r')

        while True:
            next_line = f.readline()
            if not next_line:
                break
            line = next_line.split("_-_")
            #公司标注
            sym = line[0]
            #时间序列，元素为string “时间 价格”
            dataline = line[1:]
            dataline.sort(reverse=False)
            #元祖时间序列，元素为tuple（时间，价格）
            DataTupleLine = []
            for l in dataline:
                DataTupleLine.append(tuple(l.split(" ")))
            #股票价格历史字典，元素为 标志：元祖时间序列
            self.diz_historical_stocks[sym] = DataTupleLine
        f.close()


        for sym in self.diz_historical_stocks.keys():
            DataTupleLine2 = self.diz_historical_stocks[sym]
            return_line = []
            print(sym)
            if len(DataTupleLine2) < 273:
                continue
            for i in range(1, 273):
                ReturnData = math.log(float(DataTupleLine2[i][1])) - math.log(float(DataTupleLine2[i-1][1]))
                return_line.append((DataTupleLine[i][0], ReturnData))
            self.diz_historical_returns[sym]=return_line

    #展示股票收益
    def return_show(self,sym):
        plt.xlabel("Date")
        plt.ylabel("Return of"+sym)
        x=[]
        y=[]
        for d in self.diz_historical_returns[sym]:
            x.append(d[0])
            y.append(d[1])
        plt.plot(x,y,linestyle='--',alpha=0.5,color='r')
    #平均值
    def mean(self,X):
        m=0.0
        for i in X:
            m=m+i
        return m/len(X)

    #协方差
    def covariance(self,X,Y):
        c=0.0
        m_X=self.mean(X)
        m_Y=self.mean(Y)
        for i in range(len(X)):
            c=c+(X[i]-m_X)*(Y[i]-m_Y)
        return c/len(X)

    #皮尔逊相关系数
    def pearson(self,X,Y):
        return self.covariance(X,Y)/(self.covariance(X,X)**0.5 * self.covariance(Y,Y)**0.5)