import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import pylab
from Cython import inline



class DataHandle:
    diz_historical_stocks = {}
    diz_historical_returns = {}
    corr_network = nx.Graph()
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
            if len(DataTupleLine2) < 273:
                continue
            for i in range(1, 273):
                ReturnData = math.log(float(DataTupleLine2[i][1])) - math.log(float(DataTupleLine2[i-1][1]))
                #时间收益元祖列，元素为（时间，收益）
                return_line.append((DataTupleLine[i][0], ReturnData))
            #历史收益字典，元素为 标志：时间收益元祖列
            self.diz_historical_returns[sym]=return_line

    #展示股票收益
    def return_show(self,sym):
        plt.figure(num=1, figsize=(20, 15),dpi=80)
        plt.xlabel("Date")
        plt.ylabel("Return of"+sym)
        x=[]
        y=[]
        for d in self.diz_historical_returns[sym]:
            x.append(d[0])
            y.append(d[1])
        plt.plot(x,y,linestyle='-',alpha=0.5,color='b')
        plt.xticks(x[::10],rotation=90)
        plt.show()

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

    #以皮尔逊系数得出两个股票的相关性
    def stocks_corr_coeff(self,sym1, sym2):
        if sym1 == sym2:
            return 1
        l1 = []
        l2 = []
        #h1,h2为时间收益元祖列 【（时间，收益），（）】
        h1=self.diz_historical_returns[sym1]
        h2=self.diz_historical_returns[sym2]
        for d1 in h1:
            l1.append(d1[1])
        for d2 in h2:
            l2.append(d2[1])
        return self.pearson(l1, l2)

    #计算距离
    def getNet(self):
        num_companies = len(self.diz_historical_returns.keys())
        for i1 in range(num_companies - 1):
            for i2 in range(i1 + 1, num_companies):
                #获得两只股票标志
                stock1 = list(self.diz_historical_returns.keys())[i1]
                stock2 = list(self.diz_historical_returns.keys())[i2]
                #计算距离
                metric_distance = math.sqrt(2 * (1.0 - self.stocks_corr_coeff(stock1, stock2)))
                #建立边
                self.corr_network.add_edge(stock1, stock2, weight=metric_distance)

    def showNetMST(self):
        tree_seed = "UTX"
        N_new = []
        E_new = []
        N_new.append(tree_seed)

        #prinm算法，最小生成树
        while len(N_new) < self.corr_network.number_of_nodes():
            min_weight = 10000000.0
            for n in N_new:
                for n_adj in self.corr_network.neighbors(n):
                    if not n_adj in N_new:
                        if self.corr_network[n][n_adj]['weight'] < min_weight:
                            min_weight = self.corr_network[n][n_adj]['weight']
                            min_weight_edge = (n, n_adj)
                            n_adj_ext = n_adj
            E_new.append(min_weight_edge)
            N_new.append(n_adj_ext)

        # generate the tree from the edge list
        tree_graph = nx.Graph()
        tree_graph.add_edges_from(E_new)


        # setting the color attributes for the network nodes
        # for n in tree_graph.nodes():
        #     tree_graph.node[n]['color'] = diz_colors[diz_sectors[n]]
        pos = nx.drawing(tree_graph, prog='neato', \
                                 args='-Gmodel=subset -Gratio=fill')

        plt.figure(figsize=(20, 20))
        nx.draw_networkx_edges(tree_graph, pos, width=2, \
                               edge_color='black', alpha=0.5, style="solid")
        nx.draw_networkx_labels(tree_graph, pos)
        for n in tree_graph.nodes():
            nx.draw_networkx_nodes(tree_graph, pos, [n], node_size=600, \
                                   alpha=0.5, node_color=tree_graph.node[n]['color'], \
                                   with_labels=True)
        plt.show()


dh=DataHandle()
dh.getNet()
dh.showNetMST()

f = open("./data/list_stocks_50B_6_may_2016.txt", 'r')
list_stocks_all = []
while True:
    next_line = f.readline()
    if not next_line:
        break
    #以制表符分割数据行变成列表并去掉列表最后的换行符，再强制转换列表成为元祖，并将元祖append进list_stocks_all列表
    list_stocks_all.append(tuple(next_line.split('\t')[:-1]))
f.close()


hfile = open("./data/companylist.csv", 'r')
#选择市值500亿以上的公司
cap_threshold = 50.0

list_stocks = []
nextline = hfile.readline()
while True:
    nextline = hfile.readline()
    if not nextline:
        break
    #分割数据行，line列表依次存储"标志","名称","收盘价","市值","上市年份","部门","行业","数据引用来源",
    line = nextline.split('","')
    #去掉双引号
    sym = line[0][1:]
    y_market_cap = line[3][1:]
    if y_market_cap == "/a":
        continue
    #选择市值单位是B，数字大于50的公司
    if y_market_cap[-1] == 'B' and float(y_market_cap[:-1]) > cap_threshold:
        print(sym, y_market_cap)
        #元祖内容（标志，名称，部门，行业）
        list_stocks.append((sym, line[1], line[5], line[6]))
hfile.close()

#部门字典
diz_sectors = {}
#取list_stocks中的元素元祖
for s in list_stocks:
    #加入字典 标志：部门
    diz_sectors[s[0]]=s[2]

list_ranking = []
for s in set(diz_sectors.values()):
    count = 0
    #对s部门出现的次数计数
    for v in diz_sectors.values():
        if v == s:
            count += 1
    list_ranking.append((count, s))
#以出现次数对部门由高到低排序
list_ranking.sort(reverse=True)

#list_colors=['red','green','blue','black''cyan','magenta','yellow']
list_colors=['0.0', '0.2', '0.4', '0.6','0.7', '0.8', '0.9']

#'white' is an extra color for 'n/a' and 'other' sectors

diz_colors={}

#将颜色和前七名行业练习起来
i = 7
for s in list_ranking:
    i-=1
    #未知行业和八名及以后的行业取白色
    if s[1] == 'n/a':
        continue
    if i < 0:
        break
    diz_colors[s[1]]=s[0]


plt.rc('font', size=12)
plt.figure(num=1, figsize=(15, 8),dpi=80)
x_index = list(np.arange(7))   #柱的索引
x_data = list(diz_colors.keys())[:7]
y1_data = list(diz_colors.values())[:7]
rects1 = plt.bar(x_index, y1_data, width=0.35, alpha=0.4, color='b')            #参数：左偏移、高度、柱宽、透明度、颜色、图例
plt.xticks(x_index, x_data)   #x轴刻度线
plt.xlabel("sector name")
plt.ylabel("counts")
plt.tight_layout()  #自动控制图像外部边缘，此方法不能够很好的控制图像间的间隔
for i in range(0, 7):
    data = y1_data.pop()
    plt.text(x_index.pop()-0.05, data, data)
plt.show()
