import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx


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

list_colors=["r",'coral','orange','y','greenyellow','palegreen','b']
#list_colors=['0.0', '0.2', '0.4', '0.6','0.7', '0.8', '0.9']



diz_colors={}

#association color and more represented sectors
i=0
for s in list_ranking:
    if s[1]=='n/a':
        diz_colors[s[1]]='white'
        continue
    if i>=7:
        diz_colors[s[1]]='white'
        continue
    diz_colors[s[1]]=list_colors.__getitem__(i)
    i+=1




#展示部门拥有公司数
plt.rc('font', size=12)
plt.figure(num=1, figsize=(15, 8),dpi=80)
x_index = list(np.arange(7))   #柱的索引
x_data = []
for i1 in range(0,7):
    x_data.append(list_ranking[i1][1])
y1_data = []
for i2 in range(0,7):
    y1_data.append(list_ranking[i2][0])
rects1 = plt.bar(x_index, y1_data, width=0.35, alpha=0.4, color=list_colors)            #参数：左偏移、高度、柱宽、透明度、颜色、图例
plt.xticks(x_index, x_data)   #x轴刻度线
plt.xlabel("sector name")
plt.ylabel("counts")
plt.tight_layout()  #自动控制图像外部边缘，此方法不能够很好的控制图像间的间隔
for i in range(0, 7):
    data = y1_data.pop()
    plt.text(x_index.pop()-0.05, data, data)
plt.show()

diz_historical_stocks = {}
diz_historical_returns = {}
corr_network = nx.Graph()

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
    diz_historical_stocks[sym] = DataTupleLine

f.close()

for sym in diz_historical_stocks.keys():
    DataTupleLine2 = diz_historical_stocks[sym]
    return_line = []
    if len(DataTupleLine2) < 273:
        continue
    for i in range(1, 273):
        ReturnData = math.log(float(DataTupleLine2[i][1])) - math.log(float(DataTupleLine2[i-1][1]))
        #时间收益元祖列，元素为（时间，收益）
        return_line.append((DataTupleLine[i][0], ReturnData))
    #历史收益字典，元素为 标志：时间收益元祖列
    diz_historical_returns[sym]=return_line

#展示股票收益
def return_show(sym):
    plt.figure(num=1, figsize=(20, 15),dpi=80)
    plt.xlabel("Date")
    plt.ylabel("Return of"+sym)
    x=[]
    y=[]
    for d in diz_historical_returns[sym]:
        x.append(d[0])
        y.append(d[1])
    plt.plot(x,y,linestyle='-',alpha=0.5,color='b')
    plt.xticks(x[::10],rotation=90)
    plt.show()

return_show("UTX")

#平均值
def mean(X):
    m=0.0
    for i in X:
        m=m+i
    return m/len(X)

#协方差
def covariance(X,Y):
    c=0.0
    m_X=mean(X)
    m_Y=mean(Y)
    for i in range(len(X)):
        c=c+(X[i]-m_X)*(Y[i]-m_Y)
    return c/len(X)

#皮尔逊相关系数
def pearson(X,Y):
    return covariance(X,Y)/(covariance(X,X)**0.5 * covariance(Y,Y)**0.5)


#以皮尔逊系数得出两个股票的相关性
def stocks_corr_coeff(sym1, sym2):
    if sym1 == sym2:
        return 1
    l1 = []
    l2 = []
    #h1,h2为时间收益元祖列 【（时间，收益），（）】
    h1=diz_historical_returns[sym1]
    h2=diz_historical_returns[sym2]
    for d1 in h1:
        l1.append(d1[1])
    for d2 in h2:
        l2.append(d2[1])
    return pearson(l1, l2)

#计算距离


num_companies = len(diz_historical_returns.keys())
for i1 in range(num_companies - 1):
    for i2 in range(i1 + 1, num_companies):
        #获得两只股票标志
        stock1 = list(diz_historical_returns.keys())[i1]
        stock2 = list(diz_historical_returns.keys())[i2]
        #计算距离
        metric_distance = math.sqrt(2 * (1.0 - stocks_corr_coeff(stock1, stock2)))
        #建立边
        corr_network.add_edge(stock1, stock2, weight=metric_distance)





nx.draw(corr_network, node_size=100, with_labels=True, prog='neato',  args='-Gmodel=subset -Gratio=fill')
plt.show()

tree_seed = "UTX"
N_new = []
E_new = []
N_new.append(tree_seed)

#prinm算法，最小生成树
while len(N_new) < corr_network.number_of_nodes():
    min_weight = 10000000.0
    for n in N_new:
        for n_adj in corr_network.neighbors(n):
            if not n_adj in N_new:
                if corr_network[n][n_adj]['weight'] < min_weight:
                    min_weight = corr_network[n][n_adj]['weight']
                    min_weight_edge = (n, n_adj, min_weight)
                    n_adj_ext = n_adj
    E_new.append(min_weight_edge)
    N_new.append(n_adj_ext)

# generate the tree from the edge list
tree_graph = nx.Graph()
tree_graph.add_weighted_edges_from(E_new)


# setting the color attributes for the network nodes
list_node_color = []
plt.figure(figsize=(20, 20))
for n in tree_graph.nodes():
    list_node_color.append(diz_colors[diz_sectors.setdefault(n, "n/a")])
#pos=nx.fruchterman_reingold_layout(tree_graph)
#pos=nx.kamada_kawai_layout(tree_graph)
pos=nx.kamada_kawai_layout(tree_graph)
nx.draw(tree_graph, pos, node_size=100, node_color=list_node_color, with_labels=True, prog='neato',  args='-Gmodel=subset -Gratio=fill')

plt.show()


