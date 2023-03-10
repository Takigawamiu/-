import pandas as pd
import numpy as np
import time

"""
总体来说，用一个pandas的DataFrame记录每个节点的Neighbor（邻居节点），D（邻居节点个数），L（荷载），C（容量），MaxC（最大容量），P（该节点的失效概率），F（该节点失效与否）。
其实要想进一步改进，应该是可以创建一个节点类，包含上述属性，以及删除节点、扩增节点等功能，不过我个人做不到这些，所以就靠你们啦，这样的话应该是要比以来DataFrame要高效一点。
"""


def adj_matrix_to_list(adj_matrix, adj_dataframe):  # 邻接矩阵转为邻接表
    for i in range(len(adj_matrix)):
        tmpList = []
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i][j] == 1:
                tmpList.append(j)
        tmpStr = ' '.join([str(k) for k in tmpList])
        if len(tmpStr) == 0:
            tmpStr = 'None'
            print('This Graph is not a fully-connected graph.')
        adj_dataframe.iloc[i, 0] = str(i)
        adj_dataframe.iloc[i, 1] = tmpStr


def possibility(adjDf):  # 计算过载节点失效概率
    L = adjDf.L
    C = adjDf.C
    MaxC = adjDf.MaxC
    F = adjDf.F
    pList = []
    for i in range(len(L)):
        tmpL = L[i]
        tmpC = C[i]
        tmpMax = MaxC[i]
        tmpF = F[i]
        if tmpF != 1:
            pList.append(np.piecewise(tmpL, [tmpL == 0, 0 < tmpL <= tmpC, tmpC < tmpL <= tmpMax, tmpL >= tmpMax],
                                      [1, 0, 1 / (1 + np.exp(tmpL - (tmpC + tmpMax) / 2)), 1]))
        else:
            pList.append(1)
    adjDf.P = pList


def failure_choice(adjDf):  # 根据失效概率选择失效节点
    P = adjDf.P
    choice = []
    for i in range(len(P)):
        tmpP = P[i]
        choice.append(np.random.choice([1, 0], p=[tmpP, 1 - tmpP]))
    adjDf.F = choice


# adjMat = [[0, 1, 1, 0],
#           [1, 0, 1, 0],
#           [1, 1, 0, 1],
#           [0, 0, 1, 0]]

old_time = time.time()  # 计时用
# alpha = float(input('Pleas key in alpha:'))
# gamma = float(input('Please key in gamma:'))
# beta = float(input('Please key in beta:'))

# 确定各个参数
alpha = float(1.3)
beta = float(1.3)
gamma = float(1.5)

adjMat = pd.read_excel('corr2.xlsx', header=None).to_numpy()  # 读取邻接矩阵
adj_dataframe = pd.DataFrame(index=[i for i in range(len(adjMat))],
                             columns=['Code', 'Neighbor', 'D', 'L', 'C', 'MaxC', 'P', 'F'])  # 构建DataFrame储存系欸但信息
record = []  # 记录每次仿真情况
# N = 0


for beta in np.linspace(1.71, 2.0, num=30):  # 改变beta值进行仿真
    N = 0  # 初始化节点失效数值
    print(beta)
    for node_count in range(len(adj_dataframe)):  # 依次造成各个节点失效
        # t=0 时刻，初始化节点情况
        adj_dataframe = pd.DataFrame(index=[i for i in range(len(adjMat))],
                                     columns=['Code', 'Neighbor', 'D', 'L', 'C', 'MaxC', 'P', 'F'])
        # 录入节点的各个属性
        adj_matrix_to_list(adjMat, adj_dataframe)
        adj_dataframe.D = adj_dataframe.Neighbor.apply(lambda x: len(x.split()))
        adj_dataframe.L = adj_dataframe.D.apply(lambda x: x ** alpha)
        adj_dataframe.C = adj_dataframe.L.apply(lambda x: beta * x)
        adj_dataframe.MaxC = adj_dataframe.C.apply(lambda x: gamma * x)
        # print(np.sum(adj_dataframe.L), np.sum(adj_dataframe.C))
        adj_dataframe.loc[int(node_count), 'L'] = adj_dataframe.loc[int(node_count), 'MaxC']  # 选中节点失效
        # for q in adj_dataframe.loc[14, 'Neighbor'].split():
        #     adj_dataframe.loc[int(q), 'L'] = adj_dataframe.loc[int(q), 'L'] * 1.5
        possibility(adj_dataframe)
        failure_choice(adj_dataframe)
        originalLoad = np.sum(adj_dataframe.L)
        adj_overload = adj_dataframe[(adj_dataframe.P != 0) & (adj_dataframe.F == 0)]
        adj_failure = adj_dataframe[adj_dataframe.F == 1]
        adj_failure_Code = adj_failure.Code.to_list()
        distribution = {}  # 记录荷载转移情况

        # 过载但未失效节点分担
        for i in adj_overload['Code']:
            tmpCode = i
            tmpC_sum = 0
            Neighbor_list = adj_overload.loc[int(i), 'Neighbor'].split()
            overLoad = adj_overload.loc[int(i), 'L'] - adj_overload.loc[int(i), 'C']
            Neighbor_avail_list = [k for k in Neighbor_list if k not in adj_failure_Code]
            for j in Neighbor_avail_list:
                tmpC_sum = adj_dataframe.loc[int(j), 'C'] + tmpC_sum
            for j in Neighbor_avail_list:
                tmp_distri = overLoad * (adj_dataframe.loc[int(j), 'C'] / tmpC_sum)
                if j not in distribution.keys():
                    distribution[j] = tmp_distri
                else:
                    distribution[j] += tmp_distri
        for i in adj_overload['Code']:
            adj_dataframe.loc[int(i), 'L'] = adj_dataframe.loc[int(i), 'C']
        for i in distribution.keys():
            adj_dataframe.loc[int(i), 'L'] += distribution[i]

        distribution = {}
        # 过载且失效节点分担
        for i in adj_failure['Code']:
            tmpC_sum = 0
            Neighbor_list = adj_failure.loc[int(i), 'Neighbor'].split()
            allLoad = adj_failure.loc[int(i), 'L']
            Neighbor_avail_list = [k for k in Neighbor_list if k not in adj_failure_Code]
            for j in Neighbor_avail_list:
                tmpC_sum = adj_dataframe.loc[int(j), 'C'] + tmpC_sum
            for j in Neighbor_avail_list:
                tmp_distri = allLoad * (adj_dataframe.loc[int(j), 'C'] / tmpC_sum)
                if j not in distribution.keys():
                    distribution[j] = tmp_distri
                else:
                    distribution[j] += tmp_distri

        for i in adj_failure['Code']:
            adj_dataframe.loc[int(i), 'L'] = 0

        for i in distribution.keys():
            adj_dataframe.loc[int(i), 'L'] += distribution[i]

        possibility(adj_dataframe)
        failure_choice(adj_dataframe)
        adj_still_work = adj_dataframe[adj_dataframe.F == 0]

        # t=1 之后的仿真情况
        for t in range(20):
            # print(str(t))

            adj_overload = adj_dataframe[(adj_dataframe.P != 0) & (adj_dataframe.F == 0)]
            adj_failure = adj_dataframe[adj_dataframe.F == 1]
            adj_failure_Code = adj_failure.Code.to_list()
            distribution = {}
            # 过载但未失效节点分担
            for i in adj_overload['Code']:
                tmpCode = i
                tmpC_sum = 0
                Neighbor_list = adj_overload.loc[int(i), 'Neighbor'].split()
                overLoad = adj_overload.loc[int(i), 'L'] - adj_overload.loc[int(i), 'C']
                Neighbor_avail_list = [k for k in Neighbor_list if k not in adj_failure_Code]
                for j in Neighbor_avail_list:
                    tmpC_sum = adj_dataframe.loc[int(j), 'C'] + tmpC_sum
                for j in Neighbor_avail_list:
                    tmp_distri = overLoad * (adj_dataframe.loc[int(j), 'C'] / tmpC_sum)
                    if j not in distribution.keys():
                        distribution[j] = tmp_distri
                    else:
                        distribution[j] += tmp_distri
            for i in adj_overload['Code']:
                adj_dataframe.loc[int(i), 'L'] = adj_dataframe.loc[int(i), 'C']

            for i in distribution.keys():
                adj_dataframe.loc[int(i), 'L'] += distribution[i]
            distribution = {}
            # 过载且失效节点分担
            for i in adj_failure['Code']:
                tmpC_sum = 0
                Neighbor_list = adj_failure.loc[int(i), 'Neighbor'].split()
                allLoad = adj_failure.loc[int(i), 'L']
                Neighbor_avail_list = [k for k in Neighbor_list if k not in adj_failure_Code]
                for j in Neighbor_avail_list:
                    tmpC_sum = adj_dataframe.loc[int(j), 'C'] + tmpC_sum
                for j in Neighbor_avail_list:
                    tmp_distri = allLoad * (adj_dataframe.loc[int(j), 'C'] / tmpC_sum)
                    if j not in distribution.keys():
                        distribution[j] = tmp_distri
                    else:
                        distribution[j] += tmp_distri

            for i in adj_failure['Code']:
                adj_dataframe.loc[int(i), 'L'] = 0

            for i in distribution.keys():
                adj_dataframe.loc[int(i), 'L'] += distribution[i]

            possibility(adj_dataframe)
            failure_choice(adj_dataframe)
            adj_still_work = adj_dataframe[adj_dataframe.F == 0]

            # 当所有节点失效概率之和小于0.1或正在工作节点失效概率=1，退出仿真
            if (np.mean(adj_still_work.P) == 1) or (np.sum(adj_still_work.P) <= 0.1):
                N += np.sum(adj_dataframe.F)
                break
    Sn = (N - 1) / 9900
    record.append([alpha, beta, Sn])
pd.DataFrame(record, columns=['alpha', 'beta', 'Rate']).to_excel('record1.3.xlsx', index=False)

# for beta in np.linspace(1.3, 1.6, num=31):
#     mean_rate = 0
#     mean_List = []
#     for mean_count in range(5):
#         adjMat = pd.read_excel('corr2.xlsx', header=None).to_numpy()
#         adj_dataframe = pd.DataFrame(index=[i for i in range(len(adjMat))],
#                                      columns=['Code', 'Neighbor', 'D', 'L', 'C', 'MaxC', 'P', 'F'])
#         adj_matrix_to_list(adjMat, adj_dataframe)
#         # 初始化
#         adj_dataframe.D = adj_dataframe.Neighbor.apply(lambda x: len(x.split()))
#         adj_dataframe.L = adj_dataframe.D.apply(lambda x: x ** alpha)
#         adj_dataframe.C = adj_dataframe.L.apply(lambda x: beta * x)
#         adj_dataframe.MaxC = adj_dataframe.C.apply(lambda x: gamma * x)
#         # print(np.sum(adj_dataframe.L), np.sum(adj_dataframe.C))
#         adj_dataframe.loc[29, 'L'] = adj_dataframe.loc[29, 'MaxC']
#         # for q in adj_dataframe.loc[29, 'Neighbor'].split():
#         #     adj_dataframe.loc[int(q), 'L'] = adj_dataframe.loc[int(q), 'L'] * 1.5
#         possibility(adj_dataframe)
#         failure_choice(adj_dataframe)
#         originalLoad = np.sum(adj_dataframe.L)
#         # print(np.sum(adj_dataframe.L), np.sum(adj_dataframe.C[adj_dataframe.F == 0]), np.median(adj_dataframe.D))
#
#         for t in range(20):
#             print(str(t))
#
#             adj_overload = adj_dataframe[(adj_dataframe.P != 0) & (adj_dataframe.F == 0)]
#             adj_failure = adj_dataframe[adj_dataframe.F == 1]
#             adj_failure_Code = adj_failure.Code.to_list()
#             distribution = {}
#
#             # 过载但未失效节点分担
#             for i in adj_overload['Code']:
#                 tmpCode = i
#                 tmpC_sum = 0
#                 Neighbor_list = adj_overload.loc[int(i), 'Neighbor'].split()
#                 overLoad = adj_overload.loc[int(i), 'L'] - adj_overload.loc[int(i), 'C']
#                 Neighbor_avail_list = [k for k in Neighbor_list if k not in adj_failure_Code]
#                 for j in Neighbor_avail_list:
#                     tmpC_sum = adj_dataframe.loc[int(j), 'C'] + tmpC_sum
#                 for j in Neighbor_avail_list:
#                     tmp_distri = overLoad * (adj_dataframe.loc[int(j), 'C'] / tmpC_sum)
#                     if j not in distribution.keys():
#                         distribution[j] = tmp_distri
#                     else:
#                         distribution[j] += tmp_distri
#             for i in adj_overload['Code']:
#                 adj_dataframe.loc[int(i), 'L'] = adj_dataframe.loc[int(i), 'C']
#
#             for i in distribution.keys():
#                 adj_dataframe.loc[int(i), 'L'] += distribution[i]
#             # 测试
#             # print(np.sum(adj_dataframe.L), record)
#             # test_sum = np.sum(adj_dataframe.L)
#             distribution = {}
#             # 过载且失效节点分担
#             for i in adj_failure['Code']:
#                 tmpC_sum = 0
#                 Neighbor_list = adj_failure.loc[int(i), 'Neighbor'].split()
#                 allLoad = adj_failure.loc[int(i), 'L']
#                 Neighbor_avail_list = [k for k in Neighbor_list if k not in adj_failure_Code]
#                 # if len(Neighbor_avail_list) == 0:
#                 #     print(i)
#                 for j in Neighbor_avail_list:
#                     tmpC_sum = adj_dataframe.loc[int(j), 'C'] + tmpC_sum
#                 for j in Neighbor_avail_list:
#                     tmp_distri = allLoad * (adj_dataframe.loc[int(j), 'C'] / tmpC_sum)
#                     if j not in distribution.keys():
#                         distribution[j] = tmp_distri
#                     else:
#                         distribution[j] += tmp_distri
#
#             for i in adj_failure['Code']:
#                 adj_dataframe.loc[int(i), 'L'] = 0
#
#             for i in distribution.keys():
#                 adj_dataframe.loc[int(i), 'L'] += distribution[i]
#
#             possibility(adj_dataframe)
#             failure_choice(adj_dataframe)
#             adj_still_work = adj_dataframe[adj_dataframe.F == 0]
#             # print(np.sum(adj_dataframe.L))
#             if (np.mean(adj_still_work.P) == 1) or (np.sum(adj_still_work.P) <= 0.1):
#                 finalLoad = np.sum(adj_dataframe.L)
#                 mean_List.append(finalLoad / originalLoad)
#                 break
#     mean_rate = np.mean(mean_List)
#     jilu.append([alpha, beta, mean_rate])
#
# pd.DataFrame(jilu, columns=['alpha', 'beta', 'Rate']).to_excel('record.xlsx', index=False)

# print(np.sum(adj_dataframe.L), record)
# adj_failure = adj_dataframe[adj_dataframe.F == 1]
# adj_failure_code = adj_failure.Code
# adj_failure_neighbor = adj_failure.Neighbor

# frequency = {}
# for i in range(100):
#     failure_choice(adj_dataframe)
#     count = np.sum(adj_dataframe.F)
#     if count not in frequency.keys():
#         frequency[count] = 1
#     else:
#         frequency[count] += 1
current_time = time.time()
print("运行时间为" + str(current_time - old_time) + "s")
