import pandas as pd
import numpy as np
import time


def adj_matrix_to_list(adj_matrix, adj_dataframe):
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


def possibility(adjDf):
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


def failure_choice(adjDf):
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

old_time = time.time()
# alpha = float(input('Pleas key in alpha:'))
alpha = float(1)
theta = float(1.1)
# beta = float(input('Please key in beta:'))
beta = float(1.5)
# gamma = float(input('Please key in gamma:'))
gamma = float(1.5)
adjMat = pd.read_excel('corr2.xlsx', index_col=0).to_numpy()
adj_dataframe = pd.DataFrame(index=[i for i in range(len(adjMat))],
                             columns=['Code', 'Neighbor', 'D', 'L', 'C', 'MaxC', 'P', 'F'])
record = []
N = 0
# 初始化
for theta in np.linspace(1.0, 1.5, num=51):
    N = 0
    Unumber = 0
    print(theta)
    for node_count in range(len(adj_dataframe)):
        adj_dataframe = pd.DataFrame(index=[i for i in range(len(adjMat))],
                                     columns=['Code', 'Neighbor', 'D', 'L', 'C', 'MaxC', 'P', 'F', 'U', 'kuo'])
        adj_matrix_to_list(adjMat, adj_dataframe)
        adj_dataframe.D = adj_dataframe.Neighbor.apply(lambda x: len(x.split()))
        adj_dataframe.L = adj_dataframe.D.apply(lambda x: x ** alpha)
        adj_dataframe.C = adj_dataframe.L.apply(lambda x: beta * x)
        adj_dataframe.MaxC = adj_dataframe.C.apply(lambda x: gamma * x)
        adj_dataframe.U = np.zeros((len(adjMat), 1))
        adj_dataframe.kuo = np.zeros((len(adjMat), 1))
        # print(np.sum(adj_dataframe.L), np.sum(adj_dataframe.C))
        adj_dataframe.loc[int(node_count), 'L'] = adj_dataframe.loc[int(node_count), 'MaxC']
        # for q in adj_dataframe.loc[14, 'Neighbor'].split():
        #     adj_dataframe.loc[int(q), 'L'] = adj_dataframe.loc[int(q), 'L'] * 1.5
        possibility(adj_dataframe)
        failure_choice(adj_dataframe)
        originalLoad = np.sum(adj_dataframe.L)
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

        adj_overload = adj_dataframe[(adj_dataframe.P != 0) & (adj_dataframe.F == 0)]

        for i in adj_overload['Code']:
            if adj_dataframe.loc[int(i), 'U'] != 1:
                adj_dataframe.loc[int(i), 'kuo'] = adj_dataframe.loc[int(i), 'C'] * (theta - 1)
                adj_dataframe.loc[int(i), 'C'] = adj_dataframe.loc[int(i), 'C'] * theta
                adj_dataframe.loc[int(i), 'MaxC'] = adj_dataframe.loc[int(i), 'C'] * gamma
                adj_dataframe.loc[int(i), 'U'] = 1
        possibility(adj_dataframe)

        failure_choice(adj_dataframe)
        adj_still_work = adj_dataframe[adj_dataframe.F == 0]

        for t in range(50):
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
            adj_overload = adj_dataframe[(adj_dataframe.P != 0) & (adj_dataframe.F == 0)]

            for i in adj_overload['Code']:
                if adj_dataframe.loc[int(i), 'U'] != 1:
                    adj_dataframe.loc[int(i), 'kuo'] = adj_dataframe.loc[int(i), 'C'] * (theta - 1)
                    adj_dataframe.loc[int(i), 'C'] = adj_dataframe.loc[int(i), 'C'] * theta
                    adj_dataframe.loc[int(i), 'MaxC'] = adj_dataframe.loc[int(i), 'C'] * gamma
                    adj_dataframe.loc[int(i), 'U'] = 1
            possibility(adj_dataframe)

            failure_choice(adj_dataframe)
            adj_still_work = adj_dataframe[adj_dataframe.F == 0]
            if (np.mean(adj_still_work.P) == 1) or (np.sum(adj_still_work.P) <= 0.1):
                N += np.sum(adj_dataframe.F)
                break
    Sn = (N - 1) / (len(adj_dataframe) * (len(adj_dataframe) - 1))
    Unumber = np.sum(adj_dataframe.U)
    kuoNumber = np.sum(adj_dataframe.kuo)
    record.append([beta, theta, Sn, Unumber, kuoNumber])
fileName = str(alpha) + 'simu_sensi' + str(beta) + '.xlsx'
pd.DataFrame(record, columns=['beta', 'theta', 'Sn', 'Unumber', 'kuoNumber']).to_excel(fileName, index=False)

current_time = time.time()
print("运行时间为" + str(current_time - old_time) + "s")
