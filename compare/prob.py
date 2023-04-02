import pandas as pd

from tools.PathTools import PathGetter, DataPathGetter
from tools.ReportTools import reports

# result.csv files come from 'matlab_code/Main.m'
def prob_rank(cancer, edges_with_pred_label):
    bpg = PathGetter('prob')
    dpg = DataPathGetter(cancer)
    indexs = pd.read_csv(bpg.get_path(cancer, 'result.csv'), header=None)
    indexs = (indexs - 1).iloc[:, 0].tolist()
    stages = pd.read_csv(dpg.get_path('cancer_stage.csv'))
    new_list = stages.iloc[indexs, :]
    match_r = match_rate(edges_with_pred_label, new_list)
    print('PROB与我们的方法的匹配度:', match_r)

def match_rate(edges: pd.DataFrame, new_list):
    match_list = []
    new_list = new_list.index.tolist()
    prob_pred, y = [], []
    for i, j, pred, labels in edges.to_numpy():  # i,j 是原始数据集的index
        y.append(labels)
        scorei = new_list[i]
        scorej = new_list[j]
        if scorei < scorej and pred == 1:
            match_list.append(1)
        elif scorei > scorej and pred == 0:
            match_list.append(1)
        else:
            match_list.append(0)
        if scorei < scorej:
            prob_pred.append(1)
        else:
            prob_pred.append(0)

    print('PROB的reports')
    reports(y, prob_pred)
    return sum(match_list) / len(match_list)
