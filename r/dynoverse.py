import numpy as np
import pandas as pd
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from tools.ReportTools import reports


def sc_similarity(expr, stage, edges_with_pred_label):
    importr('dyno')
    importr('tidyverse')
    for single_cell_method in ['slingshot', 'mst', 'paga', 'monocle_ica']:
    # for single_cell_method in ['slingshot']:
        with localconverter(robjects.default_converter + pandas2ri.converter):
            robjects.globalenv['expr'] = robjects.conversion.rpy2py(expr)
            robjects.globalenv['stage'] = robjects.conversion.rpy2py(stage)
        try:
            robjects.r("method = '%s'" % single_cell_method)
            robjects.r.source('r/dyno.R')
        except Exception as e:
            print(single_cell_method, '报错')
            continue

        seq1 = list(robjects.r('result').names)
        seq1 = list(map(lambda a: a.replace('.', '-'), seq1))
        new_list = stage.filter(items=seq1, axis=0)
        new_list['score'] = list(robjects.r('result'))
        match_r = match_rate(edges_with_pred_label, new_list.filter(items=stage.index, axis=0))
        print(single_cell_method, '与我们的方法的匹配度:', match_r)


def match_rate(edges: pd.DataFrame, new_list):
    match_list = []
    new_list = new_list['score'].tolist()
    dyno_pred, y = [], []
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
            dyno_pred.append(1)
        elif scorei > scorej:
            dyno_pred.append(0)
        else:
            dyno_pred.append(int(1 - labels))
    print('dyno report!!!!')
    reports(y, dyno_pred)
    return sum(match_list) / len(match_list)
