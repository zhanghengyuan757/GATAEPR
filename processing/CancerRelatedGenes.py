import os
import warnings
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

from tools.PathTools import DataPathGetter, TempPathGetter

warnings.filterwarnings('ignore')
import xgboost

p = 0.05
fc = 1.3


def find_cancer_genes(cancer: str) -> List[str]:
    dpg = DataPathGetter(cancer)
    path = TempPathGetter().get_path(cancer, 'cancer_related_genes.csv')
    if os.path.exists(path):
        return pd.read_csv(path, header=0, index_col=0)
    gene_expr = pd.read_csv(dpg.get_path('gene_expr_fpkm.csv'), index_col=0)
    stages = pd.read_csv(dpg.get_path('cancer_stage.csv'), index_col=0)
    gene_expr = gene_expr.filter(items=[i for i in gene_expr.index if not i.startswith('?')], axis=0)
    gene_expr = gene_expr[gene_expr.min(axis=1) >= 10]
    genes = gene_expr.index.tolist()
    change_marks = []
    itemsa = stages[stages['flag'] == 0].index
    itemsb = stages[stages['flag'] != 0].index
    a = gene_expr.filter(items=itemsa, axis=1).to_numpy()
    b = gene_expr.filter(items=itemsb, axis=1).to_numpy()
    change_marks.append(DGE(a, b, len(genes)))
    mark_pd = pd.DataFrame(change_marks, columns=genes,
                           index=['result']).T
    mark_pd = mark_pd[mark_pd['result'] != 0]

    x, y = gene_expr.filter(items=mark_pd.index, axis=0).T, stages['flag'].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
    xg = xgboost.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xg.fit(x.to_numpy(), y.to_numpy())
    feature_importance: np.ndarray = xg.feature_importances_
    feature_support = np.apply_along_axis(lambda n: n > 0, 0, feature_importance)
    scores = feature_importance[feature_support].tolist()
    genes = mark_pd.iloc[feature_support, :].index.tolist()
    res = pd.DataFrame(zip(genes, scores), columns=['genes', 'scores'])
    res['gene_name'] = res['genes'].map(lambda s: s.split('|')[0])
    res.to_csv(path)
    return genes


def DGE(a: np.ndarray, b: np.ndarray, genesize):
    mark = []
    for g in range(genesize):
        l = 0
        _a, _b = a[g, :], b[g, :]
        s1, p1 = stats.ranksums(_a, _b)
        ma, mb = np.mean(_a), np.mean(_b)
        _fc = ma / mb
        if p1 < p and s1 > 0 and np.log2(_fc) > np.log2(fc):
            l = 1
        if p1 < p and s1 < 0 and np.log2(_fc) < -np.log2(fc):
            l = -1
        mark.append(l)
    return mark
