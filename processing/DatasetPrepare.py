import os
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.preprocessing import LabelEncoder

from processing.CancerRelatedGenes import find_cancer_genes
from tools.PathTools import TempPathGetter, DataPathGetter

random_state = 666


def get_dataset(cancer, show_fig=False, verbose=False):
    dpg = DataPathGetter(cancer)
    expr = pd.read_csv(dpg.get_path('gene_expr_fpkm.csv'), index_col=0)
    stage = pd.read_csv(dpg.get_path('cancer_stage.csv'), index_col=0)
    genes = find_cancer_genes(cancer).loc[:, 'genes']
    expr = expr.filter(items=genes, axis=0)
    expr: pd.DataFrame = (expr + 1).apply(np.log2)
    expr = expr.filter(items=stage.index, axis=1)
    # expr.to_csv(dpg.get_path('gene_expr_fpkm_selected.csv'))
    if verbose:
        summerize = stage.groupby(['stage']).count()
        print(summerize)
    if show_fig:
        t_sne(cancer, (pd.read_csv(dpg.get_path('gene_expr_fpkm.csv'), index_col=0) + 1).apply(np.log2), stage)
        t_sne(cancer, expr, stage)
    return expr, stage


def t_sne(cancer: str, expr, stage):
    _stages = stage['stage'].drop_duplicates().size
    label = LabelEncoder().fit_transform(stage['stage'])
    colors = [plt.cm.Set2(i) for i in range(_stages)]
    color = [colors[l] for l in label]
    # 创建自定义图像
    fig = plt.figure(figsize=(5, 5))  # 指定图像的宽和高
    # plt.title("Dimensionality Reduction and Visualization of " + cancer + ' data', fontsize=14)  # 自定义图像名称

    # 绘制S型曲线的3D图像

    print('Starting compute t-SNE Embedding...')
    # t-SNE的降维与可视化
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0, learning_rate='auto')
    # 训练模型
    y = ts.fit_transform(expr.T)
    print('T-SNE Embedding computed.')
    x_min, x_max = np.min(y, 0), np.max(y, 0)
    y = (y - x_min) / (x_max - x_min)
    # plt.scatter(y[:, 0], y[:, 1], c=color, cmap=plt.cm.Spectral)
    for i in range(y.shape[0]):
        # 在图中为每个数据点画出标签
        plt.text(y[i, 0], y[i, 1], str(stage.iloc[i, 1]), color=color[i],
                 fontdict={'weight': 'bold', 'size': 7})
    # 显示图像

    # plt.legend()
    plt.show()


def cal_similarity_net(cancer: str):
    expr, label = get_dataset(cancer, verbose=True)
    dpg = TempPathGetter()
    path1 = dpg.get_path(cancer, 'nodes.csv')
    path2 = dpg.get_path(cancer, 'edges.csv')
    if os.path.exists(path1) and os.path.exists(path2):
        return
    size = expr.columns.size
    edges = []
    import time
    print(cancer, "similarity network calculating...")
    time_start = time.time()
    spearman = expr.corr(method='spearman').to_numpy()
    pearson = expr.corr(method='pearson').to_numpy()
    kendall = expr.corr(method='kendall').to_numpy()
    euclidean = expr.corr(method=lambda a, b: np.sqrt(np.sum(np.square(a - b)))).to_numpy()
    manhattan = expr.corr(method=lambda a, b: np.sum(np.abs(a - b))).to_numpy()
    chebyshev = expr.corr(method=lambda a, b: np.max(np.abs(a - b))).to_numpy()
    for i in range(1, size):
        for j in range(0, i):
            edges.append({'i': i, 'j': j,
                          'spearman': spearman[i][j], 'pearson': pearson[i][j], 'kendall': kendall[i][j],
                          'euclidean': euclidean[i][j], 'manhattan': manhattan[i][j], 'chebyshev': chebyshev[i][j]})
    time_end = time.time()
    calculate_time = time_end - time_start
    print(cancer, "similarity network calculate_time:", round(calculate_time, 2), 's')
    edges = pd.DataFrame(edges)
    edges['euclidean'] = edges['euclidean'] / (edges['euclidean'].max())
    edges['manhattan'] = edges['manhattan'] / (edges['manhattan'].max())
    edges['chebyshev'] = edges['chebyshev'] / (edges['chebyshev'].max())
    label.to_csv(path1)
    edges.to_csv(path2)


def get_directed_edges(cancer, k_nn=15, metric='euclidean'):
    tpg = TempPathGetter()
    node_path = tpg.get_path(cancer, 'nodes.csv')
    edge_path = tpg.get_path(cancer, 'edges.csv')
    if not (os.path.exists(node_path) and os.path.exists(edge_path)):
        cal_similarity_net(cancer)
    edges = pd.read_csv(edge_path, index_col=0)
    nodes = pd.read_csv(node_path, index_col=0)
    i_r = edges.sort_values([metric]).groupby('i').head(k_nn)
    i_r['node_i'] = i_r['i']
    j_r = edges.sort_values([metric]).groupby('j').head(k_nn)
    j_r['node_i'] = j_r['j']

    edges = pd.concat([i_r.reset_index(drop=True), j_r.reset_index(drop=True)]).sort_values([metric]) \
        .groupby('node_i').head(k_nn).reset_index(drop=True)
    edges = edges.drop(['node_i'], axis=1).drop_duplicates().reset_index(drop=True)
    nodes = nodes.replace('N', "A")
    stages = LabelEncoder().fit_transform(nodes['stage']).tolist()
    edge_directed, edge_undirected = [], []
    for _, l in edges.iterrows():
        i, j = int(l['i']), int(l['j'])
        si, sj = stages[i], stages[j]
        # if abs(si - sj) > 2:  # 可以看到一些IIIA IA 相较正常样本很近，这不合适
        #     continue
        if si > sj:  # edge direct using IA-IVC stage
            edge_directed.append({'i': j, 'j': i})
        elif si < sj:
            edge_directed.append({'i': i, 'j': j})
        else:
            edge_undirected.append({'i': i, 'j': j})
    edge_directed = pd.DataFrame(edge_directed, dtype=int, columns=['i', 'j'])
    # showdigraph(cancer, nodes, edge_directed)
    unknown = pd.DataFrame(edge_undirected, dtype=int, columns=['i', 'j'])
    return edge_directed, unknown


def showdigraph(cancer, nodes: pd.DataFrame, edge_directed):
    plt.figure(figsize=(20, 15))
    _stages = nodes['stage'].drop_duplicates()
    colors = [plt.cm.Set2(i) for i in range(_stages.size)]
    expr = get_dataset(cancer)[0]
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0, learning_rate='auto')
    # 训练模型
    y = ts.fit_transform(expr.T)
    print('T-SNE Embedding computed.')
    x_min, x_max = np.min(y, 0), np.max(y, 0)
    y = (y - x_min) / (x_max - x_min)
    pos = {}
    for index in range(len(y)):
        pos[index] = y[index]

    nodes = nodes.reset_index(drop=True)
    g = nx.DiGraph()
    for layer in range(_stages.size):
        g.add_nodes_from(nodes[nodes['stage'] == _stages[layer]].T, layer=layer)
    g.add_edges_from(edge_directed.to_numpy())
    # pos = nx.planar_layout(g)
    # pos = nx.multipartite_layout(g, subset_key="layer")

    label = LabelEncoder().fit_transform(nodes['stage'])
    color_list = [colors[l] for l in label]
    nx.draw_networkx(g, pos=pos, arrows=True, labels=nodes.replace('A', "N")['stage'], node_color=color_list)
    plt.show()


# from sklearn.neighbors import kneighbors_graph
# def get_knn_neighbor(cancer, k=5):
#     expr, label = get_dataset(cancer)
#     A = kneighbors_graph(expr.T, n_neighbors=5, mode='connectivity', include_self=False, p=2)
#     A.toarray()


def prepare_gae_data(train, test, unknown):
    assert type(train) == pd.DataFrame and type(test) == pd.DataFrame and type(unknown) == pd.DataFrame
    g = nx.DiGraph(pd.concat([train, test, unknown], axis=0).to_numpy().tolist())
    warnings.filterwarnings('ignore')
    g.remove_edges_from(unknown.to_numpy())
    g.remove_edges_from(test.to_numpy())
    adj = nx.adjacency_matrix(g)
    gk = list(g.nodes.keys())
    return adj, gk


def kfold_split(k, y):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k, shuffle=True)
    folds = []
    for train, test in kf.split(y):
        folds.append((train, test))
    return folds


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    coords, values, shape = sparse_to_tuple(sparse_mx)
    indices = torch.from_numpy(coords.transpose().astype(np.int64))
    values = torch.from_numpy(values.astype(np.float32))
    shape = torch.Size(shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    # Out-degree normalization of adj
    degree_mat_inv_sqrt = sp.diags(np.power(np.array(adj_.sum(1)), -1).flatten())
    adj_normalized = degree_mat_inv_sqrt.dot(adj_)
    return adj_normalized
