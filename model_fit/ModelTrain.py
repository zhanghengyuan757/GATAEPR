import time
from itertools import combinations
import scipy.sparse as sp
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

from model_fit.Models import RGANRP, Gcn, GcnVae, GAT
from processing.DatasetPrepare import get_directed_edges, kfold_split, prepare_gae_data, get_dataset, \
    sparse_mx_to_torch_sparse_tensor, normalize_adj

from r.dynoverse import sc_similarity
from tools.GraphTools import get_dag_by_predit_prob
from tools.PathTools import DataPathGetter
from tools.ReportTools import reports

learning_rate = 0.001
hid_n = 64
z_n = 32
epochs = 200
cuda = True
gpu_id = 0
verbose = True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 用作模型评估
def get_pred_from_emb(edge_pos: pd.DataFrame, emb: pd.DataFrame, epsilon=0.01):
    assert type(edge_pos) == pd.DataFrame and edge_pos.columns.tolist() == ['i', 'j']
    emb_index = emb.index.tolist()
    emb_index = dict(zip(emb_index, range(len(emb_index))))
    emb = emb.to_numpy()
    edges_pos = edge_pos.to_numpy()
    edges_neg = edge_pos[['j', 'i']].to_numpy()
    p = []
    for e0, e1 in np.vstack([edges_pos, edges_neg]):
        e0, e1 = emb_index[e0], emb_index[e1]
        dist = np.square(epsilon + np.linalg.norm(emb[e0, 0:-1] - emb[e1, 0:-1], ord=2))
        # pred_score = sigmoid(emb[e1, -1] - np.log(dist))
        # preds.append(pred_score)
        # if pred_score > 0.5:
        #     p.append(1)
        # else:
        #     p.append(0)
        pred_score_e0 = sigmoid(emb[e0, -1] - np.log(dist))
        pred_score_e1 = sigmoid(emb[e1, -1] - np.log(dist))
        if pred_score_e0 < pred_score_e1 and (pred_score_e0 >= 0.5 or pred_score_e1 >= 0.5):
            p.append(1)
        else:
            p.append(0)

    preds_all = np.array(p)
    labels_all = np.hstack([np.ones(edges_pos.shape[0]), np.zeros(edges_neg.shape[0])]).astype(int)
    return labels_all, preds_all


def cal_dist(n0, n1, emb, epsilon=0.01):
    dist = np.square(epsilon + np.linalg.norm(emb[n0, 0:-1] - emb[n1, 0:-1], ord=2))
    return sigmoid(emb[n1, -1] - np.log(dist))


# sc_simi single-cell compare need docker and R environment
def ggae_predict(cancer, k_fold=5, model_name: str = 'RGANRP', sc_simi=False):
    edge_directed, unknown = get_directed_edges(cancer)
    folds = kfold_split(k_fold, edge_directed)  # 返回k_fold个数组[(train_index,test_index),(),...]
    for k in range(k_fold):
        print('=' * 20, k + 1, 'Fold', '=' * 20)
        train, test = folds[k]
        train_y = edge_directed.iloc[train, :]
        test_y = edge_directed.iloc[test, :]
        adj, gk = prepare_gae_data(train_y, test_y, unknown)
        features = np.eye(adj.shape[0])
        # features = get_dataset(cancer)[0].iloc[:, gk].to_numpy().T
        input_dim = features.shape[1]

        if model_name == 'RGANRP':
            model = RGANRP(input_dim, hid_n, z_n)
        elif model_name == 'Gcn':
            model = Gcn(input_dim, hid_n, z_n)
        elif model_name == 'GcnVae':
            model = GcnVae(input_dim, hid_n, z_n)
        elif model_name == 'GAT':
            model = GAT(input_dim, hid_n, z_n)
        else:
            raise Exception(model_name + ' no implements!')
        emb = ggae_get_emb(adj, features, gk, model)
        y, pred = get_pred_from_emb(test_y[['i', 'j']], emb)
        reports(y, pred)
        draw_graph_result(cancer, test_y[['i', 'j']], y, pred)
        labels_all, edges_pred = get_pred_from_emb(edge_directed, emb)
        draw_graph_result_dag(cancer, edge_directed, unknown, emb)
        if sc_simi:
            dpg = DataPathGetter(cancer)
            expr = pd.read_csv(dpg.get_path('gene_expr_fpkm.csv'), index_col=0)
            stage = pd.read_csv(dpg.get_path('cancer_stage.csv'), index_col=0)
            # expr, stage = get_dataset(cancer)
            edge_directed = edge_directed.append(
                pd.DataFrame(edge_directed[['j', 'i']].to_numpy(), columns=['i', 'j'])).reset_index(drop=True)
            edges_with_pred_label = pd.concat(
                [edge_directed, pd.Series(edges_pred, name='pred'), pd.Series(labels_all, name='label')], axis=1)
            sc_similarity(expr, stage, edges_with_pred_label)


def draw_graph_result_dag(cancer, edge_directed, unknown, emb, epsilon=0.01):
    emb_index = emb.index.tolist()
    emb_index = dict(zip(emb_index, range(len(emb_index))))
    emb = emb.to_numpy()
    edges_pos = edge_directed.to_numpy()
    unknown = unknown[['i', 'j']].to_numpy()
    edges = np.vstack([edges_pos, unknown])
    preds_all = []
    scores = np.zeros((emb.shape[0], emb.shape[0]))
    for e0, e1 in edges:
        e0, e1 = emb_index[e0], emb_index[e1]
        dist = np.square(epsilon + np.linalg.norm(emb[e0, 0:-1] - emb[e1, 0:-1], ord=2))
        pred_score_e0 = sigmoid(emb[e0, -1] - np.log(dist))
        pred_score_e1 = sigmoid(emb[e1, -1] - np.log(dist))
        if pred_score_e0 < pred_score_e1 and (pred_score_e0 >= 0.5 or pred_score_e1 >= 0.5):
            preds_all.append({'i': e0, 'j': e1})
            scores[e0, e1] = pred_score_e1
        else:
            preds_all.append({'i': e1, 'j': e0})
            scores[e1, e0] = pred_score_e0
    preds_all = pd.DataFrame(preds_all)
    g = nx.DiGraph(preds_all.to_numpy().tolist())
    dag: nx.DiGraph = get_dag_by_predit_prob(g, scores)
    path = nx.algorithms.dag_longest_path(dag)
    expr, nodes = get_dataset(cancer)
    nodes = nodes.reset_index(drop=True)
    plt.figure(figsize=(15, 10))
    _stages: pd.Series = nodes['stage'].drop_duplicates()
    colors = [plt.cm.Set3(i) for i in range(_stages.size)]
    g = nx.DiGraph()
    g.add_nodes_from(path)
    for i in range(len(path) - 1):
        g.add_edge(path[i], path[i + 1])
    # ts = manifold.TSNE(n_components=2, init='pca', random_state=0, learning_rate='auto')
    # y = ts.fit_transform(expr.T)
    # x_min, x_max = np.min(y, 0), np.max(y, 0)
    # y = (y - x_min) / (x_max - x_min)
    # pos = {}
    # for index in range(len(y)):
    #     pos[index] = y[index]
    pos = nx.spiral_layout(g, equidistant=True, resolution=0.5)
    nodes_on_path = nodes['stage'].iloc[path]
    label = LabelEncoder().fit_transform(nodes_on_path)
    color_list = [colors[l] for l in label]
    nx.draw_networkx(g, pos=pos, arrows=True, labels=nodes_on_path.replace('A', "N"), node_color=color_list)
    plt.show()


def draw_graph_result(cancer, test_links, label, pred):
    expr, nodes = get_dataset(cancer)
    nodes = nodes.reset_index(drop=True)

    links = test_links.append(
        pd.DataFrame(test_links[['j', 'i']].to_numpy(), columns=['i', 'j'])).reset_index(drop=True)
    links = pd.concat([links, pd.Series(label, name='y', dtype=int),
                       pd.Series(pred, name='pred', dtype=int)], axis=1)

    plt.figure(figsize=(15, 10))
    _stages: pd.Series = nodes['stage'].drop_duplicates()
    colors = [plt.cm.Set3(i) for i in range(_stages.size)]
    g = nx.Graph()
    g.add_nodes_from(_stages)

    links['stagei'] = links['i'].map(lambda i: nodes['stage'].iloc[i])
    links['stagej'] = links['j'].map(lambda i: nodes['stage'].iloc[i])
    links['score'] = (links['y'] == links['pred']).map(int)
    pos = nx.spiral_layout(g, equidistant=True, resolution=0.5)
    for stage1, stage2 in combinations(_stages, 2):
        temp = links[(links['stagei'] == stage1) & (links['stagej'] == stage2) | (links['stagei'] == stage2) & (
                links['stagej'] == stage1)]
        if len(temp['score']) == 0:
            continue
        edge_label = str(round(sum(temp['score']) / len(temp['score']) * 100, 2)) + '%'
        g.add_edge(stage1, stage2, label=edge_label)
    nx.draw_networkx(g, pos=pos, arrows=True, node_color=colors, label=True)
    nx.draw_networkx_edge_labels(g, pos, nx.get_edge_attributes(g, 'label'))
    plt.show()


def ggae_get_emb(adj, x, gk, model):
    input_dim = x.shape[1]
    tensor_x = torch.FloatTensor(x)
    edge_sum = adj.sum()
    pos_weight = torch.FloatTensor([(input_dim ** 2 - edge_sum) / edge_sum])
    cost_norm = input_dim ** 2 / float((input_dim ** 2 - edge_sum) * 2)
    adj_norm = normalize_adj(adj)

    tensor_adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
    tensor_adj_eye = sparse_mx_to_torch_sparse_tensor(adj + sp.eye(adj.shape[0]))
    # edge_index = torch.from_numpy(edge_index)
    print('Gravity_AE Training...')
    if cuda:
        tensor_x = tensor_x.cuda(gpu_id)
        tensor_adj_norm = tensor_adj_norm.cuda(gpu_id)
        tensor_adj_eye = tensor_adj_eye.cuda(gpu_id)
        # edge_index = edge_index.cuda(gpu_id)
        pos_weight = pos_weight.cuda(gpu_id)
        model.cuda(gpu_id)
    cost = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    print("")
    for epoch in range(epochs):
        t = time.time()
        model.train()
        x2 = model(tensor_x, tensor_adj_norm)
        # x2 = model(tensor_x, edge_index)
        optimizer.zero_grad()
        loss = cost_norm * cost(x2, tensor_adj_eye.to_dense())
        loss.backward()
        optimizer.step()
        model.eval()
        if np.isnan(loss.item()):
            raise Exception('GAE Gradient disappear! Try to low "learning_rate" in GraphEmbedding.py')
        if verbose and (epoch + 1) % int(epochs / 20) == 0:
            print("\rEpoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
                  "time=", "{:.5f}".format(time.time() - t), end='', flush=True)
    print("")
    print('Gravity_AE Trained.')
    model.eval()
    emb = model.encode(tensor_x, tensor_adj_norm)
    embedding = pd.DataFrame(emb.cpu().detach().numpy(), index=gk).sort_index()
    return embedding
