import torch
from torch import nn

from model_fit.Layers import GATConv,GCNConv,MultiHeadGATConv


class Gcn(nn.Module):
    def __init__(self, input_dim, hid_dim, z_dim, normalize=False, epsilon=0.01):
        super(Gcn, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.normalize = normalize
        self.encode_act_fn = nn.ReLU()
        self.decode_act_fn = nn.Sigmoid()
        self.gc1 = GCNConv(input_dim, hid_dim)
        self.gc2 = GCNConv(hid_dim, z_dim)
        self.epsilon = epsilon

    def forward(self, x, adj):
        z = self.encode(x, adj)
        return self.decode(z)

    def encode(self, x, adj):
        x = self.gc1(x, adj)
        x = self.encode_act_fn(x)
        x = self.gc2(x, adj)
        return x

    def decode(self, z):
        if self.normalize:
            _z = nn.functional.normalize(z[:, 0:-1], p=2, dim=1)
        else:
            _z = z[:, 0:-1]
        x1 = torch.sum(_z ** 2, dim=1, keepdim=True)
        x2 = torch.matmul(_z, torch.t(_z))
        dist = x1 - 2 * x2 + torch.t(x1) + self.epsilon
        m = z[:, -1:]
        mass = torch.matmul(torch.ones([m.shape[0], 1]).cuda(), torch.t(m))
        out = mass - torch.log(dist)
        return out


class GcnVae(nn.Module):
    def __init__(self, input_dim, hid_dim, z_dim, normalize=False, epsilon=0.01):
        super(GcnVae, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.normalize = normalize
        self.encode_act_fn = nn.ReLU()
        self.decode_act_fn = nn.Sigmoid()
        self.gc1 = GCNConv(input_dim, hid_dim)
        self.gc2 = GCNConv(hid_dim, z_dim)
        self.epsilon = epsilon

    def forward(self, x, adj):
        z = self.encode(x, adj)
        z_log_std = self.encode(x, adj)
        normal = torch.normal(mean=0, std=1, size=[self.input_dim, self.z_dim]).cuda()
        z = z + normal * torch.exp(z_log_std)
        return self.decode(z)

    def encode(self, x, adj):
        x = self.gc1(x, adj)
        x = self.encode_act_fn(x)
        x = self.gc2(x, adj)
        return x

    def decode(self, z):
        if self.normalize:
            _z = nn.functional.normalize(z[:, 0:-1], p=2, dim=1)
        else:
            _z = z[:, 0:-1]
        x1 = torch.sum(_z ** 2, dim=1, keepdim=True)
        x2 = torch.matmul(_z, torch.t(_z))
        dist = x1 - 2 * x2 + torch.t(x1) + self.epsilon
        m = z[:, -1:]
        mass = torch.matmul(torch.ones([m.shape[0], 1]).cuda(), torch.t(m))
        out = mass - torch.log(dist)
        return out


class GAT(nn.Module):
    def __init__(self, input_dim, hid_dim, z_dim, normalize=False, epsilon=0.01):
        super(GAT, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.normalize = normalize
        self.encode_act_fn = nn.ReLU()
        self.decode_act_fn = nn.Sigmoid()
        self.gc1 = GATConv(input_dim, hid_dim)
        self.gc2 = GATConv(hid_dim, z_dim)
        self.epsilon = epsilon

    def forward(self, x, adj):
        z = self.encode(x, adj)
        return self.decode(z)

    def encode(self, x, adj):
        x = self.gc1(x, adj)
        x = self.encode_act_fn(x)
        x = self.gc2(x, adj)
        return x

    def decode(self, z):
        if self.normalize:
            _z = nn.functional.normalize(z[:, 0:-1], p=2, dim=1)
        else:
            _z = z[:, 0:-1]
        x1 = torch.sum(_z ** 2, dim=1, keepdim=True)
        x2 = torch.matmul(_z, torch.t(_z))
        dist = x1 - 2 * x2 + torch.t(x1) + self.epsilon
        m = z[:, -1:]
        mass = torch.matmul(torch.ones([m.shape[0], 1]).cuda(), torch.t(m))
        out = mass - torch.log(dist)
        return out


class RGANRP(nn.Module):
    def __init__(self, input_dim, hid_dim, z_dim, normalize=False, epsilon=0.01):
        super(RGANRP, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.normalize = normalize
        self.encode_act_fn = nn.ReLU()
        self.decode_act_fn = nn.Sigmoid()
        self.gc1 = MultiHeadGATConv(input_dim, hid_dim)
        self.gc2 = MultiHeadGATConv(hid_dim, z_dim)
        self.epsilon = epsilon

    def forward(self, x, adj):
        z = self.encode(x, adj)
        return self.decode(z)

    def encode(self, x, adj):
        x = self.gc1(x, adj)
        x = self.encode_act_fn(x)
        x = self.gc2(x, adj)
        return x

    def decode(self, z):
        if self.normalize:
            _z = nn.functional.normalize(z[:, 0:-1], p=2, dim=1)
        else:
            _z = z[:, 0:-1]
        x1 = torch.sum(_z ** 2, dim=1, keepdim=True)
        x2 = torch.matmul(_z, torch.t(_z))
        dist = x1 - 2 * x2 + torch.t(x1) + self.epsilon
        m = z[:, -1:]
        mass = torch.matmul(torch.ones([m.shape[0], 1]).cuda(), torch.t(m))
        out = mass - torch.log(dist)
        return out

