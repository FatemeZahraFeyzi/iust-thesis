import dgl
from dgl.data import FraudYelpDataset
import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import RGCNConv
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE


class HeteroGAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, dropout=0.5):
        super(HeteroGAT, self).__init__()

        self.conv_rsr = GraphSAGE((-1, -1), num_features, 8, dropout=dropout)
        self.dropout1 = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(160)

        self.conv_rtr = GraphSAGE((-1, -1), 8 * num_features, 64, dropout=dropout)
        self.dropout2 = nn.Dropout(p=0.2)
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.conv_rur = GraphSAGE((-1, -1), 64, 64, dropout=dropout)
        self.dropout3 = nn.Dropout(p=0.2)
        self.batchnorm3 = nn.BatchNorm1d(64)

        self.lin = torch.nn.Linear(64, num_classes)

    def forward(self, x, edge_index_rsr, edge_index_rtr, edge_index_rur):
        x = self.conv_rsr(x, edge_index_rsr)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.batchnorm1(x)

        x = self.conv_rtr(x, edge_index_rtr)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.batchnorm2(x)

        x = self.conv_rur(x, edge_index_rur)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.batchnorm3(x)
        x = self.lin(x)
        return x


class HeteroRGCN(nn.Module):
    def __init__(self, num_nodes, num_features, num_classes, num_relations):
        super(HeteroRGCN, self).__init__()

        self.rgcn1 = RGCNConv(num_features, 64, num_relations)
        self.rgcn2 = RGCNConv(64, 64, num_relations)
        self.rgcn3 = RGCNConv(64, 64, num_relations)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.rgcn1(x, edge_index, edge_type))
        x = self.bn1(x)
        x = self.dropout(x)

        x = F.relu(self.rgcn2(x, edge_index, edge_type))
        x = self.bn2(x)
        x = self.dropout(x)

        x = F.relu(self.rgcn3(x, edge_index, edge_type))
        x = self.bn3(x)
        x = self.dropout(x)

        x = self.linear(x)
        return x
    
class PairDataset(torch.utils.data.Dataset):
    def __init__(self, x, pos_pairs, neg_pairs):
        self.x = x
        self.pos_pairs = pos_pairs
        self.neg_pairs = neg_pairs

    def __len__(self):
        return len(self.pos_pairs) + len(self.neg_pairs)

    def __getitem__(self, idx):
        if idx < len(self.pos_pairs):
            pair = self.pos_pairs[idx]
            label = 1
        else:
            pair = self.neg_pairs[idx - len(self.pos_pairs)]
            label = 0

        return self.x[pair[0]], self.x[pair[1]], label


class Encoder(nn.Module):
  def __init__(self, input_dim, embed_dim):
    super(Encoder, self).__init__()
    self.model = nn.Sequential(
      nn.Linear(input_dim, 64),
      nn.ReLU(),
      nn.Linear(64, embed_dim)
    )

  def forward(self, x):
    return self.model(x)