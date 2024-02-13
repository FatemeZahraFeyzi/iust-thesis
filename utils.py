import torch
import torch_geometric
import networkx as nx
from torch_geometric.data import HeteroData
import torch_geometric.nn as py_nn
import dgl
from dgl.data import FraudYelpDataset
from dgl.dataloading import GraphDataLoader, MultiLayerFullNeighborSampler
from torch_geometric.loader import NeighborLoader, ImbalancedSampler
import torch_sparse
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import degree
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.spatial.distance import cdist
import torch
from torch_geometric.data import Data
from pygod.detector import DOMINANT
from pygod.metric import eval_roc_auc
from pygod.detector import AnomalyDAE
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.linear_model import LogisticRegression
from dgl.dataloading import GraphDataLoader
from torch_geometric.nn import GraphSAGE
from torch_geometric.data import DataLoader
from torch_geometric.nn import RGCNConv
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from sklearn.metrics import silhouette_score


def load_graph():
    dataset = FraudYelpDataset()
    return dataset.graph

def sample_nodes(graph):
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.DataLoader(graph, torch.arange(graph.num_nodes()), sampler,batch_size=64)
    input_nodes = next(iter(dataloader))
    return input_nodes

def create_subgraph(graph, input_nodes):
    return graph.subgraph(input_nodes[0])

def convert_to_pyg(graph):
    return torch_geometric.utils.convert.from_dgl(graph)

def process_graph():
    full_graph = load_graph()

    input_nodes = sample_nodes(full_graph)

    subgraph = create_subgraph(full_graph, input_nodes)

    pyg_data = convert_to_pyg(subgraph)

    return pyg_data

def add_edges(data):

  # Create edges
  data['review', 'net_rsr', 'review'].edge_index = torch.randint(0, 5104, (2, 348372))
  data['review', 'net_rtr', 'review'].edge_index = torch.randint(0, 5104, (2, 57902))
  data['review', 'net_rur', 'review'].edge_index = torch.randint(0, 5104, (2, 16666))

  return data

def define_metapath_schema():
  return [('review', 'net_rsr', 'review'),
          ('review', 'net_rtr', 'review'),
          ('review', 'net_rur', 'review')]

def create_metapath2vec_model(data, schema):
  return py_nn.models.MetaPath2Vec(data.edge_index_dict, embedding_dim=128,
                                    metapath=schema, walk_length=50,
                                    context_size=7,
                                    walks_per_node=5, num_negative_samples=5,
                                    sparse=True)

def train_methavec2path_model(model):
  model.train()

def combine_structural_features(model, data, node_type):

  num_nodes = data[node_type].feature.size(0)
  node_idx = torch.arange(num_nodes)

  embeddings = model(node_type, node_idx)

  x = data[node_type].feature

  combined = torch.cat([x, embeddings], dim=-1)

  data[node_type].feature = combined
  return data


def balanced_loader(data, node_type):

  data[node_type].num_nodes = data[node_type]['label'].size(0)

  input_nodes = (node_type, data[node_type].train_mask.long())

  sampler = ImbalancedSampler(data[node_type]['label'], num_samples=3000)

  loader = NeighborLoader(data, input_nodes=input_nodes,
                          batch_size=10000, num_neighbors=[-1, -1],
                          sampler=sampler)

  return loader

def create_neighbor_loader(data, node_type):

  data[node_type].num_nodes = data[node_type].num_nodes
  data[node_type].train_mask = data[node_type].train_mask.long()

  loader = NeighborLoader(
      data,
      num_neighbors=[30] * 2,
      batch_size=128,
      input_nodes=(node_type, data[node_type].train_mask)
  )

  return loader

def create_eval_loader(data, node_type):

  data[node_type].num_nodes = data[node_type].num_nodes
  data[node_type].val_mask = data[node_type].val_mask.long()

  loader = NeighborLoader(
      data,
      num_neighbors=[30] * 2,
      batch_size=128,
     input_nodes=(node_type, data[node_type].val_mask)
  )

  return loader

def get_device():

  if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
  else:
    device = torch.device("cpu")
    print("Using CPU")

  return device

def set_seed():
    seed = 42
    torch.manual_seed(seed)

def create_model_hetroSage(num_features, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeteroGAT(num_features=num_features, num_classes=num_classes).to(device)
    return model

def define_optimizer_hetroSuper(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    return optimizer

def define_scheduler_hetroSuper(optimizer):
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    return scheduler

def define_loss_fn_hetroSuper():
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn

# Train/valid/test split
def split_data(data):
    features_entire = data['review'].feature
    labels_entire = data['review'].label

    edge_index_rsr_entire = data.edge_index_dict[("review", "net_rsr", "review")]
    edge_index_rtr_entire = data.edge_index_dict[("review", "net_rtr", "review")]
    edge_index_rur_entire = data.edge_index_dict[("review", "net_rur", "review")]

    train_mask = data['review'].train_mask
    val_mask = data['review'].val_mask
    test_mask = data['review'].test_mask

    return (features_entire, labels_entire,
            edge_index_rsr_entire, edge_index_rtr_entire, edge_index_rur_entire,
            train_mask, val_mask, test_mask)

def train_epoch_hetroSage(model, train_loader, optimizer, loss_fn):
    model.train()
    for batch in train_loader:
        subgraph = batch.to(device)
        features = subgraph['review'].feature
        labels = subgraph['review'].label

        edge_index_rsr = subgraph.edge_index_dict[("review", "net_rsr", "review")]
        edge_index_rtr = subgraph.edge_index_dict[("review", "net_rtr", "review")]
        edge_index_rur = subgraph.edge_index_dict[("review", "net_rur", "review")]

        train_mask = subgraph['review'].train_mask.to(torch.bool)
        optimizer.zero_grad()
        out = model(features, edge_index_rsr, edge_index_rtr, edge_index_rur)
        out = out.detach()
        out.requires_grad = True
        loss = loss_fn(out[train_mask], labels[train_mask])
        loss.backward()
        out.detach()
        optimizer.step()

def eval_model_hetroSage(model, eval_loader):
    model.eval()
    logits = []
    for batch in eval_loader:
        subgraph = batch.to(device)

        features = subgraph['review'].feature
        labels = subgraph['review'].label

        edge_index_rsr = subgraph.edge_index_dict[("review", "net_rsr", "review")]
        edge_index_rtr = subgraph.edge_index_dict[("review", "net_rtr", "review")]
        edge_index_rur = subgraph.edge_index_dict[("review", "net_rur", "review")]

        val_mask = subgraph['review'].val_mask.to(torch.bool)

        with torch.no_grad():
            logit = model(features, edge_index_rsr, edge_index_rtr, edge_index_rur)
            logits.append(logit)

    logits = torch.cat(logits)
    val_probs = logits[:, 1] > 0.5
    val_auc = roc_auc_score(val_mask.cpu(), val_probs.cpu())
    return val_auc

def train_model_hetroSage(model, train_loader, eval_loader, optimizer, loss_fn, num_epochs):
    best_val_auc = 0.0
    patience = 20
    early_stop_counter = 0

    for epoch in range(num_epochs):
        train_epoch_hetroSage(model, train_loader, optimizer, loss_fn)

        # Eval model
        val_auc = eval_model_hetroSage(model, eval_loader)

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping. Validation performance has not improved.")
            break

        # Update lr scheduler
        scheduler.step()

def test_model_hetroSage(model, features, edge_index_rsr_entire, edge_index_rtr_entire, edge_index_rur_entire, labels, test_mask, device):

  model.eval()
  model.to(device)

  test_logits = model(features.to(device),
                      edge_index_rsr_entire.to(device),
                      edge_index_rtr_entire.to(device),
                       edge_index_rur_entire.to(device))

  test_mask = test_mask.to(torch.bool)

  test_probs = test_logits.softmax(1)[:, 1][test_mask].detach().cpu().numpy()
  test_labels = labels[test_mask].cpu().numpy()

  test_auc = roc_auc_score(test_labels, test_probs)

  return test_auc,test_probs

def save_predictions(preds, filepath):

  data = {'Probability': preds.detach().numpy()}

  df = pd.DataFrame(data)

  df.to_csv(filepath, index=False)

def create_model_hetroRGCN(num_nodes, num_features, num_classes, num_relations):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeteroRGCN(num_nodes, num_features, num_classes, num_relations).to(device)
    return model

def train_epoch_hetroRGCN(model, train_loader, optimizer, loss_fn):
  model.train()

  for batch in train_loader:

    subgraph = batch.to(device)
    features = subgraph['review'].feature
    labels = subgraph['review'].label

    edge_index_rsr = subgraph.edge_index_dict[("review", "net_rsr", "review")]
    edge_index_rtr = subgraph.edge_index_dict[("review", "net_rtr", "review")]
    edge_index_rur = subgraph.edge_index_dict[("review", "net_rur", "review")]

    train_mask = subgraph['review'].train_mask.to(torch.bool)

    edge_index = torch.cat([edge_index_rsr, edge_index_rtr, edge_index_rur], dim=1)
    num_rsr = edge_index_rsr.shape[1]
    num_rtr = edge_index_rtr.shape[1]
    num_rur = edge_index_rur.shape[1]

    num_edges = edge_index.shape[1]

    edge_type = torch.zeros(num_edges, dtype=torch.long)

    edge_type[:num_rsr] = 0
    edge_type[num_rsr:num_rsr+num_rtr] = 1
    edge_type[num_rsr+num_rtr:] = 2

    optimizer.zero_grad()
    out = model(features, edge_index,edge_type)
    out = out.detach()
    out.requires_grad = True
    loss = loss_fn(out[train_mask], labels[train_mask])
    loss.backward()
    out.detach()
    optimizer.step()


def eval_model_hetroRGCN(model, eval_loader):
  model.eval()
  logits = []
  for batch in eval_loader:
      subgraph = batch.to(device)
      features = subgraph['review'].feature
      labels = subgraph['review'].label

      edge_index_rsr = subgraph.edge_index_dict[("review", "net_rsr", "review")]
      edge_index_rtr = subgraph.edge_index_dict[("review", "net_rtr", "review")]
      edge_index_rur = subgraph.edge_index_dict[("review", "net_rur", "review")]

      val_mask = subgraph['review'].val_mask.to(torch.bool)

      edge_index = torch.cat([edge_index_rsr, edge_index_rtr, edge_index_rur], dim=1)
      num_rsr = edge_index_rsr.shape[1]
      num_rtr = edge_index_rtr.shape[1]
      num_rur = edge_index_rur.shape[1]

      num_edges = edge_index.shape[1]

      edge_type = torch.zeros(num_edges, dtype=torch.long)

      edge_type[:num_rsr] = 0
      edge_type[num_rsr:num_rsr+num_rtr] = 1
      edge_type[num_rsr+num_rtr:] = 2


      with torch.no_grad():
          logit  = model(features, edge_index, edge_type)
          logits.append(logit)

      logits = torch.cat(logits)
      val_probs = logits[:, 1] > 0.5
      val_auc = roc_auc_score(val_mask.cpu(), val_probs.cpu().numpy())
      return val_auc
      # print(f'Val AUC: {val_auc:.4f}')


def train_model_hetroRGCN(model, train_loader, eval_loader, optimizer, loss_fn, num_epochs):

  best_val_auc = 0.0
  patience = 20
  early_stop_counter = 0

  for epoch in range(num_epochs):

    train_epoch_hetroRGCN(model, train_loader, optimizer, loss_fn)

    val_auc = eval_model_hetroRGCN(model, eval_loader)

    # Early stopping
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= patience:
        print("Early stopping.")
        break

    scheduler.step()

def test_model_hetroRGCN(model, features, edge_index_rsr, edge_index_rtr, edge_index_rur, labels, test_mask,device):

  model.eval()
  model.to(device)

  # Flatten edge indices
  edge_index = torch.cat([edge_index_rsr, edge_index_rtr, edge_index_rur], dim=1).to(device)

  # Create edge types
  num_rsr = edge_index_rsr.shape[1]
  num_rtr = edge_index_rtr.shape[1]
  num_edges = edge_index.shape[1]

  edge_type = torch.zeros(num_edges, dtype=torch.long).to(device)
  edge_type[:num_rsr] = 0
  edge_type[num_rsr:num_rsr+num_rtr] = 1
  edge_type[num_rsr+num_rtr:] = 2

  features = features.to(device)

  test_logits = model(features, edge_index, edge_type)

  test_mask = test_mask.to(torch.bool)
  test_probs = test_logits.softmax(1)[:, 1][test_mask].detach().cpu().numpy()
  test_labels = labels[test_mask].cpu().numpy()

  test_auc = roc_auc_score(test_labels, test_probs)

  return test_auc,test_probs


def create_subgraphs(subgraph):

  g_rsr = subgraph.edge_type_subgraph(['net_rsr'])
  g_rtr = subgraph.edge_type_subgraph(['net_rtr'])
  g_rur = subgraph.edge_type_subgraph(['net_rur'])

  g_rsr_nx = dgl.to_networkx(g_rsr)
  g_rtr_nx = dgl.to_networkx(g_rtr)
  g_rur_nx = dgl.to_networkx(g_rur)

  g_rsr_nx = nx.Graph(g_rsr_nx)
  g_rtr_nx = nx.Graph(g_rtr_nx)
  g_rur_nx = nx.Graph(g_rur_nx)

  return g_rsr_nx, g_rtr_nx, g_rur_nx

def calculate_topology_features(g_rsr_nx, g_rtr_nx, g_rur_nx):

  clustering_g_rsr_nx = nx.clustering(g_rsr_nx)
  clustering_g_rtr_nx = nx.clustering(g_rtr_nx)
  clustering_g_rur_nx = nx.clustering(g_rur_nx)

  betweenness_g_rtr_nx = nx.betweenness_centrality(g_rtr_nx)

  closeness_g_rsr_nx = nx.closeness_centrality(g_rsr_nx)
  closeness_g_rtr_nx = nx.closeness_centrality(g_rtr_nx)
  closeness_g_rur_nx = nx.closeness_centrality(g_rur_nx)

  eigenvector_g_rsr_nx = nx.eigenvector_centrality(g_rsr_nx)
  eigenvector_g_rur_nx = nx.eigenvector_centrality(g_rur_nx)

  triangle_count_g_rur_nx = nx.triangles(g_rur_nx)
  triangle_count_g_rtr_nx = nx.triangles(g_rtr_nx)
  triangle_count_g_rsr_nx = nx.triangles(g_rsr_nx)

  return [clustering_g_rsr_nx, clustering_g_rtr_nx, clustering_g_rur_nx,
          betweenness_g_rtr_nx,
          closeness_g_rsr_nx, closeness_g_rtr_nx, closeness_g_rur_nx,
          eigenvector_g_rsr_nx, eigenvector_g_rur_nx,
          triangle_count_g_rur_nx, triangle_count_g_rtr_nx, triangle_count_g_rsr_nx]

def add_topology_features(data, topology_features):

  nodes = list(range(len(data['review'].feature)))

  for topology_feature in topology_features:

    cluster_list = [topology_feature[n] for n in nodes]
    cluster_tensor = torch.tensor(cluster_list)
    cluster_tensor = cluster_tensor.unsqueeze(1)

    data['review'].feature = torch.cat([data['review'].feature, cluster_tensor], dim=1)

  return data

def create_structural_features(subgraph):

  data = torch_geometric.utils.convert.from_dgl(subgraph)

  g_rsr_nx, g_rtr_nx, g_rur_nx = create_subgraphs(subgraph)

  topology_features = calculate_topology_features(g_rsr_nx, g_rtr_nx, g_rur_nx)

  data = add_topology_features(data, topology_features)

  return data

def add_degree_features(data):

  # Extract edge indices
  edge_index_rsr_entire = data.edge_index_dict[("review", "net_rsr", "review")]
  edge_index_rtr_entire = data.edge_index_dict[("review", "net_rtr", "review")]
  edge_index_rur_entire = data.edge_index_dict[("review", "net_rur", "review")]

  # Calculate degrees
  deg_rsr = degree(edge_index_rsr_entire[0])
  deg_rtr = degree(edge_index_rtr_entire[0])
  deg_rur = degree(edge_index_rur_entire[0])

  # Get max length
  max_len = max(len(deg_rsr), len(deg_rtr), len(deg_rur))

  # Pad tensors
  deg_rsr = F.pad(deg_rsr, (0, max_len - len(deg_rsr)), 'constant', 0)
  deg_rtr = F.pad(deg_rtr, (0, max_len - len(deg_rtr)), 'constant', 0)
  deg_rur = F.pad(deg_rur, (0, max_len - len(deg_rur)), 'constant', 0)

  # Calculate total degree
  deg_total = deg_rsr + deg_rtr + deg_rur

  # Concatenate new features
  new_features = torch.cat([
      deg_total.view(-1,1),
      deg_rsr.view(-1,1),
      deg_rtr.view(-1,1),
      deg_rur.view(-1,1)], dim=1)

  # Add to data
  data['review'].feature = torch.cat([data['review'].feature, new_features], dim=1)

  return data


def create_spectral_clusters(data):
  # Extract features
  review_features = data['review'].feature[data['review'].train_mask & (data['review'].label==False)].detach().numpy()

  # Calculate similarity
  similarity = cosine_similarity(review_features)

  # Create graph
  similarity_graph = nx.from_numpy_array(similarity)

  # Cluster
  spectral = SpectralClustering(n_clusters=14, affinity='precomputed')
  spectral.fit(similarity)
  labels = spectral.labels_

  # Get cluster centers
  centers = [np.where(labels==i)[0][0] for i in range(spectral.n_clusters)]

  # Get cluster indices
  cluster_indices = [np.where(labels==i)[0] for i in range(spectral.n_clusters)]

  return labels,centers, cluster_indices


def sample_positive_pairs(cluster_indices, num_pairs):

  pos_pairs = []
  for cluster in cluster_indices:
    for i in range(num_pairs):
      idx1 = np.random.choice(cluster)
      idx2 = np.random.choice(cluster)
      if idx1 != idx2:
        pos_pairs.append((idx1, idx2))

  return pos_pairs

def sample_negative_pairs(cluster_indices, cluster_centers, node_features, num_pairs):

  neg_pairs = []

  for i in range(num_pairs):

    # Sample cluster
    cluster_idx = np.random.choice(len(cluster_centers))

    # Sample node
    node_indices = cluster_indices[cluster_idx]
    node_idx = np.random.choice(node_indices)

    # Get node cluster
    node_cluster_features = node_features[node_idx].detach().numpy().reshape(1,-1)
    center_features = node_features[cluster_centers].detach().numpy()
    distances = cdist(node_cluster_features, center_features)[0]

    # Get farthest cluster
    neg_cluster = np.argmax(distances)

    # Sample node from farthest cluster
    neg_cluster_indices = cluster_indices[neg_cluster]
    neg_index = np.random.choice(neg_cluster_indices)

    neg_pairs.append((node_idx, neg_index))

  return neg_pairs


def create_pairs_dataframe(dataset):

  pairs_df = []
  for i in range(len(dataset)):
      pairs_df.append(dataset[i])

  df = pd.DataFrame(pairs_df, columns=['Feature1', 'Feature2', 'Label'])
  return df

def train_model_contrastive(model, dataloader, criterion, optimizer,input_dim):

  for x1, x2, label in dataloader:

    z1 = model(x1.view(-1, input_dim))
    z2 = model(x2.view(-1, input_dim))

    loss = criterion(z1, label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  return loss

def get_cluster_centers(centers, embeddings):

  cluster_center_embeddings =[]
  for i in range(len(centers)):
    center_idx = centers[i]
    center_emb = embeddings[center_idx].detach().numpy()
    cluster_center_embeddings.append(center_emb)

  cluster_center_embeddings = np.array(cluster_center_embeddings)

  return cluster_center_embeddings

def calculate_similarity(embeddings, cluster_center_embeddings):
    sim = cosine_similarity(embeddings, cluster_center_embeddings)
    return sim

def get_closest_clusters(sim):
    closest_cluster = np.argmax(sim, axis=1)
    return closest_cluster

def calculate_distances(sim, clusters):
    distances = 1 - sim[:,clusters]
    return distances

def detect_anomalies(embeddings, cluster_center_embeddings):

    sim = cosine_similarity(embeddings, cluster_center_embeddings)

    # Get index of closest cluster for each node
    closest_cluster = np.argmax(sim, axis=1)

    # Get distances to closest cluster
    cluster_distances = 1 - sim[np.arange(len(sim)), closest_cluster]

    # Nodes with largest distance are more anomalous
    anomaly_scores = cluster_distances

    # Sort by anomaly score
    anomaly_order = np.argsort(anomaly_scores)[::-1]

    return anomaly_order

def create_anomaly_df(anomaly_order, node_labels):

  df = pd.DataFrame()

  df['node_index'] = anomaly_order
  df['labels'] = node_labels

  # Add indicator for top anomalies
  df['Probability'] = 0
  df.loc[:499, 'Probability'] = 1

  # Sort by anomaly score
  df = df.sort_values('node_index', ascending=True)
  df = df.reset_index(drop=True)

  return df

def save_predictions_array(preds, filepath):

  data = {'Probability': preds}

  df = pd.DataFrame(data)

  df.to_csv(filepath, index=False)

def extract_edges(data):

  edge_index_rsr = data.edge_index_dict[("review", "net_rsr", "review")]
  edge_index_rtr = data.edge_index_dict[("review", "net_rtr", "review")]
  edge_index_rur = data.edge_index_dict[("review", "net_rur", "review")]

  return edge_index_rsr, edge_index_rtr, edge_index_rur

def extract_node_features(data):

  features = data['review'].feature
  return features

def create_graph(edge_index, features):

  graph = Data(edge_index=edge_index, x=features)
  return graph

def add_labels_and_masks(graph, data):

  # Add labels, masks
  graph.y = data['review'].label
  graph.train_mask = data['review'].train_mask
  graph.val_mask = data['review'].val_mask
  graph.test_mask = data['review'].test_mask

  return graph
# Usage

def train_model_unsuper(model, graph):
  model.fit(graph)
  return model

def evaluate_model_unsuper(model, graph):
  pred, scores, prob, conf = model.predict(graph_rtr,
                                           return_pred=True,
                                           return_score=True,
                                           return_prob=True,
                                           return_conf=True)
  auc = eval_roc_auc(graph.y, scores)
  return auc

def tune_hyperparameters_dominant(graph, hidden_dims, num_layers, learning_rates, batch_sizes, epochs):

  best_params = None
  best_auc = 0

  for hid_dim in hidden_dims:
    for num_layer in num_layers:
      for lr in learning_rates:
        for batch_size in batch_sizes:

          model = DOMINANT(hid_dim, num_layer, lr, batch_size)

          model = train_model_unsuper(model, graph)

          auc = evaluate_model_unsuper(model, graph)

          if auc > best_auc:
            best_auc = auc
            best_params = {'hid_dim': hid_dim,
                        'num_layers': num_layers,
                        'lr': lr,
                        'batch_size': batch_size}

  return best_params

def create_model_dominant(params):
  model = DOMINANT(**params)
  return model

def evaluate_model(model, graph):
  preds, scores, probs, confs = model.predict(graph, return_pred=True,
                                           return_score=True,
                                           return_prob=True,
                                           return_conf=True)
  auc = eval_roc_auc(graph.y, scores)
  return auc, preds, scores, probs, confs

def save_predictions_unsupervised(labels, scores, probs, confs, filepath):

  data = {
    'Labels': labels,
    'Raw Scores': scores,
    'Probability': probs,
    'Confidence': confs
  }

  df = pd.DataFrame(data)

  df.to_csv(filepath, index=False)


def tune_hyperparameters_anomalyDAE(graph, hidden_dims, num_layers_list, learning_rates, batch_sizes, epochs):

  best_params = None
  best_auc = 0

  for hid_dim in hidden_dims:
      for num_layers in num_layers_list:
          for lr in learning_rates:
              for batch_size in batch_sizes:
                # for epoch in epochs_list:
                    # Create model with current hyperparameters
                    model = AnomalyDAE(hid_dim=hid_dim, num_layers=num_layers, lr=lr, batch_size=batch_size)

                    # Train the model
                    model = train_model_unsuper(model, graph)

                    auc = evaluate_model_unsuper(model, graph)

                    if auc > best_auc:
                        best_auc = auc
                        best_params = {
                            'hid_dim': hid_dim,
                            'num_layers': num_layers,
                            'lr': lr,
                            'batch_size': batch_size
                            # 'epoch':epoch
                        }

  return best_params


def load_prediction_data(path):

  dominant_rsr = pd.read_csv(path + 'Dominant1_5104_rsr.csv')
  dominant_rur = pd.read_csv(path + 'Dominant1_5104_rur.csv')
  dominant_rtr = pd.read_csv(path + 'Dominant1_5104_rtr.csv')
  AnomalyDAE1_rsr = pd.read_csv(path + 'AnomalyDAE1_5104_rsr.csv')
  AnomalyDAE1_rur = pd.read_csv(path + 'AnomalyDAE1_5104_rur.csv')
  AnomalyDAE1_rtr = pd.read_csv(path + 'AnomalyDAE1_5104_rtr.csv')
  graph_sage = pd.read_csv(path + 'graphSage1_5104_all.csv')
  contrastive = pd.read_csv(path + 'contrastive_5104_all.csv')

  return dominant_rsr,dominant_rur, dominant_rtr,AnomalyDAE1_rsr,AnomalyDAE1_rur,AnomalyDAE1_rtr,graph_sage,contrastive


def create_dataframe(graph,dominant_rsr,dominant_rur, dominant_rtr,AnomalyDAE1_rsr,AnomalyDAE1_rur,AnomalyDAE1_rtr,graph_sage):

  df = pd.DataFrame()
  df['dominant_rsr']=dominant_rsr['Probability']
  df['dominant_rur']=dominant_rur['Probability']
  df['dominant_rtr']=dominant_rtr['Probability']
  df['AnomalyDAE1_rsr']=AnomalyDAE1_rsr['Probability']
  df['AnomalyDAE1_rur']=AnomalyDAE1_rur['Probability']
  df['AnomalyDAE1_rtr']=AnomalyDAE1_rtr['Probability']
  df['graph_sage']=graph_sage['Probability']
  df['contrastive']=contrastive['Probability']
  df['test']=graph.test_mask
  df['grand_truth']=graph.y

  return df

def split_data(df):

  train_set = df[df['test'] == 0]
  
  train_set = train_set.drop('test', axis=1)
  
  return train_set

def resample_data(X, y, random_state=42):
  sm = SMOTEENN(random_state=random_state)
  X_res, y_res = sm.fit_resample(X, y)
  return X_res, y_res

def evaluate_model_lr(clf, X, y):
  predictions = clf.predict(X)
  auc = roc_auc_score(y, predictions)
  return auc

def split_test_set(df):

  test_df = df[df['test'] == 1]
  test_df = test_df.drop('test', axis=1)

  return test_df

def make_predictions(df, weights):

  cols = ['dominant_rsr','dominant_rur', 'dominant_rtr','AnomalyDAE1_rsr','AnomalyDAE1_rur','AnomalyDAE1_rtr','graph_sage','contrastive']
  
  df['weighted_prob'] = df[cols] * weights

  return df

def make_predictions_lr(df, weights):

  df['weighted_prob'] =(df['dominant_rsr']*weights['dominant_rsr']+
  df['dominant_rur']*weights['dominant_rur']+
  df['dominant_rtr']*weights['dominant_rtr']+
  df['AnomalyDAE1_rsr']*weights['AnomalyDAE1_rsr']+
  df['AnomalyDAE1_rur']*weights['AnomalyDAE1_rur']+
  df['AnomalyDAE1_rtr']*weights['AnomalyDAE1_rtr']+
  df['graph_sage']*weights['graph_sage']+
  df['contrastive']*weights['contrastive'])

  return df