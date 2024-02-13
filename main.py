import models
import utils
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from pygod.detector import AnomalyDAE
from pygod.metric import eval_roc_auc
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import torch.nn as nn

g = load_graph()
data =convert_to_pyg(g)
input_nodes = sample_nodes(g)
subgraph= create_subgraph(g,input_nodes)
data =convert_to_pyg(subgraph)


edge_indices = extract_edges(data)
features = extract_node_features(data)

graph_rsr = create_graph(edge_indices[0], features)
graph_rtr = create_graph(edge_indices[1], features)
graph_rur = create_graph(edge_indices[2], features)

graph_rsr =add_labels_and_masks(graph_rsr, data)
graph_rtr =add_labels_and_masks(graph_rtr, data)
graph_rur =add_labels_and_masks(graph_rur, data)

best_params = tune_hyperparameters_dominant(graph_rtr,[32, 64, 128], [3, 4, 5], [0.001, 0.01, 0.1], [32, 64, 128], [10, 100, 200])

model = create_model_dominant(best_params)

model = train_model_unsuper(model, graph_rsr)
auc_rsr, preds_rsr, scores_rsr, probs_rsr, confs_rsr = evaluate_model(model, graph_rsr)

save_path = '/content/drive/MyDrive/'
filename = 'Dominant1_5104_rsr.csv'
save_predictions_unsupervised(preds_rsr, scores_rsr, probs_rsr, confs_rsr, save_path + filename)

model = train_model_unsuper(model, graph_rtr)
auc_rtr, preds_rtr, scores_rtr, probs_rtr, confs_rtr = evaluate_model(model, graph_rtr)

save_path = '/content/drive/MyDrive/'
filename = 'Dominant1_5104_rtr.csv'
save_predictions_unsupervised(preds_rtr, scores_rtr, probs_rtr, confs_rtr, save_path + filename)

model = train_model_unsuper(model, graph_rur)
auc_rur, preds_rur, scores_rur, probs_rur, confs_rur = evaluate_model(model, graph_rur)

save_path = '/content/drive/MyDrive/'
filename = 'Dominant1_5104_rur.csv'
save_predictions_unsupervised(preds_rur, scores_rur, probs_rur, confs_rur, save_path + filename)


best_params_anomalyDAE = tune_hyperparameters_anomalyDAE(graph_rtr,[32, 64, 128], [3, 4, 5], [0.001, 0.01, 0.1], [32, 64, 128], [10, 100, 200])
model_anomalyDAE = AnomalyDAE(**best_params_anomalyDAE)


model_anomalyDAE = train_model_unsuper(model_anomalyDAE, graph_rsr)
auc_rsr, preds_rsr, scores_rsr, probs_rsr, confs_rsr = evaluate_model(model_anomalyDAE, graph_rsr)
save_path = '/content/drive/MyDrive/'
filename = 'AnomalyDAE1_5104_rsr.csv'
save_predictions_unsupervised(preds_rsr, scores_rsr, probs_rsr, confs_rsr, save_path + filename)


model_anomalyDAE = train_model_unsuper(model_anomalyDAE, graph_rtr)
auc_rtr, preds_rtr, scores_rtr, probs_rtr, confs_rtr = evaluate_model(model_anomalyDAE, graph_rtr)
save_path = '/content/drive/MyDrive/'
filename = 'AnomalyDAE1_5104_rtr.csv'
save_predictions_unsupervised(preds_rtr, scores_rtr, probs_rtr, confs_rtr, save_path + filename)


model_anomalyDAE = train_model_unsuper(model_anomalyDAE, graph_rur)
auc_rur, preds_rur, scores_rur, probs_rur, confs_rur = evaluate_model(model_anomalyDAE, graph_rur)
save_path = '/content/drive/MyDrive/'
filename = 'AnomalyDAE1_5104_rur.csv'
save_predictions_unsupervised(preds_rur, scores_rur, probs_rur, confs_rur, save_path + filename)


data = create_structural_features(subgraph)
data = add_degree_features(data)

data = add_edges(data)
schema = define_metapath_schema()
model = create_metapath2vec_model(data, schema)
train_methavec2path_model(model)
data= combine_structural_features(model,data,'review')
labels,centers, cluster_indices = create_spectral_clusters(data)

pos_pairs = sample_positive_pairs(cluster_indices, 10000)
features= data['review'].feature
neg_pairs = sample_negative_pairs(cluster_indices, centers, features, 150000)

final_features=data['review'].feature[data['review'].train_mask & (data['review'].label==False)].detach().numpy()
dataset = PairDataset(final_features, pos_pairs, neg_pairs)

df = create_pairs_dataframe(dataset)

dataloader = DataLoader(dataset, batch_size=32)
model_contrastive = Encoder(176, 32)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_contrastive.parameters())
loss = train_model_contrastive(model_contrastive, dataloader, criterion, optimizer,176)

all_nodes = data['review'].feature
embeddings = model_contrastive(all_nodes)

cluster_centers_embeddings = get_cluster_centers(centers, embeddings)

anomaly_order= detect_anomalies(embeddings.detach().numpy() ,cluster_centers_embeddings)

df= create_anomaly_df(anomaly_order,data['review'].label )

save_path = '/content/drive/MyDrive/'
filename = 'contrastive1_5104_all.csv'

save_predictions_array(df['Probability'], save_path + filename)


balance_loader= balanced_loader(data, 'review')
data= next(iter(balance_loader))

data['review'].num_nodes = 5104
data['review'].train_mask = data['review'].train_mask.long()
data['review'].val_mask = data['review'].val_mask.long()
data['review'].test_mask = data['review'].test_mask.long()

train_loader = create_neighbor_loader(data,'review')
eval_loader =  create_eval_loader(data,'review')

device = get_device()

# define model_hetroSage
model_hetroSage = create_model_hetroSage(num_features=160, num_classes=2)
optimizer = define_optimizer_hetroSuper(model_hetroSage)
scheduler = define_scheduler_hetroSuper(optimizer)
loss_fn = define_loss_fn_hetroSuper()
features_entire, labels_entire, edge_index_rsr_entire, edge_index_rtr_entire, edge_index_rur_entire, train_mask, val_mask, test_mask = split_data(data)

# Train model_hetroSage
num_epochs = 10
train_model_hetroSage(model_hetroSage, train_loader, eval_loader, optimizer, loss_fn, num_epochs)
hetroSage_auc,test_probs = test_model_hetroSage(model_hetroSage,features_entire,edge_index_rsr_entire, edge_index_rtr_entire, edge_index_rur_entire,labels_entire,test_mask,device)

save_path = '/content/drive/MyDrive/'
filename = 'graphSage1_5104_all.csv'

save_predictions(test_probs, save_path + filename)

# define model_hetroRGCN
model_hetroRGCN = create_model_hetroRGCN(7724, 160, 2, 3)
optimizer = define_optimizer_hetroSuper(model_hetroRGCN)
scheduler = define_scheduler_hetroSuper(optimizer)
loss_fn = define_loss_fn_hetroSuper()
features_entire, labels_entire, edge_index_rsr_entire, edge_index_rtr_entire, edge_index_rur_entire, train_mask, val_mask, test_mask = split_data(data)

train_model_hetroRGCN(model_hetroRGCN, train_loader, eval_loader, optimizer, loss_fn, num_epochs)
hetroRGCN_auc,test_probs = test_model_hetroRGCN(model_hetroRGCN,features_entire,edge_index_rsr_entire, edge_index_rtr_entire, edge_index_rur_entire,labels_entire,test_mask,device)

save_path = '/content/drive/MyDrive/'
filename = 'graphRGCN1_5104_all.csv'

save_predictions(test_probs, save_path + filename)


load_path = '/content/drive/My Drive/'
dominant_rsr,dominant_rur, dominant_rtr,AnomalyDAE1_rsr,AnomalyDAE1_rur,AnomalyDAE1_rtr,graph_sage,contrastive = load_prediction_data(load_path)

df = create_dataframe(graph_rtr,dominant_rsr,dominant_rur, dominant_rtr,AnomalyDAE1_rsr,AnomalyDAE1_rur,AnomalyDAE1_rtr,graph_sage,contrastive)

train_set = split_data(df)

X = train_set.drop(['grand_truth'], axis=1)
y = train_set['grand_truth']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,random_state=95)
X_over, y_over = resample_data(X_train, y_train)
for i in range(10):
  X_over, y_over = resample_data(X_over, y_over)

clf = LogisticRegression()
clf.fit(X_over, y_over)
val_auc = evaluate_model_lr(clf, X_val, y_val)

test_df = split_test_set(df)
weights = dict(zip(X.columns, clf.coef_[0]))
test_df = make_predictions_lr(test_df, weights)
auc = roc_auc_score(test_df['grand_truth'], test_df['weighted_prob'])