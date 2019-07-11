from arxiv_public_data.embeddings.util import load_embeddings, fill_zeros
from arxiv_public_data.config import DIR_OUTPUT, DIR_BASE, LOGGER
from arxiv_public_data.oai_metadata import load_metadata
from torch_geometric.nn import GCNConv, SGConv, AGNNConv
from torch_geometric.data import Data
import torch.nn.functional as F
from torch.nn import ModuleList
import matplotlib.pyplot as plt
import analysis.intra_citation as ia
import os, time, json, gzip, torch
import networkx as nx
import pickle as pkl
import numpy as np
from datetime import datetime


def make_train_val_test_masks(num_nodes,train_fraction,val_fraction):
    """
    
    Makes the train, validation, and testing masks.
    where each mask has length num_nodes, and each
    elements has value 1 or 0.
        
    If train_fraction = 0.8
    And val_fraction = 0.1
    Then test_fraction = 1 - 0.8 - 0.1 = 0.1
    
    INPUT: int, float, float

    OUTPUT: torch.tensor, torch.tensor, torch.tensor
    
    """
    
    train_size = int(train_fraction*num_nodes)  #so 80-10-10 split
    val_size = int(val_fraction*num_nodes)

    np.random.seed(14850)
    indicies = np.array(range(num_nodes))
    np.random.shuffle(indicies)

    train_mask = indicies[:train_size]
    train_mask = np.array([1 if i in train_mask else 0 for i in range(num_nodes)])
    train_mask = torch.tensor(train_mask, dtype=torch.uint8)

    val_mask = indicies[train_size:train_size+val_size]
    val_mask = np.array([1 if i in val_mask else 0 for i in range(num_nodes)])
    val_mask = torch.tensor(val_mask, dtype=torch.uint8)

    test_mask = indicies[train_size:train_size+val_size]
    test_mask = np.array([1 if i in test_mask else 0 for i in range(num_nodes)])
    test_mask = torch.tensor(test_mask, dtype=torch.uint8)
    
    return train_mask, val_mask, test_mask



class Net_GCN(torch.nn.Module):
    def __init__(self, data, num_layers = 2, hidden_dim = 128):
        super(Net_GCN, self).__init__()
        
        self.num_features = data.x.shape[1]
        self.num_classes = max(data.y).item() + 1   #assume each class is present in full dataset
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        #Make sizes of layers
        self.layers = ModuleList()
        sizes = [self.num_features]
        for i in range(self.num_layers-1):
            sizes.append(self.hidden_dim)
        sizes.append(self.num_classes)
        
        #Make layers
        for k in range(self.num_layers):
            self.layers.append(GCNConv(sizes[k], sizes[k+1]))
        
                             
    def forward(self,data):
        x,edge_index = data.x, data.edge_index
        
        for layer in self.layers[:-1]:
            x = layer(x,edge_index)
            x = F.relu(x)
            #x = F.dropout(x, training=self.training)  COME BACK TO THIS
            
        output_layer = self.layers[-1]
        x = output_layer(x,edge_index)
        return F.log_softmax(x, dim=1)
    
    
    
class Net_SAGE(torch.nn.Module):
    def __init__(self, data, num_layers = 2, hidden_dim = 128):
        super(Net_SAGE, self).__init__()
        
        self.num_features = data.x.shape[1]
        self.num_classes = max(data.y).item() + 1   #assume each class is present in full dataset
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        #Make sizes of layers
        self.layers = ModuleList()
        sizes = [self.num_features]
        for i in range(self.num_layers-1):
            sizes.append(self.hidden_dim)
        sizes.append(self.num_classes)
        
        #Make layers
        for k in range(self.num_layers):
            self.layers.append(SGConv(sizes[k], sizes[k+1], K = 2))
        
                             
    def forward(self,data):
        x,edge_index = data.x, data.edge_index
        
        for layer in self.layers[:-1]:
            x = layer(x,edge_index)
            x = F.relu(x)
            #x = F.dropout(x, training=self.training)  COME BACK TO THIS
            
        output_layer = self.layers[-1]
        x = output_layer(x,edge_index)
        return F.log_softmax(x, dim=1)
    

    
class Net_AGNN(torch.nn.Module):
    def __init__(self, data, num_layers = 2, hidden_dim = 16):
        super(Net_AGNN, self).__init__()
        self.num_features = data.x.shape[1]
        self.num_classes = max(data.y).item() + 1   #assume each class is present in full dataset
        self.lin1 = torch.nn.Linear(self.num_features, hidden_dim)
        
        #Do gating in the inner layers
        inner_layers = []
        for k in range(hidden_dim-1):
            inner_layers.append(AGNNConv(requires_grad=False))
        inner_layers.append(AGNNConv(requires_grad=False))
        self.layers = inner_layers
        
        self.lin2 = torch.nn.Linear(hidden_dim, self.num_classes)

    def forward(self,data):
        #x = F.dropout(data.x, training=self.training)
        x = F.relu(self.lin1(data.x))
        for layer in self.layers:
            x = layer(x, data.edge_index)
        #x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


def build_GCN(data, num_layers = 2, hidden_dim = 512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net_GCN(data, num_layers = num_layers, hidden_dim = hidden_dim).to(device)
    data = data.to(device)
    return model    


def build_SAGE(data, num_layers = 2, hidden_dim = 512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net_SAGE(data, num_layers = num_layers, hidden_dim = hidden_dim).to(device)
    data = data.to(device)
    return model 


def build_AGNN(data, num_layers = 2, hidden_dim = 512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net_AGNN(data, num_layers = num_layers, hidden_dim = hidden_dim).to(device)
    data = data.to(device)
    return model 


def in_top_n(prob, target, n=5):
    intopn = 0
    labels = np.arange(prob.shape[1])
    for p, t in zip(prob, target):
        if t in sorted(labels, key = lambda i: p[i])[-n:]:
            intopn += 1
    return intopn/prob.shape[0]    



def get_edge_index(G,metadata):
    """
    
    Graphs the edge list in format required by
    Pytorch geometric
    
    Input: nx DiGraph
    Output: torch tensor, [edge1, edge2, ...]
            where edge is [node1, node]
    
    """
    
    #Need to make sure the graph is labeled in the same way
    m_aid_order = {x['id']:i for i,x in enumerate(metadata)}
    G = nx.relabel_nodes(G, m_aid_order)

    
    #GCN breaks if there are no edges
    #So in this case, I'll link the first
    #Node to itself. This is a hack.
    if G.number_of_edges() == 0:
        return torch.tensor([[0,1],[1,2]])
    
    edge_index = []
    for node in G:
        neighbours = G[node]
        for n in neighbours:
            edge_index.append([node,n])
    edge_index = torch.tensor(edge_index)
    return edge_index


def make_ys(metadata):
    """
    Makes the labels (y in torch-geometric notation)
    """
    
    #Load in metadata
    categories = np.array([m['categories'][0].split()[0] for m in metadata],
                      dtype='object')   # ['hep-th', ...] i.e. labeled by strings  
    labels = list(set(categories))
    
    #Make class_labels class['hep=th'] = 4
    class_labels = {}
    for i, x in enumerate(labels):
        class_labels[x] = i
    
    # I need labels in form [1,3,1,17,1]
    y = torch.tensor([class_labels[cat] for cat in categories])
    return y


def shuffle(arr, seed=14850):
    """ Deterministic in-place shuffling """
    rng = np.random.RandomState(seed)
    rng.shuffle(arr)
    return arr


def sync_G_with_metadata(G, m, logger):
    """ Citation graph G is missing some articles
        (the ones for which we have no full-text data)
        So, I need to add these missing ones as isolated
        nodes into G.
        
        G = nx.Graph, citation graph
        m = meta data  (get from load_metadata function)
    """
    
    #Check if G subset of m. 
    G_nodes = set(list(G.nodes()))
    ids = set([x['id'] for x in m])
    bad_nodes = G_nodes - ids
    for node in bad_nodes:
        G.remove_node(node)
        
    #Then find missing nodes & add them
    missing = ids - G_nodes
    for node in missing:
        if not G.has_node(node):
            G.add_node(node)
    
    #Check if they're sync'd up
    if G.number_of_nodes() == len(m):
        logger.info('Graph nodes match metadata')
    else:
        logger.warning('Inconsistency between graph and metadata')
    return G


def get_train_acc(model,data):
    _, y_pred = model(data)[data.train_mask].max(dim=1)
    y_true = data.y[data.train_mask]
    acc = (y_pred == y_true).sum().item() / (1.0*data.train_mask.sum().item())
    return acc


def get_val_acc(model,data):
    _, y_pred = model(data)[data.val_mask].max(dim=1)
    y_true = data.y[data.val_mask]
    acc = (y_pred == y_true).sum().item() / (1.0*data.val_mask.sum().item())
    return acc