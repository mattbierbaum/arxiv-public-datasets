""" 

VGAE link prediction. Code adapted from

https://github.com/rusty1s/pytorch_geometric/blob/master/examples/autoencoder.py

Python: version 3.7.3 
Pytorch: version 1.1.0
Pytorch geometric: version 1.2.1

"""

from arxiv_public_data.embeddings.util import load_embeddings, fill_zeros
from arxiv_public_data.config import DIR_OUTPUT, DIR_BASE, LOGGER
from arxiv_public_data.oai_metadata import load_metadata
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch.nn import ModuleList
import matplotlib.pyplot as plt
import analysis.intra_citation as ia
import os, time, json, gzip, torch, argparse, math
import networkx as nx
import pickle as pkl
import numpy as np
from datetime import datetime
from utils import *



class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        if args.model in ['GAE']:
            self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)
        elif args.model in ['VGAE']:
            self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
            self.conv_logvar = GCNConv(
                2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        if args.model in ['GAE']:
            return self.conv2(x, edge_index)
        elif args.model in ['VGAE']:
            return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)


def sample_negative_edges(edge_index, num_negative_edges, MAXITER=500):
    """ 
    Randomly pick edges from all possible edges (of which there are N*(N-1) / 2)
    until you have the desired number of negative edges (a negative edge being
    a pair of nodes WITHOUT an edge between them)
    
    While this method of choosing negative edges is potentially time consuming
    in a dense graph -- because the odds of selecting a negative edge is small
    when there are lots of positive edges -- it has the benefit of being space 
    efficient; in a sparse graph, the total number of negative elements is 
    large -- O(N^2) -- which is difficult to store.
    
    Input: edge_index, torch tensor in format required by pytorch geometric
           num_negative_edges = int, the number of neg edges you want 
           MAXITER = int, the total number of times a 
           
    Output: neg_edged = torch.tensor([e1,e2,e3,..], [e1,e2,e3,...] )
    
    """

    num_nodes = edge_index.shape[1]
    nodes = range(num_nodes)
    neg_row = np.zeros(num_negative_edges)
    neg_col = np.zeros(num_negative_edges)

    for i in range(num_negative_edges):

        #sample from all possible edges  (will break if you have the complete graph...)
        isEdge = True
        ctr = 0
        while isEdge and ctr <= MAXITER:
            r,c = np.random.choice(nodes,2,replace = False)  

            #check it is a negative edge
            isEdge = (edge_index[0] == r).any() and (edge_index[1] == c).any()
            ctr += 1

        neg_row[i], neg_col[i] = r,c
        ctr = 0
        
        if ctr > MAXITER:
            print('Failed to find permissible negative edge; max iterations reached')

    neg_edges=torch.stack([torch.tensor(neg_row,dtype=torch.int64), \
                           torch.tensor(neg_col,dtype=torch.int64)])
    return neg_edges


def split_edges_custom(data, val_ratio=0.05, test_ratio=0.1):
        """
        Home made replacement of model.split_edges
        """

        assert 'batch' not in data  # No batch-mode.

        row, col = data.edge_index

        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]

        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))

        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]

        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)

        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

        # Negative edges.
        num_negative_edges = n_v + n_t
        neg_edges = sample_negative_edges(data.edge_index, num_negative_edges)
        data.val_neg_edge_index = neg_edges[:,:n_v]
        data.test_neg_edge_index = neg_edges[:,n_v:n_v+n_t]
        
        #Remove old index
        data.edge_index = None
        
        assert data.test_neg_edge_index.shape == data.test_pos_edge_index.shape
        
        return data
    

def train(epoch):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    if args.model in ['VGAE']:
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    
    #Save model
    if epoch % 50 == 0:
        nowdate = str(datetime.now()).split()[0]
        PATH = os.path.join(DIR_OUTPUT, 'VGAE/models/' + nowdate +'-epoch-' + str(epoch))
        torch.save(model.state_dict(), PATH)


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)
    
    
#dataset = Planetoid(root='/tmp/Cora', name='Cora')
#data = dataset[0]
torch.manual_seed(14850)
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GAE')
parser.add_argument('--dataset', type=str, default='Cora')
args = parser.parse_args()
assert args.model in ['GAE', 'VGAE']
assert args.dataset in ['Cora', 'CiteSeer', 'PubMed']
kwargs = {'GAE': GAE, 'VGAE': VGAE}

logger = LOGGER.getChild('VGAE')
EMB_DIR = os.path.join(DIR_OUTPUT, 'embeddings')


###################################################################################################


if __name__ == '__main__':
                                                                 
    #Load data
    N = 10**5
    #N = 0   # N = 0 means the entire graph
    EPOCHS = 400

    #Make directories
    PATH = os.path.join(DIR_OUTPUT, 'VGAE')
    if not os.path.exists(PATH):
        os.mkdir(PATH)
        os.mkdir(os.path.join(PATH,'test_stats'))
        os.mkdir(os.path.join(PATH,'models'))

    #Load graph and metadata
    logger.info('Loading metadata')
    metadata = shuffle(np.array(load_metadata(), dtype='object'))
    logger.info('Finished loading metadata')


    logger.info('Loading graph')
    graph_saved = False    #for debugging, I've already made the full graph
    if graph_saved == True:
        dirname = 'data'
        G = nx.read_gpickle(dirname + '/full_graph.gpickle')
    else:
        GRAPH_FILE = os.path.join(DIR_OUTPUT, 'internal-citations.json.gz')
        G = ia.makegraph(json.load(gzip.open(GRAPH_FILE, 'rt', encoding='utf-8')))
        # G is missing some articles (see func for more details)
        G = sync_G_with_metadata(G, metadata, logger)   

    #Select subgraph if specified
    if N != 0:
        metadata = metadata[:N]  #has been shuffled, so just take first n
        node_strings = np.array([i['id'] for i in metadata])
        G = nx.subgraph(G, node_strings)

    #covert to format we need & then delete G
    edge_index = get_edge_index(G,metadata)
    del G
    logger.info('Finished loading graph: N = {}'.format(N))
    
    #Load labels
    logger.info('Loading labels')
    y = make_ys(metadata)
    logger.info('Finished loading labels')

  
    #Load titles
    TITLE_FILE = os.path.join(EMB_DIR, 'title-embedding-usel-2019-03-19.pkl')
    logger.info('Loading title vectors')
    title_vecs = shuffle(load_embeddings(TITLE_FILE)['embeddings'])
    if N != 0:
        title_vecs = title_vecs[:N]
    title_vecs = torch.tensor(title_vecs)
    logger.info('Finished loading title vectors')
    
    #Load abstracts
    ABSTRACT_FILE = os.path.join(EMB_DIR, 'abstract-embedding-usel-2019-03-19.pkl')
    logger.info('Loading abstract vectors')
    abstract_vecs = shuffle(load_embeddings(ABSTRACT_FILE)['embeddings'])
    if N != 0:
        abstract_vecs = abstract_vecs[:N]
    abstract_vecs = torch.tensor(abstract_vecs)
    logger.info('Finished loading abstract vectors')
    
    #Load fulltext
    FULLTEXT_FILE = os.path.join(
        EMB_DIR, 'fulltext-embedding-usel-2-headers-2019-04-05.pkl')
    logger.info('Loading fulltext vectors')
    fulltext_vecs = shuffle(fill_zeros(load_embeddings(FULLTEXT_FILE, 2)))
    if N != 0:
        fulltext_vecs = fulltext_vecs[:N]
    fulltext_vecs = torch.tensor(fulltext_vecs)
    fulltext_vecs = fulltext_vecs.float()
    logger.info('Finished loading fulltext vectors')
    
    
                                                                 
    # ----------------------------------------------------------------------
    #Do link prediction with features  = title + abstract + fulltext
    logger.info('Starting link prediction ')

    #Do test training split
    logger.info('Prepping data')
    x = torch.cat((title_vecs, abstract_vecs, fulltext_vecs),1)
    num_nodes = x.shape[0]
    train_fraction, val_fraction = 0.8, 0.1  #so 80-10-10 split
    train_mask, val_mask, test_mask = make_train_val_test_masks(num_nodes,train_fraction,val_fraction)

    #Put data into format for torch
    data = Data(x = x, y = y, edge_index = edge_index.t().contiguous(), train_mask= train_mask, \
               val_mask = val_mask, test_mask = test_mask)
    logger.info('Finished prepping data')
                                                                 

    logger.info('Starting training')
    #Build model
    channels = 16
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = kwargs[args.model](Encoder(data.num_features, channels)).to(dev)
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    #data = model.split_edges(data)
    data = split_edges_custom(data)
    x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    #Train
    results = {}
    for epoch in range(1, EPOCHS+1): #401
        train(epoch)
        auc_test, ap_test = test(data.test_pos_edge_index, data.test_neg_edge_index)
        auc_val, ap_val = test(data.val_pos_edge_index, data.val_neg_edge_index)
        #auc_val, auc_test = test(data.test_pos_edge_index, data.test_neg_edge_index)
        print('Epoch: {:03d}, AUC_test: {:.4f}, AUC_val: {:.4f}'.format(epoch, auc_test, auc_val))
    logger.info('Finished training')
    results['AUC'], results['AP'] = auc_test, ap_test


    # -----------------------------------------------------------------------
    # SAVE
    nowdate = str(datetime.now()).split()[0]
    filename = "VGAE/VGAE-link-prediction-N-{}-{}.json".format(N,nowdate)
    with open(os.path.join(DIR_OUTPUT, filename), 'w') as fout:
        json.dump(results, fout)
     
        
 # ----------------------------------------------------------------------

    #Do featureless link prediction 
    logger.info('Starting featureless link prediction ')
    x = torch.cat((title_vecs, abstract_vecs, fulltext_vecs),1)
    data = Data(x = x, y = y, edge_index = edge_index.t().contiguous(), train_mask= train_mask, \
               val_mask = val_mask, test_mask = test_mask)
    data.x = torch.eye(data.x.shape[0], data.x.shape[1])   #set features to identity matrix                                    

    logger.info('Starting training')
    #Build model
    channels = 16
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = kwargs[args.model](Encoder(data.num_features, channels)).to(dev)
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    #data = model.split_edges(data)
    data = split_edges_custom(data)
    x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    #Train
    results = {}
    for epoch in range(1, EPOCHS+1): #401
        train(0)  #set to zero so I don't overwrite the saved models
        auc_test, ap_test = test(data.test_pos_edge_index, data.test_neg_edge_index)
        auc_val, ap_val = test(data.val_pos_edge_index, data.val_neg_edge_index)
        #auc_val, auc_test = test(data.test_pos_edge_index, data.test_neg_edge_index)
        print('Epoch: {:03d}, AUC_test: {:.4f}, AUC_val: {:.4f}'.format(epoch, auc_test, auc_val))
    logger.info('Finished training')
    results['AUC'], results['AP'] = auc_test, ap_test


    # -----------------------------------------------------------------------
    # SAVE
    nowdate = str(datetime.now()).split()[0]
    filename = "VGAE/VGAE-link-prediction-N-{}-featureless-{}.json".format(N,nowdate)
    with open(os.path.join(DIR_OUTPUT, filename), 'w') as fout:
        json.dump(results, fout)
