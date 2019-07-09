""" 

Python: version 3.7.3 
Pytorch: version 1.1.0
Pytorch geometric: version 1.2.1

"""


from arxiv_public_data.embeddings.util import load_embeddings, fill_zeros
from arxiv_public_data.config import DIR_OUTPUT, DIR_BASE, LOGGER
from arxiv_public_data.oai_metadata import load_metadata
from torch_geometric.nn import GCNConv
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

    test_mask = indicies[train_size+val_size:]
    test_mask = np.array([1 if i in test_mask else 0 for i in range(num_nodes)])
    test_mask = torch.tensor(test_mask, dtype=torch.uint8)
    
    return train_mask, val_mask, test_mask


class Net(torch.nn.Module):
    def __init__(self, data, num_layers = 2, hidden_dim = 128):
        super(Net, self).__init__()
        
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
            #x = F.dropout(x, training=self.training)  #COME BACK TO THIS
            
        output_layer = self.layers[-1]
        x = output_layer(x,edge_index)
        return F.log_softmax(x, dim=1)


def build_GCN(data, num_layers = 2, hidden_dim = 512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(data, num_layers = num_layers, hidden_dim = hidden_dim).to(device)
    data = data.to(device)
    return model    
    
    

def in_top_n(prob, target, n=5):
    intopn = 0
    labels = np.arange(prob.shape[1])
    for p, t in zip(prob, target):
        if t in sorted(labels, key = lambda i: p[i])[-n:]:
            intopn += 1
    return intopn/prob.shape[0]    


def train_test(model, data):
    
    #Train model
    model.train()
    num_epochs = 500
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-8)
    trains, vals = [], []
    for epoch in range(1,num_epochs+1):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step() 
        
        #Check the accurarcy
        acc_train, acc_val = get_train_acc(model,data), get_val_acc(model,data)
        trains.append(acc_train);vals.append(acc_val)
        if epoch % 50 == 0:
            print('(epoch, acc_train, acc_val) = ({}, {:.2f}, {:.2f})'.format(epoch,acc_train,acc_val))

            #Save model
            nowdate = str(datetime.now()).split()[0]
            PATH = os.path.join(DIR_OUTPUT, 'GCN/models/{}-epoch-{}'.format(N,epoch))
            torch.save(model.state_dict(), PATH)
            
    #Save stuff
    nowdate = str(datetime.now()).split()[0]
    plt.plot(trains);plt.plot(vals)
    plt.hlines(1,0,num_epochs,linestyle='dashed')
    plt.savefig(os.path.join(DIR_OUTPUT,'GCN/test_stats/{}-acc-N-{}.png'.format(nowdate,N)))
    plt.close()
        
    #Find acc -- note, the GCN acts on the entire dataset
    #So you have to pull out the test part manually
    model.eval() 
    log_prob = model(data)[data.test_mask]        #the model returns the log prob
    log_prob_y_pred, y_pred = log_prob.max(dim=1) 
    y_true = data.y[data.test_mask]
    acc = (y_pred == y_true).sum().item() / (1.0*data.test_mask.sum().item())

    #Find acc in top3 and top5
    acc_top3 = in_top_n(log_prob,y_true, 3)
    acc_top5 = in_top_n(log_prob,y_true, 5)

    #Loglikligood
    loglikelihood = log_prob_y_pred.sum().item()
    perplexity = 2 ** ( - loglikelihood / len(y_true) / np.log(2))
    
    return dict(top1=acc,top3=acc_top3, top5=acc_top5, \
                loglikelihood=loglikelihood,perplexity=perplexity)


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


def sync_G_with_metadata(G, m):
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


logger = LOGGER.getChild('kipf-welling')
EMB_DIR = os.path.join(DIR_OUTPUT, 'embeddings')


###################################################################################################


if __name__ == '__main__':

    N = 10**2
    #N = 0   # N = 0 means the entire graph

    #Make directories
    PATH = os.path.join(DIR_OUTPUT, 'GCN')
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
        G = sync_G_with_metadata(G, metadata)   

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


    results = {}
    # -----------------------------------------------------------------------
    #Do classification on just title vectors
    logger.info('Start classification on just title vectors')
    TITLE_FILE = os.path.join(EMB_DIR, 'title-embedding-usel-2019-03-19.pkl')
    logger.info('Loading title vectors')
    title_vecs = shuffle(load_embeddings(TITLE_FILE)['embeddings'])
    if N != 0:
        title_vecs = title_vecs[:N]
    title_vecs = torch.tensor(title_vecs)
    logger.info('Finished loading title vectors')

    #Do test training split
    logger.info('Prepping data')
    x = title_vecs
    num_nodes = x.shape[0]
    train_fraction, val_fraction = 0.8, 0.1  #so 80-10-10 split
    train_mask, val_mask, test_mask = make_train_val_test_masks(num_nodes,train_fraction,val_fraction)

    #Ok, now do GCN bit
    data = Data(x = x, y = y, edge_index = edge_index.t().contiguous(), train_mask= train_mask, \
               val_mask = val_mask, test_mask = test_mask)
    logger.info('Finished prepping data')

    logger.info('Starting training')
    model = build_GCN(data)
    temp_res = train_test(model,data)
    results['title'] = temp_res
    acc = temp_res['top1']
    logger.info('Finished training: acc = {:.2f}'.format(acc))



    # -----------------------------------------------------------------------
    #Do classification on just abstract vectors
    logger.info('Start classification on just abstract vectors')
    ABSTRACT_FILE = os.path.join(EMB_DIR, 'abstract-embedding-usel-2019-03-19.pkl')
    logger.info('Loading abstract vectors')
    abstract_vecs = shuffle(load_embeddings(ABSTRACT_FILE)['embeddings'])
    if N != 0:
        abstract_vecs = abstract_vecs[:N]
    abstract_vecs = torch.tensor(abstract_vecs)
    logger.info('Finished loading abstract vectors')

    #Do test training split
    logger.info('Prepping data')
    x = abstract_vecs

    #Now do GCN bit
    data = Data(x = x, y = y, edge_index = edge_index.t().contiguous(), train_mask= train_mask, \
               val_mask = val_mask, test_mask = test_mask)
    logger.info('Finished prepping data')

    logger.info('Starting training')
    model = build_GCN(data)
    temp_res = train_test(model,data)
    results['abstract'] = temp_res
    acc = temp_res['top1']
    logger.info('Finished training: acc = {:.2f}'.format(acc))



    # -----------------------------------------------------------------------
    #Do classification on just fulltext vectors
    logger.info('Start classification on just fulltext vectors')
    FULLTEXT_FILE = os.path.join(
        EMB_DIR, 'fulltext-embedding-usel-2-headers-2019-04-05.pkl')
    logger.info('Loading fulltext vectors')
    fulltext_vecs = shuffle(fill_zeros(load_embeddings(FULLTEXT_FILE, 2)))
    if N != 0:
        fulltext_vecs = fulltext_vecs[:N]
    fulltext_vecs = torch.tensor(fulltext_vecs)
    fulltext_vecs = fulltext_vecs.float()
    logger.info('Finished loading fulltext vectors')

    logger.info('Prepping data')
    x = fulltext_vecs
    data = Data(x = x, y = y, edge_index = edge_index.t().contiguous(), train_mask= train_mask, \
               val_mask = val_mask, test_mask = test_mask)
    logger.info('Finished prepping data')

    logger.info('Starting training')
    model = build_GCN(data)
    temp_res = train_test(model,data)
    results['fulltext'] = temp_res
    acc = temp_res['top1']
    logger.info('Finished training: acc = {:.2f}'.format(acc))



    # -----------------------------------------------------------------------
    #Do classification on all vectors
    logger.info('Start classification on all vectors')

    #Do test training split
    logger.info('Prepping data')
    x = torch.cat((title_vecs, abstract_vecs, fulltext_vecs),1)

    #Ok, now do GCN bit
    data = Data(x = x, y = y, edge_index = edge_index.t().contiguous(), train_mask= train_mask, \
               val_mask = val_mask, test_mask = test_mask)
    logger.info('Finished prepping data')

    logger.info('Starting training')
    model = build_GCN(data)
    temp_res = train_test(model,data)
    results['all'] = temp_res
    acc = temp_res['top1']
    logger.info('Finished training: acc = {:.2f}'.format(acc))



    # -----------------------------------------------------------------------
    #Do classification on all - title
    logger.info('Start classification all - title')

    #Do test training split
    logger.info('Prepping data')
    x = torch.cat((abstract_vecs, fulltext_vecs),1)

    #Ok, now do GCN bit
    data = Data(x = x, y = y, edge_index = edge_index.t().contiguous(), train_mask= train_mask, \
               val_mask = val_mask, test_mask = test_mask)
    logger.info('Finished prepping data')

    logger.info('Starting training')
    model = build_GCN(data)
    temp_res = train_test(model,data)
    results['all-title'] = temp_res
    acc = temp_res['top1']
    logger.info('Finished training: acc = {:.2f}'.format(acc))



    # -----------------------------------------------------------------------
    #Do classification on all - abstract
    logger.info('Start classification all - abstract')

    #Do test training split
    logger.info('Prepping data')
    x = torch.cat((title_vecs, fulltext_vecs),1)

    #Ok, now do GCN bit
    data = Data(x = x, y = y, edge_index = edge_index.t().contiguous(), train_mask= train_mask, \
               val_mask = val_mask, test_mask = test_mask)
    logger.info('Finished prepping data')

    logger.info('Starting training')
    model = build_GCN(data)
    temp_res = train_test(model,data)
    results['all-abstract'] = temp_res
    acc = temp_res['top1']
    logger.info('Finished training: acc = {:.2f}'.format(acc))



    # -----------------------------------------------------------------------
    #Do classification on all - fulltext
    logger.info('Start classification all - fulltext')

    #Do test training split
    logger.info('Prepping data')
    x = torch.cat((title_vecs, abstract_vecs),1)

    #Ok, now do GCN bit
    data = Data(x = x, y = y, edge_index = edge_index.t().contiguous(), train_mask= train_mask, \
               val_mask = val_mask, test_mask = test_mask)
    logger.info('Finished prepping data')

    logger.info('Starting training')
    model = build_GCN(data)
    temp_res = train_test(model,data)
    results['all-fulltext'] = temp_res
    acc = temp_res['top1']
    logger.info('Finished training: acc = {:.2f}'.format(acc))




    # -----------------------------------------------------------------------
    # SAVE
    nowdate = str(datetime.now()).split()[0]
    filename = "GCN/GCN-classification-{}.json".format(nowdate)
    with open(os.path.join(DIR_OUTPUT, filename), 'w') as fout:
        json.dump(results, fout)
