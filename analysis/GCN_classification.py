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
from utils import *


def train_test(model, data, N, num_epochs = 200):
    
    #Train model
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    trains, vals = [], []
    for epoch in range(num_epochs):
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
            PATH = os.path.join(DIR_OUTPUT, 'GCN/models/' + nowdate +'-epoch-' + str(epoch))
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


logger = LOGGER.getChild('GCN')
EMB_DIR = os.path.join(DIR_OUTPUT, 'embeddings')


###################################################################################################


if __name__ == '__main__':

    N = 10**2
    #N = 0   # N = 0 means the entire graph
    EPOCHS = 500

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
    
    
    results = {}

    # ----------------------------------------------------------------------
    #Do classification on all vectors
    logger.info('Start classification on all vectors')

    #Do test training split
    logger.info('Prepping data')
    x = torch.cat((title_vecs, abstract_vecs, fulltext_vecs),1)
    num_nodes = x.shape[0]
    train_fraction, val_fraction = 0.8, 0.1  #so 80-10-10 split
    train_mask, val_mask, test_mask = make_train_val_test_masks(num_nodes,train_fraction,val_fraction)

    #Ok, now do GCN bit
    data = Data(x = x, y = y, edge_index = edge_index.t().contiguous(), train_mask= train_mask, \
               val_mask = val_mask, test_mask = test_mask)
    logger.info('Finished prepping data')

    logger.info('Starting training')
    model = build_GCN(data)
    temp_res = train_test(model,data,N, num_epochs = EPOCHS)
    results['all'] = temp_res
    acc = temp_res['top1']
    logger.info('Finished training: acc = {:.2f}'.format(acc))



    # -----------------------------------------------------------------------
    #Do classification on all - title
    logger.info('Start classification all - title')

    #Do test training split
    logger.info('Prepping data')
    x = torch.cat((abstract_vecs, fulltext_vecs),1)
    num_nodes = x.shape[0]
    train_fraction, val_fraction = 0.8, 0.1  #so 80-10-10 split
    train_mask, val_mask, test_mask = make_train_val_test_masks(num_nodes,train_fraction,val_fraction)
    
    #Ok, now do GCN bit
    data = Data(x = x, y = y, edge_index = edge_index.t().contiguous(), train_mask= train_mask, \
               val_mask = val_mask, test_mask = test_mask)
    logger.info('Finished prepping data')

    logger.info('Starting training')
    model = build_GCN(data)
    temp_res = train_test(model,data,N,num_epochs = EPOCHS)
    results['all-title'] = temp_res
    acc = temp_res['top1']
    logger.info('Finished training: acc = {:.2f}'.format(acc))



    # -----------------------------------------------------------------------
    #Do classification on all - abstract
    logger.info('Start classification all - abstract')

    #Do test training split
    logger.info('Prepping data')
    x = torch.cat((title_vecs, fulltext_vecs),1)
    num_nodes = x.shape[0]
    train_fraction, val_fraction = 0.8, 0.1  #so 80-10-10 split
    train_mask, val_mask, test_mask = make_train_val_test_masks(num_nodes,train_fraction,val_fraction)

    #Ok, now do GCN bit
    data = Data(x = x, y = y, edge_index = edge_index.t().contiguous(), train_mask= train_mask, \
               val_mask = val_mask, test_mask = test_mask)
    logger.info('Finished prepping data')

    logger.info('Starting training')
    model = build_GCN(data)
    temp_res = train_test(model,data,N, num_epochs = EPOCHS)
    results['all-abstract'] = temp_res
    acc = temp_res['top1']
    logger.info('Finished training: acc = {:.2f}'.format(acc))



    # -----------------------------------------------------------------------
    #Do classification on all - fulltext
    logger.info('Start classification all - fulltext')

    #Do test training split
    logger.info('Prepping data')
    x = torch.cat((title_vecs, abstract_vecs),1)
    num_nodes = x.shape[0]
    train_fraction, val_fraction = 0.8, 0.1  #so 80-10-10 split
    train_mask, val_mask, test_mask = make_train_val_test_masks(num_nodes,train_fraction,val_fraction)

    #Ok, now do GCN bit
    data = Data(x = x, y = y, edge_index = edge_index.t().contiguous(), train_mask= train_mask, \
               val_mask = val_mask, test_mask = test_mask)
    logger.info('Finished prepping data')

    logger.info('Starting training')
    model = build_GCN(data)
    temp_res = train_test(model,data,N, num_epochs = EPOCHS)
    results['all-fulltext'] = temp_res
    acc = temp_res['top1']
    logger.info('Finished training: acc = {:.2f}'.format(acc))




    # -----------------------------------------------------------------------
    # SAVE
    nowdate = str(datetime.now()).split()[0]
    filename = "GCN/GCN-classification-{}.json".format(nowdate)
    with open(os.path.join(DIR_OUTPUT, filename), 'w') as fout:
        json.dump(results, fout)
