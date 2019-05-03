""" Put arXiv data into right form for the GCN """

import os
import numpy as np
import networkx as nx
import time, json, gzip
import pickle as pkl

import analysis.intra_citation as ia
from arxiv_public_data.oai_metadata import load_metadata
from arxiv_public_data.embeddings.util import load_embeddings, fill_zeros
from arxiv_public_data.config import LOGGER, DIR_BASE, DIR_OUTPUT

logger = LOGGER.getChild('kipf-welling')
EMB_DIR = os.path.join(DIR_OUTPUT, 'embeddings')

# Location of saved data for Kipf-Welling
SAVE_DIR = os.path.join(DIR_OUTPUT, 'kipf-welling', 'data')
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def shuffle(arr, seed=14850):
    """ Deterministic in-place shuffling """
    rng = np.random.RandomState(seed)
    rng.shuffle(arr)
    return arr

def onehot(labels, label_order):
    """ 
    Encode labels into one-hot unit vectors
    Parameters
    ----------
        labels : list
            list of string labels to be encoded
        label_order : list
            unique list of labels to determine unique mapping
    Returns
    -------
        labels_categorical : array_like
            one-hot vectors of shape (len(labels, len(label_order)))
    Examples
    --------
    >>> onehot(['hep-th', 'hep-th', 'hep-ex'], ['hep-th', 'hep-ex'])
    array([[1, 0],
           [1, 0],
           [0, 1]])
    """
    class_labels = {}
    for i,x in enumerate(label_order):
        class_labels[x] = i
    
    labels_categorical = np.zeros((len(labels), len(label_order)))
    for i, label in enumerate(labels):
        labels_categorical[i, class_labels[label]]
    return labels_categorical

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

def save_data(G, nodes_int, nodes_string, vector_label, vector_train, 
              vector_test, vector, labels_train, labels_test, onehot_labels,
              cutoff1, cutoff2):
    """ 
    Saves data in format required by Kipfs and Welling
    
        nodes_int = list, list of nodes labeled by integers
        dirname = string, where to save the data
        vector_type = string, = 'title', 'abstract', 'full-text'
    """
    FNAME_TEMPLATE = os.path.join(
        SAVE_DIR, 'ind.arXiv-{}.{{}}'.format(vector_label)
    )
    with open(FNAME_TEMPLATE.format('x'), 'wb') as fout:
        pkl.dump(vector_train, fout, protocol=4)

    with open(FNAME_TEMPLATE.format('tx'), 'wb') as fout:
        pkl.dump(vector_test, fout, protocol=4)

    with open(FNAME_TEMPLATE.format('allx'), 'wb') as fout:
        pkl.dump(vector[:cutoff2], fout, protocol=4)

    with open(FNAME_TEMPLATE.format('y'), 'wb') as fout:
        pkl.dump(labels_train, fout, protocol=4)

    with open(FNAME_TEMPLATE.format('ty'), 'wb') as fout:
        pkl.dump(labels_test, fout, protocol=4)

    with open(FNAME_TEMPLATE.format('ally'), 'wb') as fout:
        pkl.dump(onehot_labels[:cutoff2], fout, protocol=4)

    test_nodes = nodes_int[cutoff2:]
    with open(FNAME_TEMPLATE.format('test.index'), 'wt') as f:
        for node in test_nodes:
            f.write(str(node))
            f.write('\n')
    
    #Save graph in format required by Kipf-Welling -- nodes labeled as ints
    #Also, need to save in same order as metadata
    #COLIN: I relabeled the nodes to matche metadata before sending here
    
    #Save 
    graph_dict = {}     
    for node in G.nodes():  
        graph_dict[node] = list(G.neighbors(node))
    pkl.dump(
        graph_dict, open(FNAME_TEMPLATE.format('graph'), 'wb'), 
        protocol=4
    )

def cast_data_into_right_form(N):
    """
    Puts the arXivdata into right form for kipf Welling GCN classifier.
    
    N = int, number of nodes in citation graph
             N = 0 means take all the nodes
        
    """
    
    #Meta data -- need this later
    logger.info('Loading metadata')
    metadata = np.array(load_metadata(), dtype='object')
    shuffle(metadata)
    md_aid_order = {m.get('id'): i for i, m in enumerate(metadata)}
    logger.info('Finished loading metadata')
    
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
        comps = nx.weakly_connected_components(G)
        biggest = max(comps, key=len)
        G_cc = G.subgraph(biggest)
        nodes = list(G_cc.nodes())[:N]
        G_sub = G_cc.subgraph(nodes)

        slicer = np.s_[np.array([md_aid_order[n] for n in nodes])]
        metadata = metadata[slicer]
        logger.info('Using {} nodes'.format(len(nodes)))
    else:
        slicer = np.s_[:]
        G_sub = G
        
    G_sub = G
        
    nodes_string = [m['id'] for m in metadata]

    # NOTE: label nodes by shuffled index order, will match metadata
    nodes_relabel = {l: i for i, l in enumerate(nodes_string)}
    nx.relabel_nodes(G_sub, nodes_relabel, copy=False)
    nodes_int = np.arange(len(nodes_string))
    shuffle(nodes_int)
    logger.info('Finished loading graph')

    TITLE_FILE = os.path.join(EMB_DIR, 'title-embedding-usel-2019-03-19.pkl')

    logger.info('Loading title vectors')
    title_vecs = load_embeddings(TITLE_FILE)['embeddings'][slicer]
    shuffle(title_vecs)
    logger.info('Finished loading title vectors')

    ABSTRACT_FILE = os.path.join(
        EMB_DIR, 'abstract-embedding-usel-2019-03-19.pkl'
    )

    logger.info('Loading abstract vectors')
    abstract_vecs = load_embeddings(ABSTRACT_FILE)['embeddings'][slicer]
    shuffle(abstract_vecs)
    logger.info('Finished loading abstract vectors')

    #fulltext_vecs = load_fulltext(G, metadata, nodes_string, dirname)
    FULLTEXT_FILE = os.path.join(
        EMB_DIR, 'fulltext-embedding-usel-2-headers-2019-04-05.pkl'
    )

    logger.info('Loading fulltext vectors')
    fulltext_vecs = fill_zeros(load_embeddings(FULLTEXT_FILE, 2))[slicer]
    shuffle(fulltext_vecs)
    logger.info('Finished loading fulltext vectors')
    
    #Load labels
    categories = [m['categories'][0].split()[0] for m in metadata]
    shuffle(categories)
    labels = list(set(categories))
    onehot_labels = onehot(categories, labels)
    logger.info('finished loading metadata')

    #dirname = '/home/kokeeffe/research/arxiv-public-datasets/arxiv-data'
    #labels_cat = load_labels(nodes_string, m)
    #t2 = time.time()
    #print( 'Loading features & labels took ' + str((t2-t1)/60.0) + ' mins')
    
    #Split into test & train & ulabeled portion
    #For now, I'll assume that nothing is unlabeled
    #That means cutoff1 and cutoff2 are the same
    #t1 = time.time()
   
    #Shuffle
    #indicies = list(range(len(title_vecs)))
    #np.random.shuffle(indicies)
    #title_vecs = title_vecs[indicies]
    #abstract_vecs = abstract_vecs[indicies]
    #fulltext_vecs = fulltext_vecs[indicies] 
    #nodes_int = nodes_int[indicies]
    #nodes_string = nodes_string[indicies]

    #Split
    cutoff1 = 1200000  # int(0.9*title_vecs.shape[0]) 
    cutoff2 = 1200000  # int(0.9*title_vecs.shape[0])
    title_vec_train, title_vec_test = title_vecs[:cutoff1], title_vecs[cutoff2:]
    abstract_vec_train, abstract_vec_test = abstract_vecs[:cutoff1], abstract_vecs[cutoff2:]
    fulltext_vec_train, fulltext_vec_test = fulltext_vecs[:cutoff1], fulltext_vecs[cutoff2:]
    labels_train, labels_test = onehot_labels[:cutoff1], onehot_labels[cutoff2:]

    #Save data
    save_data(
        G_sub, nodes_int, nodes_string, 'title', title_vec_train, title_vec_test, title_vecs,
        labels_train, labels_test, onehot_labels, cutoff1, cutoff2,
    )
    
    save_data(
        G_sub, nodes_int, nodes_string, 'abstract', abstract_vec_train, abstract_vec_test, 
        abstract_vecs, labels_train, labels_test, onehot_labels, cutoff1, cutoff2
    )
    
    save_data(
        G_sub, nodes_int, nodes_string, 'fulltext', fulltext_vec_train, fulltext_vec_test, 
        fulltext_vecs,labels_train, labels_test, onehot_labels, cutoff1, cutoff2
    )
    
    #Combine all
    vecs = np.concatenate([title_vecs, abstract_vecs, fulltext_vecs], axis=1)
    vecs_train, vecs_test = vecs[:cutoff1], vecs[cutoff2:]
    save_data(
        G_sub, nodes_int, nodes_string, 'all', vecs_train, vecs_test, vecs, 
        labels_train, labels_test, onehot_labels, cutoff1, cutoff2,
    )
    logger.info('Finished saving data')
