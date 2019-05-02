""" Put arXiv data into right form for the GCN """

import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import intra_citation as ia 
import time, json, gzip
import pickle as pkl
from arxiv_public_data.oai_metadata import load_metadata
from arxiv_public_data.embeddings.util import load_embeddings, fill_zeros



def sync_G_with_metadata(G,m):
    """ Citation graph G is missing some articles
        (the ones for which we have no full-text data)
        So, I need to add these missing ones as isolated
        nodes into G.
        
        G = nx.Graph, citation graph
        m = meta data  (get from load_metadata function)
    """
    
    #Check if G subset of m. 
    #It should be; if not, remove the bad noes
    G_nodes = list(G.nodes())
    ids = [x['id'] for x in m]
    bad_nodes = set(G_nodes) - set(ids)
    for node in bad_nodes:
        G.remove_node(node)
        
    #Then find missing nodes & add them
    missing = set(ids) - set(G_nodes)
    for node in missing:
        if not G.has_node(node):
            G.add_node(node)
    
    #Check if they're sync'd up
    print('(# Nodes in G, # articles in meta-data) = ' + str((G.number_of_nodes(),len(m))))
    if G.number_of_nodes() == len(m):
        print('So the syncing worked')
    else:
        print('So the syncing did not work')
    return G


def clean_labels(labels):
    """ Some labels have multiple listings
        so I take the first one
        
        Input: list of strings
    """

    for i,label in enumerate(labels):

        #If multiple listings, take first
        label = label[0].split()[0]

        #Merge sub-classes
        label = label[:label.find('.')]

        labels[i] = label
    return labels


def labels2categorical(labels):
    """ labels are strings -- have form
        'hep-th' -- So need to covert to
        categoricals
    """
        
    #Create mapping
    classes = set(labels)
    class_labels = {}
    for i,x in enumerate(classes):
        class_labels[x] = i
    class_labels
    
    #change
    labels_categorical = []
    for label in labels:
        vec = np.zeros(len(classes))
        temp = class_labels[label]
        vec[temp] = 1
        labels_categorical.append(vec)
    return np.array(labels_categorical)


def load_titles(G, m, nodes_string, dirname):
    
    #Load the full feature matrix
    filename = dirname + '/title-embedding-usel-2019-03-19.pkl'
    title_vecs =  load_embeddings(filename)['embeddings']
    
    #Then select the subset corresponding to the sub-graph we're examining
    if len(nodes_string) == G.number_of_nodes():
        return title_vecs
    else:
        indicies = []
        all_nodes = [x['id'] for x in m]
        for i,node in enumerate(nodes_string):
            index = all_nodes.index(node)
            indicies.append(index)
        return title_vecs[indicies]


def load_abstracts(G, m, nodes_string, dirname):
    
    #Load the full feature matrix
    filename = dirname + '/abstract-embedding-usel-2019-03-19.pkl'
    abstract_vecs =  load_embeddings(filename)['embeddings']
    
    #Then select the subset corresponding to the sub-graph we're examining
    if len(nodes_string) == G.number_of_nodes():
        return abstract_vecs
    else:
        indicies = []
        all_nodes = [x['id'] for x in m]
        for i,node in enumerate(nodes_string):
            index = all_nodes.index(node)
            indicies.append(index)
        return abstract_vecs[indicies]


def load_fulltext(G, m, nodes_string, dirname):
    
    #Load the full feature matrix
    dirname = '/home/kokeeffe/research/arxiv-public-datasets/arxiv-data/output/embeddings'
    filename = dirname + '/fulltext-embedding-usel-2-headers-2019-04-05.pkl'
    fulltext_vecs = fill_zeros(load_embeddings(filename, 2))

    #Then select the subset corresponding to the sub-graph we're examining
    if len(nodes_string) == G.number_of_nodes():
        return fulltext_vecs
    else:
        indicies = []
        all_nodes = [x['id'] for x in m]
        for i,node in enumerate(nodes_string):
            index = all_nodes.index(node)
            indicies.append(index)
        return fulltext_vecs[indicies]


def load_labels(nodes_string, m):
    if len(nodes_string) == len(m):
        labels = [x['categories'] for x in m]
    else:
        labels = [x['categories'] for x in m if x['id'] in nodes_string]
    labels_cl = clean_labels(labels)
    labels_cat = labels2categorical(labels_cl)
    return labels_cat


def save_data(nodes_int, dirname,vector_type, vector_train, vector_test, vector, labels_train, \
              labels_test, labels_cat, G_sub, m, cutoff1, cutoff2):
    """ Saves data in format required by Kipfs and Welling
    
        nodes_int = list, list of nodes labeled by integers
        dirname = string, where to save the data
        vector_type = string, = 'title', 'abstract', 'full-text'
    
    """
    
    #Save vectors
    dirname = 'data'
    fname = dirname + '/ind.arXiv-' + vector_type + '.x'
    pkl.dump(vector_train, open(fname,'wb'), protocol=4)

    fname = dirname + '/ind.arXiv-' + vector_type + '.tx'
    pkl.dump(vector_test, open(fname,'wb'), protocol=4)

    fname = dirname + '/ind.arXiv-' + vector_type + '.allx'
    pkl.dump(vector[:cutoff2], open(fname,'wb'), protocol=4)

    fname = dirname + '/ind.arXiv-' + vector_type + '.y'
    pkl.dump(np.array(labels_train), open(fname,'wb'), protocol=4)

    fname = dirname + '/ind.arXiv-' + vector_type + '.ty'
    pkl.dump(np.array(labels_test), open(fname,'wb'), protocol=4)

    fname = dirname + '/ind.arXiv-' + vector_type + '.ally'
    pkl.dump(np.array(labels_cat[:cutoff2]), open(fname,'wb'), protocol=4)

    test_nodes = nodes_int[cutoff2:]
    with open(dirname + '/ind.arXiv-' + vector_type + '.test.index','wt') as f:
        for node in test_nodes:
            f.write(str(node))
            f.write('\n')
            
            
    #Save graph in format required by Kipf-Welling -- nodes labeled as ints
    #Also, need to save in same order as metadata
    #So need to relabel the nodes
    
    #Find mapping: article_id -->int
    #mapping = {}
    #for i,item in enumerate(m):
    #    article_id = item['id']
    #    mapping[article_id] = i

    #Apply mapping
    #nx.relabel_nodes(G_sub, mapping, copy = False)
    
    #Save
    G_sub = nx.convert_node_labels_to_integers(G_sub)
    graph_dict = {}     
    for node in G_sub.nodes():
        graph_dict[node] = list(G_sub.neighbors(node))
    pkl.dump(graph_dict, open(dirname + '/ind.arXiv-' + vector_type + '.graph', 'wb'), protocol=4) 
    return
    
    
    

def cast_data_into_right_form(N):
    """
    Puts the arXivdata into right form for kipf Welling GCN classifier.
    
    N = int, number of nodes in citation graph
             N = 0 means take all the nodes
        
    """

    
    #Load graph & metadata
    t1 = time.time()
    
    #Meta data -- need this later
    dirname = '/home/kokeeffe/research/arxiv-public-datasets/arxiv-data'
    m = load_metadata(dirname + '/oai-arxiv-metadata-2019-03-01.json.gz')
    
    graph_saved = False    #for debugging, I've already made the full graph
    if graph_saved == True:
        dirname = 'data'
        G = nx.read_gpickle(dirname + '/full_graph.gpickle')
    else:
        dirname = '/home/kokeeffe/research/arxiv-public-datasets/arxiv-data/output'
        fname = dirname + '/internal-citations.json.gz'
        q = json.load(gzip.open(fname, 'rt', encoding='utf-8'))
        G = ia.makegraph(q)
        G = sync_G_with_metadata(G,m)   #G is missing some articles (see func for more details)

    #Select subgraph if specified
    if N != 0:
        comps = nx.weakly_connected_components(G)
        biggest = max(comps, key=len)
        G_cc = G.subgraph(biggest)
        nodes = list(G_cc.nodes())[:N]
        G_sub = G_cc.subgraph(nodes)
    else:
        G_sub = G
        
    nodes_string = np.array(G_sub.nodes())            #nodes labeled by strings
    nodes_int = np.array(range(G_sub.number_of_nodes()))       #nodes labeled by ints
    t2 = time.time()
    print('Loading graph took ' + str((t2-t1)/60.0) + ' mins')

    #Load features
    t1 = time.time()
    dirname = '/home/kokeeffe/research/arxiv-public-datasets/arxiv-data/output/embeddings'
    title_vecs = load_titles(G, m, nodes_string, dirname)
    abstract_vecs = load_abstracts(G, m, nodes_string, dirname)
    fulltext_vecs = load_fulltext(G, m, nodes_string, dirname)
    
    #Load labels
    dirname = '/home/kokeeffe/research/arxiv-public-datasets/arxiv-data'
    labels_cat = load_labels(nodes_string, m)
    t2 = time.time()
    print( 'Loading features & labels took ' + str((t2-t1)/60.0) + ' mins')

    
     #Split into test & train & ulabeled portion
    #For now, I'll assume that nothing is unlabeled
    #That means cutoff1 and cutoff2 are the same
    t1 = time.time()
   
    #Shuffle
    indicies = list(range(len(title_vecs)))
    np.random.shuffle(indicies)
    title_vecs = title_vecs[indicies]
    abstract_vecs = abstract_vecs[indicies]
    fulltext_vecs = fulltext_vecs[indicies] 
    nodes_int = nodes_int[indicies]
    nodes_string = nodes_string[indicies]

    #Split
    cutoff1 = int(0.9*title_vecs.shape[0]) 
    cutoff2 = int(0.9*title_vecs.shape[0])
    title_vec_train, title_vec_test = title_vecs[:cutoff1], title_vecs[cutoff2:]
    abstract_vec_train, abstract_vec_test = abstract_vecs[:cutoff1], abstract_vecs[cutoff2:]
    fulltext_vec_train, fulltext_vec_test = fulltext_vecs[:cutoff1], fulltext_vecs[cutoff2:]
    labels_train, labels_test = labels_cat[:cutoff1], labels_cat[cutoff2:]

    #Save data
    dirname = 'data'
    save_data(nodes_int, dirname, 'title', title_vec_train, title_vec_test, title_vecs,labels_train, labels_test, labels_cat, G_sub, m, cutoff1, cutoff2)
    
    save_data(nodes_int, dirname, 'abstract', abstract_vec_train, abstract_vec_test, abstract_vecs, labels_train, labels_test, labels_cat, G_sub, m, cutoff1, cutoff2)
    
    save_data(nodes_int, dirname, 'fulltext', fulltext_vec_train, fulltext_vec_test, fulltext_vecs,labels_train, labels_test, labels_cat, G_sub, m, cutoff1, cutoff2)
    
    #Combine all
    title_vecs = np.concatenate([title_vecs, abstract_vecs, fulltext_vecs], axis=1)
    title_vec_train, title_vec_test = title_vecs[:cutoff1], title_vecs[cutoff2:]
    save_data(nodes_int, dirname, 'all', title_vec_train, title_vec_test, title_vecs, labels_train, labels_test, labels_cat, G_sub, m, cutoff1, cutoff2)
    t2 = time.time()
    print( 'Saving data took ' + str((t2-t1)/60.0) + ' mins')
    
    return