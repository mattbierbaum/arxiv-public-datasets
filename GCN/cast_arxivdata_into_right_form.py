""" Put arXiv data into right form for the GCN """

import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import arxiv_public_data.tests.intra_citation as ia 
import time, json, gzip
import pickle as pkl
from arxiv_public_data.oai_metadata import load_metadata


                                #Auxiliary
#-----------------------------------------------------------------------


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


def load_titles(G, nodes_string, dirname):
    
    #Load the full feature matrix
    filename = dirname + '/title-embedding-usel-2019-03-19.pkl'
    out = []
    with open(filename, 'rb') as f:
        while True:
            try:
                out.extend(pkl.load(f))
            except EOFError as e:
                break
    title_vecs = np.array(out)
    
    #Then select the subset corresponding to the sub-graph we're examining
    indicies = []
    all_nodes = list(G.nodes())
    for i,node in enumerate(nodes_string):
        index = all_nodes.index(node)
        indicies.append(index)
    return title_vecs[indicies]



def load_abstracts(G, nodes_string, dirname):
    
    #Load the full feature matrix
    filename = dirname + '/abstract-embedding-usel-2019-03-19.pkl'
    out = []
    with open(filename, 'rb') as f:
        while True:
            try:
                out.extend(pkl.load(f))
            except EOFError as e:
                break
    abstract_vecs = np.array(out)
    
    #Then select the subset corresponding to the sub-graph we're examining
    indicies = []
    all_nodes = list(G.nodes())
    for i,node in enumerate(nodes_string):
        index = all_nodes.index(node)
        indicies.append(index)
    return abstract_vecs[indicies]


def load_fulltext(G, nodes_string, dirname):
    
    #Load the full feature matrix
    filename = dirname + '/fulltext-embedding-usel-2-headers-2019-04-05.pkl'
    out = []
    with open(filename, 'rb') as f:
        while True:
            try:
                out.extend(pkl.load(f))
            except EOFError as e:
                break
    fulltext_vecs = np.array(out)
    
    #Then select the subset corresponding to the sub-graph we're examining
    indicies = []
    all_nodes = list(G.nodes())
    for i,node in enumerate(nodes_string):
        index = all_nodes.index(node)
        indicies.append(index)
    return fulltext_vecs[indicies]


def load_labels(nodes_string, dirname):
    m = load_metadata( dirname + '/oai-arxiv-metadata-2019-03-01.json.gz')
    labels = [x['categories'] for x in m if x['id'] in nodes_string]
    labels_cl = clean_labels(labels)
    labels_cat = labels2categorical(labels_cl)
    return labels_cat


def save_data(nodes_int, dirname,vector_type, vector_train, vector_test, vector, G_sub):
    """ Saves data in format required by Kipfs and Welling
    
        nodes_int = list, list of nodes labeled by integers
        dirname = string, where to save the data
        vector_type = string, = 'title', 'abstract', 'full-text'
    
    """

    #Save vectors
    dirname = 'data'
    fname = dirname + '/ind.arXiv-' + vector_type + '.x'
    pkl.dump(vector_train, open(fname,'wb'))

    fname = dirname + '/ind.arXiv-' + vector_type + '.tx'
    pkl.dump(vector_test, open(fname,'wb'))

    fname = dirname + '/ind.arXiv-' + vector_type + '.allx'
    pkl.dump(vector[:cutoff2], open(fname,'wb'))

    fname = dirname + '/ind.arXiv-' + vector_type + '.y'
    pkl.dump(np.array(labels_train), open(fname,'wb'))

    fname = dirname + '/ind.arXiv-' + vector_type + '.ty'
    pkl.dump(np.array(labels_test), open(fname,'wb'))

    fname = dirname + '/ind.arXiv-' + vector_type + '.ally'
    pkl.dump(np.array(labels_cat[:cutoff2]), open(fname,'wb'))

    test_nodes = nodes_int[cutoff2:]
    with open(dirname + '/ind.arXiv-' + vector_type + '.test.index','wt') as f:
        for node in test_nodes:
            f.write(str(node))
            f.write('\n')
            
            
    #Save graph
    graph_dict = {}       #put into their format
    for node in G_sub.nodes():
        graph_dict[node] = [x for x in G_sub.neighbors(node)]
    pkl.dump(graph_dict, open(dirname + '/ind.arXiv-' + vector_type + '.graph', 'wb')) 
    return
    


                                #Main
#-----------------------------------------------------------------------


def main():

    N = 10**2  #size of subgraph to be used (using a subraph during testing phase)

    #Graph
    t1 = time.time()
    fname = '/home/kokeeffe/research/arxiv-public-datasets/data/internal-citations.json.gz'
    q = json.load(gzip.open(fname, 'rt', encoding='utf-8'))
    G = ia.makegraph(q)

    #Select subgraph if specified
    if N != 0:
        comps = nx.weakly_connected_components(G)
        biggest = max(comps, key=len)
        G_cc = G.subgraph(biggest)
        nodes = list(G_cc.nodes())[:N]
        G_sub = G_cc.subgraph(nodes)
    else:
        G_sub = G

    nodes_string = list(G_sub.nodes())  #nodes labeled by ints
    G_sub = nx.convert_node_labels_to_integers(G_sub)
    nodes_int = list(G_sub.nodes())     #nodes labeled in strings
    t2 = time.time()
    print('Loading graph took ' + str((t2-t1)/60.0) + ' mins')


    #Feature and label matrices
    t1 = time.time()
    dirname = '/home/kokeeffe/research/arxiv-public-datasets/data'
    title_vecs = load_titles(G, nodes_string, dirname)
    abstract_vecs = load_abstracts(G, nodes_string, dirname)
    #fulltext_vecs = load_fulltext(G, nodes_string, dirname)
    labels_cat = load_labels(nodes_string, dirname)
    t2 = time.time()
    print( 'Loading features & labels took ' + str((t2-t1)/60.0) + ' mins')

    #Split into test & train & ulabeled portion
    #For now, I'll assume that nothing is unlabeled
    #That means cutoff1 and cutoff2 are the same
    cutoff1 = int(0.9*title_vecs.shape[0]) 
    cutoff2 = int(0.9*title_vecs.shape[0])
    title_vec_train, title_vec_test = title_vecs[:cutoff1], title_vecs[cutoff2:]
    abstract_vec_train, abstract_vec_test = abstract_vecs[:cutoff1], abstract_vecs[cutoff2:]
    #fulltext_vec_train, fulltext_vec_test = fulltext_vecs[:cutoff1], fulltext_vecs[cutoff2:]
    labels_train, labels_test = labels_cat[:cutoff1], labels_cat[cutoff2:]

    #Save data
    dirname = 'data'
    vector_type = 'title'
    save_data(nodes_int, dirname, 'title', title_vec_train, title_vec_test, title_vecs, G_sub)
    save_data(nodes_int, dirname, 'abstract', abstract_vec_train, abstract_vec_test, abstract_vecs, G_sub)

    #Combine title and abstract vecs
    title_vecs = np.concatenate((title_vecs, abstract_vecs), axis=1)
    title_vec_train, title_vec_test = title_vecs[:cutoff1], title_vecs[cutoff2:]
    save_data(nodes_int, dirname, 'title', title_vec_train, title_vec_test, title_vecs, G_sub)

    return

if __name__ == "__main__":
    main()