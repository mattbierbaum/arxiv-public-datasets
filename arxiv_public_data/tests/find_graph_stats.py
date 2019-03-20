import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import arxiv_public_data.tests.intra_citation as ia 
import time
from powerlaw import Fit


def main():
    
    """ Computes various graph statistics """

    #Load subgraph here
    #G = nx.read_adjlist("data/sub_graph_networkx_graph")
    #G = G.to_directed()
    
    #Load graph   
    name = 'data/internal-references-pdftotext.json.gz'
    q = ia.loaddata(fname=name)
    G, badrefs = ia.makegraph(q)

    #basic stats
    N_nodes, N_edges = G.number_of_nodes(), G.number_of_edges()

    #Degree
    t1 = time.time()
    in_deg = [d for n, d in G.in_degree()]
    out_deg = [d for n, d in G.out_degree()]
    np.savetxt('data/in_degree.txt', in_deg)
    np.savetxt('data/out_degree.txt', out_deg)
    mean_k = 2*np.mean(in_deg)
    t2 = time.time()
    print('degree took ' + str((t2-t1)/60.0) + ' mins')

    #Find powerlaw fits
    fit_in, fit_out = Fit(in_deg,xmin=0), Fit(out_deg,xmin=0)
    alpha_in, xmin_in = np.round(fit_in.power_law.alpha,2), np.round(fit_in.power_law.xmin,2)
    alpha_out, xmin_out = np.round(fit_out.power_law.alpha,2), np.round(fit_out.power_law.xmin,2)
    print('For power law fitting in-degree: x_min = ' + str(xmin_in))
    print('For power law fitting out-degree: x_min = ' + str(xmin_out) + '\n')

    #Clustering coeff
    t1 = time.time()
    cs = list(nx.clustering(G).values())
    np.savetxt('data/clustering_c.txt', cs)
    mean_C = np.round(np.mean(cs), 2)
    t2 = time.time()
    print('cluster coeff took ' + str((t2-t1)/60.0) + ' mins')

    #Size-biggest
    t1 = time.time()
    comps = nx.weakly_connected_components(G)
    biggest = max(comps, key=len)
    G_cc = G.subgraph(biggest)
    size_WCC = 1.0*G_cc.number_of_nodes()
    fraction_WCC = np.round(size_WCC / N_nodes, 2)

    #Num isolated
    num_isolated = 0
    comps = nx.weakly_connected_components(G)
    for cc in comps:
        if len(cc) == 1:
            num_isolated += 1
    fraction_isolated = np.round((1.0*num_isolated) / N_nodes, 2)

    t2 = time.time()
    print('cluster size dist ' + str((t2-t1)/60.0) + ' mins')
 

    #results
    stats = [N_nodes, N_edges, mean_k, alpha_in, alpha_out, mean_C, fraction_WCC, fraction_isolated]
    print(stats)

    #Stuff for tables
    headings = ['','$N_nodes$', 'N_edges', '$\langle k \rangle$', '\alpha_in', '\alpha_out', '\langle C \rangle', '\% WCC' \
               '% isolated']
    
    row1 = ['openArXiv', N_nodes, N_edges, mean_k, alpha_in, alpha_out, mean_C, fraction_WCC, fraction_isolated]
    row2 = ['WoS', '$1.40 \times 10^5$', '$6.4 \times 10^5$', '9.11', '2.39', '3.88 ', '--', '97 \% ', '--']
    row3 = ['CiteSeer', '$3.84 \times 10^5$', '$1.74 \times 10^6$', '9.08' , '2.28', '3.82 ', '--' '95 \% ', '--']
    row4 = ['ArXiv', '$3.34 \times 10^4$', '$4.21 \times 10^5$', '24.50', '2.54', '3.45', '--', '99.6 \%', '--']
    
    
    #### MAKE FIGURE
    tick_size = 20
    axis_size = 30
    label_size = 28
    label_y_position = 1.10
    inset_size = 18
    plt.figure(figsize=(20,5))

    #Histogram in-degree
    n_bins = 30
    ax1 = plt.subplot(131)
    plt.hist(in_deg, alpha=0.75,bins=n_bins)
    #plt.hist(out_deg, alpha=0.75,bins=n_bins)
    plt.xlabel('$k_{in}$',fontsize=axis_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.rc('font', size=15) 
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.spines["top"].set_visible(False)  
    ax1.spines["right"].set_visible(False)
    ax1.text(-0.025,label_y_position, 'a', transform=ax1.transAxes,
          fontsize=label_size, fontweight='bold', va='top', ha='right')
    ax1.text(0.9, 0.55, '', transform=ax1.transAxes,
          fontsize=inset_size, va='top', ha='right')

    #Histogram out-degree
    ax2 = plt.subplot(132)
    plt.hist(out_deg, alpha=0.75,bins=n_bins)
    plt.xlabel('$k_{out}$',fontsize=axis_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.rc('font', size=15) 
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.spines["top"].set_visible(False)  
    ax2.spines["right"].set_visible(False)
    ax2.text(-0.025,label_y_position, 'b', transform=ax2.transAxes,
          fontsize=label_size, fontweight='bold', va='top', ha='right')
    ax2.text(0.9, 0.55, '', transform=ax1.transAxes,
          fontsize=inset_size, va='top', ha='right')

    #Histogram clustering coefficients
    ax3 = plt.subplot(133)
    plt.hist(cs, alpha=0.75, bins=n_bins)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    plt.xlabel('$C$',fontsize=axis_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)  
    plt.rc('font', size=15) 
    ax3.spines["top"].set_visible(False)  
    ax3.spines["right"].set_visible(False)
    ax3.text(-0.025,label_y_position, 'c', transform=ax3.transAxes,
          fontsize=label_size, fontweight='bold', va='top', ha='right')
    ax3.text(0.9, 0.55, '', transform=ax2.transAxes,
          fontsize=inset_size, va='top', ha='right')
    plt.tight_layout()
    plt.savefig('figures/histograms_onerow.png')

    return 

if __name__ == '__main__':
    main()
