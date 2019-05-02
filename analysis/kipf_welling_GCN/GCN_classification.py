""" Runs the Kipf-Welling GCN. First, I put the arxiv data into their
    format. Then I simply call their train.py file. See README.
        
"""

import os, time
import GCN_utils as u


def main():
    
    #N = 10**2       #number of nodes in graph to take  (if you want to experiment on smaller graph)
    N = 0          #this means take the full dataset
    epochs = 2*10**2  #number of epochs in training, used by kipf-welling https://arxiv.org/pdf/1609.02907.pdf

    
    #Options
    data_made = False     #might want to reuse the data that's been used already
    delete_data = True    #delete the intermediary data files after they've been used
        
    #Put data into right format
    if data_made == False:
        print('Preparing data \n ')
        if not os.path.exists('data/'):
            os.system('mkdir data')
        if not os.path.exists('results'):
            os.system('mkdir results')
        u.cast_data_into_right_form(N)
    
    #Then do the GCN classification
    print('\n Now starting classification')
    
    #title vectors --- see the train.py file for other command file arguments
    print('Starting title vectors')
    os.system('python train.py --dataset arXiv-title --epochs ' + str(epochs))

    #abstract vectors
    print('\n Staring abstract vectors')
    os.system('python train.py --dataset arXiv-abstract --epochs ' + str(epochs))
    
    #fulltext
    print('\n Starting fulltext vectors')
    os.system('python train.py --dataset arXiv-fulltext -- epochs ' + str(epochs))
    
    #all together
    print('\n Starting all vectors')
    os.system('python train.py --dataset arXiv-all -- epochs ' + str(epochs))
    
     
    #Delete the data I don't need
    if delete_data == True:
        os.system('rm -r data/')
    
    return
    
  
if __name__ == '__main__':
    main()
