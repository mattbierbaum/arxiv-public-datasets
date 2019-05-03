""" Runs the Kipf-Welling GCN. First, I put the arxiv data into their
    format. Then I simply call their train.py file. See README.
        
"""

import os, time
import GCN_utils as u

from arxiv_public_data.config import LOGGER, DIR_OUTPUT

logger = LOGGER.getChild('GCN-classify')
SAVE_DIR = os.path.join(DIR_OUTPUT, 'kipf-welling')

def main():
    N = 10**2       #number of nodes in graph to take  (if you want to experiment on smaller graph)
    #N = 0          #this means take the full dataset
    epochs = 2*10**2  #number of epochs in training, used by kipf-welling https://arxiv.org/pdf/1609.02907.pdf
    
    #Options
    data_made = False     #might want to reuse the data that's been used already
    delete_data = True    #delete the intermediary data files after they've been used
        
    #Put data into right format
    if data_made == False:
        logger.info('Preparing data \n ')
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        RES_DIR = os.path.join(SAVE_DIR, 'results')
        if not os.path.exists(RES_DIR):
            os.makedirs(RES_DIR)
        u.cast_data_into_right_form(N)
    
    #Then do the GCN classification
    logger.info('\n Now starting classification')
    
    cmd = 'python analysis/kipf_welling_GCN/train.py --dataset arXiv-{} --epochs {}'
    #title vectors --- see the train.py file for other command file arguments
    logger.info('Starting title vectors')
    os.system(cmd.format('title', epochs))

    #abstract vectors
    logger.info('\n Starting abstract vectors')
    os.system(cmd.format('abstract', epochs))
    
    #fulltext
    logger.info('\n Starting fulltext vectors')
    os.system(cmd.format('fulltext', epochs))
    
    #all together
    logger.info('\n Starting all vectors')
    os.system(cmd.format('all', epochs))
     
    #Delete the data I don't need
    #if delete_data == True:
    #    os.system('rm -r data/')
    
    return
    
  
if __name__ == '__main__':
    main()
