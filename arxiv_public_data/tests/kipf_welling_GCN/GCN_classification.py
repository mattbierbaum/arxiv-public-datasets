""" Runs the Kipf-Welling GCN. Assumes the data has been prepared in the right way
    (by running cast_arxivdata_into_right_format.py)
"""

import os, time


def main():
    
    
    epochs = 10**3
    
    #title vectors --- see the train.py file for other command file arguments
    t1 = time.time()
    print('Starting title vectors')
    os.system('python train.py --dataset arXiv-title --epochs ' + str(epochs))
    t2 = time.time()
    print('\n Sim took ' + str((t2-t1)/60.0) + ' mins')

    #abstract vectors
    #print('\n Staring abstract vectors')
    #os.system('python train.py --dataset arXiv-abstract --epochs ' + str(epochs))
    
    #title and abstracts
    #print('\n Starting title and abstract vectors')
    #os.system('python train.py --dataset arXiv-title-abstract -- epochs ' + str(epochs))
    
    return
    
  
if __name__ == '__main__':
    main()
