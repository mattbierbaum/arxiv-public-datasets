""" Runs the Kipf-Welling GCN. Assumes the data has been prepared in the right way
    (by running cast_arxivdata_into_right_format.py)
"""

import os


def main():
    
    #title vectors --- see the train.py file for other command file arguments
    os.system('python3 train.py --dataset arXiv-title')
    
    #abstract vectors
    os.system('python3 train.py --dataset arXiv-abstract')
    
    #title and abstracts
    os.system('python3 train.py --dataset arXiv-title-abstract')
    
    return
    
  
if __name__ == '__main__':
    main()
