""" Run the classification """

import os 

def main():
    
    #title vecs
    os.system('python3 train.py --dataset arXiv-title')
    
    #abstract vecs
    os.system('python3 train.py --dataset arXiv-abstract')
    
    #title + abstract 
    #os.system('python3 train.py --dataset arXiv-abstract-and-title')

    return
    
  
if __name__ == '__main__':
    main()
