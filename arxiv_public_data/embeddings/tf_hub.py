"""
elmo.py

https://tfhub.dev/google/elmo/2

"""

import os
import pickle
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

from arxiv_public_data.config import ARXIV_DIR

def embed_strings(strings, filename, batchsize=32,
                  model_url="https://tfhub.dev/google/elmo/2",
                  model_kwargs = dict(signature='default', as_dict=True),
                  dictkey='default'):
    """
    Compute and save vector embeddings of lists of strings in batches
    Parameters
    ----------
        strings : list(str)
            list of strings to be embedded
        filename : str
            filename to store embeddings            
        (optional)
        batchsize : int
            size of batches
        model_url : str
            TensorFlow hub model url
        model_kwargs : dict
            Dictionary of kwargs that the model specifies
        dictkey : str
            Many models return dicts, this specifies which part to select
    """
    batches = np.array_split(
        np.array(strings), len(strings)//batchsize
    )
    with tf.Graph().as_default():
        embd = hub.Module(model_url, trainable=True)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            for i, batch in enumerate(batches):
                # grab mean-pooling of contextualized word reps
                embeddings = embd(batch, **model_kwargs)[dictkey]
                print("Computing/saving batch {}".format(i))
                with open(filename, 'ab') as fout:
                    pickle.dump(sess.run(embeddings), fout)

def load_embeddings(filename):
    """
    Loads vector embeddings
    Parameters
    ----------
        filename : str
            path to vector embeddings saved by `create_save_embeddings`
    Returns
    -------
        embeddings : array_like
    """
    out = []
    with open(filename, 'rb') as fin:
        while True:
            try:
                out.extend(pickle.load(fin))
            except EOFError as e:
                break 
    return np.array(out)

def create_save_embeddings(strings, filename, SAVEDIR=ARXIV_DIR, **kwargs):
    """
    Create vector embeddings of strings and save them to filename
    Parameters
    ----------
        strings: list(str)
        filename: str
            embeddings will be saved in ARXIV_DIR/embeddings/filename
    """
    filepath = os.path.join(ARXIV_DIR, "embeddings")
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    print("Saving embeddings to {}".format(os.path.join(filepath, filename)))
    embed_strings(strings, os.path.join(filepath, filename), **kwargs)
