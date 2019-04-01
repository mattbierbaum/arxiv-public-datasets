"""
tf_hub.py

Find text embeddings using pre-trained TensorFlow Hub models
"""

import os
import pickle
import numpy as np

from arxiv_public_data.config import DIR_OUTPUT, LOGGER

logger = LOGGER.getChild('embds')

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    import sentencepiece as spm
except ImportError as e:
    logger.info("This module requires 'tensorflow', 'tensorflow-hub', and"
                "'sentencepiece'\n"
                'Please install these modules to use tf_hub.py')


UNIV_SENTENCE_ENCODER_URL = ('https://tfhub.dev/google/'
                             'universal-sentence-encoder/2')

ELMO_URL = "https://tfhub.dev/google/elmo/2"
ELMO_KWARGS = dict(signature='default', as_dict=True)
ELMO_MODULE_KWARGS = dict(trainable=True)
ELMO_DICTKEY = 'default'

DIR_EMBEDDING = os.path.join(DIR_OUTPUT, 'embeddings')
if not os.exists(DIR_EMBEDDING):
    os.mkdir(DIR_EMBEDDING)

def elmo_strings(strings, filename, batchsize=32):
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
    """
    batches = np.array_split(
        np.array(strings, dtype='object'), len(strings)//batchsize
    )

    g = tf.Graph()
    with g.as_default():
        module = hub.Module(ELMO_URL, **ELMO_MODULE_KWARGS)
        text_input = tf.placeholder(dtype=tf.string, shape=[None])
        embeddings = module(text_input, **ELMO_KWARGS)
        init_op = tf.group([tf.global_variables_initializer(),
                            tf.tables_initializer()])
    g.finalize()

    with tf.Session(graph=g) as sess:
        sess.run(init_op)

        for i, batch in enumerate(batches):
            # grab mean-pooling of contextualized word reps
            logger.info("Computing/saving batch {}".format(i))
            with open(filename, 'ab') as fout:
                pickle.dump(sess.run(
                    embeddings, feed_dict={text_input: batch}
                )[ELMO_DICTKEY], fout)

UNIV_SENTENCE_LITE = "https://tfhub.dev/google/universal-sentence-encoder-lite/2"

def get_sentence_piece_model():
    with tf.Session() as sess:
        module = hub.Module(UNIV_SENTENCE_LITE)
        return sess.run(module(signature="spm_path"))

def process_to_IDs_in_sparse_format(sp, sentences):
    """
    An utility method that processes sentences with the sentence piece
    processor
    'sp' and returns the results in tf.SparseTensor-similar format:
    (values, indices, dense_shape)
    """
    ids = [sp.EncodeAsIds(x) for x in sentences]
    max_len = max(len(x) for x in ids)
    dense_shape=(len(ids), max_len)
    values=[item for sublist in ids for item in sublist]
    indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
    return (values, indices, dense_shape)

def universal_sentence_encoder_lite(strings, filename, spm_path, batchsize=32):
    """
    Compute and save vector embeddings of lists of strings in batches
    Parameters
    ----------
        strings : list(str)
            list of strings to be embedded
        filename : str
            filename to store embeddings            
        spm_path : str
            path to sentencepiece model from `get_sentence_piece_model`
        (optional)
        batchsize : int
            size of batches
    """
    batches = np.array_split(
        np.array(strings, dtype='object'), len(strings)//batchsize
    )
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)

    g = tf.Graph()
    with g.as_default():
        module = hub.Module(UNIV_SENTENCE_LITE)
        input_placeholder = tf.sparse_placeholder(
            tf.int64, shape=(None, None)
        )
        embeddings = module(
            inputs=dict(
                values=input_placeholder.values, indices=input_placeholder.indices,
                dense_shape=input_placeholder.dense_shape
            )
        )
        init_op = tf.group([tf.global_variables_initializer(),
                            tf.tables_initializer()])
    g.finalize()

    with tf.Session(graph=g) as sess:
        sess.run(init_op)
        for i, batch in enumerate(batches):
            values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, batch)
            logger.info("Computing/saving batch {}".format(i))
            emb = sess.run(
                embeddings, 
                feed_dict={
                    input_placeholder.values: values, 
                    input_placeholder.indices: indices, 
                    input_placeholder.dense_shape: dense_shape
                }
            )
            with open(filename, 'ab') as fout:
                    pickle.dump(emb, fout)

def batch_fulltext():
    pass

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

def create_save_embeddings(strings, filename, encoder, encoder_args=(),
                           encoder_kwargs={}, SAVEDIR=DIR_EMBEDDING):
    """
    Create vector embeddings of strings and save them to filename
    Parameters
    ----------
        strings: list(str)
        filename: str
            embeddings will be saved in DIR_EMBEDDING/embeddings/filename
    Usage
    -----
    Universal Sentence Encoder Lite:
    spm_path = get_sentence_piece_model()
    create_save_embeddings(strings, filename, 
                           universal_sentence_encoder_lite,
                           (spm_path,), encoder_kwargs=dict(batchsize=512))

    ELMO:
    create_save_embeddings(strings, filename, elmo_strings,
                           encoder_kwargs=dict(batchsize=64))
    """
    filepath = os.path.join(DIR_EMBEDDING, "embeddings")
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    logger.info("Saving embeddings to {}".format(os.path.join(filepath, filename)))
    encoder(strings, os.path.join(filepath, filename), *encoder_args, 
            **encoder_kwargs)
