"""
elmo.py

https://tfhub.dev/google/elmo/2

"""
import tensorflow as tf
import tensorflow_hub as hub

def embed_strings(strings):

    with tf.Graph().as_default():
        model_url = "https://tfhub.dev/google/elmo/2"
        elmo = hub.Module(model_url, trainable=True)

        # grab mean-pooling of contextualized word reps
        embeddings = elmo(strings, signature="default", as_dict=True)['default']

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            return sess.run(embeddings)
