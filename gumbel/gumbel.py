import numpy as np

from keras import backend as K

def random_gumbel(shape, eps=1e-20):
    U = K.random_uniform(shape)
    return  -K.log(-K.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature): 
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + random_gumbel(K.shape(logits))
    return K.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = K.shape(logits)[-1]
        y_hard = K.cast(K.one_hot(K.argmax(y, 1), k), K.floatx)
        y = K.stop_gradient(y_hard - y) + y
    return y
