'''This script demonstrates how to build a variational autoencoder with Keras.

Reference: 
"Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
"CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX" https://arxiv.org/abs/1611.01144
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Reshape, Softmax, Flatten
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from keras import objectives
from keras.datasets import mnist

from gumbel import gumbel_softmax

np.random.seed(1111)  # for reproducibility

batch_size = 100 
nb_classes = 10
n = 784
m = 1
hidden_dim = 256
epochs = 50 
epsilon_std = 1.0
use_loss = 'xent' # 'mse' or 'xent'

decay = 1e-4 # weight decay, a.k. l2 regularization
use_bias = True

## Encoder
def build_encoder(temperature, hard):
    x = Input(batch_shape=(batch_size, n))
    h_encoded = Dense(hidden_dim, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=use_bias, activation='tanh')(x)
    z = Dense(m * nb_classes, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=use_bias)(h_encoded)

    logits_z = Reshape((m, nb_classes))(z)  # batch x m * nb_classes -> batch x m x nb_classes
    q_z = Softmax()(logits_z)
    log_q_z = Lambda(lambda x: K.log(x + K.epsilon()))(q_z)

    z = Lambda(lambda x: gumbel_softmax(x, temperature, hard))(logits_z)

    z = Flatten()(z)
    q_z = Flatten()(q_z)
    log_q_z = Flatten()(log_q_z)

    return x, z, q_z, log_q_z


def build_decoder(z):
    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(hidden_dim, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=use_bias, activation='tanh')
    decoder_mean = Dense(n, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=use_bias, activation='sigmoid')

    h_decoded = decoder_h(z)
    x_hat = decoder_mean(h_decoded)

    return x_hat, decoder_h, decoder_mean


#temperature = K.variable(np.asarray([1])) 
temperature = 1
hard = False
x, z, q_z, log_q_z = build_encoder(temperature, hard)
x_hat, decoder_h, decoder_mean = build_decoder(z)


## loss
def vae_loss(x, x_hat):
    kl_loss = 0.01 + K.mean(q_z * (log_q_z - K.log(1.0 / nb_classes)))
    xent_loss = n * objectives.binary_crossentropy(x, x_hat)
    mse_loss = n * objectives.mse(x, x_hat) 
    if use_loss == 'xent':
        return xent_loss - kl_loss
    elif use_loss == 'mse':
        return mse_loss - kl_loss
    else:
        raise Expception, 'Nonknow loss!'

vae = Model(x, x_hat)
vae.compile(optimizer='rmsprop', loss=vae_loss)

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))


# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(m * nb_classes,))
_h_decoded = decoder_h(decoder_input)
_x_hat = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_hat)

n = nb_classes
digit_size = 28
figure = np.zeros((digit_size, digit_size * n))
for i in range(nb_classes):
    z_sample = np.zeros([1, nb_classes])
    z_sample[0, i] = 1
    x_decoded = generator.predict(z_sample)
    digit = x_decoded[0].reshape(digit_size, digit_size)
    figure[:, i * digit_size: (i + 1) * digit_size] = digit

fig = plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
fig.savefig('x_{}.png'.format(use_loss))


# data imputation
n = 15  # figure with 15x15 digits
figure = np.zeros((digit_size * 3, digit_size * n))
x = x_test[:batch_size,:]
x_corupted = np.copy(x)
x_corupted[:, 300:400] = 0
x_encoded = vae.predict(x_corupted, batch_size=batch_size).reshape((-1, digit_size, digit_size))
x = x.reshape((-1, digit_size, digit_size))
x_corupted = x_corupted.reshape((-1, digit_size, digit_size))
for i in range(n):
    xi = x[i]
    xi_c = x_corupted[i]
    xi_e = x_encoded[i]
    figure[:digit_size, i * digit_size:(i+1)*digit_size] = xi
    figure[digit_size:2 * digit_size, i * digit_size:(i+1)*digit_size] = xi_c
    figure[2 * digit_size:, i * digit_size:(i+1)*digit_size] = xi_e

fig = plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
fig.savefig('i_{}.png'.format(use_loss))
