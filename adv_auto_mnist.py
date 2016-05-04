from models.adv_autoencoder import AdversarialAutoencoder
import source.layers as L
from theano.tensor.shared_randomstreams import RandomStreams
import numpy, theano
import theano.tensor as T
import operator


def get_normalized_vector(v):
    v = v / (1e-20 + T.max(T.abs_(v), axis=1, keepdims=True))
    v_2 = T.sum(v ** 2, axis=1, keepdims=True)
    return v / T.sqrt(1e-6 + v_2)


class AdversarialAutoencoderMNIST(AdversarialAutoencoder):
    def __init__(self, n_in=784, n_hidden_g=[1000, 1000], n_hidden_d=[500, 500], latent_dim=2,
                 z_prior='gaussian'):

        self.n_in = n_in
        self.z_prior = z_prior

        self.enc = []
        for i, _ in enumerate(n_hidden_g):
            if 0 == i:
                self.enc.append(L.Linear((n_in, n_hidden_g[i])))
            elif len(n_hidden_g) - 1 > i:
                self.enc.append(L.Linear((n_hidden_g[i], n_hidden_g[i + 1])))
            else:
                self.enc.append(L.Linear((n_hidden_g[i], latent_dim)))

        self.dec = []
        for i, _ in enumerate(n_hidden_g):
            if 0 == i:
                self.dec.append(L.Linear((latent_dim, n_hidden_g[-1])))
            elif len(n_hidden_g) - 1 > i:
                self.dec.append(L.Linear((n_hidden_g[-(i + 1)], n_hidden_g[-(i + 2)])))
            else:
                self.dec.append(L.Linear((n_hidden_g[-(i + 1)], n_in)))

        self.discriminator = []
        for i, _ in enumerate(n_hidden_d):
            if 0 == i:
                self.discriminator.append(L.Linear((latent_dim, n_hidden_d[i])))
            elif len(n_hidden_d) - 1 > i:
                self.discriminator.append(L.Linear((n_hidden_d[i], n_hidden_d[i + 1])))
            else:
                self.discriminator.append(L.Linear((n_hidden_d[i], 1)))

        self.model_params = []
        for i in xrange(len(self.enc)):
            self.model_params += self.enc[i].params
        for i in xrange(len(self.dec)):
            self.model_params += self.dec[i].params
        self.D_params = []
        for i in xrange(len(self.discriminator)):
            self.model_params += self.discriminator[i].params
        self.rng = RandomStreams(seed=numpy.random.randint(1234))

    def encode(self, input, train=True):
        h = input
        for e in self.enc:
            h = e(h)
            h = L.sigmoid(h)
        return h

    def decode(self, input, train=True):
        h = input
        for d in self.dec:
            h = d(h)
            h = L.sigmoid(h)
        return h

    def D(self, input, train=True):
        h = input
        for d in self.discriminator:
            h = d(h)
            h = L.sigmoid(h)
        return h

    def sample_from_prior(self, z):

        ###### gausssian #######
        if (self.z_prior is 'gaussian'):
            return 1.0 * self.rng.normal(size=z.shape, dtype=theano.config.floatX)

        ###### uniform ########
        elif (self.z_prior is 'uniform'):
            v = get_normalized_vector(self.rng.normal(size=z.shape, dtype=theano.config.floatX))
            r = T.power(
                self.rng.uniform(size=z.sum(axis=1, keepdims=True).shape, low=0, high=1.0, dtype=theano.config.floatX),
                1. / z.shape[1])
            r = T.patternbroadcast(r, [False, True])
            return 2.0 * r * v

        else:
            raise NotImplementedError()
