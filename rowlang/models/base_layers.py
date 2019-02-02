from abc import ABC, abstractmethod

import tensorflow as tf

from rowlang.utils import graph_def


class LayerABC(ABC):
    def __init__(self, name):
        self.name = name
        self.outputs = None

    @abstractmethod
    def on(self, X,  *args, **kwargs):
        pass


class LinearLayer(LayerABC):
    def __init__(self, out_dim, name):
        super(LinearLayer, self).__init__(name)
        self.out_dim = out_dim

    @graph_def
    def on(self, X):
        return tf.layers.dense(X, self.out_dim, activation=None, name=self.name)


class DropoutLayer(LayerABC):
    def __init__(self, dropout, name="dropout"):
        super(DropoutLayer, self).__init__(name)
        self.dropout = dropout

    @graph_def
    def on(self, X):
        return tf.nn.dropout(X, 1 - self.dropout, name='dropped')


class LayerNormLayer(LayerABC):
    def __init__(self, name="layernorm"):
        super(LayerNormLayer, self).__init__(name)
        self._eps = 1e-6  # for numerical stability
        self.mean = None
        self.std = None

    @graph_def
    def on(self, X):
        """
        X: [minibatch x seq x dims]
        """
        self.mean, self.std = tf.nn.moments(X, axes=-1, keep_dims=True)
        return (X - self.mean) / (self.std + self._eps)
