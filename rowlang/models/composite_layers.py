from collections import namedtuple

import tensorflow as tf
import numpy as np

from rowlang.utils import graph_def
from rowlang.models.base_layers import LayerABC, DropoutLayer, LayerNormLayer, LinearLayer


class FeedForwardLayer(LayerABC):
    def __init__(self, dropout, d_model, d_ff, name="f_fwd"):
        super(FeedForwardLayer, self).__init__(name)

        self.d_ff = d_ff
        self.linear1 = LinearLayer(self.d_ff, "ff_up")
        self.linear2 = LinearLayer(d_model, "ff_down")
        self.dropout = DropoutLayer(dropout)

    @graph_def
    def on(self, X):
        relu_d = tf.nn.relu(self.linear1.on(X), name="relu")
        return self.linear2.on(self.dropout.on(relu_d))


class ScaledDotProdAttentionLayer(LayerABC):
    def __init__(self, scale, dropout, name):
        super(ScaledDotProdAttentionLayer, self).__init__(name)
        self.scale = scale
        self.scores = None
        self.dot = None
        self.dropout = DropoutLayer(dropout)

    @graph_def
    def on(self, Q, K, V):
        '''
        Q: queries [ minibatch x queries x dim_k]
        K: keys    [ minibatch x keys x dim_k]
        V: values  [ minibatch x keys x dim_v]
        '''
        self.dot = tf.einsum('mqd,mkd->mqk', Q, K, name='dot')
        self.scores = tf.nn.softmax(self.scale * self.dot, name='scores')
        dropped_scores = self.dropout.on(self.scores)
        A = tf.einsum('mqk,mkd->mqd', dropped_scores, V, name='a')
        return A


class MultiHeadAttention(LayerABC):
    def __init__(self, dropout, d_model, h, name="multihead"):
        '''Implement the multiheaded self attention
        '''
        super(MultiHeadAttention, self).__init__(name)
        self.h = h
        self.d_k = d_model // h
        self.d_model = d_model
        self.scale = 1 / np.sqrt(self.d_k)

        self.attentions = []
        self.heads = []
        Head = namedtuple("Head", ["to_q", "to_k", "to_v", "attn"])
        for i in range(h):
            q = LinearLayer(self.d_k, "q")
            k = LinearLayer(self.d_k, "k")
            v = LinearLayer(self.d_k, "v")
            attn = ScaledDotProdAttentionLayer(self.scale, dropout, "attn")
            self.heads.append(Head(q, k, v, attn))
        self.a = None
        self.out_layer = LinearLayer(self.d_model, "O")

    @graph_def
    def on(self, X):
        for i, h in enumerate(self.heads):
            with tf.variable_scope("h{}".format(i)):
                q = h.to_q.on(X)
                k = h.to_k.on(X)
                v = h.to_v.on(X)
                a = h.attn.on(q, k, v)
                self.attentions.append(a)

        self.a = tf.concat(self.attentions, axis=-1, name="A")
        return self.out_layer.on(self.a)
