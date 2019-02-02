from rowlang.models.base_layers import LayerABC, DropoutLayer, LayerNormLayer
from rowlang.models.composite_layers import MultiHeadAttention, FeedForwardLayer
from rowlang.utils import graph_def


class EncoderSubLayer(LayerABC):
    def __init__(self, sublayer, dropout, d_model,  name, *args, **kwargs):
        super(EncoderSubLayer, self).__init__(name)
        self.dropout = DropoutLayer(dropout)
        self.sublayer = sublayer(dropout, d_model, *args, **kwargs)
        self.layer_norm = LayerNormLayer()

    @graph_def
    def on(self, X):
        return X + self.dropout.on(self.sublayer.on(self.layer_norm.on(X)))


class EncoderLayer(LayerABC):
    def __init__(self, dropout, d_model, heads, d_ff, name):
        super(EncoderLayer, self).__init__(name)
        self.mha_sublayer = EncoderSubLayer(MultiHeadAttention, dropout, d_model,
                                            "self_attn", heads)
        self.ffwd_sublayer = EncoderSubLayer(FeedForwardLayer, dropout, d_model,
                                             "feed_fwd", d_ff)

    @graph_def
    def on(self, X):
        return self.ffwd_sublayer.on(self.mha_sublayer.on(X))
