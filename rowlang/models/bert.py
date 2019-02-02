from rowlang.models.base_layers import LayerABC
from rowlang.models.encoder_layers import EncoderLayer
from rowlang.utils import graph_def


class BERTModel(LayerABC):
    def __init__(self, layers, dropout, d_model, heads, d_ff, name="BERT"):
        super(BERTModel, self).__init__(name)
        self.layers = []
        for i in range(layers):
            enc = EncoderLayer(dropout, d_model, heads,
                               d_ff, "layer{}".format(i))
            self.layers.append(enc)

    @graph_def
    def on(self, X):
        for layer in self.layers:
            X = layer.on(X)
        return X
