from functools import wraps

import tensorflow as tf


def graph_def(func):
    '''Cache layer (w self.outputs) and autoscope (w self.name)'''
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.outputs is None:
            with tf.variable_scope(self.name):
                self.outputs = func(self, *args, **kwargs)
        return self.outputs
    return wrapper
