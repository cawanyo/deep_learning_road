import numpy as np

"""
    Definition of the main initializer
"""

class ZeroInitializer:
    def __init__(self):
        self.name = "zeros initializer"
    def __call__(self, in_shape:int, out_shape:int):
        """
            in_shape: the input shape of the layer
            out_shape: the output shape of the layer
        """
        W = np.zeros((out_shape, in_shape))
        b = np.zeros((out_shape, 1))

        return W, b


class RandomInitializer:
    def __init__(self, seed=None):
        self.name = "random initializer"
        self.seed = seed

    def __call__(self, in_shape:int, out_shape:int):
        """
            in_shape: the input shape of the layer
            out_shape: the output shape of the layer
        """
        if self.seed:
            np.random.seed(self.seed)
        W = np.random.randn(out_shape, in_shape)
        b = np.zeros((out_shape, 1))

        return W, b


class HeInitializer:
    def __init__(self, seed=None):
        self.seed = seed


    def __call__(self, in_shape:int, out_shape:int):
        """
            in_shape: the input shape of the layer
            out_shape: the output shape of the layer
        """
        if self.seed:
            np.random.seed(self.seed)
        W = np.random.randn(out_shape, in_shape) * np.sqrt(2/in_shape)
        b = np.zeros((out_shape, 1))

        return W, b

class XavierInitializer:
    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, in_shape:int, out_shape:int):
        """
            in_shape: the input shape of the layer
            out_shape: the output shape of the layer
        """
        if self.seed:
            np.random.seed(self.seed)
        W = np.random.randn(out_shape, in_shape) * (2/(in_shape+out_shape))
        b = np.zeros((out_shape, 1))

        return W, b
