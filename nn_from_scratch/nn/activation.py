import numpy as np

"""
    Definition of the differents activation functions
    We add also the derivative that we will use in the training 
"""

class Linear:
    def __init__(self):
        super().__init__()
        self.name = "Linear"
    def __call__(self, x):
        return x
    def derivative(self, x):
        return 1

class Relu:
    def __init__(self):
        super().__init__()
        self.name = "Relu"
    def __call__(self, x):
        x = np.array(x)
        return np.where(x>0, x, 0)
    def derivative(self, x):
        x = np.array(x)
        return (x>0).astype(int)

class Sigmoid:
    def __init__(self):
        super().__init__()
        self.name = "Sigmoid"
    def __call__(self, x):
        x = np.array(x)
        return 1 / (1 + np.exp(-x))
    def derivative(self, x):
        return self(x)*(1 - self(x))

class Tanh:
    def __init__(self):
        super().__init__()
        self.name = "Tanh"
    def __call__(self, x):
        x = np.array(x)
        return np.tanh(x)
    def derivative(self, x):
        return 1 - np.square(self(x))
