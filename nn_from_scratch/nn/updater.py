import numpy as np
"""
    Definition of the main updater
        ***     Simple gradient descent   ***
        ***     Moment algorithm          ***
        ***     RMSProp  algorithm        ***
        ***     Adam algorithm            ***
"""

class GradientUpdater:
    """
        For the gradient updater we just take the grad and update the weights
        @learning_rate: the learning rate of the gradient descent
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def __call__(self, layers, grads, iteration=0):
        for layer in layers:
            (W,b,activation) = layers[layer]
            (dW, db) = grads[layer]

            W -= self.learning_rate*dW
            b -= self.learning_rate*db

            layers[layer] = (W,b,activation)

        return layers, grads

class MomentumUpdater:
    """
        For the momentum we need to compute the moving average of the grads before
        updating the weights
        @learning_rate:
        @self.beta: the beta value to compute the moving average
        @self.v if the moving average of the grads
    """
    def __init__(self, learning_rate=0.1, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self.v = {}
        self.already_called = False

    def initialize(self, layers):
        self.already_called = True
        for layer in layers:
            (W,b,activation) = layers[layer]
            dW_v = np.zeros(W.shape)
            db_v = np.zeros(b.shape)

            self.v[layer] = (dW_v, db_v)

    def __call__(self, layers, grads, iteration=0):
        """
            Update the value of the parameters W and b of each layer
        """

        if self.already_called == False:
            self.initialize(layers)

        for layer in layers:
            (dW, db) = grads[layer]
            (dW_v, db_v) = self.v[layer]
            (W,b,activation) = layers[layer]

            dW_v = self.beta*dW_v + (1-self.beta)*dW
            db_v = self.beta*db_v + (1-self.beta)*db

            #correction
            #scaler = round(1 - self.beta**iteration,2)
            #dW_v = dW_v / np.array(scaler)
            #db_v = db_v / scaler

            W -= self.learning_rate*dW_v
            b -= self.learning_rate*db_v

            layers[layer] = (W,b,activation)
            self.v[layer] = (dW_v, db_v)


class RMSPropUpdater:
    """
        In the RMSProp we scale the grads before doing the update
    """
    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.s = {}
        self.already_called = False

    def initialize(self, layers):
        self.already_called = True
        for layer in layers:
            (W,b,activation) = layers[layer]
            dW_s = np.zeros(W.shape)
            db_s = np.zeros(b.shape)

            self.s[layer] = (dW_s, db_s)

    def __call__(self, layers, grads, iteration=0):
        """
            Update the value of the parameters W and b of each layer
        """

        if self.already_called == False:
            self.initialize(layers)

        for layer in layers:
            (dW, db) = grads[layer]
            (dW_s, db_s) = self.s[layer]
            (W,b,activation) = layers[layer]

            dW_s = self.beta*dW_s + (1-self.beta)*np.square(dW)
            db_s = self.beta*db_s + (1-self.beta)*np.square(db)

            #correction
            #scaler = round(1 - self.beta**iteration,2)
            #dW_s = dW_s / np.array(scaler)
            #db_s = db_s / scaler

            W -= self.learning_rate*(dW / (np.sqrt(dW_s) + self.epsilon))
            b -= self.learning_rate*(db / (np.sqrt(db_s) + self.epsilon))

            layers[layer] = (W,b,activation)
            self.s[layer] = (dW_s, db_s)


class AdamUpdater:
    """
        In the adam we combine the momentum and the RMSProp
        we also apply decay to it
    """
    def __init__(self, learning_rate, beta1=0.9, beta2=0.9, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.already_called = False

        self.v = {}
        self.s = {}

    def initialize(self, layers):
        self.already_called = True
        for layer in layers:
            (W,b,activation) = layers[layer]
            dW_v = np.zeros(W.shape)
            db_v = np.zeros(b.shape)

            dW_s = np.zeros(W.shape)
            db_s = np.zeros(b.shape)

            self.v[layer] = (dW_v, db_v)
            self.s[layer] = (dW_s, db_s)

    def __call__(self, layers, grads, iteration=0):
        """
            Update the value of the parameters W and b of each layer
        """
        if self.already_called == False:
            self.initialize(layers)

        for layer in layers:
            (dW, db) = grads[layer]
            (dW_s, db_s) = self.s[layer]
            (dW_v, db_v) = self.v[layer]
            (W,b,activation) = layers[layer]

            dW_v = self.beta1*dW_v + (1-self.beta1)*dW
            db_v = self.beta1*db_v + (1-self.beta1)*db

            #correction
            #scaler = round(1 - self.beta**iteration,2)
            #dW_v = dW_v / np.array(scaler)
            #db_v = db_v / scaler

            dW_s = self.beta2*dW_s + (1-self.beta2)*np.square(dW)
            db_s = self.beta2*db_s + (1-self.beta2)*np.square(db)

            #correction
            #scaler = round(1 - self.beta1**iteration,2)
            #dW_s = dW_s / np.array(scaler)
            #db_s = db_s / scaler

            W -= self.learning_rate*(dW_v / (np.sqrt(dW_s) + self.epsilon))
            b -= self.learning_rate*(db_v / (np.sqrt(db_s) + self.epsilon))

            layers[layer] = (W,b,activation)
            self.v[layer] = (dW_v, db_v)
            self.s[layer] = (dW_s, db_s)
