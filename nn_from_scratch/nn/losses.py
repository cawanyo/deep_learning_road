import numpy as np
from nn.activation import Sigmoid

"""
    Definition of the differents losses and their derivative
"""


class BinaryCrossEntropy:
    """
        Binary cross entropy to compute the loss for binary classification
    """
    def __init__(self):
        self.name = "binary_cross_entropy"

    def __call__(self, y, y_pred, from_logits=False):
        m = y.shape[1]
        if from_logits:
            y_pred = Sigmoid()(y_pred)

        cost = -1/m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return np.squeeze(cost)
    def derivative(self, y, y_pred, from_logits=False):
        if from_logits == True:
            y_pred = Sigmoid()(y_pred)

        result = - (np.divide(y, y_pred) - np.divide(1-y, 1-y_pred))

        if from_logits:
            result = Sigmoid().derivative(result)
        return result


def main():
    y = np.random.randn(1,10)>0
    y = y.astype(int)
    y_pred = np.random.rand(1,10)

    loss = BinaryCrossEntropy()
    print(loss(y,y_pred, from_logits=True))
    print(loss.derivative(y,y_pred, from_logits=True))


if __name__ == '__main__':
    main()
