import numpy as np
import nn.initializer as init
from nn.activation import Relu,Linear
from  nn.updater import GradientUpdater
from nn.losses import BinaryCrossEntropy

class NeuralNetwork:
    """
        Define a dense neural network to perform classification or regression
    """
    def __init__(self,input_shape, name:str = "NN", initializer=init.RandomInitializer):
        """
            @input_shape: the shape of the input data
            @initializer: The type of initializer to use to init the different layers
            @layers: the list of layers
            @last_out_shape: to keep track of the output of the last layer
            @layer_index
        """
        self.input_shape = input_shape
        self.initializer = initializer()

        self.layers = {}
        self.last_out_shape = input_shape
        self.layer_index = 1

    def add_layer(self, units:int, activation=Linear):
        """
            Add new layer to the network
        """
        W ,b = self.initializer(self.last_out_shape, units)
        self.layers["layer"+str(self.layer_index)]= (W,b,activation())

        self.last_out_shape = units
        self.layer_index += 1

    def define_loss(self, loss):
        """
            Define the loss to use to evaluate the model
        """
        self.loss = loss()

    def define_updater(self, updater=GradientUpdater, learning_rate=0.01):
        """
            Define the type of upater to use
        """
        self.updater = updater(learning_rate=learning_rate)

    def train(self, x_train, y_train, max_iter, lambd=0, verbose=False):
        """
            Training of the model
        """
        error_list = []

        x_train, y_train = np.array(x_train).T, np.array(y_train).T
        for i in range(1,max_iter+1):
            self.output = self.forward(x_train)
            self.backprop(x_train, y_train, i, lambd)


            error_list.append(self.loss(y_train,self.output))
            if verbose and i%100 == 0:
                print(f"The error at the iteration {i} is {error_list[i-1]}")

        print(f"The train error  is {self.loss(y_train,self.output)}")
        return error_list

    def forward(self,A):
        """
            Define the forward process
        """
        self.cache = {}
        last_output = A
        for layer in self.layers:
            W, b, activation = self.layers[layer]
            Z = np.dot(W,last_output) + b
            A = activation(Z)
            self.cache[layer] = (Z,A)
            last_output = A

        return last_output

    def backprop(self,x_train, y_train,iteration, lambd):
        """
            Do the backpropagation to update the parameters of the model
        """
        self.compute_grad(x_train, y_train, lambd)
        self.updater(self.layers,self.grads,iteration)

    def compute_grad(self,X, y,lambd=0):
        """
            Compute the gradient of each parameter of the model
            @X: the training data
            @y: the training label
            @lambd: the l2 regularization term
        """
        dA = self.loss.derivative(y, self.output)
        layers = list(self.layers.keys())
        m = X.shape[1]

        self.grads = {}
        for i in range(len(layers)-1,-1,-1):
            (W,b,activation) = self.layers[layers[i]]
            (Z,A) = self.cache[layers[i]]
            (Z_prec, A_prec) = self.cache[layers[i-1]]

            if i==0:
                A_prec = X

            dZ = dA * activation.derivative(Z)
            dW = 1/m * np.dot(dZ, A_prec.T) + lambd*W
            db = 1/m * np.sum(dZ, axis=1, keepdims=True) + lambd*b

            self.grads[layers[i]] = (dW, db)

            dA = np.dot(W.T, dZ)

    def predict(self, X):
        """
            Make prediction on a new data
        """
        X = np.array(X.T)
        y_pred = self.forward(X)
        return y_pred

    def describe(self):
        print("Layer \tInput_shape \tOutput_shape \tActivation")
        for layer in self.layers:
            (W,b,activation) = self.layers[layer]
            print(f"{layer} \t({W.shape[1]}, n_data) \t({W.shape[0]}, n_data) \t{activation.name}")

    def show_loss(self, X, y):
        y = np.array(y).T
        y_pred = self.predict(X)
        self.loss(y, y_pred)

class BinaryClassifier(NeuralNetwork):
    """
        A Neural network to do binary classification
    """
    def __init__(self,input_shape, name:str = "BinaryClassifier", initializer=init.RandomInitializer):
        super().__init__(input_shape, name, initializer)
        super().define_loss(loss=BinaryCrossEntropy)

    def predict(self, X):
        y_pred = super().predict(X)
        y_class = (y_pred>0.5).astype(int)
        return y_class

    def score(self, X, y):
        y_class = self.predict(X)
        y = np.array(y).T

        return np.average(y == y_class)
