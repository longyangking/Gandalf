import numpy as np 
import sklearn
from sklearn.neural_networks import MLPRegressor

class NeuralNetworks:
    def __init__(self, input_size, output_size, hidden_layers, **args):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers

        self.activation = "relu"
        self.solver = "adam" 
        self.alpha = 0.0001 
        self.batch_size = "auto"  
        self.learning_rate_init = 0.001 
        self.max_iter = 200  
        self.tol = 0.0001
        self.verbose = False 
        self.momentum = 0.9
        self.validation_fraction = 0.1 
        self.epsilon = 1e-08
    
        if "verbose" in args:
            self.verbose = args["verbose"]
        if "activation" in args:
            self.activation = args["activation"]
        if "solver" in args:
            self.solver = args["solver"]
        if "alpha" in args:
            self.solver = args["alpha"]
        if "batch_size" in args:
            self.batch_size = args["batch_size"]
        if "learning_rate" in args:
            self.learning_rate_init = args["learning_rate"]
        if "max_iter" in args:
            self.max_iter = args["max_iter"]
        if "tol" in args:
            self.tol = args["tol"]
        if "momentum" in args:
            self.momentum = args["momentum"]
        if "validation_fraction" in args:
            self.validation_fraction = args["validation_fraction"]
        if "epsilon" in args:
            self.epsilon = args["epsilon"]

        self.model = MLPRegressor(hidden_layer_sizes=self.hidden_layers)
    
    def predict(self, X):
        self.model.predict(X)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def update(self, X_train, y_train):
        self.model.partial_fit(X_train, y_train)

    def evaluate(self, X, y):
        score = self.model.score(X, y)
        return score