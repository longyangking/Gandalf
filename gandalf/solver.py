import numpy as np 
import sklearn
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

class Solver:
    def __init__(self, func, scopes):
        self.func = func
        self.scopes = np.array(scopes)

        self.model = None
    
    def train(self, epochs=1e3, verbose=False):
        self.model =  MultiOutputRegressor(MLPRegressor(solver='lbfgs', alpha=1e-5,
                hidden_layer_sizes=(100, 30), random_state=1))
        
        n_variables = len(self.scopes)
        xmin = self.scopes[:,0]
        xmax = self.scopes[:,1]

        Xs = list()
        Ys = list()
        if verbose:
            print("Generating training data...",end="")
        for i in range(int(epochs)):
            x = xmin + (xmax - xmin)*np.random.random(n_variables)
            Xs.append(self.func(x))
            Ys.append(x)
            if (i+1)%int(epochs/10)==0 and verbose:
                print(" {value:0.0f}% ".format(value=(i+1)/int(epochs)*100),end="")

        if verbose:
            print("Complete!")
        #Xs = np.array(Xs)
        #Ys = np.array(Ys)
        if verbose:
            print("Training model...",end='')
        self.model.fit(Xs,Ys)
        if verbose:
            print("End with R^2: {value:0.4f}".format(value=self.model.score(Xs,Ys)))

    def evaluate(self, bs):
        return self.model.predict(bs)

    def evaluate_single(self, b):
        return self.model.predict([b])[0]