import numpy as np 

class Dataset:
    def __init__(self, func, nvar, scopes=None, alpha=0.1, beta=0.0):
        self.func = func
        self.nvar = nvar
        self.scopes = scopes

        self.alpha = alpha
        self.beta = beta
    
    def generate(self, nb_set):
        if self.scopes is None:
            Xs =  self.__sample_without_scope(nb_set)
        else:
            Xs = self.__sample_with_scope(nb_set)

        bs = [self.func(X) for X in Xs]
        return Xs,bs

    def __sample_with_scope(self, nb_set):
        data = np.random.random((nb_set,self.nvar))
        self.scopes = np.array(scopes)
        lowerband = self.scopes[:,0]
        upperband = self.scopes[:,1]
        for i in range(nb_set):
            data[i] = (upperband - lowerband)*data[i] + lowerband
        return data

    def __sample_without_scope(self, nb_set):
        xs = np.random.random((nb_set,self.nvar))

        # To remove in-valid ill point
        n_zero = n_one = 1
        while (n_zero > 0) and (n_one > 0):
            n_zero = np.sum(xs==0)
            n_one = np.sum(xs==1)
            if n_zero > 0:
                pos = np.where(xs==0)
                xs[pos] = np.random.random(n_zero)
            if n_one > 0:
                pos = np.where(xs==1)
                xs[pos] = np.random.random(n_one)

        data = 1/self.alpha*np.log(1/xs - 1) + self.beta
        return data

class DataGenerator:
    def __init__(self, model):
        self.model = model

    def 
    

if __name__=="__main__":
    def func(x):
        A = np.array([[2,1],[1,1]])
        return A.dot(x)
    nvar = 2
    dataset = Dataset(func=func, nvar=nvar)
    Xs,bs = dataset.generate(nb_set=4)