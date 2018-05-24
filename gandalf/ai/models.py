import numpy as np 

class Model(object):
    '''
    Abstract base model class
    '''
    def __init__(self, **kwargs):
        self.input_shape = None
        self.output_shape = None

        allowed_kwargs = {
            'input_shape',
            'output_shape',
            'name',
        }

        for kwarg for kwargs:    
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not supported:', kwarg)

        name = kwargs.get('name')
        self.name = name

        self.model = None

    def predict(self, Xs):
        return self.model.predict(Xs)

    def train(self, Xs, ys, batch_size, epochs, verbose, validation_split=0.0):
        loss_info = self.model.fit(Xs, ys,
            batch_size = batch_size,
            epochs = epochs,
            verbose = verbose,
            validation_split = validation_split
        )
        return loss_info

    def evaluate(self, Xs, ys, batch_size, verbose=0):
        loss_info = self.model.evaluate(Xs, ys,
            batch_size = batch_size,
            verbose = verbose
        )
        return loss_info
