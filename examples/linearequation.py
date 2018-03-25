import sys
sys.path.append("..")

import numpy as np
from gandalf.solver import Solver
import matplotlib.pyplot as plt

if __name__=='__main__':
    n_dims = 20
    A = np.random.random((n_dims,n_dims))
    def func(x):
        x = np.array(x)
        return A.dot(x)

    scopes = [[0,1] for i in range(n_dims)]
    solver = Solver(func=func,scopes=scopes)
    solver.train(verbose=True)

    x0 = np.random.random(n_dims)
    b = func(x0)
    x = solver.evaluate_single(b)
    error = np.average(np.square(x-x0)/np.square(x0))
    print("Res Error: {error:0.4f}".format(error=error))
    
    plt.plot(range(n_dims),x0,'r',label="x0")
    plt.plot(range(n_dims),x,'b',label="x")
    plt.legend()
    plt.title("Linear Reconstruction")
    plt.show()