import sys
sys.path.append("..")

import numpy as np
from gandalf.solver import Solver
import matplotlib.pyplot as plt

if __name__=='__main__':
    n_input = 8
    n_points = n_input*n_input
    A = np.random.random((n_points, n_points))
    def func(x):
        x = np.array(x)
        return A.dot(x)

    scopes = [[0,1] for i in range(n_points)]
    solver = Solver(func=func,scopes=scopes)
    solver.train(epochs=3000,verbose=True)

    x0 = np.random.random(n_points)
    b = func(x0)
    x = solver.evaluate_single(b)
    error = np.average(np.square(x-x0)/np.square(x0))
    print("Res Error: {error:0.4f}".format(error=error))

    plt.subplot(121) 
    plt.imshow(x0.reshape((n_input,n_input)),cmap="hot")
    plt.title("X0")
    plt.subplot(122) 
    plt.imshow(x.reshape((n_input,n_input)),cmap="hot")
    plt.title("X")
    plt.show()
    