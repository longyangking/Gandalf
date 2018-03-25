import sys
sys.path.append("..")

import numpy as np
from gandalf.solver import Solver

if __name__=='__main__':
    def func(x):
        x = np.array(x)
        A = np.array([[2,1],[1,2]])
        return A.dot(x)

    scopes = [[-1,1],[-1,1]]
    solver = Solver(func=func,scopes=scopes)
    solver.train(verbose=False)

    x0 = [0.5,0.5]
    b = func(x0)
    x = solver.evaluate_single(b)
    error = np.average(np.square(x-x0)/np.square(x0))
    print("Test Case:\r\n X0 = {x0}\r\n X = {x}\r\n Res Error: {error:0.4f}".format(x0=x0,x=x,error=error))
    