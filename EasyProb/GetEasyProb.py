import numpy as np

def rosenbrock(x):
    '''
    input:
    x -- ndarray; current point
    '''
    a = 1.0
    b = 100.0
    return np.sum(b * (x[1:] - x[:-1] ** 2) ** 2 + (a - x[:-1]) ** 2)


# this class is defined for an easy unconstrained problem created by myself
class UnconstrProb:
    def __init__(self,name='self constructed rosenbrock'):
        self.prob_name = name # string; the name of the problem in cutest
        self.dim = None # integer; problem dimension
        self.x0 = None # ndarray; start point of the iteration
        self.obj_func = None # function; objective function of the problem

    def init_prob_dim(self,dim=10):
        '''
        initialize problem dimension at first and initialize a default start point(0)
        '''
        self.dim = dim
        self.x0 = np.zeros(dim)

    def init_prob_x0(self,x0):
        '''
        initialize start point of iteration
        '''
        self.x0 = x0

    def get_obj_func(self):
        def obj_func(x):
            '''
            given the present point, output the function value at this point
            we use rosenbrock function as default, other function can be used by modifying the code here(TODO)
            input:
            x -- ndarray; current point
            output:
            float; function value at this point
            '''
            return rosenbrock(x)
        self.obj_func = obj_func