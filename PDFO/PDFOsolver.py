import pdfo

# pdfo for unconstrained problem
class PDFOsolver:
    def __init__(self,prob):
        '''
        input:
        prob -- the prob instance defined by us(i.e., UnconstrProb, CutestUnconstrProb); contain all the information of the problem
        (including x0 and a method to get function vslue at certain point:get_func_val(self,x))
        '''
        self.prob = prob
        self.res = None
        self.max_nfev = None
        self.max_iter = None

    def init_solver(self,max_iter,max_nfev):
        self.max_iter = max_iter
        self.max_nfev = max_nfev

    def solve(self):
        '''
        run the pdfo solver
        '''
        if self.prob.prob_name == 'self constructed rosenbrock':
            self.res = pdfo.pdfo(self.prob.obj_func, self.prob.x0,options={'ftarget':1e-5,'maxfev':self.max_nfev})
        else:
            self.res = pdfo.pdfo(self.prob.obj_func, self.prob.x0,options={'maxfev':self.max_nfev})
    
    def display_result(self):
        '''
        the output result is:
        message; str
        Description of the exit status specified in the status field (i.e., the cause of the termination of the solver).

        success; bool
        Whether the optimization procedure terminated successfully.

        status; int
        Termination status of the optimization procedure.

        fun; float
        Objective function value at the solution point.

        x; numpy.ndarray, shape (n,)
        Solution point.

        nfev; int
        Number of function evaluations.

        fun_history; numpy.ndarray, shape (nfev,)
        History of the objective function values.

        method; str
        Name of the Powell method used.
        '''
        print(self.res)

    def draw_result(self,ax):
        '''
        input:
        ax
        '''
        # marker='o'
        ax.plot(self.res.fun_history, linestyle='-', color='b', label='pdfo')
        ax.set_xlabel('number of function evaluations')
        ax.set_ylabel('function value at each function evaluation')
        ax.legend()


