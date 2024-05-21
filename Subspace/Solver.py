import numpy as np
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# record all the needed information for output
class his_manager:
    def __init__(self):
        self.message = None # str; Description of the exit status specified in the status field
        self.success = None # bool; Whether the optimization procedure terminated successfully
        self.y_star = None # float; Objective function value at the solution point
        self.x_star = None # numpy.ndarray, shape (n,); Solution point
        self.nfev = 0 # int; Number of function evaluations
        self.niter = 0 # int; number of iterations
        self.x_iter_his = [] # list of ndarray; history of iteration points
        self.y_iter_his = [] # list of float(niter); history of function values at iteration points
        self.x_total_his = [] # list of ndarray; history of function evaluation points
        self.y_total_his = [] # list of float(nfev); history of function values at function evaluation points

# generate random points on sphere       
def generate_rand_points_on_sphere(n, num_points=1):
    '''
    input:
    n -- int; dimension of the space
    num_points -- int; number of points
    '''
    points = np.random.randn(n, num_points)
    points /= np.linalg.norm(points, axis=0)
    return points

# get quadratic regeression model
def quadratic_regression(X, y):
    '''
    input:
    X -- ndarray; sample points, every row is a point, so the input in solver should be transposed
    y -- 1D array; corresponding function value
    output:
    constant_term -- float; constant of the quadratic model
    linear_params -- ndarray, shape(n,1); gradient of the quadratic model
    Hessian -- ndarray, shape(n,n); hessian matrix of the quadratic model
    '''
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    model = LinearRegression().fit(X_poly, y)

    params = model.coef_
    intercept = model.intercept_

    constant_term = intercept

    n = X.shape[1]
    linear_params = params[:n].reshape(-1, 1)
    quadratic_params = params[n:]
    
    Hessian = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i, n):
            Hessian[i, j] = quadratic_params[idx]
            Hessian[j, i] = quadratic_params[idx]
            idx += 1

    return constant_term, linear_params, Hessian

# a function that is needed in the truncated CG
# a is s, b is p, c is trust region radius
def solve_for_t(a, b, c):
    A = np.dot(a.T, a)
    B = np.dot(a.T, b)
    C = np.dot(b.T, b)

    # Coefficients for the quadratic equation
    a_coef = C
    b_coef = 2 * B
    c_coef = A - c**2

    # Calculate the discriminant
    discriminant = b_coef**2 - 4 * a_coef * c_coef

    if discriminant < 0:
        raise ValueError("No real solution exists for the given c.")

    # Calculate the two possible values of t
    t1 = (-b_coef + np.sqrt(discriminant)) / (2 * a_coef)

    return t1


# subspace solver
class solver:
    def __init__(self,prob):
        self.prob = prob
        self.his = his_manager()
        
        # basic information
        self.x0 = prob.x0
        self.dim = prob.dim
        self.obj_func = prob.obj_func
        
        # parameters needed to be initialized
        self.max_iter = None # int; max number of iterations
        self.rou_1 = None # function; sufficient decrease function, if satisfied, put it into direction basis of the subspace
        self.rou_2 = None # function; sufficient decrease function, if satisfied, adopt the current point and jump to next iteration immediately
        # mom
        self.mom_num = None # int; number of momentums in consideration
        # ds
        self.ds_point_pair_num = None # int; number of random points pairs generated on the unit sphere when doing direct search trial
        self.ds_step_size_tol = None # float; step size tolerance of direct search trial
        # gd
        self.gd_num = None # int; number of estimated gradient(based on probability method estimation)
        self.gd_sample_num = None # int; number of sample points to estimate each gradient
        # trust region
        self.tr_radius_max = None
        self.neta = None
        self.rou_1_bar = None
        self.rou_2_bar = None
        self.gamma_1 = None
        self.gamma_2 = None


        # attributes needed to be initialized and are keep changing during the iteration
        self.x_current = None
        self.y_current = None
        # mom
        self.x_previous = None # ndarray; previous points, every list is a point, the left is the newest
        self.mom_step_size = None # float; step size when doing momentum trial
        self.mom_directions = np.zeros((self.dim,1)) # ndarray; momentums which will be used in subspace construction 
        # ds
        self.ds_step_size = None # float; step size of direct search
        self.ds_directions = np.zeros((self.dim,1)) # ndarray; direct search directions which will be used in subspace construction(NORMALIZED)
        # gd
        self.gd_step_size = None # float; step size of gradient trial
        self.gd_sample_size = None # float; the radius of the sphere on which random samples are generated to estimate gradient
        self.gd_directions = np.zeros((self.dim,1)) # ndarray; gradients which will be used in subspace construction
        # subspace
        self.sub_basis = None # ndarray; subspace basis consist of mom_directions, ds_directions, gd_directions
        self.sub_dim = None # int; subspace dimension
        # model
        self.const = None # constant term of the quadratic model constricted on subspace
        self.g = None # gradient term
        self.H = None # hessian term
        self.tr_radius = None # trust region radius

    def get_func_val(self,x):
        self.his.nfev += 1
        self.his.x_total_his.append(x)
        self.his.y_total_his.append(self.obj_func(x))
        return self.obj_func(x)
    
    def next_iter(self,x,y):
        self.update_x_previous()
        self.his.niter += 1
        self.his.x_iter_his.append(x)
        self.his.y_iter_his.append(y)
        self.x_current = x
        self.y_current = y

    # update_x_previous before next_iter, which has been integrated into next_iter
    def update_x_previous(self):
        self.x_previous[:,-1] = self.x_current[:,0]
        self.x_previous = self.x_previous[:, [-1, *range(self.mom_num - 1)]]
        
    def init_solver(self,max_iter = 100000,
                    mom_num=1,mom_step_size_0=1,
                    ds_point_pair_num=1,ds_step_size_tol=1e-8,ds_step_size_0=10,
                    gd_num=1,gd_sample_num=20,gd_step_size_0=1,gd_sample_size_0=1,
                    tr_radius_0=1,
                    tr_radius_max=10,neta=1e-2,rou_1_bar=0.25,rou_2_bar=0.75,gamma_1=0.25,gamma_2=2):
        '''
        input:
        max_iter
        mom_num
        mom_step_size_0 -- float; initial step size when doing momentum trial
        ds_point_pair_num
        ds_step_size_tol
        ds_step_size_0 -- float; the initial step size for direct search trial
        gd_num
        gd_sample_num -- int; number of sample points/float smaller than 1; proportion of sample points to n
        gd_step_size_0 -- float; initial step size when doing gradient trial
        gd_sample_size_0 -- float; initial step size of estimating gradient
        tr_radius_0 -- float; initial trust region radius
        '''
        # parameters needed to be initialized
        self.max_iter = max_iter
        def rou_1(alpha):
            return 0.1*(alpha**2)
        self.rou_1 = rou_1
        def rou_2(alpha):
            return 0.5*(alpha**2)
        self.rou_2 = rou_2
        # mom
        self.mom_num = mom_num
        # ds
        self.ds_point_pair_num = ds_point_pair_num
        self.ds_step_size_tol = ds_step_size_tol
        # gd
        self.gd_num = gd_num
        if gd_sample_num < 1:
            self.gd_sample_num = math.ceil(gd_sample_num*self.dim)
        else:
            self.gd_sample_num = gd_sample_num
        # trust region
        self.tr_radius_max = tr_radius_max
        self.neta = neta
        self.rou_1_bar = rou_1_bar
        self.rou_2_bar = rou_2_bar
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2

        # attributes that are keep changing during the iteration
        self.x_current = self.x0.reshape(self.dim,1)
        self.y_current = self.get_func_val(self.x_current)
        # ds
        self.ds_step_size = ds_step_size_0
        # self.ds_directions = None
        # mom
        self.x_previous = np.random.rand(self.dim,mom_num)
        self.mom_step_size = mom_step_size_0
        # self.ds_directions = None
        # gd
        self.gd_step_size = gd_step_size_0
        self.gd_sample_size = gd_sample_size_0
        # model
        self.tr_radius = tr_radius_0

    def mom_step(self):
        mom_flag = 0 
        # 0: just default momentum has been chosen into subspace basis(the newest/left)
        # 1: some sufficient decrease momentums have been chosen into subsapce basis
        # 2: immediate adoption takes place
        momentum = self.x_current-self.x_previous
        X_temp = self.x_current+self.mom_step_size*momentum
        mom_directions_index = []
        for i in range(self.mom_num):
            x_temp = X_temp[:,i].reshape(self.dim,1)
            y_temp = self.get_func_val(x_temp)

            if y_temp < self.y_current - self.rou_2(self.mom_step_size):
                mom_flag = 2
                self.next_iter(x_temp,y_temp)
                return mom_flag
            elif y_temp < self.y_current - self.rou_1(self.mom_step_size):
                mom_flag = 1
                mom_directions_index.append(i)
        if mom_flag == 1:
            self.mom_directions = momentum[:,mom_directions_index]
        else:
            self.mom_directions = momentum[:,[0]]
        return mom_flag
    
    def ds_step(self):
        D_positive = generate_rand_points_on_sphere(self.dim, num_points=self.ds_point_pair_num)
        D_negative = D_positive * (-1)
        D = np.hstack((D_positive, D_negative))

        ds_flag = 0
        # 0: no direct search direction has been chosen into subspace basis
        # 1: some sufficient decrease directions have been chosen into subsapce basis
        # 2: immediate adoption takes place
        ds_directions_index = []
        for i in range(2*self.ds_point_pair_num):
            d = D[:,[i]]
            x_temp = self.x_current + self.ds_step_size*d
            y_temp = self.get_func_val(x_temp)

            if y_temp < self.y_current - self.rou_2(self.ds_step_size):
                ds_flag = 2
                self.next_iter(x_temp,y_temp)
                return ds_flag
            
            if y_temp < self.y_current - self.rou_1(self.ds_step_size):
                ds_flag = 1
                ds_directions_index.append(i) 

        if ds_flag == 1:
            self.ds_directions = D[:,ds_directions_index]
        else:
            pass
        return ds_flag
    
    def gd_step(self):
        gd_flag = 0
        # 0: just total average gradient has been chosen into subspace basis(only 1)
        # 1: some sufficient decrease gradients have been chosen into subsapce basis
        # 2: immediate adoption takes place
        gd_directions = []
        grad_sum = np.zeros((self.dim,1))
        for i in range(self.gd_num):
            B = generate_rand_points_on_sphere(self.dim, num_points=self.gd_sample_num)
            grad = np.zeros((self.dim,1))
            for j in range(self.gd_sample_num):
                d = B[:,[j]]
                x_sample = self.x_current + self.gd_sample_size*d
                y_sample = self.get_func_val(x_sample)
                grad += ((y_sample-self.y_current)/self.gd_sample_size)*d
            grad /= self.gd_sample_num
            grad_sum += grad

            x_temp = self.x_current + self.gd_step_size*grad
            y_temp = self.get_func_val(x_temp)
            if y_temp < self.y_current - self.rou_2(self.gd_step_size):
                gd_flag = 2
                self.next_iter(x_temp,y_temp)
                return gd_flag
            
            if y_temp < self.y_current - self.rou_1(self.gd_step_size):
                gd_flag = 1
                gd_directions.append(grad)

        if gd_flag == 1:
            self.gd_directions = np.column_stack(gd_directions)
        else:
            self.gd_directions = grad_sum/self.gd_num
        return gd_flag
    
    # preprocess subspace basis, which has been integrated into construct_model
    def preprocess(self):
        epsilon = 1e-8
        combined = np.hstack((self.mom_directions, self.ds_directions, self.gd_directions))
        norms = np.linalg.norm(combined, axis=0)
        filtered = combined[:, norms > epsilon]
        # make this process more robust, filtered can be empty, then we can generate a vec randomly
        # TODO: make this process more robust, if there are too many subspace basis, then throw away some of the basis
        if filtered.shape[1] == 0:
            print('zero subspace basis, generate one randomly')
            self.sub_basis = generate_rand_points_on_sphere(self.dim,1)
            self.sub_dim = 1
        elif filtered.shape[1] > self.dim:
            print('too many subspace basis, throw some of them away')
            self.sub_basis = filtered[:,:self.dim]
            self.sub_dim = self.dim
        else:
            self.sub_basis = filtered / np.linalg.norm(filtered, axis=0)
            self.sub_dim = self.sub_basis.shape[1]

    def construct_model(self):
        self.preprocess() # get subspace basis

        I = np.eye(self.sub_dim)

        q = 2*(1+math.ceil((self.sub_dim**2)/4))
        R = generate_rand_points_on_sphere(self.sub_dim,q)
        X = np.hstack((I,-1*I,R))
        P = np.hstack((np.zeros((self.sub_dim,1)),self.tr_radius*X,0.5*self.tr_radius*X))
        y = np.apply_along_axis(self.get_func_val, axis=0, arr=self.x_current+self.sub_basis@P) # 1D array
        # TODO: to make mom,ds,gd parallel computing
        self.const, self.g, self.H = quadratic_regression(P.T, y)

    def truncated_CG(self):
        epsilon = 1e-4
        max_iter = self.sub_dim

        s = np.zeros((self.sub_dim,1))
        r = self.g
        r_norm_0 = np.linalg.norm(r)
        p = -self.g
        k = 0
        while k<max_iter:
            if p.T@self.H@p <=0:
                t = solve_for_t(s,p,self.tr_radius)
                return s + t*p
            alpha = (r.T@r)/(p.T@self.H@p)
            s_new = s + alpha*p
            if np.linalg.norm(s_new) >= self.tr_radius:
                t = solve_for_t(s,p,self.tr_radius)
                return s + t*p
            r_new = r + alpha*(self.H@p)
            if np.linalg.norm(r_new) < epsilon*r_norm_0:
                return s_new
            beta = (r_new.T@r_new)/(r.T@r)
            p = -r_new + beta*p
            k += 1
            s = s_new
            r = r_new
        return s
     
     # update parameter
    def para_upd(self,ds_flag,mom_flag,gd_flag):
        if mom_flag == 2:
            self.mom_step_size *= 1.5
        elif mom_flag == 1:
            self.mom_step_size *= 0.6
        else:
            pass

        if ds_flag == 2:
            self.ds_step_size *= 2
        elif ds_flag == 0:
            self.ds_step_size *= 0.5
        else:
            pass

        if gd_flag == 2:
            self.gd_step_size *= 2
        elif gd_flag == 0:
            self.gd_step_size *= 0.5
        else:
            pass
    
    def get_model_val(self,p):
        return self.const+p.T@self.g+p.T@self.H@p
    
    def test_model(self,p):
        model_val = self.get_model_val(p)
        true_val = self.get_func_val(self.x_current+self.sub_basis@p)
        print('test model accuracy')
        print('model value vs true value: ',model_val,true_val)
        print('\n')

    def solve(self):
        while True:
            mom_flag = self.mom_step()
            if mom_flag != 2:
                ds_flag = self.ds_step()
                if ds_flag != 2:
                    gd_flag = self.gd_step()
                    if gd_flag != 2:
                        self.construct_model()

                        # p_test = generate_rand_points_on_sphere(self.sub_dim,1)*self.tr_radius
                        # self.test_model(p_test)

                        p_star = self.truncated_CG()
                        x_temp = self.x_current+self.sub_basis@p_star
                        y_temp = self.get_func_val(x_temp)

                        y_temp_model = self.get_model_val(p_star)
                        y_current_model = self.get_model_val(np.zeros((self.sub_dim,1)))

                        # print('truncated CG step, true value: y_star vs y current')
                        # print(y_temp,self.y_current)
                        # print('truncated CG step, model value: y_star vs y current')
                        # print(y_temp_model,y_current_model)
                        # print('trust region radius: ',self.tr_radius)

                        rou = (self.y_current-y_temp)/max(1e-8,(y_current_model-y_temp_model))
                        
                        # print('rou: ',rou)
                        # print('\n')

                        if rou < self.rou_1_bar:
                            self.tr_radius *= self.gamma_1
                            # TODO: this can use different paras
                            self.gd_sample_size *= 0.5
                        else:
                            if rou > self.rou_2_bar and np.linalg.norm(p_star) >= 0.5*self.tr_radius:
                                self.tr_radius = min(self.tr_radius*self.gamma_2,self.tr_radius_max)
                                # TODO: this can use different paras
                                self.gd_sample_size *= 1.5
                            else:
                                pass
                        if rou >self.neta:
                            self.next_iter(x_temp,y_temp)
                        else:
                            self.next_iter(self.x_current,self.y_current)

            self.para_upd(ds_flag,mom_flag,gd_flag)

            if self.his.niter >= self.max_iter:
                self.his.x_star = self.x_current
                self.his.y_star = self.y_current
                self.his.message = 'Return from solver because max iteration number has achieved'
                self.his.success = False
                return self.his
            
            if self.ds_step_size < self.ds_step_size_tol:
                self.his.x_star = self.x_current
                self.his.y_star = self.y_current
                self.his.message = 'Return from solver because step size stopping criterion for Random Direct Search has achieved'
                self.his.success = True
                return self.his
        
    def display_result(self):
        res = self.his
        print('\nmessage: ',res.message)
        print('\nsuccess: ',res.success)
        print('\nnfev: ',res.nfev) # number of total function evaluations
        print('\nniter: ',res.niter) # number of total iterations
        # print('\nx: ',res.x_star) # solution point
        print('\nfun: ',res.y_star) # objective function value at solution point