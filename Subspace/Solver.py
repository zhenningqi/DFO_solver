import numpy as np
import math

import sys
import os
'''
When running the code under the project root directory, getcwd will return the project root directory no matter what directory the code file is in.
However, python will set the directory where the code file is to default path, the project root directory will not be included in the default path.
Thus, if we want to use import, we should first set the project root directory into default path.
'''
current_path = os.getcwd()
print(current_path)
sys.path.append(current_path)

from Subspace.func import generate_rand_points_on_sphere
from Subspace.func import quadratic_regression
from Subspace.func import solve_for_tau

# record all the needed information for output
class his_manager:
    def __init__(self):
        self.message = None # str; Description of the exit status specified in the status field
        self.success = None # bool; Whether the optimization procedure terminated successfully
        self.y_star = None # float; Objective function value at the solution point
        self.x_star = None # numpy.ndarray, shape (n,); solution point
        self.nfev = 0 # int; number of function evaluations
        self.niter = 0 # int; number of iterations
        self.x_iter_his = [] # list of ndarray; history of iteration points
        self.y_iter_his = [] # list of float(niter); history of function values at iteration points
        self.x_total_his = [] # list of ndarray; history of function evaluation points
        self.y_total_his = [] # list of float(nfev); history of function values at function evaluation points

        self.info = [] # list of string; history of interation information: 'mom','ds','gd','tr','stumble','implicit'
        self.mom_step_size_his = [] # list of float; history of momentum search step size
        self.ds_step_size_his = [] # list of float; history of direct search step size
        self.gd_step_size_his = [] # list of float; history of gradient search step size
        self.tr_radius_his = [] # list of float; history of trust region radius

        self.n_mom_explicit = 0 # int; number of successful momentum search
        self.n_mom_implicit = 0 # int; number of implicit momentum search
        self.n_ds_explicit = 0 # int; number of successful direct search
        self.n_ds_implicit = 0 # int; number of implicit direct search
        self.n_gd_explicit = 0 # int; number of successful gradient search
        self.n_gd_implicit = 0 # int; number of implicit gradient search
        self.n_model_implicit = 0
        self.n_stumble = 0 # int; number of stumble step
        self.n_tr = 0 # int; number of trust region step

# subspace solver
class solver:
    def __init__(self,prob):
        self.prob = prob
        self.his = his_manager()
        
        # basic information
        self.x0 = prob.x0.reshape(-1,1)
        self.dim = prob.dim
        self.obj_func = prob.obj_func

        # options
        self.trick_option = None
        self.subspace_option = None
        self.gd_sample_num_option = None
        
        # parameters needed to be initialized
        self.max_iter = None # int; max number of iterations
        self.max_nfev = None # int; max number of function evaluations
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
        self.gd_sample_num_small = None
        self.gd_sample_num_large = None
        # trust region
        self.tr_radius_tol = None # float; stop criterion for the trust region radius
        self.tr_radius_max = None # float; max trust region radius
        self.rou_1_bar = None
        self.rou_2_bar = None
        self.gamma_1 = None
        self.gamma_2 = None


        # attributes needed to be initialized and are keep changing during the iteration
        self.x_current = None
        self.y_current = None
        # mom
        self.inert = None
        self.x_previous = None # ndarray; previous points, every list is a point, the left is the newest
        self.mom_step_size = None # float; step size when doing momentum trial
        self.mom_directions = None # ndarray; momentums which will be used in subspace construction 
        self.mom_flag = None
        self.mom_temp_x = None
        self.mom_temp_y = None
        self.mom_min_x = None
        self.mom_min_y = None
        # ds
        self.ds_step_size = None # float; step size of direct search
        self.ds_directions = None # ndarray; direct search directions which will be used in subspace construction(NORMALIZED)
        self.ds_flag = None
        self.ds_temp_x = None
        self.ds_temp_y = None
        self.ds_min_x = None
        self.ds_min_y = None
        # gd
        self.gd_step_size = None # float; step size of gradient trial
        self.gd_sample_size = None # float; the radius of the sphere on which random samples are generated to estimate gradient
        self.gd_directions = None # ndarray; gradients which will be used in subspace construction
        self.gd_flag = None
        self.gd_temp_x = None
        self.gd_temp_y = None
        self.gd_min_x = None
        self.gd_min_y = None
        # subspace
        self.sub_basis = None # ndarray; subspace basis consist of mom_directions, ds_directions, gd_directions
        self.sub_dim = None # int; subspace dimension
        # model
        self.const = None # float; constant term of the quadratic model constricted on subspace
        self.gradient = None # ndarray, shape(sub_dim,1); gradient vector
        self.hessian = None # ndarray, shape(sub_dim,sub_dim); hessian matrix
        self.model_min_x = None
        self.model_min_y = None
        # TCG(trust region)
        self.tr_radius = None # trust region radius
        self.tcg_x = None 
        self.tcg_y = None 
        self.tr_flag = None
        self.p_star_norm = None

    def init_solver(self,max_iter=100,max_nfev=1000,
                    subspace_option=[1,1,1],trick_option=[0,0,0,0,0],
                    gd_sample_num_option=[2,10],
                    mom_step_size_0=0.1,
                    ds_step_size_tol=1e-8,ds_step_size_0=0.1,
                    gd_step_size_0=0.5,gd_sample_size_0=1e-6,
                    tr_radius_0=0.2,tr_radius_tol=1e-6,
                    tr_radius_max=10,rou_1_bar=0.25,rou_2_bar=0.75,gamma_1=0.8,gamma_2=2):
        '''
        input:
        max_iter
        max_nfev
        mom_num
        mom_step_size_0 -- float; initial step size when doing momentum trial
        ds_point_pair_num
        ds_step_size_tol
        ds_step_size_0 -- float; the initial step size for direct search trial
        gd_num
        gd_sample_num -- int; number of sample points
        gd_step_size_0 -- float; initial step size when doing gradient trial
        gd_sample_size_0 -- float; initial step size of estimating gradient
        tr_radius_0 -- float; initial trust region radius
        tr_radius_tol
        tr_radius_max
        rou_1_bar
        rou_2_bar
        gamma_1
        gamma_2
        '''
        # options
        self.subspace_option = subspace_option
        self.trick_option = trick_option
        self.gd_sample_num_option = gd_sample_num_option
        
        # parameters needed to be initialized
        self.max_iter = max_iter
        self.max_nfev = max_nfev

        def rou_1(alpha):
            return 0.1*(alpha**2)
        self.rou_1 = rou_1
        def rou_2(alpha):
            return 2*(alpha**2)
        self.rou_2 = rou_2
        # mom
        self.mom_num = subspace_option[0]
        # ds
        self.ds_point_pair_num = subspace_option[1]
        self.ds_step_size_tol = ds_step_size_tol
        # gd
        self.gd_num = subspace_option[2]
        if trick_option[1] == 1:
            self.gd_sample_num_small = math.ceil(gd_sample_num_option[0]*self.dim)
            self.gd_sample_num_large = math.ceil(gd_sample_num_option[1]*self.dim)
            self.gd_sample_num = self.gd_sample_num_large
        else:
            self.gd_sample_num = math.ceil(gd_sample_num_option*self.dim)
        # trust region
        self.tr_radius_tol = tr_radius_tol
        self.tr_radius_max = tr_radius_max
        self.rou_1_bar = rou_1_bar
        self.rou_2_bar = rou_2_bar
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2

        # attributes that are keep changing during the iteration
        self.x_current = self.x0.reshape(self.dim,1)
        self.y_current = self.get_func_val(self.x_current)
        # mom
        self.inert = np.zeros((self.dim,1))
        self.x_previous = np.zeros((self.dim,10))
        self.x_previous[:,[0]] = self.x0
        self.mom_step_size = mom_step_size_0
        self.mom_directions = np.zeros((self.dim,1))
        self.mom_flag = 0
        self.mom_min_y = float('inf')
        # ds
        self.ds_step_size = ds_step_size_0
        self.ds_directions = np.zeros((self.dim,1))
        self.ds_flag = 0
        self.ds_min_y = float('inf')
        # gd
        self.gd_step_size = gd_step_size_0
        self.gd_sample_size = gd_sample_size_0
        self.gd_directions = np.zeros((self.dim,1))
        self.gd_flag = 0
        self.gd_min_y = float('inf')
        # model
        self.model_min_y = float('inf')
        # trust region
        self.tr_radius = tr_radius_0
        self.tr_flag = 0

    def get_func_val(self,x):
        self.his.nfev += 1
        self.his.x_total_his.append(x)
        self.his.y_total_his.append(self.obj_func(x))
        return self.obj_func(x)

   # update_x_previous before next_iter, which has been integrated into next_iter
    def update_x_previous(self):
        self.x_previous[:,[-1]] = self.x_current
        self.x_previous = self.x_previous[:, [-1, *range(self.x_previous.shape[1] - 1)]]

    def next_iter(self,x,y):
        self.update_x_previous()
        self.his.niter += 1
        self.his.x_iter_his.append(x)
        self.his.y_iter_his.append(y)
        self.x_current = x
        self.y_current = y
        if self.trick_option[0] == 1:
            self.inert = 0.8*self.inert + 0.2*(self.x_current-self.x_previous[:,[0]])

    def mom_step(self):
        momentum = self.x_current-self.x_previous[:,:self.mom_num]
        if self.trick_option[0] == 1:
            momentum = 0.8*momentum + 0.2*self.inert
        X_temp = self.x_current+self.mom_step_size*momentum

        mom_directions_index = []
        for i in range(self.mom_num):
            x_temp = X_temp[:,i].reshape(self.dim,1)
            y_temp = self.get_func_val(x_temp)

            if y_temp < self.y_current - self.rou_2(self.mom_step_size):
                self.mom_flag = -1
                self.mom_temp_x = x_temp
                self.mom_temp_y = y_temp
                break
            else:
                self.mom_flag += 1
                mom_directions_index.append(i)
                if y_temp <= self.mom_min_y:
                    self.mom_min_x = x_temp
                    self.mom_min_y = y_temp
        if self.mom_flag >= 1:
            self.mom_directions = momentum[:,mom_directions_index]
    
    def ds_step(self):
        D = generate_rand_points_on_sphere(self.dim, num_points=self.ds_point_pair_num)
        ds_directions_index = []
        for i in range(self.ds_point_pair_num):
            d = D[:,[i]]
            x_temp = self.x_current + self.ds_step_size*d
            y_temp = self.get_func_val(x_temp)

            if y_temp < self.y_current - self.rou_2(self.ds_step_size):
                self.ds_flag = -1
                self.ds_temp_x = x_temp
                self.ds_temp_y = y_temp
                break
            else:
                if y_temp <= self.ds_min_y:
                    self.ds_min_x = x_temp
                    self.ds_min_y = y_temp
                if self.trick_option[4] == 1:
                    if y_temp < self.y_current - self.rou_1(self.ds_step_size):
                        self.ds_flag += 1
                        ds_directions_index.append(i)
                        continue 

                x_temp = self.x_current - self.ds_step_size*d
                y_temp = self.get_func_val(x_temp)

                if y_temp < self.y_current - self.rou_2(self.ds_step_size):
                    self.ds_flag = -1
                    self.ds_temp_x = x_temp
                    self.ds_temp_y = y_temp
                    break
            
                else:
                    if y_temp <= self.ds_min_y:
                        self.ds_min_x = x_temp
                        self.ds_min_y = y_temp
                    if self.trick_option[4] == 1:
                        if y_temp < self.y_current - self.rou_1(self.ds_step_size):
                            self.ds_flag += 1
                            ds_directions_index.append(i)
                            continue 
                    else:
                        self.ds_flag += 1
                        ds_directions_index.append(i)
                        continue 

        if self.ds_flag >= 1:
            self.ds_directions = D[:,ds_directions_index]

    def gd_step(self):
        gd_directions = []
        for _ in range(self.gd_num):
            B = generate_rand_points_on_sphere(self.dim, num_points=self.gd_sample_num)
            x_sample = self.x_current + self.gd_sample_size*B
            y_sample = np.apply_along_axis(self.get_func_val, axis=0, arr=x_sample) # 1D array
            
            index = np.argmin(y_sample)
            self.gd_min_x = x_sample[:,[index]]
            self.gd_min_y = y_sample[index]

            grad = np.mean(((y_sample-self.y_current)/self.gd_sample_size)*B, axis=1).reshape(-1,1) # using broadcast
            x_temp = self.x_current - self.gd_step_size*grad
            y_temp = self.get_func_val(x_temp)

            if y_temp < self.y_current - self.rou_2(self.gd_step_size):
                self.gd_flag = -1
                self.gd_temp_x = x_temp
                self.gd_temp_y = y_temp
                break
            
            else:
                self.gd_flag += 1
                gd_directions.append(-1*grad)
                if y_temp <= self.gd_min_y:
                    self.gd_min_x = x_temp
                    self.gd_min_y = y_temp

        if self.gd_flag >= 1:
            self.gd_directions = np.column_stack(gd_directions)
    
    # preprocess subspace basis, which has been integrated into construct_model
    def preprocess(self):
        epsilon = 1e-8
        combined = np.hstack((self.mom_directions, self.ds_directions, self.gd_directions))
        norms = np.linalg.norm(combined, axis=0)
        filtered = combined[:, norms > epsilon]
        Q, _ = np.linalg.qr(filtered, mode='reduced')
        # make this process more robust, filtered can be empty, then we can generate a vec randomly
        # make this process more robust, if there are too many subspace basis, then throw away some of the basis
        if Q.shape[1] == 0:
            print('zero subspace basis, generate one randomly')
            self.sub_basis = generate_rand_points_on_sphere(self.dim,1)
            self.sub_dim = 1
        elif Q.shape[1] > self.dim:
            print('too many subspace basis, throw some of them away')
            self.sub_basis = Q[:,:self.dim]
            self.sub_dim = self.dim
        else:
            self.sub_basis = Q
            self.sub_dim = self.sub_basis.shape[1]
            # print('subspace dimension',self.sub_dim)

    def construct_model(self):
        self.preprocess() # get subspace basis and subspace dimension

        I = np.eye(self.sub_dim)

        q = math.ceil((self.sub_dim**2)/2-self.sub_dim/2) # number of additional random directions needed for construct quadratic model
        R = generate_rand_points_on_sphere(self.sub_dim,q)
        X = np.hstack((I,-1*I,R))

        shuffled_indices = np.random.permutation(X.shape[1])
        split_point = len(shuffled_indices) // 2
        indices_part1 = shuffled_indices[:split_point]
        indices_part2 = shuffled_indices[split_point:]

        r_1 = 1e-4 + np.random.normal(0, 1e-5)
        r_2 = 0.5*1e-4 + np.random.normal(0,1e-6)

        P = np.hstack((np.zeros((self.sub_dim,1)),r_1*self.tr_radius*X[:,indices_part1],r_2*self.tr_radius*X[:,indices_part2]))
        y = np.apply_along_axis(self.get_func_val, axis=0, arr=self.x_current+self.sub_basis@P) # 1D array

        index = np.argmin(y)
        self.model_min_x = self.x_current + self.sub_basis@P[:,[index]]
        self.model_min_y = y[index]

        self.const, self.gradient, self.hessian = quadratic_regression(P.T, y)
    
    def get_model_val(self,p):
        return (self.const+p.T@self.gradient+0.5*p.T@self.hessian@p).item()
    
    def test_model(self,p):
        model_val = self.get_model_val(p)
        true_val = self.get_func_val(self.x_current+self.sub_basis@p)
        print('test model accuracy')
        print('model value vs true value: ',model_val,true_val)
        print('\n')

    def truncated_CG(self):
        epsilon = 1e-4
        max_iter = self.sub_dim*100

        s = np.zeros((self.sub_dim,1))
        r = self.gradient
        r_norm_0 = np.linalg.norm(r)
        p = -self.gradient
        k = 0
        while k<max_iter:
            if p.T@self.hessian@p <=0:
                t = solve_for_tau(s,p,self.tr_radius)
                return s + t*p
            alpha = (r.T@r)/(p.T@self.hessian@p)
            s_new = s + alpha*p
            if np.linalg.norm(s_new) >= self.tr_radius:
                t = solve_for_tau(s,p,self.tr_radius)
                return s + t*p
            r_new = r + alpha*(self.hessian@p)
            if np.linalg.norm(r_new) < epsilon*r_norm_0:
                return s_new
            beta = (r_new.T@r_new)/(r.T@r)
            p = -1*r_new + beta*p
            k += 1
            s = s_new
            r = r_new
        return s

    def tr_step(self):
        self.construct_model()

        # test the accuracy of the constructed model
        # p_test = generate_rand_points_on_sphere(self.sub_dim,1)*self.tr_radius*0.5
        # self.test_model(p_test)

        p_star = self.truncated_CG()
        self.p_star_norm = np.linalg.norm(p_star)
        self.tcg_x = self.x_current+self.sub_basis@p_star
        self.tcg_y = self.get_func_val(self.tcg_x)

        tcg_y_model = self.get_model_val(p_star)
        y_current_model = self.get_model_val(np.zeros((self.sub_dim,1)))

        # output some relative information for observation
        # print('truncated CG step, true value: y_star vs y current')
        # print(y_temp,self.y_current)
        # print('truncated CG step, model value: y_star vs y current')
        # print(y_temp_model,y_current_model)
        # print('trust region radius: ',self.tr_radius)

        if y_current_model-tcg_y_model > 1e-8:
                self.tr_flag = (self.y_current-self.tcg_y)/(y_current_model-tcg_y_model)
        else:
            print('TCG generate a wrong point')
            self.tr_flag= -1
            
            # output some relative information for observation
            # print('rou: ',self.tr_flag)

# update parameter   
    def para_upd(self):
        # record parameter
        self.his.mom_step_size_his.append(self.mom_step_size)
        self.his.ds_step_size_his.append(self.ds_step_size)
        self.his.gd_step_size_his.append(self.gd_step_size)
        self.his.tr_radius_his.append(self.tr_radius)

        # update search step size for mom, ds, gd
        if self.mom_flag == -1:
            self.mom_step_size *= 2
        elif self.mom_flag >= 1:
            self.mom_step_size *= 0.8
        else:
            pass

        if self.ds_flag == -1:
            self.ds_step_size *= 2
        elif self.trick_option[4] == 0 and self.ds_flag >= 1:
            self.ds_step_size *= 0.8
        elif self.trick_option[4] == 1 and self.ds_flag >= 0:
            self.ds_step_size *= 0.8

        if self.gd_flag == -1:
            self.gd_step_size *= 2
        elif self.gd_flag >= 1:
            self.gd_step_size *= 0.8
        else:
            pass

        # update tr_dadius
        if self.tr_flag < self.rou_1_bar:
            self.tr_radius *= self.gamma_1
        else:
            if self.tr_flag > self.rou_2_bar and self.p_star_norm >= 0.5*self.tr_radius:
                self.tr_radius = min(self.tr_radius*self.gamma_2,self.tr_radius_max)

        # 0: this step has been jumped
        # n > 0: n directions have been chosen into subsapce basis
        # -1: immediate adoption takes place
        self.mom_flag = 0
        self.ds_flag = 0
        self.gd_flag = 0
        self.tr_flag = 0
        self.mom_min_y = float('inf')
        self.ds_min_y = float('inf')
        self.gd_min_y = float('inf')
        self.model_min_y = float('inf')

        # use cyclic gradient estimation
        if self.trick_option[1] == 1:
            if self.his.niter % 10 == 0:
                self.gd_sample_num = self.gd_sample_num_large
            elif self.his.niter % 10 == 1:
                self.gd_sample_num = self.gd_sample_num_small

        # restart
        if self.trick_option[2] == 1:
            if self.his.niter % 5 == 0:
                self.mom_step_size *= 1.5
                self.ds_step_size *= 1.5
                # self.gd_step_size *= 1.5
                
        if self.trick_option[3] == 1:
            if self.his.niter % 20 == 0:
                self.tr_radius *= 1.5

    def check_stop(self):
        if self.his.niter >= self.max_iter:
            self.his.x_star = self.x_current
            self.his.y_star = self.y_current
            self.his.message = 'Return from solver because max iteration number has achieved'
            self.his.success = False
            return 1
        
        if self.his.nfev >= self.max_nfev:
            self.his.x_star = self.x_current
            self.his.y_star = self.y_current
            self.his.message = 'Return from solver because max function evaluations number has achieved'
            self.his.success = False
            return 1
        
        if self.ds_step_size < self.ds_step_size_tol:
            self.his.x_star = self.x_current
            self.his.y_star = self.y_current
            self.his.message = 'Return from solver because step size stopping criterion for Random Direct Search has achieved'
            self.his.success = True
            return 1
        
        if self.tr_radius < self.tr_radius_tol:
            self.his.x_star = self.x_current
            self.his.y_star = self.y_current
            self.his.message = 'Return from solver because step size stopping criterion for trust region radius has achieved'
            self.his.success = True
            return 1

        return 0

    def solve(self):
        while True:
            self.mom_step()
            if self.mom_flag == -1:
                self.next_iter(self.mom_temp_x,self.mom_temp_y)

                self.his.n_mom_explicit += 1
                self.his.info.append('mom')

                self.para_upd()
                stop = self.check_stop() # 1 means stop; 0 means continue
                if stop == 0:
                    continue
                else:
                    return 0

            self.ds_step()
            if self.ds_flag == -1:
                self.next_iter(self.ds_temp_x,self.ds_temp_y)
                self.his.n_ds_explicit += 1  
                self.his.info.append('ds')
                self.para_upd()
                stop = self.check_stop() # 1 means stop; 0 means continue
                if stop == 0:
                    continue
                else:
                    return 0
                
            self.gd_step()
            if self.gd_flag == -1:
                self.next_iter(self.gd_temp_x,self.gd_temp_y) 
                self.his.n_gd_explicit += 1 
                self.his.info.append('gd')
                self.para_upd()
                stop = self.check_stop() # 1 means stop; 0 means continue
                if stop == 0:
                    continue
                else:
                    return 0

            self.tr_step()
            # choose final point to be adopted
            x_ls = [self.x_current,self.tcg_x,self.mom_min_x,self.ds_min_x,self.gd_min_x,self.model_min_x]
            y_array = np.array([self.y_current,self.tcg_y,self.mom_min_y,self.ds_min_y,self.gd_min_y,self.model_min_y])
            ind = np.argmin(y_array)
            self.next_iter(x_ls[ind],y_array[ind])
            if ind == 0:
                self.his.n_stumble += 1
                self.his.info.append('stumble')
            elif ind == 1:
                self.his.n_tr += 1
                self.his.info.append('tr')
            elif ind == 2:
                self.his.n_mom_implicit += 1
                self.his.info.append('implicit')
            elif ind == 3:
                self.his.n_ds_implicit += 1
                self.his.info.append('implicit')
            elif ind == 4:
                self.his.n_gd_implicit += 1
                self.his.info.append('implicit')
            elif ind == 5:
                self.his.n_model_implicit += 1
                self.his.info.append('implicit')
            else:
                pass

            self.para_upd()
            stop = self.check_stop() # 1 means stop; 0 means continue
            if stop == 0:
                continue
            else:
                return 0
        
    def display_result(self):
        res = self.his
        print('\nmessage: ',res.message)
        print('\nsuccess: ',res.success)
        print('\nnfev: ',res.nfev) # number of total function evaluations
        print('\nniter: ',res.niter) # number of total iterations
        # print('\nx_star: ',res.x_star) # solution point
        print('\ny_star: ',res.y_star) # objective function value at solution 
        
        print('----------------')
        print('final trust region step size', self.tr_radius)
        print('\n')
        print('number of trust region step: ', res.n_tr)
        print('number of stumble step: ', res.n_stumble)
        print('number of explicit momentum step: ', res.n_mom_explicit)
        print('number of implicit momentum step: ', res.n_mom_implicit)
        print('number of explicit direct search step: ', res.n_ds_explicit)
        print('number of implicit direct search step: ', res.n_ds_implicit)
        print('number of explicit gradient step: ', res.n_gd_explicit)
        print('number of implicit gradient step: ', res.n_gd_implicit)

    def draw_result(self,ax,content):
        '''
        input:
        ax
        content -- str; the content of the drawing, the choice is listed below
        content = 
        'niter': y_iter_his vs niter
        'nfev': y_total_his vs nfev
        'mom_step_size'
        'ds_step_size'
        'gd_step_size'
        'tr_radius'
        '''
        color_map = {
            'mom': 'red',
            'ds': 'green',
            'gd': 'blue',
            'tr': 'gold',
            'stumble': 'black',
            'implicit': 'purple'
        }
        color_labels = self.his.info
        colors = [color_map[label] for label in color_labels]

        # marker='o'
        if content == 'niter':
            ax.plot(self.his.y_iter_his, linestyle='-', linewidth=1, color='Turquoise', label='subspace method')
            ax.scatter(range(self.his.niter),self.his.y_iter_his, color=colors, marker='o', s = 30)
            ax.set_xlabel('number of iterations')
            ax.set_ylabel('function value at each iteration')
            ax.set_ylim(0, self.dim)
        elif content == 'nfev':
            ax.plot(self.his.y_total_his, linestyle='-', linewidth=0.5, color='DeepSkyBlue', label='subspace method')
            ax.set_xlabel('number of function evaluations')
            ax.set_ylabel('function value at each function evaluation')
            ax.set_ylim(0, self.dim)
        elif content == 'mom_step_size':
            ax.plot(self.his.mom_step_size_his, linestyle='-', color='red', label='subspace method')
            ax.scatter(range(self.his.niter),self.his.mom_step_size_his, color=colors, marker='o', s = 30)
            ax.set_xlabel('number of iterations')
            ax.set_ylabel('momentum search step size at each iteration')
            ax.set_ylim(0, 0.2)
        elif content == 'ds_step_size':
            ax.plot(self.his.ds_step_size_his, linestyle='-', color='green', label='subspace method')
            ax.scatter(range(self.his.niter),self.his.ds_step_size_his, color=colors, marker='o', s = 30)
            ax.set_xlabel('number of iterations')
            ax.set_ylabel('direct search step size at each iteration')
            ax.set_ylim(0, 0.2)
        elif content == 'gd_step_size':
            ax.plot(self.his.gd_step_size_his, linestyle='-', color='blue', label='subspace method')
            ax.scatter(range(self.his.niter),self.his.gd_step_size_his, color=colors, marker='o', s = 30)
            ax.set_xlabel('number of iterations')
            ax.set_ylabel('gradient search step size at each iteration')
            ax.set_ylim(0, 0.6)
        elif content == 'tr_radius':
            ax.plot(self.his.tr_radius_his, linestyle='-', color='gold', label='subspace method')
            ax.scatter(range(self.his.niter),self.his.tr_radius_his, color=colors, marker='o', s = 30)
            ax.set_xlabel('number of iterations')
            ax.set_ylabel('trust region radius at each iteration')
            ax.set_ylim(0, 1)
        else:
            print('error: no such content')

        ax.legend()