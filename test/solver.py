import sys
import os
'''
When running the code under the project root directory, getcwd will return the project root directory no matter what directory the code file is in.
However, python will set the directory where the code file is to default path, the project root directory will not be included in the default path.
Thus, if we want to use import, we should first set the project root directory into default path.
'''
current_path = os.getcwd()
sys.path.append(current_path)

from EasyProb import GetEasyProb
import numpy as np
# create an easy problem instance
dim = 1000
easy_prob = GetEasyProb.UnconstrProb()
# initialize problem
easy_prob.init_prob_dim(dim)
easy_prob.init_prob_x0(np.zeros(dim))
easy_prob.get_obj_func()

from Subspace import Solver
# use stochastic direct search solver to solve the easy problem
# create solver
solver = Solver.solver(easy_prob)
# initialize solver
'''
it is important to choose:
mom_num
ds_point_pair_num
gd_num
'''
max_iter = 1e6
mom_num=3
mom_step_size_0=1
ds_point_pair_num =10
ds_step_size_tol=1e-8
ds_step_size_0=10
gd_num=5
gd_sample_num=20
gd_step_size_0=1
gd_sample_size_0=1
tr_radius_0 = 0.01
tr_radius_max=10
neta=0.01
rou_1_bar=0.25
rou_2_bar=0.75
gamma_1=0.25
gamma_2=2
solver.init_solver(max_iter,mom_num,mom_step_size_0,
                   ds_point_pair_num,ds_step_size_tol,ds_step_size_0,
                   gd_num,gd_sample_num,gd_step_size_0,gd_sample_size_0,
                   tr_radius_0,
                   tr_radius_max,neta,rou_1_bar,rou_2_bar,gamma_1,gamma_2)
# solve problem by the solver
solver.solve()
# show result
solver.display_result()