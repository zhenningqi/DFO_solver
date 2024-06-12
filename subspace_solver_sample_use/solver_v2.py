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
# create an easy problem instance
dim = 100
easy_prob = GetEasyProb.UnconstrProb()
# initialize problem
easy_prob.init_prob_dim(dim)
easy_prob.get_obj_func()

from Subspace import Solver_v2
# use subspace solver to solve the easy problem
# create solver
solver = Solver_v2.solver(easy_prob)
# initialize solver
'''
it is important to choose:
mom_num
ds_point_pair_num
gd_num
'''
max_iter=1e10
max_nfev=5e4

subspace_option = [4,0,1]
trick_option = [1,0,1,1,0]
# use proportion
if trick_option[1] == 1:
    gd_sample_num_option = [0.1,1]
else:
    gd_sample_num_option = 1
gd_estimator_index = 'FFD'

mom_step_size_0=0.1
ds_step_size_tol=1e-8
ds_step_size_0=0.1
gd_step_size_0=0.5
gd_sample_size_0=1e-6

tr_radius_0 = 0.2
tr_radius_tol=1e-3
tr_radius_max=10

rou_1_bar=0.25
rou_2_bar=0.75
gamma_1=0.8
gamma_2=2

solver.init_solver(max_iter,max_nfev,
                   subspace_option,trick_option,
                   gd_sample_num_option,gd_estimator_index,
                   mom_step_size_0,
                   ds_step_size_tol,ds_step_size_0,
                   gd_step_size_0,gd_sample_size_0,
                   tr_radius_0,tr_radius_tol,
                   tr_radius_max,rou_1_bar,rou_2_bar,gamma_1,gamma_2)
# solve problem by the solver
solver.init_gd_estimator()
solver.solve()
# show result
solver.display_result()

# draw result of the solver
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(14,7)) # 7*14
(ax1, ax2, ax3), (ax4, ax5, ax6) = axes
solver.draw_result(ax1,'niter')
solver.draw_result(ax2,'nfev')
solver.draw_result(ax3,'mom_step_size')
solver.draw_result(ax4,'ds_step_size')
solver.draw_result(ax5,'gd_step_size')
solver.draw_result(ax6,'tr_radius')
fig.suptitle(f'subspace method result on {easy_prob.prob_name} with dimension = {easy_prob.dim}')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()