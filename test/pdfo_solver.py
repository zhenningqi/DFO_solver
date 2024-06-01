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
dim = 500
easy_prob = GetEasyProb.UnconstrProb()
# initialize problem
easy_prob.init_prob_dim(dim)
easy_prob.init_prob_x0(np.zeros(dim))
easy_prob.get_obj_func()

from PDFO import PDFOsolver
# use pdfo solver to solve the easy problem
pdfosolver = PDFOsolver.PDFOsolver(easy_prob)
pdfosolver.solve()
# res = pdfosolver.res
# print(res.nfev)
# print(type(res.fun_history))

# show result
pdfosolver.display_result()
# draw result
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(15,6)) # 6*15
pdfosolver.draw_result(ax)
fig.suptitle(f'pdfo result on {easy_prob.prob_name} with dimension = {easy_prob.dim}')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()