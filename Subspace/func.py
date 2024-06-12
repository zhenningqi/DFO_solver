import numpy as np
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# generate random points on sphere       
def generate_rand_points_on_sphere(n, num_points=1):
    '''
    input:
    n -- int; dimension of the space
    num_points -- int; number of points
    '''
    points = np.random.multivariate_normal(np.zeros((n)), cov = np.eye(n), size = num_points).T
    norms = np.linalg.norm(points, axis=0)
    if np.sum(norms==0) > 0:
        print('a zero radius point is generated randomly, return points without normalization')
    else:
        points /= norms
    return points

def linear_regression(X,y):
    '''
    input:
    X -- ndarray; sample points, every row is a point, so the input in solver should be transposed
    y -- 1D array; corresponding function value gap
    output:
    gradient -- ndarray, shape(n,1); gradient of the linear model
    '''
    model = LinearRegression(fit_intercept=False).fit(X, y)
    params = model.coef_
    gradient = params.reshape(-1, 1)
    return gradient

# get quadratic regeression model
def quadratic_regression(X, y):
    '''
    input:
    X -- ndarray; sample points, every row is a point, so the input in solver should be transposed
    y -- 1D array; corresponding function value
    output:
    constant_term -- float; constant of the quadratic model
    gradient -- ndarray, shape(n,1); gradient of the quadratic model
    hessian -- ndarray, shape(n,n); hessian matrix of the quadratic model
    '''
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    model = LinearRegression().fit(X_poly, y)

    params = model.coef_
    intercept = model.intercept_

    constant_term = intercept

    n = X.shape[1]
    linear_params = params[:n]
    quadratic_params = params[n:]
    
    gradient = linear_params.reshape(-1, 1)
    hessian = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i, n):
            hessian[i, j] += quadratic_params[idx]
            hessian[j, i] += quadratic_params[idx]
            idx += 1

    return constant_term, gradient, hessian

# a function that is needed in the truncated CG
def solve_for_tau(s, p, tr_radius):
    # Coefficients for the quadratic equation
    a_coef = np.dot(p.T,p)
    b_coef = 2*np.dot(s.T,p)
    c_coef = np.dot(s.T,s) - tr_radius**2

    # Calculate the discriminant
    discriminant = b_coef**2 - 4 * a_coef * c_coef # it is definitely >=0
    if math.isnan(discriminant):
        print('the discriminant is a nan due to some numerical reasons, set tau to 0')
        tau = 0
        return tau 
    elif discriminant < 0:
        print("the discriminant is negative due to numerical reaons")
        discriminant = 0

    # Calculate the possible value of tau
    if a_coef > 0:
        tau = (-b_coef + np.sqrt(discriminant)) / (2 * a_coef)
    else:
        tau = 0
    return tau

def infinite_batch_generator(lst, batch_size):
    length = len(lst)
    idx = 0
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(lst[idx])
            idx = (idx + 1) % length
        yield batch