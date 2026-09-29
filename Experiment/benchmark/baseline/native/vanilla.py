from ag_vs.utils import trial_seed, sobol_engine
from Experiment.benchmark.baseline.reference_infrastructure import BO_REFERENCE_SPEC
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import torch
from torch.quasirandom import SobolEngine
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from sklearn.exceptions import ConvergenceWarning
import warnings
# warnings.filterwarnings('error', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)


def generate_initial_data(func, n, lb, ub):
    train_x = [np.random.uniform(lb, ub) for _ in range(n)]
    train_y = [func(x) for x in train_x]
    return train_x, train_y


def get_gpr_model():
    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=m52, alpha=BO_REFERENCE_SPEC.alpha, random_state=trial_seed())
    return gpr


def expected_improvement(gpr, X_sample, Y_sample, X, xi=0.0001, use_ei=True):
    ''' 
    Computes the EI at points X based on existing samples X_sample and 
        Y_sample using a Gaussian process surrogate model.
    Args: 
        gpr: A GaussianProcessRegressor fitted to samples.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1). 
        X: Points at which EI shall be computed (m x d). 
        xi: Exploitation-exploration trade-off parameter.
        Returns: Expected improvements at points X. 
    '''
    X_sample = np.asarray(X_sample)
    Y_sample = np.asarray(Y_sample).reshape(-1, 1)
    mu, sigma = gpr.predict(X, return_std=True)

    if not use_ei:
        return mu
    else:
        #calculate EI
        mu_sample = gpr.predict(X_sample)
        sigma = sigma.reshape(-1, 1)
        mu_sample_opt = np.max(mu_sample)
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            imp = imp.reshape((-1, 1))
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei
    
    
# def upper_confidence_bound(gpr, X_sample, Y_sample, X, beta=0.1):
#     X_sample = np.asarray(X_sample)
#     Y_sample = np.asarray(Y_sample).reshape(-1, 1)
#     # mu, sigma = gpr.predict(X, return_std=True)
#     mean = X.mean(axis=0)
#     X_ucb = mean + np.sqrt(beta * np.pi / 2) * (X - mean)
#     print(mean.shape)
#     print(X.shape)
#     print(X_ucb.shape)
#     return X_ucb
    
    
def propose_rand_samples_sobol(dims, n, lb, ub):
    seed = trial_seed()
    sobol = sobol_engine(dims, role="native") 
    cands = sobol.draw(n).to(dtype=torch.float64).cpu().detach().numpy()
    cands = cands * (ub - lb) + lb
    return cands


def optimize_acqf(dims, gpr, X_sample, Y_sample, n, lb, ub):
    # maximize acquisition function
    X = propose_rand_samples_sobol(dims, 1024, lb, ub)
    X_acqf = expected_improvement(gpr, X_sample, Y_sample, X, xi=0.0001, use_ei=True)
    # X_acqf = upper_confidence_bound(gpr, X_sample, Y_sample, X, beta=0.1)
    X_acqf = X_acqf.reshape(-1)
    indices = np.argsort(X_acqf)[-n: ]
    proposed_X, proposed_X_acqf = X[indices], X_acqf[indices]
    return proposed_X, proposed_X_acqf


