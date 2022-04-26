"""

Created on 2 May 2021

@author: Giulio Isacchini

Functions for mcmc, copied and adapted from multiple sources:
    - https://github.com/montefiore-ai/hypothesis
    - https://github.com/mackelab/sbi

"""


import numpy as np
import warnings
import numpy.random as rng
from copy import copy
from tqdm import tqdm

class NormalTransition:
    """ 
    
    Symmetric normal transition with std sigma, jump from theta0 
    
    """

    def __init__(self, sigma):
        self.sigma = sigma 

    def sample(self, theta0):
        return rng.randn(*theta0.shape)*self.sigma + theta0

class Chain:
    """ 
    
    Process MCMC chain
    
    """

    def __init__(self, samples, acceptances):
        self.acceptances = np.array(acceptances)
        self.samples = np.array(samples)

    def acc_ratio(self):
        return np.mean(self.acceptances)

    def autocorrelations(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples = np.atleast_1d(self.samples)
            axis = 0
            m = [slice(None), ] * len(samples.shape)
            n = samples.shape[axis]
            f = np.fft.fft(samples - np.mean(samples, axis=axis), n=2 * n, axis=axis)
            m[axis] = slice(0, n)
            samples = np.fft.ifft(f * np.conjugate(f), axis=axis)[m].real
            m[axis] = 0
            acf = samples / samples[m]

        return acf
    
    def trim(self,threshold=0.1):
        self.auto_corr=self.autocorrelations()
        min_value=np.arange(len(self.auto_corr))[self.auto_corr.mean(axis=1)[:,0]<threshold].min()
        selection=(np.arange(len(self.auto_corr))%min_value)==0
        self.samples_trimmed = self.samples[selection]
        self.samples_trimmed_flat = self.samples_trimmed.reshape(self.samples.shape[1]*np.sum(selection),self.samples.shape[2])
        
    def best_theta(self,mcmc,xs):
        samples_flat = self.samples[-1,:,:]
        self.ratios = mcmc.compute_ratio(samples_flat,xs)
        return samples_flat[self.ratios.argmax()]
    
class MCMC:
    """
    
    Multi-dim MCMC class
    
    """
    
    def __init__(self, prior, ratio_estimator, transition):
        self.denominator = None
        self.prior = prior
        self.ratio_estimator = ratio_estimator
        self.transition = transition

    def compute_ratio(self, theta, xs):
        ntheta,dtheta = theta.shape
        nx = len(xs)
        dx = len(xs[0])
        all_thetas = np.array([theta]*nx).reshape(nx*ntheta,dtheta)
        all_xs = np.array([xs]*ntheta)
        all_xs = np.moveaxis(all_xs, 0, 1)
        all_xs = all_xs.reshape(nx*ntheta,dx)
        
        return self.ratio_estimator.log_ratio(all_thetas,all_xs).reshape(nx,ntheta).sum(axis=0)

    def step(self, theta0, xs):
        theta1 = self.transition.sample(theta0)
        numerator = self.prior.log_prob(theta1) + self.compute_ratio(theta1, xs)
        if self.denominator is None:
            self.denominator = self.prior.log_prob(theta0) + self.compute_ratio(theta0, xs)
        acceptance_ratio = numerator - self.denominator
        accepted = (np.random.uniform(size=theta0.shape[0]) <= np.exp(acceptance_ratio))
        
        theta0[accepted] = copy(theta1[accepted])
        self.denominator[accepted] = copy(numerator[accepted])

        return copy(theta0), accepted

    def reset(self):
        self.denominator = None
        
    def sample(self, theta, xs, length):
        acceptances = []
        samples = []
        self.reset()  
        for _ in tqdm(range(length)):
            theta, accepted = self.step(theta, xs)
            samples.append(theta)
            acceptances.append(accepted)
        chain = Chain(samples, acceptances)
        return chain