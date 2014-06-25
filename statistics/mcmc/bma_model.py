import numpy as np
from pymc import stochastic, DiscreteMetropolis, MCMC
import statsmodels.api as sm
import pandas as pd
import random

def pack(alist, rank):
    
    binary = [str(1) if i in alist else str(0) for i in xrange(0,rank)]
    string = '0b1'+''.join(binary)
    return int(string, 2)

def unpack(integer):
    
    string = bin(integer)[3:]
    
    return [int(i) for i in xrange(len(string)) if string[i]=='1']
    

def make_bma():
    
    # Simulating Data
    size = 100
    rank = 20  
    
    X = 10*np.random.randn(size, rank)
    error = 30*np.random.randn(size,1)
    coefficients = np.array([10, 2, 2, 2, 2, 2]).reshape((6,1))
    y = np.dot(sm.add_constant(X[:,:5], prepend=True), coefficients) + error
    
    # Number of allowable regressors    
    predictors = [3,4,5,6,7]
    
    @stochastic(dtype=int)
    def regression_model():
        
        def logp(value):

            columns = unpack(value)
            
            x = sm.add_constant(X[:,columns], prepend=True)
            
            corr = np.corrcoef(x[:,1:], rowvar=0)
        
            prior = np.linalg.det(corr)
    
            ols = sm.OLS(y,x).fit()
            
            posterior = np.exp(-0.5*ols.bic)*prior
    
            return np.log(posterior)
    
        def random():
            
            k = np.random.choice(predictors)
            
            columns = sorted(np.random.choice(xrange(0,rank), size=k, replace=False))
            
            return pack(columns, rank)
    
    class ModelMetropolis(DiscreteMetropolis):
        def __init__(self, stochastic):
            DiscreteMetropolis.__init__(self, stochastic)
            
        def propose(self):
            '''considers a neighborhood around the previous model, 
            defined as having one regressor removed or added, provided
            the total number of regressors coincides with predictors
            '''
            
            # Building set of neighboring models
            last = unpack(self.stochastic.value)
            last_indicator = np.zeros(rank)
            last_indicator[last] = 1
            last_indicator = last_indicator.reshape((-1,1))
            neighbors = abs(np.diag(np.ones(rank)) - last_indicator)
            neighbors = neighbors[:,np.any([neighbors.sum(axis=0) == i \
                                for i in predictors], axis=0)]
            neighbors = pd.DataFrame(neighbors)
            
            # Drawing one model at random from the neighborhood
            draw = random.choice(xrange(neighbors.shape[1]))
    
            self.stochastic.value = pack(list(neighbors[draw][neighbors[draw]==1].index), rank)
        
#        def step(self):
#            
#            logp_p = self.stochastic.logp
#            
#            self.propose()
#            
#            logp = self.stochastic.logp
#            
#            if np.log(random.random()) > logp_p - logp:
#                
#                self.reject()
                
            
            
            
    
    return locals()
    
if __name__ == '__main__':
    
    model = make_bma()
    M = MCMC(model)
    M.use_step_method(model['ModelMetropolis'], model['regression_model'])
    M.sample(iter=5000, burn=1000, thin=1)
    
    model_chain = M.trace("regression_model")[:]
    
    from collections import Counter
    
    counts = Counter(model_chain).items()
    counts.sort(reverse=True, key=lambda x: x[1])
    
    for f in counts[:10]:
        columns = unpack(f[0])
        print('Visits:', f[1])
        print(np.array([1. if i in columns else 0 for i in range(0,M.rank)]))
        print(M.coefficients.flatten())
        X = sm.add_constant(M.X[:, columns], prepend=True)
        corr = np.corrcoef(X[:,1:], rowvar=0)
        prior = np.linalg.det(corr)
        fit = sm.OLS(model['y'],X).fit()
        posterior = np.exp(-0.5*fit.bic)*prior
        print(fit.params)
        print('R-squared:', fit.rsquared)
        print('BIC', fit.bic)
        print('Prior', prior)
        print('Posterior', posterior)
        print(" ")