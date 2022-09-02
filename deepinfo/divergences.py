"""

Created on 2 May 2021

@author: None

"""

from deepinfo.density_ratio_estimator import DensityRatioEstimator
import numpy as np

class DklDivergence(DensityRatioEstimator):
    """
    
    Kullback Leibler Divergence through density ratio estimation.
    It is a simple wrapping of the DensityRatioEstimator class
    
    """
    
    
    def __init__(self, numerator = [], denominator = [], load_dir = None, 
                 l2_reg = 0., l1_reg=0., nodes_number=50, objective='BCE',
                 seed=None,validation_split=0.1,optimizer='RMSprop'):
        super().__init__(numerator = numerator, denominator = denominator, load_dir = load_dir, 
                         l2_reg = l2_reg, l1_reg=l1_reg, nodes_number=nodes_number, 
                         objective=objective,seed=seed,validation_split=validation_split,optimizer=optimizer)
    
    def evaluate(self):
        _,MI,BCE=np.array(self.model.evaluate(self.X_test,self.Y_test,batch_size=self.batch_size))*1.44
        return -MI, BCE
    
class DjsDivergence(object):
    """
    
    Jensen-Shannon Divergence through density ratio estimation.
    It is a composition of two DklDivergence class for density ratio estimation.
    
    """
    
    def __init__(self, values1 = [], values2 = [], load_dir = None, 
                 l2_reg = 0., l1_reg=0., nodes_number=50, objective='BCE',
                 seed=None,validation_split=0.1,optimizer='RMSprop'):
        
        n_num=len(values1)
        shuffle=np.random.choice(np.arange(2*n_num),2*n_num)
        # only same size of num and denominator are now implemented.
        self.m=np.concatenate([values1,values2])[shuffle][:n_num]
        
        self.dkl1=DklDivergence(numerator = values1, denominator = self.m, load_dir = load_dir, 
                                l2_reg = l2_reg, l1_reg=l1_reg, nodes_number=nodes_number, objective=objective,
                                seed=seed,validation_split=validation_split,optimizer=optimizer)
        
        self.dkl2=DklDivergence(numerator = values2, denominator = self.m, load_dir = load_dir, 
                                l2_reg = l2_reg, l1_reg=l1_reg, nodes_number=nodes_number, objective=objective,
                                seed=seed,validation_split=validation_split,optimizer=optimizer)
        
    def fit(self, epochs = 10, batch_size=5000, seed = None, verbose=0, callbacks=None,weights_name='weights.h5',early_stopping=False,patience=5):
        
        self.dkl1.fit(epochs = epochs, batch_size=batch_size, seed = seed, verbose=verbose, callbacks=callbacks,weights_name=weights_name,early_stopping=early_stopping,patience=patience)
        self.dkl2.fit(epochs = epochs, batch_size=batch_size, seed = seed, verbose=verbose, callbacks=callbacks,weights_name=weights_name,early_stopping=early_stopping,patience=patience)
    
    def evaluate(self):
        MI1,BCE1=self.dkl1.evaluate()
        MI2,BCE2=self.dkl2.evaluate()
        return 0.5*(MI1+MI2), 0.5*(BCE1+BCE2)
    
    
class MutualInformation(DensityRatioEstimator):
    """
    
    Mutual Information through density ratio estimation.
    It wraps around the DensityRatioEstimator class but it is specific
    to the ratio of joint over independent distributions.
    
    """
    
    def __init__(self, values1 = [], values2 = [], load_dir = None, 
                 l2_reg = 0., l1_reg=0., nodes_number=50, objective='BCE',
                 seed=None,validation_split=0.1,optimizer='RMSprop',gamma=0.001,lr=0.001):
        
        self.values1 = values1
        self.values2 = values2
        self.L1_converge_history = []
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.likelihood_train=[]
        self.likelihood_test=[]
        self.optimizer_=optimizer
        self.objective=objective
        self.nodes_number = nodes_number 
        self.gamma=gamma
        self.lr=lr
        self.Z=1.
        self.validation_split=validation_split
        self.batch_size=10000
        
        if seed is not None: np.random.seed(seed = seed)
        
        if not load_dir is None:
            self.load_model(load_dir = load_dir)
        else:
            self.prepare_data() 
            self.update_model_structure(initialize=True)

    def prepare_data(self,k=5):
        """
        
        Prepare inputs for training and performs validation split

        """
        n_simulations=len(self.values1)

        self.numerator=np.concatenate([self.values1,self.values2],axis=1)
        
        #multiple reshuflles for bigger negative set
        dens=[]
        for _ in range(k):
            shuffling=np.random.choice(np.arange(n_simulations),n_simulations)
            dens.append(np.concatenate([self.values1,self.values2[shuffling]],axis=1))
            
        self.denominator=np.concatenate(dens,axis=0)

        x=np.concatenate([self.numerator,self.denominator],axis=0)
        
        y=np.zeros((1+k)*n_simulations)
        
        # indep is class1 joint is class1
        y[n_simulations:]=1
        
        shuffle=np.random.choice(np.arange((1+k)*n_simulations),(1+k)*n_simulations)
        self.X=x[shuffle][int(self.validation_split*(1+k)*n_simulations):]
        self.Y=y[shuffle][int(self.validation_split*(1+k)*n_simulations):]
        self.X_test=x[shuffle][:int(self.validation_split*(1+k)*n_simulations)]
        self.Y_test=y[shuffle][:int(self.validation_split*(1+k)*n_simulations)]
        
    def log_ratio(self, values1, values2):
        """
        
        log ratio for mcmc
        
        """
        return -self.compute_energy(np.concatenate([values1,values2],axis=1))-np.log(self.Z)
    
    def evaluate(self,values1,values2):
        """
        
        evaluate estimator on test data
                
        Parameters
        ----------
        
        values1 : ndarray
            List of samples from first variable
        
        values2 : ndarray
            List of samples from second variable
       
        Returns
        -------
        
        MI: float
            estimated mutual information between variables
            
        BCE: float
            binary cross entropy loss for the classification between
            independent and joint distribution
        
        """
        # NB different behaviour than other two. Decide what you want.
        
        n_negs=len(values2)
        total_length=len(values2)+len(values1)
        shuffling=np.random.choice(np.arange(n_negs),n_negs)

        numerator=np.concatenate([values1,values2],axis=1)
        denominator=np.concatenate([values1,values2[shuffling]],axis=1)

        x=np.concatenate([numerator,denominator],axis=0)
        y=np.zeros(total_length)
        y[len(values1):]=1
        shuffling=np.random.choice(np.arange(total_length),total_length)
        _,MI,BCE=np.array(self.model.evaluate(x[shuffling],y[shuffling],batch_size=self.batch_size))*1.44
        return -MI,BCE