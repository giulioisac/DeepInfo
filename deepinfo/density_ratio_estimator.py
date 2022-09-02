"""

Created on 2 May 2021

@author: None

"""


from __future__ import print_function, division,absolute_import
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input,Dense,Lambda
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.backend import sum as ksum
from tensorflow.keras.backend import update

from tensorflow.keras.backend import log as klog
from tensorflow.keras.backend import exp as kexp
from tensorflow.keras.backend import clip as kclip
from tensorflow import cast,boolean_mask,Variable
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import pandas as pd
from tensorflow import math
#from tensorflow.math import reduce_logsumexp, logical_not,reduce_mean
from copy import copy
from tqdm import tqdm
import pandas as pd

#Set input = raw_input for python 2
try:
    import __builtin__
    input = getattr(__builtin__, 'raw_input')
except (ImportError, AttributeError):
    pass


class DensityRatioEstimator(object):
    """
    
    Class used to infer the density ratio estimator. We use it for the specific case of
    joint vs independent but can in principle be used for any density ratio. 
    This ratio is by default inferred using a 2 hidden layer vanilla network.
    
    
    Attributes
    ----------
    numerator: list/ndarray
        samples from the distribution in the numerator of the ratio

    denominator: list/ndarray
        samples from the distribution in the denominator of the ratio
        
    load_dir: string
        path to the directory for loading the model
        
    l2_reg: float
        strength of l2 regularization
    
    l1_reg: float
        strength of l1 regularization
        
    nodes_number: int
        number of nodes in the hidden layers of the NN
        
    objective: string
        objective to infer ratio, possibilities: MINE,fMINE, BCE
    
    seed: int
        seed number for reproducibility
        
    validation_split: float
        relative size of validation data in the training
    
    optimizer: string
        choice of optimizer. At the moment the choiches are RMSprop and Adam
        
    gamma: float
        relative strength of the regularization parameter.
        
    lr: float
        parameter defining the learning rate
        
    
    Methods
    -------
    
    compute_energy(data=None)
        Compute energy of multiple samples
        
    fit(epochs, batch_size=5000, seed = None, verbose=0, callbacks=None,weights_name='weights.h5',early_stopping=True,patience=3)
        infer the density ratio estimator
    
    update_model_structure(output_layer=[],input_layer=[],initialize=False)
        define architecture of the network. For custom artchitectures use the input_layer and output_layer paramters
        
    save_model(save_dir, force=True, data=False)
        save model to directory
        
    load_model(load_dir = None, verbose = True,load_data=False)
        load model from directory
    
    """
    
    def __init__(self, numerator = [], denominator = [], load_dir = None, 
                 l2_reg = 0., l1_reg=0., nodes_number=50, objective='BCE',
                 seed=None,validation_split=0.1,optimizer='RMSprop',gamma=0.001,lr=0.001):
        
        self.numerator = numerator
        self.denominator = denominator
        self.L1_converge_history = []
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.likelihood_train=[]
        self.likelihood_test=[]
        self.optimizer_=optimizer
        self.objective=objective
        self.nodes_number = nodes_number 
        self.gamma=gamma
        self.Z=1.
        self.lr=lr
        self.validation_split=validation_split
        self.batch_size=10000
        
        if seed is not None: np.random.seed(seed = seed)
           
        
        if not load_dir is None:
            self.load_model(load_dir = load_dir)
        else:
            self.prepare_data()
            self.update_model_structure(initialize=True)
        
    def prepare_data(self):
        
        """
        
        Prepare inputs for training and performs validation split

        """
        
        n_num=len(self.numerator)
        n_den=len(self.denominator)
        
        if n_num!= n_den:
            print('different sizes not yet implemented. Please rerun with same size.')
            return 1

        x=np.concatenate([self.numerator,self.denominator],axis=0)
        y=np.zeros(2*n_num)
        y[n_num:]=1
        
        shuffle=np.random.choice(np.arange(2*n_num),2*n_num)
        
        self.X=x[shuffle][int(self.validation_split*2*n_num):]
        self.Y=y[shuffle][int(self.validation_split*2*n_num):]
        self.X_test=x[shuffle][:int(self.validation_split*2*n_num)]
        self.Y_test=y[shuffle][:int(self.validation_split*2*n_num)]
            
    def compute_energy(self,data=None):
        
        """
        
        Compute energy of multiple samples
        
        Parameters
        ----------
        
        data : list/ndarray
            List of samples 
       
        Returns
        -------
        
        ndarray
            energies for each sample
            
        """
        return self.model.predict(data)[:, 0]
        
    def fit(self, epochs = 1000, batch_size=5000, verbose=0, callbacks=None,weights_name='weights.h5',early_stopping=True,patience=30):
        """
        
        infer the density ratio estimator        
        
        
        Parameters
        ----------
        
        epochs : int
            max number of epochs
        
        batch_size: int
            size of minibatch
        
        verbose: int
            verbose paramter of keras fit function
            
        callbacks: list
            list of custom callbacks
        
        weights_name: str
            name of weights for the early stopping callback
        
        early_stopping: bool
            early stopping by monitoring the likelihood
        
        patience: int
            patience of the early stopping callback
            
        """
        self.batch_size=batch_size
        if callbacks is None: callbacks=[ModelCheckpoint(filepath=weights_name,save_best_only=True)]
        if early_stopping: callbacks+=[EarlyStopping(patience=patience)]
        
        self.learning_history = self.model.fit(self.X, self.Y, epochs=epochs, batch_size=batch_size, 
                                               validation_data=(self.X_test, self.Y_test), verbose=verbose, callbacks=callbacks)
        
        self.likelihood_train=-np.array(self.learning_history.history['likelihood'])*1.44
        self.likelihood_test=-np.array(self.learning_history.history['val_likelihood'])*1.44
        self.BCE_train=np.array(self.learning_history.history['binary_crossentropy'])*1.44
        self.BCE_test=np.array(self.learning_history.history['val_binary_crossentropy'])*1.44
        
        self.mi_value=self.likelihood_test[-1]
        self.bce_value=self.BCE_test[-1]

        self.model.load_weights(weights_name)

        # set Z    
        self.energies=self.compute_energy(self.X)
        self.energies_denominator=self.energies[self.Y.astype(np.bool)]
        self.energies_numerator=self.energies[np.logical_not(self.Y.astype(np.bool))]

        self.Z=np.sum(np.exp(-self.energies_denominator))/len(self.energies_denominator)        
        if np.abs(np.log(self.Z))>20: print ('Z us huuuuuuge')
            
    def update_model_structure(self,output_layer=[],input_layer=[],initialize=False):
        
        """
        
        define architecture of the network. For custom artchitectures use the input_layer and output_layer paramters
        
        
        Parameters
        ----------
        
        output_layer : keras layer
            custom keras output layer
        
        input_layer: keras layer
            custom keras input layer
            
        initialize: bool
            if True it reinitializes the network to basic structure
        
            
        """
        length_input=self.X.shape[1]
        l2_reg=copy(self.l2_reg)
        l1_reg=copy(self.l1_reg)

        if initialize:
            input_layer = Input(shape=(length_input,))
            middle1=Dense(self.nodes_number,activation='tanh',kernel_regularizer=l1_l2(l2=l2_reg,l1=l1_reg))(input_layer)
            middle2=Dense(self.nodes_number,activation='tanh',kernel_regularizer=l1_l2(l2=l2_reg,l1=l1_reg))(middle1)
            output_layer = Dense(1,activation='linear',kernel_regularizer=l1_l2(l2=l2_reg,l1=l1_reg))(middle2) #normal glm model

        # Define model
        self.model = Model(inputs=input_layer, outputs=output_layer)
        
        if self.optimizer_=='adam':
            self.optimizer = Adam(learning_rate=self.lr)
        elif self.optimizer_=='RMSprop':
            self.optimizer = RMSprop(learning_rate=self.lr)
        else: 
            print('only adam and RMSprop available. Retry.')
            return 1
            
        if self.objective=='BCE':
            self.model.compile(optimizer=self.optimizer, loss=BinaryCrossentropy(from_logits=True),metrics=[self.likelihood, BinaryCrossentropy(from_logits=True,name='binary_crossentropy')])
        elif self.objective=='fMINE':
            self.model.compile(optimizer=self.optimizer, loss=self.f_loss,
                               metrics=[self.likelihood, BinaryCrossentropy(from_logits=True,name='binary_crossentropy')])     
        elif self.objective=='MINE':
            self.model.compile(optimizer=self.optimizer, loss=self.loss, 
                               metrics=[self.likelihood, BinaryCrossentropy(from_logits=True,name='binary_crossentropy')])
        else: 
            print('only MINE, BCE and f-MINE available. Retry.')
        return True

    def loss(self, y_true, y_pred):
        
        """
        
        This is the "I" loss in the arxiv paper with added regularization
        
        """
        y=cast(y_true,dtype='bool')
        data= math.reduce_mean(boolean_mask(y_pred,math.logical_not(y)))
        gen= math.reduce_logsumexp(-boolean_mask(y_pred,y))-klog(ksum(y_true))
        return gen+data+self.gamma*gen*gen
    
    def f_loss(self, y_true, y_pred):
        
        """
        
        This is the "L_f" loss in the arxiv paper ... 
        
        """
        y=cast(y_true,dtype='bool')
        data= math.reduce_mean(boolean_mask(y_pred,math.logical_not(y)))
        gen=math.reduce_mean(kexp(-boolean_mask(y_pred,y)))
        loggen=klog(gen)-2.71828182846
        return gen/2.71828182846+data+self.gamma*loggen*loggen

    def likelihood(self, y_true, y_pred):
        
        """
        
        This is the "I" loss in the arxiv paper with added regularization
        
        """
        y=cast(y_true,dtype='bool')
        data= math.reduce_mean(boolean_mask(y_pred,math.logical_not(y)))
        gen= math.reduce_logsumexp(-boolean_mask(y_pred,y))-klog(ksum(y_true))
        return gen+data
    
    def log_ratio(self, data):
        """
        
        log ratio for mcmc
        
        """
        return -self.compute_energy(data)-np.log(self.Z)

    def save_model(self, save_dir, force=True, data=False):
        """
        
        save model to directory
        
        
        Parameters
        ----------
        
        save_dir : str
            path of the directory to save
        
        force: bool
            force the saving if the directory already contains a model. Otherwise asks for permission.
            
        data: bool
            if True saves dataset as well
            
        """
        # save model
        
        if os.path.isdir(save_dir):
            if not force:
                if not input('The directory ' + save_dir + ' already exists. Overwrite existing model (y/n)? ').strip().lower() in ['y', 'yes']:
                    print('Exiting...')
                    return None
        else:
            os.mkdir(save_dir)
        if data:  pd.DataFrame(zip(self.numerator,self.denominator),columns=['numerator','denominator']).to_csv(os.path.join(save_dir, 'data.csv.gz'),index=False,compression='gzip')

        with open(os.path.join(save_dir, 'log.txt'), 'w') as L1_file:
            L1_file.write('Z ='+str(self.Z)+'\n')
            L1_file.write('MI ='+str(self.mi_value)+'\n')
            L1_file.write('BCE ='+str(self.bce_value)+'\n')
            L1_file.write('likelihood_train,likelihood_test,BCE_train,BCE_test\n')
            for i in range(len(self.likelihood_train)):
                L1_file.write(str(self.likelihood_train[i])+','+str(self.likelihood_test[i])+','+str(self.BCE_train[i])+','+str(self.BCE_test[i])+'\n')
                
        self.model.save(os.path.join(save_dir, 'model.h5'))

        return 0

    def load_model(self, load_dir = None,load_data=False):
        """
        
        load model from directory
        
        
        Parameters
        ----------
        
        load_dir : str
            path of the directory to load the model from
        
        load_data: bool
            if True loads data as well
            
        """
        if load_dir is not None:
            if not os.path.isdir(load_dir):
                print('Directory for loading model does not exist (' + load_dir + ')')
                print('Exiting...')
                return None
            model_file = os.path.join(load_dir, 'model.h5')
            data_file = os.path.join(load_dir, 'data.csv.gz')
            log_file = os.path.join(load_dir, 'log.txt')
        else:
            print('give a directory name. Retry.')
            return 1



        with open(log_file, 'r') as L1_file:    
            for i,line in enumerate(L1_file):
                self.likelihood_train=[]
                self.likelihood_test=[]
                self.BCE_train=[]
                self.BCE_test=[]
                if i==0: self.Z=float(line.strip().split('=')[1])
                elif i==1: self.mi_value=float(line.strip().split('=')[1])
                elif i==2: self.bce_value=float(line.strip().split('=')[1])
                elif len(line.strip())>0 and i>1: 
                    try:
                        self.likelihood_train.append(float(line.strip().split(',')[0]))
                        self.likelihood_test.append(float(line.strip().split(',')[1]))
                        self.BCE_train.append(float(line.strip().split(',')[2]))
                        self.BCE_test.append(float(line.strip().split(',')[3]))
                    except:
                        continue
                        
        # load features and model not so good for re-inference
        self.model = load_model(model_file, compile = False, custom_objects={'loss': self.loss,'likelihood': self.likelihood,
                                                            'binary_crossentropy': BinaryCrossentropy(from_logits=True,name='binary_crossentropy')})
        self.optimizer = Adam()
        self.model.compile(optimizer=self.optimizer, loss=self.loss,metrics=[self.likelihood, BinaryCrossentropy(from_logits=True,name='binary_crossentropy')])

        if load_data:
            d=pd.read_csv(data_file)
            self.numerator=np.array([i for i in d['numerator'].values])
            self.denominator=np.array([i for i in d['denominator'].values])
        return 0
    

