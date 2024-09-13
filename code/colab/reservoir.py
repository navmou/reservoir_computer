import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams['figure.titlesize'] = 25 
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.labelsize'] = 22 
mpl.rcParams['xtick.labelsize'] = 22 
mpl.rcParams['ytick.labelsize'] = 22 
mpl.rcParams['legend.fontsize'] = 18


np.random.seed(10)

class RESERVOIR():

    def __init__(self, data , n_reservoir=200, n_inputs=1 , n_outputs=1 ,
                 spectral_radius = 0.95, sparsity=0 , learning_rate=0.2 , beta = 0.1 , n_trainings=100):
        """
        Args:
            n_reservoir: number of reservoir neurons
            spectral_radius: spectral radius of the recurrent weight matrix
            sparsity: proportion of recurrent weights set to zero
            
        """
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.data = data
        self.test_data = data[int(0.8*len(data)):]
        self.learning_rate = learning_rate
        self.beta = beta
        self.n_trainings = n_trainings
        
    def initweights(self):
        # initialize reservoir weights:
        W = np.random.rand(self.n_reservoir, self.n_reservoir)*2-1
        # delete the fraction of connections given by (self.sparsity):
        for i in range(self.n_reservoir):
            for j in range(self.n_reservoir):
                if np.random.uniform() > self.sparsity:
                    W[i,j] = 0
        print(f'np.linalg.norm(W): {np.linalg.norm(W)}')
        W = W / (np.linalg.norm(W)+0.001)    
        print(f'np.linalg.norm(W): {np.linalg.norm(W)}')                
        # compute the spectral radius of these weights:
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        # rescale them to reach the requested spectral radius:
        self.W = W * (self.spectral_radius / radius)
        # random input weights:
        self.W_in  = np.random.rand(self.n_reservoir , self.n_inputs )*0.2 - 0.1
        print(np.linalg.norm(self.W_in))
        self.W_in = self.W_in/(np.linalg.norm(self.W_in)+0.001)
        print(np.linalg.norm(self.W_in))
        #for i in range(self.n_reservoir):
        #    if np.random.uniform() < 0.8:
        #        self.W_in[i] = 0
        print(self.W)
        print(np.count_nonzero(self.W))
        print(np.count_nonzero(self.W)/(self.n_reservoir*self.n_reservoir))
        # random output weights:
        self.W_out = np.random.rand(self.n_reservoir , self.n_outputs)*0.2 - 0.1
        # initializing the neurons with random values
        self.state = np.random.rand(self.n_reservoir , 1)
        print(np.linalg.norm(self.state))
        self.state = self.state/(np.linalg.norm(self.state)+0.001)
        print(np.linalg.norm(self.state))
        self.bs = np.zeros((self.n_reservoir , 1))
                              
    def _update(self, target_pattern, output_pattern):
        """Updates the output weights"""
        self.W_out += self.learning_rate*((target_pattern - output_pattern)*(self.state) +
         (self.beta*self.W_out))
        
        
    def _predict(self , input_pattern):
        """Calculates the output pattern"""
        b = np.dot(self.W , self.state) + (self.W_in*input_pattern)
        self.state = np.tanh(b)
        #self.bs = np.append(self.bs , b , axis=1)
        #self.state = new_state
        return np.dot(np.transpose(self.W_out) , self.state)[0,0]

    def _train(self):
        """Training the reservoir with 80% of the trajectory"""
        error = 100
        counter = 0
        while(error > 0.005):
            self.diff = []
            path = []
            for i in range(2000):
            #i = np.random.choice(np.arange(0,2000))
                output_pattern = self._predict(self.data[i])
                path.append(output_pattern)
                self._update(self.data[i+1], output_pattern)
                self.diff.append(output_pattern-self.data[i+1])
            counter += 1
            error = (np.sum(self.diff))
            if counter % 100 == 0:
                print(f'training {counter}: difference: {error}')
                self.learning_rate = np.max([0.9*self.learning_rate , 0.0001])
                print(self.learning_rate)
                plt.figure(figsize=(22,7))
                plt.plot(self.data[1:2001] , label = 'data')
                plt.plot(path , label = 'training')
                plt.legend()
                plt.show()
                plt.savefig('training.png')
                
            #print(self.error)
        

    def _test(self):
        x = self.test_data[0]
        self.prediction = [x]
        self.prediction2 = [self.data[0]]
        #to only have from the data point after training
        for i in range(int(0.2*len(self.data))-1):
            y = self._predict(x)
            self.prediction.append(y)
            x = y

        #to have frm the begining
        for i in range(len(self.data)-1):
            y = self._predict(self.data[i])
            self.prediction2.append(y)
    
    def _plot(self):
        landa1 = 0.906
        T = np.linspace(0,10,500)*landa1
        plt.figure(figsize=(20,9))
        plt.plot(T , self.prediction , label='Prediction')
        plt.plot(T , self.test_data , label='Actual')
        plt.plot(np.ones(15)*landa1 , np.linspace(-30,30,15) , 'r--' , label='Lyaponov time')
        plt.legend()
        plt.ylim(-30,30)
        plt.xlabel(r'$X_2(t)$')
        plt.ylabel(r'$\lambda_1 t$')
        plt.savefig('comparison.png')
        
        T = np.linspace(0,50,500)*landa1
        plt.figure(figsize=(20,9))
        plt.plot(T , self.prediction2 , label='Prediction')
        plt.plot(T , self.data[1:] , label='Actual')
        plt.plot(np.ones(15)*landa1 , np.linspace(-30,30,15) , 'r--' , label='Lyaponov time')
        plt.legend()
        plt.ylim(-30,30)
        plt.xlabel(r'$X_2(t)$')
        plt.ylabel(r'$\lambda_1 t$')
        plt.savefig('comparison2.png')

    def _get_J(self):
        
        max_ind = int(0.8*self.bs.shape[1])
        landas = []
        Js = np.eye(self.n_reservoir , self.n_reservoir)
        
        for t in range(1,max_ind):    
            J = np.diag(1-np.tanh(self.bs[:,t])**2)*self.W * Js[:,(t-1)*self.n_reservoir:t*self.n_reservoir]
            Js = np.append(Js , J , axis=1)
            
            landas.append(np.max(np.linalg.eigvals(np.transpose(J)*J)))
        print(self.bs[:,0])
        print(self.bs[:,1])
        print(self.bs.shape)
        print(self.bs.shape[1])
        print(landas)
        #plt.figure(figsize=(15,10))
        #n = np.arange(1,max_ind)
        #plt.plot(n , 1/n*np.log(landas))
        
    def _save_reservoir(self):
        with open('res_weights.npy' , 'wb') as f:
            np.save(f, self.W)
        with open('in_weights.npy' , 'wb') as f:
            np.save(f, self.W_in)
        with open('out_weights.npy' , 'wb') as f:
            np.save(f, self.W_out)


    def run(self):
        self.initweights()
        self._train()
        self._save_reservoir()
        self._test()
        self._plot()
        #self._get_J()
