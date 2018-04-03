# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 23:29:47 2018

@author: keding
"""
import numpy as np


class PLDA(object):
    def __init__(self, type='inv'):
        '''Two Covariance PLDA.
        Args:
            type: full, diag
        '''
        self.type = type
        
        if self.type == 'full':
            self.B = None  # between-class covariance
            self.W = None  # within-class covariance
            self.mu = None  # between-class center
        elif self.type == 'diag':
            self.V = None  # transform matrix
            self.psi = None  # diagnolized between-class covariance
            self.mu = None  # between-class center
        elif self.type == 'inv':
            self.invB = None
            self.invW = None
            self.mu = None
            
    def covert(self, target):
        '''Covert between types
        '''
        if target == self.type:
            return
        
        if self.type == 'full' and target == 'diag':
            raise RuntimeError('Not Implemented yet!')
        elif self.type == 'diag' and target == 'full':
            raise RuntimeError('Not Implemented yet!')
        else:
            raise RuntimeError('Invalid type convertion!')
            
        self.type = target
 
    def compute_log_likelihood(self, data):
        """Comute the log likelihood for the whole dataset.
        
        Args:
            data: An array of the shape (number_of_features, number_of_samples).
        """
        if self.type == 'full':
            return self._compute_llk_full(data, self.mu, self.B, self.W)
        elif self.type == 'diag':
            return self._compute_llk_diag(data, self.mu, self.psi, self.V)
        elif self.type == 'inv':
            return self._compute_llk_full(data, self.mu, self.invB, self.invW)
    
    def _compute_llk_full(self, data, mu, B, W):
        d, n = data.shape

        centered_data = data - mu
            
        # Total covariance matrix for the model with integrated out latent 
        # variables
        Sigma_tot = B + W
        
        # Compute log-determinant of the Sigma_tot matrix
        E, _ = np.linalg.eig(Sigma_tot)
        log_det = np.sum(np.log(E))
        
        return -0.5 * (n * d * np.log(2 * np.pi) + n * log_det + 
               np.sum(np.dot(centered_data.T, np.linalg.inv(Sigma_tot)) *
               centered_data.T))
    
    def _compute_llk_diag(self, data, mu, psi, V):
        d, n = data.shape

        u = np.dot(V, data - mu)
            
        # Total covariance matrix for the model with integrated out latent 
        # variables
        Sigma_tot = psi + 1
        
        # Compute log-determinant of the Sigma_tot matrix
        log_det = np.sum(np.log(Sigma_tot))
        
        return -0.5 * (n * d * np.log(2 * np.pi) + n * log_det + 
               np.sum(u ** 2 / Sigma_tot[:, np.newaxis]))

             
def preprocessing(data):
    '''
    '''
    # Sort the speakers by the number of utterances for the faster E-step
    data.sort(key=lambda x: x.shape[1]) 
    
    # Pool all the data for the more efficient M-step
    pooled_data = np.hstack(data)
        
    N = pooled_data.shape[1]  # total number of files
    K = len(data)  # number of classes
        
    mu = pooled_data.mean(axis=1, keepdims=True)
        
    # Calc first and second moments
    f = [spk_data.sum(axis=1) for spk_data in data]
    f = np.asarray(f).T
    S = np.dot(pooled_data, pooled_data.T)
    
    return pooled_data, N, K, f, S, mu


def initialize(plda, N, S, mu):
    cov = S / N - np.dot(mu, mu.T)
    
    if plda.type == 'full':
        plda.mu = mu
        plda.B = plda.W = cov
        plda.W = cov
    elif plda.type == 'inv':
        plda.mu = mu
        plda.invB = plda.invW = cov


def inv_e_step(plda, data, N, f, S):
    dim_d = data[0].shape[0]
    
    B = np.linalg.inv(plda.invB)
    W = np.linalg.inv(plda.invW)
    mu = plda.mu
    
    # Initialize output matrices
    T = np.zeros((dim_d, dim_d))
    R = np.zeros((dim_d, dim_d))
    Y = np.zeros((dim_d, 1))
    
    # Set auxiliary matrix
    Bmu = np.dot(B, mu)

    n_previous = 0  # number of utterances for a previous person
    for i in range(len(data)):
        n = data[i].shape[1]  # number of utterances for a particular person
        if n != n_previous: 
            # Update matrix that is dependent on the number of utterances
            invL_i = np.linalg.inv(B + n * W)
            n_previous = n
            
        gamma_i = Bmu + np.dot(W, f[:,[i]])
        Ey_i = np.dot(invL_i, gamma_i) 
        
        T += np.dot(Ey_i, f[:, [i]].T)
        R += n * (invL_i + np.dot(Ey_i, Ey_i.T))
        Y += n * Ey_i
        
    return T, R, Y

    
def inv_m_step(plda, T, R, Y, N, S):
    plda.mu = Y / N
    plda.invB = (R - np.dot(Y, Y.T) / N) / N
    plda.invW = (S - (T + T.T) + R) / N
    
    
def full_e_step(plda, data, N, f, S):
    dim_d = data[0].shape[0]
        
    invB = np.linalg.inv(plda.B)
    invW = np.linalg.inv(plda.W)
    mu = plda.mu

    # Initialize output matrices
    T = np.zeros((dim_d, dim_d))
    R = np.zeros((dim_d, dim_d))
    P = np.zeros((dim_d, dim_d))
    E = np.zeros((dim_d, dim_d))
        
    # Set auxiliary matrix
    invBmu = np.dot(invB, mu)

    n_previous = 0  # number of utterances for a previous person
    for i in range(len(data)):
        n = data[i].shape[1]  # number of utterances for a particular person
        if n != n_previous: 
            # Update matrix that is dependent on the number of utterances
            Phi = np.linalg.inv(invB + n * invW)
            n_previous = n
            
        gamma_i = invBmu + np.dot(invW, f[:, [i]])
        Ey_i = np.dot(Phi, gamma_i) 
        Eyyt_i = Phi + np.dot(Ey_i, Ey_i.T) 
        Ey_immu = Ey_i - mu

        T += np.dot(Ey_i, f[:, [i]].T)
        R += n * Eyyt_i
        P += Phi
        E += np.dot(Ey_immu, Ey_immu.T)
        
    return T, R, P, E
 

def full_m_step(plda, T, R, P, E, N, K, S):
    plda.B = (P + E) / K
    plda.W = (S - (T + T.T) + R) / N 
    
    
def print_progress(plda, pooled_data, cur_iter, total_iters):
    progress_message = '%d-th\titeration out of %d.' % (cur_iter+1,
                       total_iters)
    progress_message += ('  Log-likelihood is %f' % 
    plda.compute_log_likelihood(pooled_data))
    print progress_message
    
  
def train(plda, data, iterations):   
    pooled_data, N, K, f, S, mu  = preprocessing(data)
    initialize(plda, N, S, mu) 

    for i in range(iterations):
        if plda.type == 'inv':
            T, R, Y = inv_e_step(plda, data, N, f, S)
            inv_m_step(plda, T, R, Y, N, S)
        elif plda.type == 'full':
            T, R, P, E = full_e_step(plda, data, N, f, S)
            full_m_step(plda, T, R, P, E, N, K, S)
        
        # Print current progress
        print_progress(plda, pooled_data, i, iterations)
        

def test_plda():
    types = ['full', 'inv', 'diag']
    for t in types:
        plda = PLDA(t)        
 

def fake_data(D=2, K=3, n=10):
    sqrtB = np.diag(np.random.rand(D) + 0.1)
    B = np.dot(sqrtB, sqrtB.T)
    
    sqrtW = np.diag(np.random.rand(D) * 0.1 + 0.01)
    W = np.dot(sqrtW, sqrtW.T)
    
    m = np.random.randn(D, 1)
    
    data = []
    for i in range(K):
        ni = n + i
        X = m + np.dot(sqrtB.T, np.random.randn(D, ni)) + np.dot(sqrtW.T, np.random.randn(D, ni))
        data.append(X)
        
    return data, m, B, W
    
    
def test_train():
    
    D, K, n, iters = 2, 3, 10, 10
    
    np.random.seed(1111)
    data, m, B, W = fake_data(D, K, n)
    print(m)
    print(B)
    print(W)
    
    np.random.seed(1111)
    plda = PLDA('inv')
    train(plda, data, iterations=iters)
    print(plda.mu)
    print(plda.invB)
    print(plda.invW)
    
    np.random.seed(1111)
    plda = PLDA('full')
    train(plda, data, iterations=iters)
    print(plda.mu)
    print(plda.B)
    print(plda.W)
     
        
if __name__ == '__main__':
    test_plda()
    test_train()