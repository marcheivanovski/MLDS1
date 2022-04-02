import numpy as np
'''
import os
import sys
sys.path.append(os.path.join(os.sep, "C:" + os.sep, "Users", "marko" + os.sep,
  "anaconda3" + os.sep,  "envs" + os.sep,  "ids-clone" + os.sep,  
  "Lib" + os.sep,  "site-packages"+ os.sep, 'cvxopt') )'''

import cvxopt

class Polynomial:

    def __init__(self, M):
        self.M = M

    def __call__(self, A, B):
        if A.ndim==1 and B.ndim==1:
            return (np.dot(A,B)+1)**self.M
        elif A.ndim==1 and B.ndim==2:
            return (np.dot(B,A)+1)**self.M
        elif A.ndim==2 and B.ndim==1:
            return (np.dot(A,B)+1)**self.M
        else:
            return (np.dot(A,B.T))**self.M


class RBF:

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, A, B):
        #(a-b)^2 = a.dot(a) - 2*a.dot(b) + b.dot(b)
        if A.ndim==1 and B.ndim==1:
            return np.exp(np.sum((A-B)**2)/(-2*self.sigma))
        elif (A.ndim==1 and B.ndim==2) or (A.ndim==2 and B.ndim==1):
            if B.ndim==1: t = A; A = B; B = t
            A_repeated = np.repeat([A], B.shape[0], axis=0)
            inter = np.sum((A_repeated-B)**2, axis=1)
            return np.exp(inter/(-2*self.sigma))
        else:
            A_norm = np.sum(A ** 2, axis = -1)
            B_norm = np.sum(B ** 2, axis = -1)
            numerator = A_norm[:,None] + B_norm[None,:] - 2 * np.dot(A, B.T) 
            denominator = -2*self.sigma
            return np.exp(numerator/denominator)

class KernelizedRidgeRegression:
    def __init__(self,kernel , lambda_):
        self.lbd = lambda_
        self.kernel = kernel


    def fit(self, X, y):
        self.X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        A = self.kernel(self.X, self.X) + self.lbd * np.eye(self.X.shape[0])
        self.alpha = np.linalg.solve(A,y)
        #self.alpha = np.dot(np.linalg.inv(A), y)
        return self

    def predict(self, X):
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        k = self.kernel(X, self.X)
        return np.sum(self.alpha*k, axis = -1)

class SVR:
    def __init__(self, kernel, lambda_, epsilon) -> None:
        self.kernel = kernel
        self.lamb = lambda_
        self.epsilon = epsilon

    def compute_b(self):
        K_matrix = self.kernel(self.X, self.X)
        ai_minus_ai_star_vector = np.dot(self.alpha, self.help_mtx1)
        inter_step = self.y - np.dot(ai_minus_ai_star_vector, K_matrix)

        min_value = (-self.epsilon + inter_step).max()
        max_value = (+self.epsilon + inter_step).min()
        self.b = np.random.uniform(min_value, max_value, size=1)[0]

    def fit(self, X, y):
        #X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        self.X = X
        self.y = y

        n = self.X.shape[0]
        self.help_mtx1 = np.zeros((2*n, n))
        help_mtx2 = np.zeros(2*n)
        help_mtx3 = np.zeros(2*n)
        for i,j in zip(range(0, 2*n, 2), range(n)):
            self.help_mtx1[i:i+2, j]=np.array([1, -1])
            help_mtx2[i:i+2]=np.array([y[j], -y[j]])
            help_mtx3[i:i+2]=np.array([1, -1])

        G = np.zeros((4*n, 2*n))
        G[0:2*n, :]= np.eye(2*n)
        G[2*n: , :]= -1*np.eye(2*n)

        h = np.zeros(4*n)
        h[0:2*n]=np.ones(2*n)/self.lamb

        P = cvxopt.matrix (np.linalg.multi_dot((self.help_mtx1, self.kernel(self.X, self.X), self.help_mtx1.T)) )
        q = cvxopt.matrix (self.epsilon*np.ones(2*n) - help_mtx2)
        A = cvxopt.matrix (help_mtx3, (1, 2*n))
        b = cvxopt.matrix (0.0)
        G = cvxopt.matrix (G)
        h = cvxopt.matrix (h)

        #cvxopt.solvers.qp.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alpha = np.ravel(solution['x'])

        self.compute_b()


        return self

    def predict(self, X):
        K_matrix = self.kernel(self.X, X)
        ai_minus_ai_star_vector = np.dot(self.alpha, self.help_mtx1)
        return np.dot(ai_minus_ai_star_vector, K_matrix) + self.b

    def get_alpha(self):
        n = self.X.shape[0]
        return self.alpha.reshape((int(n), 2))

    def get_b(self):
        return self.b



        


