from statistics import mean
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import os

import cvxopt

class Linear:
    """An example of a kernel."""

    def __init__(self):
        # here a kernel could set its parameters
        pass

    def __call__(self, A, B):
        """Can be called with vectors or matrices, see the
        comment for test_kernel"""
        return A.dot(B.T)

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
            return (np.dot(A,B.T)+1)**self.M


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

    def name():
        return "KRR"

    def fit(self, X, y):
        self.X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        I_n = np.eye(self.X.shape[0])
        I_n[0,0] = 0

        A = self.kernel(self.X, self.X) + self.lbd * np.eye(self.X.shape[0])
        self.alpha = np.linalg.solve(A,y[...,None])
        #self.alpha = np.dot(np.linalg.inv(A), y)
        return self

    def predict(self, X):
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        k = self.kernel(X, self.X)
        return np.sum(self.alpha.T*k, axis=1)


class SVR:
    def __init__(self, kernel, lambda_, epsilon) -> None:
        self.kernel = kernel
        self.lamb = lambda_
        self.epsilon = epsilon

    def name():
        return "SVR"

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
            help_mtx2[i:i+2]=np.array([self.y[j], -self.y[j]])
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

        cvxopt.solvers.options['show_progress'] = False
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


def load_sine():
    #C:\Users\marko\OneDrive\Desktop\MLDS1\Homework4
    file = os.path.join(os.sep, "C:" + os.sep, "Users", "marko" + os.sep,
    "OneDrive" + os.sep,  "Desktop" + os.sep,  "MLDS1" + os.sep,  
    "Homework4" + os.sep, 'sine.csv')
    df = pd.read_csv(file, sep=',')
    data = df.to_numpy()
    X, y = data[:,0], data[:,1]
    return X[...,None], y


def load_housing():
    file = os.path.join(os.sep, "C:" + os.sep, "Users", "marko" + os.sep,
    "OneDrive" + os.sep,  "Desktop" + os.sep,  "MLDS1" + os.sep,  
    "Homework4" + os.sep, 'housing2r_standardized.csv')
    df = pd.read_csv(file, sep=',')
    data = df.to_numpy()
    X, y = data[:,:5], data[:,5]
    return X[:160,:], y[:160], X[160:,:], y[160:]

def parameter_tuning(X, y, model):
    print("Fitting params to ", model)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    #kernels = [Linear(), RBF(sigma=0.5), Polynomial(M=5)]
    lambdas = [0.001, 0.01, 0.1, 0.5, 1]
    sigmas = [0.5, 1, 3, 5, 7, 10]
    Ms = [2,3,4,5,6,7,8,9,10]
    epsilons = [0.0001, 0.001, 0.01, 0.1]
    
    #-----------Tuning linear kernel------------
    minimal_rmse = 10000
    optimal_lambda = 0
    optimal_epsilon = 0
    if model=="KRR":
        for lbd in lambdas:
            fitter = KernelizedRidgeRegression(kernel=Linear(), lambda_=lbd)
            m = fitter.fit(X_train, y_train)
            pred = m.predict(X_test)
            if mean_squared_error(y_test, pred)<minimal_rmse:
                optimal_lambda=lbd
    else:
        for lbd in lambdas:
            for epsilon in epsilons:
                fitter = SVR(kernel=Linear(), lambda_=lbd, epsilon=epsilon)
                m = fitter.fit(X_train, y_train)
                pred = m.predict(X_test)
                if mean_squared_error(y_test, pred)<minimal_rmse:
                    optimal_lambda=lbd
                    optimal_epsilon=epsilon

    print("Optimal lambda (and epsilon) on linear kernel is:", optimal_lambda,"(", optimal_epsilon,")")
    #-----------Tuning RBF kernel------------
    minimal_rmse = 10000
    optimal_lambda = 0
    optimal_sigma = 0
    optimal_epsilon = 0
    if model=="KRR":
        for lbd in lambdas:
            for sigma in sigmas:
                fitter = KernelizedRidgeRegression(kernel=RBF(sigma=sigma), lambda_=lbd)
                m = fitter.fit(X_train, y_train)
                pred = m.predict(X_test)
                if mean_squared_error(y_test, pred)<minimal_rmse:
                    optimal_lambda=lbd
                    optimal_sigma=sigma
    else:
        for epsilon in epsilons:
            for lbd in lambdas:
                for sigma in sigmas:
                    fitter = SVR(kernel=RBF(sigma=sigma), lambda_=lbd, epsilon=epsilon)
                    m = fitter.fit(X_train, y_train)
                    pred = m.predict(X_test)
                    if mean_squared_error(y_test, pred)<minimal_rmse:
                        optimal_lambda=lbd
                        optimal_sigma=sigma
                        optimal_epsilon=epsilon

    print("Optimal lambda and sigma (and epsilon) on RBF kernel is:", optimal_lambda, optimal_sigma, "(", optimal_epsilon,")")
    #-----------Tuning polynomial kernel------------
    minimal_rmse = 10000
    optimal_lambda = 0
    optimal_m = 0
    optimal_epsilon = 0
    if model=="KRR":
        for lbd in lambdas:
            for M in Ms:
                fitter = KernelizedRidgeRegression(kernel=Polynomial(M=M), lambda_=lbd)
                m = fitter.fit(X_train, y_train)
                pred = m.predict(X_test)
                if mean_squared_error(y_test, pred)<minimal_rmse:
                    optimal_lambda=lbd
                    optimal_m=M
    else:
        for epsilon in epsilons:
            for lbd in lambdas:
                for M in Ms:
                    fitter = SVR(kernel=Polynomial(M=M), lambda_=lbd, epsilon=epsilon)
                    m = fitter.fit(X_train, y_train)
                    pred = m.predict(X_test)
                    if mean_squared_error(y_test, pred)<minimal_rmse:
                        optimal_lambda=lbd
                        optimal_m=M
                        optimal_epsilon=epsilon
    print("Optimal lambda and M (and epsilon) on polynomial kernel is:", optimal_lambda, optimal_m, "(", optimal_epsilon,")")
    

def evaluate_model_all_kernels(X_train, y_train, X_test, y_test, model, lbd=1, sigma=3, M=3, epsilon=0.01):
    
    #SVR(kernel=Linear(), lambda_=0.0001, epsilon=0.1)
    kernels = [Linear(), RBF(sigma=sigma), Polynomial(M=M)]
    for kernel in kernels:
        if model=="KRR":
            fitter = KernelizedRidgeRegression(kernel=kernel, lambda_=lbd)
        else:
            fitter = SVR(kernel=kernel, lambda_=lbd, epsilon=epsilon)
        
        m = fitter.fit(X_train, y_train)
        pred = m.predict(X_test)
        print("MSE:", mean_squared_error(y_test, pred))

def evaluate_model(X_train, y_train, X_test, y_test, model, kernel, lbd=1, epsilon=0.01):
    if model=="KRR":
        fitter = KernelizedRidgeRegression(kernel=kernel, lambda_=lbd)
    else:
        fitter = SVR(kernel=kernel, lambda_=lbd, epsilon=epsilon)
    
    m = fitter.fit(X_train, y_train)
    pred = m.predict(X_test)
    return round(mean_squared_error(y_test, pred),2)
    

def sine_data():
    X, y = load_sine()
    #plt.scatter(X,y)
    #plt.show()
    
    #parameter_tuning(X, y, "KRR")
    evaluate_model_all_kernels(X, y, X, y, "KRR", lbd = 1, sigma=3, M=5)

    #parameter_tuning(X, y, "SVR")
    evaluate_model_all_kernels(X, y, X, y, "SVR", lbd = 0.1, sigma=3, M=3, epsilon=0.1)
    #parameter_tuning(X, y, SVR)

def housing_data():
    X_train, y_train, X_test, y_test = load_housing()
    fig, ax = plt.subplots(2,2, figsize=(12,12))

    #First is KRR
    all_rmse=[]
    all_rmse_tuned=[]
    best_lambdas=[]
    for M in [i for i in range(1,9)]:
        kernel = Polynomial(M=M)
        all_rmse.append(evaluate_model(X_train, y_train, X_test, y_test, model = "KRR", kernel=kernel))
        rmse, best_lambda = housing_with_cross_val(X_train, y_train, X_test, y_test, "KRR", kernel, epsilon=0.01)
        all_rmse_tuned.append(round(rmse,2))
        best_lambdas.append(round(best_lambda,2))

    print(best_lambdas)    
    ax[0,0].plot([i for i in range(1,9)], all_rmse, marker='o', label='lambda=1')
    ax[0,0].plot([i for i in range(1,9)], all_rmse_tuned, marker='o', label='tuned lambda')
    ax[0,0].legend()
    for j in range(1,9):
        ax[0,0].annotate(all_rmse[j-1], 
                     (j,all_rmse[j-1]), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center',
                     fontsize=10) 
        ax[0,0].annotate(all_rmse_tuned[j-1], 
                     (j,all_rmse_tuned[j-1]), 
                     textcoords="offset points", 
                     xytext=(0,-10), 
                     ha='center',
                     fontsize=10) 
    ax[0,0].set_xticks([i for i in range(1,9)])
    ax[0,0].set_title("KRR model with polynomial kernel\n using different degrees", fontsize=15)
    ax[0,0].set_xlabel("polynomial degree")
    ax[0,0].set_ylabel("RMSE")
    
    
    #Then SVR
    all_rmse=[]
    all_rmse_tuned=[]
    best_lambdas=[]
    for M in [i for i in range(1,9)]:
        kernel = Polynomial(M=M)
        all_rmse.append(evaluate_model(X_train, y_train, X_test, y_test, model = "SVR", kernel=kernel))
        rmse, best_lambda = housing_with_cross_val(X_train, y_train, X_test, y_test, "SVR", kernel, epsilon=0.01)
        all_rmse_tuned.append(round(rmse,2))
        best_lambdas.append(round(best_lambda,2))

    print(best_lambdas)
    
    ax[0,1].plot([i for i in range(1,9)], all_rmse, marker='o', label='lambda=1')
    ax[0,1].plot([i for i in range(1,9)], all_rmse_tuned, marker='o', label='tuned lambda')
    ax[0,1].legend()
    for j in range(1,9):
        ax[0,1].annotate(all_rmse[j-1], 
                     (j,all_rmse[j-1]), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center',
                     fontsize=10) 
        ax[0,1].annotate(all_rmse_tuned[j-1], 
                     (j,all_rmse_tuned[j-1]), 
                     textcoords="offset points", 
                     xytext=(0,-10), 
                     ha='center',
                     fontsize=10) 
    ax[0,1].set_xticks([i for i in range(1,9)])
    ax[0,1].set_title("SVR model with polynomial kernel\n using different degrees", fontsize=15)
    ax[0,1].set_xlabel("polynomial degree")
    ax[0,1].set_ylabel("RMSE")
    
    

    #Then KRR with RBF
    sigmas = [0.01,0.1,1,3,5,10,15,20,40,60,100]
    all_rmse=[]
    all_rmse_tuned=[]
    best_lambdas=[]
    for sigma in sigmas:
        kernel = RBF(sigma=sigma)
        all_rmse.append(evaluate_model(X_train, y_train, X_test, y_test, model = "KRR", kernel=kernel))
        rmse, best_lambda = housing_with_cross_val(X_train, y_train, X_test, y_test, "KRR", kernel, epsilon=0.01)
        all_rmse_tuned.append(round(rmse,2))
        best_lambdas.append(round(best_lambda,2))

    print(best_lambdas)

    ax[1,0].plot([i for i in range(len(sigmas))], all_rmse, marker='o', label='lambda=1')
    ax[1,0].plot([i for i in range(len(sigmas))], all_rmse_tuned, marker='o', label='tuned lambda')
    ax[1,0].legend()
    for j in range(len(sigmas)):
        ax[1,0].annotate(all_rmse[j], 
                     (j,all_rmse[j]), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center',
                     fontsize=10) 
        ax[1,0].annotate(all_rmse_tuned[j], 
                     (j,all_rmse_tuned[j]), 
                     textcoords="offset points", 
                     xytext=(0,-10), 
                     ha='center',
                     fontsize=10) 
    ax[1,0].set_xticks([i for i in range(len(sigmas))])
    ax[1,0].set_xticklabels(sigmas)
    ax[1,0].set_title("KRR model with RBF kernel\n using different sigmas", fontsize=15)
    ax[1,0].set_xlabel("sigma value")
    ax[1,0].set_ylabel("RMSE")
    
    

    #Finally SVR with RBF
    all_rmse=[]
    all_rmse_tuned=[]
    best_lambdas=[]
    for sigma in sigmas:
        kernel = RBF(sigma=sigma)
        all_rmse.append(evaluate_model(X_train, y_train, X_test, y_test, model = "SVR", kernel=kernel))
        rmse, best_lambda = housing_with_cross_val(X_train, y_train, X_test, y_test, "SVR", kernel, epsilon=0.01)
        all_rmse_tuned.append(round(rmse,2))
        best_lambdas.append(round(best_lambda,2))
    
    print(best_lambdas)

    ax[1,1].plot([i for i in range(len(sigmas))], all_rmse, marker='o', label='lambda=1')
    ax[1,1].plot([i for i in range(len(sigmas))], all_rmse_tuned, marker='o', label='tuned lambda')
    ax[1,1].legend()
    for j in range(len(sigmas)):
        ax[1,1].annotate(all_rmse[j], 
                     (j,all_rmse[j]), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center',
                     fontsize=10) 
        ax[1,1].annotate(all_rmse_tuned[j], 
                     (j,all_rmse_tuned[j]), 
                     textcoords="offset points", 
                     xytext=(0,-10), 
                     ha='center',
                     fontsize=10) 
    ax[1,1].set_xticks([i for i in range(len(sigmas))])
    ax[1,1].set_xticklabels(sigmas)
    ax[1,1].set_title("SVR model with RBF kernel\n using different sigmas", fontsize=15)
    ax[1,1].set_xlabel("sigma value")
    ax[1,1].set_ylabel("RMSE")

    fig.tight_layout()
    plt.show()


#FOR CROSS VAL
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def divide_data(X, y, i):
    X, y = unison_shuffled_copies(X, y)
    X_train1, y_train1 = X[:i,:], y[:i]
    X_val, y_val = X[i:i+32,:], y[i:i+32]
    X_train2, y_train2 = X[i+32:,:], y[i+32:]

    if i==0:
        return X_train2, X_val, y_train2, y_val
    elif i==128:
        return X_train1, X_val, y_train1, y_val
    else:
        return np.vstack((X_train1,X_train2)), X_val, np.concatenate((y_train1,y_train2)), y_val
        
def cross_validation(X,y, model, kernel, epsilon):
    lambdas=[]
    for i in range(5):
        X_train, X_val, y_train, y_val = divide_data(X, y, i*32)
        best_lambda=0
        best_RMSE=10000000000000

        for lbd in [0.01,0.1,1,3,5,10,15,20,40,60,100]:
            m = KernelizedRidgeRegression(kernel=kernel, lambda_=lbd) if model=="KRR" else SVR(kernel=kernel, lambda_=lbd, epsilon=epsilon)
            m.fit(X_train, y_train)
            pred=m.predict(X_val)
            rmse = mean_squared_error(y_val, pred) #np.sqrt(np.dot((pred-y_val), (pred-y_val))/len(y_val))
            if rmse<best_RMSE:
                best_RMSE = rmse
                best_lambda = lbd
                #best_model = model
            del m

        lambdas.append(best_lambda)
    
    return lambdas

def housing_with_cross_val(X, y, X_test, y_test, model, kernel, epsilon=0.01):
    #X_train, y_train, X_test, y_test, model = "KRR", kernel=kernel
    best_lambdas=[]
    for i in range(20):
        best_lambdas+=cross_validation(X,y, model, kernel, epsilon)

    #print(best_lambdas)
    best_lambda=sum(best_lambdas)/len(best_lambdas)
    lambda_std=np.std(np.array(best_lambdas))
    
    #KernelizedRidgeRegression(kernel=kernel, lambda_=lbd)
    #SVR(kernel=kernel, lambda_=lbd, epsilon=epsilon)

    m = KernelizedRidgeRegression(kernel=kernel, lambda_=best_lambda) if model=="KRR" else SVR(kernel=kernel, lambda_=best_lambda, epsilon=epsilon)
    m.fit(X, y)
    pred=m.predict(X_test)
    
    #np.sqrt(np.dot((pred-y_test), (pred-y_test))/len(y_test)))
    return mean_squared_error(y_test, pred), best_lambda


housing_data()