from audioop import cross
import unittest
import numpy as np
from scipy.optimize import minimize
import pandas as pd

from sklearn.model_selection import train_test_split
from sympy import Q
import random

def minimize_L1(beta, *args):
    X, y, lamb = args[0], args[1], args[2]

    residuals = np.dot(X, beta) - y
    residuals_sum_squared=np.dot(residuals, residuals)
    return residuals_sum_squared+lamb*np.sum(np.abs(beta[1:]))


class RidgeReg:
    def __init__(self, lamb):
        self.lamb = lamb

    def fit(self, X, y):

        X=np.insert(X, 0, np.ones(X.shape[0]), axis=1)

        A = np.dot(X.T, X) 
        I = np.eye(A.shape[0])
        c = np.dot(X.T, y)

        lamb_I = self.lamb * I
        lamb_I[0,0]=0
        B = A + lamb_I
        # Solve for beta
        self.beta = np.linalg.solve(B,c)

    def predict(self, X):
        X=np.array(X)
        X=np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        return np.dot(X,self.beta)

class LassoReg:
    def __init__(self, lamb):
        self.lamb = lamb

    def fit(self, X, y):
        X=np.insert(X, 0, np.ones(X.shape[0]), axis=1)

        res = minimize(
            fun = minimize_L1, 
            x0 = np.zeros( X.shape[1] ),
            args = (X, y, self.lamb),
            method='Powell'
            )
        self.beta=res.x

    def predict(self, X):
        X=np.array(X)
        X=np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        return np.dot(X,self.beta)

class RegularizationTest(unittest.TestCase):

    def test_ridge_simple(self):
        X = np.array([[1],
                      [10],
                      [100]])
        y = 10 + 2*X[:,0]
        model = RidgeReg(1)
        model.fit(X, y)
        y = model.predict([[10],
                           [20]])
        self.assertAlmostEqual(y[0], 30, delta=0.1)
        self.assertAlmostEqual(y[1], 50, delta=0.1)
    # ... add your tests


def load(fname):
    df = pd.read_csv(fname, delimiter=',', decimal='.')
    data = df.to_numpy()             

    return data[:200, :-1], data[:200, -1], data[200:, :-1], data[200:, -1]

def superconductor(X_train, y_train, X_test, y_test):
    lamb=100 #0.05
    model = LassoReg(lamb)
    model.fit(X_train, y_train)
    pred=model.predict(X_test)

    print('Lasso',np.sqrt(np.dot((pred-y_test), (pred-y_test))/len(y_test)))

    model = RidgeReg(lamb)
    model.fit(X_train, y_train)
    pred=model.predict(X_test)

    print('Ridge',np.sqrt(np.dot((pred-y_test), (pred-y_test))/len(y_test)))

    return np.sqrt(np.dot((pred-y_test), (pred-y_test))/len(y_test))

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def divide_data(X, y, i):
    X, y = unison_shuffled_copies(X, y)
    X_train1, y_train1 = X[:i,:], y[:i]
    X_val, y_val = X[i:i+40,:], y[i:i+40]
    X_train2, y_train2 = X[i+40:,:], y[i+40:]

    if i==0:
        return X_train2, X_val, y_train2, y_val
    elif i==160:
        return X_train1, X_val, y_train1, y_val
    else:
        return np.vstack((X_train1,X_train2)), X_val, np.concatenate((y_train1,y_train2)), y_val
        

def cross_validation(X,y, ridge=False):
    lambdas=[]
    for i in range(5):
        X_train, X_val, y_train, y_val = divide_data(X, y, i*40)
        best_lambda=0
        best_RMSE=10000

        for lbd in [0.1, 0.5, 0.7, 0.9, 1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70]:
            model = RidgeReg(lbd) if ridge else LassoReg(lbd)
            model.fit(X_train, y_train)
            pred=model.predict(X_val)
            rmse = np.sqrt(np.dot((pred-y_val), (pred-y_val))/len(y_val))
            if rmse<best_RMSE:
                best_RMSE = rmse
                best_lambda = lbd
                #best_model = model
            del model

        lambdas.append(best_lambda)
    
    return lambdas

def superconductor_with_cross_val(X, y, X_test, y_test, ridge=False):
    best_lambdas=[]
    for i in range(100):
        best_lambdas+=cross_validation(X,y, ridge)

    #print(best_lambdas)
    best_lambda=sum(best_lambdas)/len(best_lambdas)
    lambda_std=np.std(np.array(best_lambdas))
    
    model = RidgeReg(best_lambda) if ridge else LassoReg(best_lambda)
    model.fit(X, y)
    pred=model.predict(X_test)
    model_name = 'Ridge' if ridge else 'Lasso'
    print(model_name, " RMSE on test using lambda=",best_lambda,"+/-",lambda_std," is ",np.sqrt(np.dot((pred-y_test), (pred-y_test))/len(y_test)))
    return model


def bootstrap_std(model, X_test, y_test,repetitions, ridge=False):
    n=X_test.shape[0]

    all_instances=[i for i in range(n)]
    RMSEs=[]
    for _ in range(repetitions):
        boostrap_sample_indices=random.choices(all_instances, k=int(n))
        bootstrap_sample_X=X_test[boostrap_sample_indices, :]
        bootstrap_sample_y=y_test[boostrap_sample_indices]

        pred=model.predict(bootstrap_sample_X)
        rmse = np.sqrt(np.dot((pred-bootstrap_sample_y), (pred-bootstrap_sample_y))/len(bootstrap_sample_y))
        RMSEs.append(rmse)

    std_error=np.std(np.array(RMSEs))
    model_name = 'Ridge' if ridge else 'Lasso'
    print(model_name, " RMSE STD is:",std_error) 
        

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load("superconductor_processed.csv")
    #superconductor(X_train, y_train, X_test, y_test)
    
    final_model = superconductor_with_cross_val(X_train, y_train, X_test, y_test, ridge=True)
    bootstrap_std(final_model, X_test, y_test, 100, True)
    
    final_model = superconductor_with_cross_val(X_train, y_train, X_test, y_test)
    bootstrap_std(final_model, X_test, y_test, 100)

    #unittest.main()