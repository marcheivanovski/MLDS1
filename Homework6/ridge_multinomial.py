import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fmin_l_bfgs_b
import math

def minimize_L1(beta, *args):
    X, y, lamb = args[0], args[1], args[2]

    residuals = np.dot(X, beta) - y
    residuals_sum_squared=np.dot(residuals, residuals)
    return residuals_sum_squared+lamb*np.sum(np.abs(beta[1:]))

def Multinomial_LogReg_Cost_Optimized(beta, *args):
    X, y, num_categories = args[0], args[1], args[2]
    #X...   matrix which rows are the learn instances and columns are the features
    #y...   target variable vector
    #beta...matrix which rows are rows are #cat-1 (betas for every category) and columns are the #features
    
    #num_categories=len(np.unique(y))-1
    num_features=X.shape[1]
    num_instances=X.shape[0]
    beta=np.reshape(beta, (num_categories-1, num_features))
    beta = np.vstack ((beta, np.zeros(num_features)) )

    linear_predictors=np.dot(X, np.transpose(beta))

    exp_linear_predictos=np.exp(linear_predictors)

    exp_sum=np.sum(exp_linear_predictos, axis=1)

    selected_exponents=exp_linear_predictos[[i for i in range(num_instances)], y]
    results=selected_exponents/(exp_sum)

    return -np.sum(np.log(results))

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


class MultinomialLogReg:
    def __init__(self):
        ...

    def build(self, X, y):
        return MultinomialLogRegNode(X, y) # dummy output


class MultinomialLogRegNode:

    def __init__(self, X, y):
        self.X=np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        self.y=y
        self.build()

    def build(self):
        self.num_categories=len(np.unique(self.y))
        self.num_features=self.X.shape[1]
        num_instances=self.X.shape[0]

        beta, _, _ = fmin_l_bfgs_b(
            func = Multinomial_LogReg_Cost_Optimized, 
            x0 = np.zeros( (self.num_categories-1)*self.num_features ),
            args = (self.X, self.y, self.num_categories),
            approx_grad = True
            )
        self.beta=np.reshape(beta, (self.num_categories-1, self.num_features))


    def predict(self,X_new):
        X_new=np.insert(X_new, 0, np.ones(X_new.shape[0]), axis=1)
        y_new=[]
        for i in range(X_new.shape[0]):
            x_i=X_new[i,:]

            linear_predictors=np.array([np.sum(self.beta[j,:]*x_i) for j in range(self.num_categories-1)] + [0]) #0 is added at the end since u^m=0 for reference
            denominator_sum=np.sum(np.exp(linear_predictors)) 
            probabilities_vector=np.array([math.exp(linear_predictors[j])/denominator_sum for j in range(self.num_categories)] )
            #y_new.append(probabilities_vector)
            y_new.append(np.argmax(probabilities_vector))

        return np.array(y_new)

    def return_lr_coefficients(self):
        return self.beta