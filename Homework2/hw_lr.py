from pickle import TRUE
from matplotlib.pyplot import axis
import numpy as np
import time
import math
from scipy.optimize import fmin_l_bfgs_b
import pandas as pd
import random
from sklearn.model_selection import train_test_split



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




def Multinomial_LogReg_Cost(beta, *args):
    X, y, num_categories = args[0], args[1], args[2]
    #X...   matrix which rows are the learn instances and columns are the features
    #y...   target variable vector
    #beta...matrix which rows are rows are #cat-1 (betas for every category) and columns are the #features
    
    #num_categories=len(np.unique(y))-1
    num_features=X.shape[1]
    num_instances=X.shape[0]
    beta=np.reshape(beta, (num_categories-1, num_features))

    log_likelihood=0
    for i in range(num_instances):
        x_i=X[i,:]
        y_i=y[i]
        
        linear_predictors=np.array([np.sum(beta[j,:]*x_i) for j in range(num_categories-1)] + [0]) #0 is added at the end since u^m=0 for reference
        denominator_sum=np.sum(np.exp(linear_predictors)) 
        p_yi = math.exp(linear_predictors[y_i])/denominator_sum         
        
        log_likelihood-=math.log(p_yi)

    return log_likelihood

def OrdinalLogRegCost(params, *args):
    X, y = args[0], args[1]
    num_categories=len(np.unique(y))
    num_features=X.shape[1]
    num_instances=X.shape[0]
    deltas, beta = params[:num_categories-2], params[num_categories-2:]
    bounds= np.array([0] + list(np.cumsum(deltas)))

    log_likelihood=0
    for i in range(num_instances):
        x_i=X[i,:]
        y_i=y[i]

        linear_predictor = np.sum(x_i*beta)

        if y_i==0:
            right = 0
        else:
            right = 1/(1+math.exp(linear_predictor-bounds[y_i-1]))
        
        if y_i==len(np.unique(y))-1:
            left = 1
        else:
            left =  1/(1+math.exp( linear_predictor-bounds[y_i]))

        #add_this=left - right
        log_likelihood=math.log( (left - right + 0.0000001) )
    return -log_likelihood



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
            y_new.append(probabilities_vector)
            #y_new.append(np.argmax(probabilities_vector))

        return np.array(y_new)

    def return_lr_coefficients(self):
        return self.beta


class OrdinalLogReg:
    def __init__(self):
        ...

    def build(self, X, y):
        return OrdinalLogRegNode(X,y)

class OrdinalLogRegNode:
    def __init__(self, X, y):
        self.X=np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        self.y=y
        self.build()

    def build(self):
        self.num_categories=len(np.unique(self.y))
        self.num_features=self.X.shape[1]

        deltas=[i+1 for i in range(self.num_categories-2)]
        betas=list(np.ones(self.num_features))

        bounds_delta=[(0.0001,None) for i in range(self.num_categories-2)]
        bounds_betas=[(None, None) for i in range(self.num_features)]

        params, _, _ = fmin_l_bfgs_b(
            func = OrdinalLogRegCost, 
            x0 = np.array(deltas+betas),
            args = (self.X, self.y),
            approx_grad = True,
            bounds=np.array(bounds_delta+bounds_betas)
            )

        self.delta, self.betas = params[:self.num_categories-2], params[self.num_categories-2:]
        self.bounds= np.array([0] + list(np.cumsum(deltas)))

    def predict(self,X_new):
        X_new=np.insert(X_new, 0, np.ones(X_new.shape[0]), axis=1)
        y_new=[]
        for i in range(X_new.shape[0]):
            x_i=X_new[i,:]
            linear_predictor = np.sum(x_i*self.betas)
            probabilities_vector=np.zeros(self.num_categories)
            for j in range(self.num_categories):
                if j==0:
                    right = 0
                else:
                    right = 1/(1+math.exp(linear_predictor-self.bounds[j-1]))
                
                if j==self.num_categories-1:
                    left = 1
                else:
                    left =  1/(1+math.exp( linear_predictor-self.bounds[j]))
                
                probabilities_vector[j]=left-right

            y_new.append(probabilities_vector)
            

        #print(self.bounds, self.betas)
        return np.array(y_new)


def read_dataset():
    df = pd.read_csv('dataset_processed_onehot.csv', delimiter=';')
    df.drop(['TwoLegged'], axis=1, inplace=True)
    #df.drop(['movement_no'], axis=1, inplace=True)

    data = df.to_numpy()                         
    data=data[:,1:]
    return data[:,1:], data[:,0].astype(int)

def bootstrap_ci(X,y, repetitions, alpha, rand):
    n=X.shape[0]
    features=X.shape[1]+1
    categories=len(np.unique(y))

    all_instances=[i for i in range(n)]
    coefficients=np.zeros((repetitions, categories-1, features)) #self.num_categories*self.num_features
    
    for i in range(repetitions):
        boostrap_sample_indices=rand.choices(all_instances, k=int(n))
        bootstrap_sample_X=X[boostrap_sample_indices, :]
        bootstrap_sample_y=y[boostrap_sample_indices]

        l = MultinomialLogReg()
        c = l.build(bootstrap_sample_X, bootstrap_sample_y)
        coefficients[i,:,:]=c.return_lr_coefficients()

    
    for i in range(categories-1):
        for j in range(features):
            left = np.percentile(coefficients[:,i,j], alpha/2*100)
            right = np.percentile(coefficients[:,i,j], 100-alpha/2*100)
            print(f'({left:.2f},{right:.2f}) ' , end ="" )

        print('')

MBOG_TRAIN = 10

def multinomial_bad_ordinal_good(n, rand):
    x1 = np.array([[i+rand.gauss(0,1)] for i in range(n)])

    y = []
    for i, var in enumerate(x1):
        y.append(int(var)%2)


    y=np.array(y)
    return x1, y

def train_multi_bad_ordi_good(rand):
    X, y = multinomial_bad_ordinal_good(MBOG_TRAIN,rand)
    y=y[..., None]

    l = MultinomialLogReg()
    c1 = l.build(X, y)

    l = OrdinalLogReg()
    c2 = l.build(X, y)

    X_test, y_test = multinomial_bad_ordinal_good(1000,rand)
    y_test=y_test[..., None]

    print("Multinomial log loss:", Multinomial_LogReg_Cost_Optimized(
        np.reshape(c1.beta, (1, (c1.num_categories-1)*c1.num_features)),
        np.insert(X_test, 0, np.ones(X_test.shape[0]), axis=1),
        y_test,
        len(np.unique(y))
    ))
    
    print("Ordinal log loss:", OrdinalLogRegCost(
        np.concatenate((c2.delta, c2.betas)),
        np.insert(X_test, 0, np.ones(X_test.shape[0]), axis=1),
        y_test
    ))
    


def train_test_eval():
    X, y = read_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    l = MultinomialLogReg()
    c = l.build(X_train, y_train)

    prob = c.predict(X_test)
    suma=0
    for i,j in zip(prob, y_test):
        if i==j: suma+=1

    print(suma/len(prob))

def train_whole_dataset(bootstrap=False):
    X, y = read_dataset()

    l = MultinomialLogReg()
    c = l.build(X, y)
    betas=c.return_lr_coefficients()

    '''
    for i in range(betas.shape[1]):
        for j in range(betas.shape[0]):
            print(f"{betas[j,i]:.2f} ", end=' ')
        print('')
    '''
    for i in betas:
        for j in i:
            print(f"{j:.2f} ", end=' ')
        print('')

    if bootstrap:
        bootstrap_ci(X, y, 100, 0.05, rand)

if __name__ == "__main__":
    start = time.time()
    rand=random.Random(0)
    
    train_test_eval()
    #train_whole_dataset(TRUE)
    #train_multi_bad_ordi_good(rand)
    
    print("--- %s seconds ---" % (time.time() - start))