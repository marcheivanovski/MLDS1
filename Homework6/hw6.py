from hashlib import new
import pandas as pd
import numpy as np

def init_starting_weights(nrow, ncol):
    return np.random.rand(nrow, ncol)

def get_one_hot(targets, nb_classes): #target has to start with 0
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def softmax(arr):
    return np.exp(arr) / (np.exp(arr).sum())

def softmaxZ(Z):
    A = np.apply_along_axis(softmax, 1, Z)
    return A

def relu(Z):
    #Z[Z<0]=0
    return np.maximum(0,Z)

def relu_derivative(x):
    x[x<=0]=0
    x[x>0]=1
    return x

class ANNClassification:
    def __init__(self, units, lambda_):
        self.units = units
        self.lambda_ = lambda_

    
    def fit(self, X, y):
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        num_classes = len(np.unique(y))
        y = get_one_hot(y, num_classes)
        y = np.array(y, bool)
        learning_rate = 0.1
        
        m, n = X.shape[0], X.shape[1]
        W = []
        for i in range(len(self.units)+1):
            if len(self.units)==0:
                W.append(init_starting_weights(n, num_classes))
            elif i == 0:
                W.append(init_starting_weights(n, self.units[0]))
            elif i == len(self.units):
                W.append(init_starting_weights(self.units[i-1]+1, num_classes))
            else:
                W.append(init_starting_weights(self.units[i-1]+1, self.units[i]))

        for _ in range(50000):
            A = []; A.append(X)

            #Feed-forward
            for i in range(len(self.units)+1):
                Z=np.dot(A[-1], W[i])
                
                if i == len(self.units):
                    A.append(softmaxZ(Z))
                else:
                    sigmoidZ = 1/(1+np.exp(-1*Z))
                    #reluZ = relu(Z)
                    A.append(np.insert(sigmoidZ, 0, np.ones(sigmoidZ.shape[0]), axis=1))

            #Backprop
            dZs = []
            for i in range(len(self.units), -1, -1):
                if i == len(self.units): #Last layer
                    dZ = A[i+1] - y
                    dW =  1/m * ( np.dot( np.transpose(A[i]), dZ) )
                    dZs.append(dZ)
                else: 
                    dA = np.dot(dZs[-1] , W[i+1].T) 
                    dZ = dA * A[i+1] * (1 - A[i+1])
                    #dZ = dA * relu_derivative(dA)
                    dW = 1/m * ( np.dot( np.transpose(A[i]),dZ) )
                    dW = dW[:,1:]
                    dZ = dZ[:,1:]
                    dZs.append(dZ)
                    
                W[i][1:,] = W[i][1:,]*(1-self.lambda_/m) - learning_rate * dW[1:,]
                #W[i][1:,] = W[i][1:,] - learning_rate * dW[1:,]
                W[i][0,] = W[i][0,] - learning_rate * dW[0,]
                

        self.W = W
        return self


    def predict(self, X):
        #X=np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        
        prev = X
        for i in range(len(self.units)+1):
            prev = np.insert(prev, 0, np.ones(prev.shape[0]), axis=1)
            if i == len(self.units):
                prev = softmaxZ( np.dot(prev, self.W[i]) )
            else:
                prev = 1/(1+np.exp(-np.dot(prev, self.W[i]))) 
                #prev = relu(np.dot(prev, self.W[i]))
            
        
        return prev
    
    
class ANNRegression:
    def __init__(self, units, lambda_):
        self.units = units
        self.lambda_ = lambda_

    def fit(self, X, y):
        X=np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        y = y[ ... , None]
        learning_rate = 0.1

        m, n = X.shape[0], X.shape[1]
        W = []
        for i in range(len(self.units)+1):
            if len(self.units)==0:
                W.append(init_starting_weights(n, 1))
            elif i == 0:
                W.append(init_starting_weights(n, self.units[0]))
            elif i == len(self.units):
                W.append(init_starting_weights(self.units[i-1]+1, 1)) #+1 for biases
            else:
                W.append(init_starting_weights(self.units[i-1]+1, self.units[i])) #+1 for biases

        for _ in range(10000):
            A = []; A.append(X)
            #Z = []

            #Feed-forward
            for i in range(len(self.units)+1):
                if i == len(self.units): #last layer does not have a sigmoid
                    A.append(np.dot(A[-1], W[i]))
                else:
                    Z = np.dot(A[-1], W[i])
                    sigmoidZ=1/(1+np.exp(-1*Z))
                    A.append(np.insert(sigmoidZ, 0, np.ones(sigmoidZ.shape[0]), axis=1)) #add first column of ones

            #Backprog
            dZs = []
            for i in range(len(self.units), -1, -1):
                if i == len(self.units):
                    dZ = (A[i+1]-y)
                    dW = 1/m * ( np.dot( A[i].T, dZ) )
                    dZs.append(dZ)
                else:
                    dA = np.dot(dZs[-1] , W[i+1].T)
                    dZ = dA * A[i+1] * (1 - A[i+1])
                    dW = 1/m * ( np.dot( np.transpose(A[i]),dZ) )
                    dW = dW[:,1:]
                    dZ = dZ[:,1:]
                    dZs.append(dZ)
                    
                W[i][1:,] = W[i][1:,]*(1-self.lambda_/m) - learning_rate * dW[1:,]
                W[i][0,] = W[i][0,] - learning_rate * dW[0,]

        self.W = W
        return self

    def predict(self, X):
        #X=np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        
        prev = X
        for i in range(len(self.units)+1):
            prev = np.insert(prev, 0, np.ones(prev.shape[0]), axis=1)
            if i == len(self.units):
                prev = np.dot(prev, self.W[i])
            else:
                prev = 1/(1+np.exp(-1*np.dot(prev, self.W[i]))) 
            
        return prev[:,0]

    def weights(self):
        return self.W


if __name__ == "__main__":
    X = np.array([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]])
    y = np.array([0, 1, 2, 3])
    #y = np.array([0, 1, 1, 0])

    #fitter = ANNRegression(units=[10, 5], lambda_=0.0001)
    fitter = ANNClassification([10, 20], lambda_=0.0001)
    m = fitter.fit(X, y)
    pred = m.predict(X)
    print(pred)