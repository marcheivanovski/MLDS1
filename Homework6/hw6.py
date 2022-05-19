import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ridge_multinomial import RidgeReg, MultinomialLogReg

def init_starting_weights(nrow, ncol):
    return np.random.rand(nrow, ncol)

#Classes must start with 0 and go to n-1 !!!
def get_one_hot(targets, nb_classes): #target has to start with 0
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def softmax(arr):
    return np.exp(arr) / (np.exp(arr).sum())

def softmaxZ(Z):
    A = np.apply_along_axis(softmax, 1, Z)
    return A

def relu(Z):
    return np.maximum(0,Z)

def relu_derivative(x):
    x[x<=0]=0
    x[x>0]=1
    return x

def root_mean_squared_error(y_test, y_pred):
    return np.sqrt(np.mean(((y_test-y_pred)**2)))

def classification_accuracy(y_test, y_pred):
    return np.sum(y_pred==y_test)/len(y_test)

def compute_multiclass_loss(Y, p):   # Y -> actual, Y_hat -> predicted
    #L_sum = np.sum(np.multiply(Y, np.log(p)))
    L_sum = np.sum(np.log(np.choose(Y, p.T)))
    m = len(Y)
    L = -(1/m) * L_sum

    L = np.squeeze(L)
    return L

def compute_regression_loss(y,y_hat):
    return np.sum((y_hat - y)**2)


def feed_forward(regression, units, W, X):
    A = []; A.append(X)
    #Feed-forward
    for i in range(len(units)+1):
        Z=np.dot(A[-1], W[i])
        
        if i == len(units):
            if regression:
                ...
            else:
                A.append(softmaxZ(Z))
        else:
            sigmoidZ = 1/(1+np.exp(-1*Z))
            #reluZ = relu(Z)
            A.append(np.insert(sigmoidZ, 0, np.ones(sigmoidZ.shape[0]), axis=1))

    return A


class ANNClassification:
    def __init__(self, units, lambda_):
        self.units = units
        self.lambda_ = lambda_


    #def check_gradient_cost(self):
         

    
    def fit(self, X, y, early_stop=False):
        if early_stop:
            X, x_val, y, y_val = train_test_split(X, y, test_size=0.3, random_state=123)

        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        num_classes = len(np.unique(y))
        y = get_one_hot(y, num_classes)
        y = np.array(y, bool)
        learning_rate = 0.2
        
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

        self.W = W
        continous_close=0
        previous_loss=0
        for i in range(100000):
            if early_stop:
                y_val_pred=self.predict(x_val)
                current_loss = compute_multiclass_loss(y_val, y_val_pred)
                if abs(current_loss-previous_loss)<0.00001:
                    continous_close+=1
                else:
                    continous_close=0

                if continous_close>=10:
                    print("Stopping after",i,"iterations.")
                    break
                previous_loss=current_loss

            '''A = []; A.append(X)

            #Feed-forward
            for i in range(len(self.units)+1):
                Z=np.dot(A[-1], W[i])
                
                if i == len(self.units):
                    A.append(softmaxZ(Z))
                else:
                    sigmoidZ = 1/(1+np.exp(-1*Z))
                    #reluZ = relu(Z)
                    A.append(np.insert(sigmoidZ, 0, np.ones(sigmoidZ.shape[0]), axis=1))'''
            A = feed_forward(False, self.units, W, X)

            #Backprop
            dZs = []
            dWs = []
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
                    
                dWs.append(dW)
                #W[i][1:,] = W[i][1:,]*(1-self.lambda_/m) - learning_rate * dW[1:,]
                #W[i][0,] = W[i][0,] - learning_rate * dW[0,]

            for i in range(len(self.units), -1, -1):
                W[i][1:,] = W[i][1:,]*(1-self.lambda_/m) - learning_rate * dWs[len(self.units)-i][1:,]
                W[i][0,] = W[i][0,] - learning_rate * dWs[len(self.units)-i][0,]

            self.W = W

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

    def predict_class(self, X):
        #X=np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        
        prev = X
        for i in range(len(self.units)+1):
            prev = np.insert(prev, 0, np.ones(prev.shape[0]), axis=1)
            if i == len(self.units):
                prev = softmaxZ( np.dot(prev, self.W[i]) )
            else:
                prev = 1/(1+np.exp(-np.dot(prev, self.W[i]))) 
                #prev = relu(np.dot(prev, self.W[i]))

        classes_predictions = []
        for i in prev:
            classes_predictions.append(i.argmax())

        return np.array(classes_predictions)
    
    
class ANNRegression:
    def __init__(self, units, lambda_):
        self.units = units
        self.lambda_ = lambda_

    def fit(self, X, y, early_stop=False):
        if early_stop:
            X, x_val, y, y_val = train_test_split(X, y, test_size=0.3, random_state=123)

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

        self.W = W
        previous_loss=0
        continous_close=0
        for i in range(100000):
            if early_stop:
                y_val_pred=self.predict(x_val)
                current_loss = compute_regression_loss(y_val, y_val_pred)
                if abs(current_loss-previous_loss)<0.00001:
                    continous_close+=1
                else:
                    continous_close=0

                if continous_close>=10:
                    print("Stopping after",i,"iterations.")
                    break
                previous_loss=current_loss

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
            dWs = []
            for i in range(len(self.units), -1, -1):
                if i == len(self.units):
                    dZ = (A[i+1]-y)
                    dW = 1/m * ( np.dot( A[i].T, dZ) )
                    dZs.append(dZ)
                else:
                    dA = np.dot(dZs[-1] , W[i+1].T)
                    dZ = A[i+1] * (1 - A[i+1]) * dA
                    dW = 1/m * ( np.dot( np.transpose(A[i]),dZ) )
                    dW = dW[:,1:]
                    dZ = dZ[:,1:]
                    dZs.append(dZ)
                  
                dWs.append(dW) 
                #W[i][1:,] = W[i][1:,]*(1-self.lambda_/m) - learning_rate * dW[1:,]
                #W[i][0,] = W[i][0,] - learning_rate * dW[0,]
            
            for i in range(len(self.units), -1, -1):
                W[i][1:,] = W[i][1:,]*(1-self.lambda_/m) - learning_rate * dWs[len(self.units)-i][1:,]
                W[i][0,] = W[i][0,] - learning_rate * dWs[len(self.units)-i][0,]
            
            self.W = W

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


def housing2r():
    df = pd.read_csv('housing2r_standardized.csv')
    X, y = df.iloc[:,0:5].to_numpy(), df.iloc[:,5].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    #Neural network
    fitter = ANNRegression(units=[5,5], lambda_=0.0001)
    m = fitter.fit(X, y, True)
    pred = m.predict(X)
    print("Neural network", root_mean_squared_error(y, pred))

    #Ridge Regression
    model = RidgeReg(0.01)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print("Ridge regression:",root_mean_squared_error(y_test, pred))

def housing3():
    df = pd.read_csv('housing3_standardized.csv')
    X, y = df.iloc[:,0:13].to_numpy(), df.iloc[:,13].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    #Neural network
    fitter = ANNClassification(units=[], lambda_=0.1)
    m = fitter.fit(x_train, y_train, early_stop=True)
    pred = m.predict_class(x_test)
    print("Neural network with no hidden layers:",classification_accuracy(y_test, pred))

    #Multinomial
    l = MultinomialLogReg()
    c = l.build(x_train, y_train)
    pred = c.predict(x_test)
    print("Multinomial with:",classification_accuracy(y_test, pred))

    



if __name__ == "__main__":
    #housing2r()
    housing3()

    '''
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
    '''