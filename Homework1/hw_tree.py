from re import L
import numpy as np
import math
import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import Counter
import time

def all_columns(X, rand):
    return range(X.shape[1])


def random_sqrt_columns(X, rand):
    c = rand.sample([i for i in range(X.shape[1])], int(math.sqrt(X.shape[1]))) # select random columns
    if len(np.unique(X[:,c[0]]))==1: #hardcode -> fix
        c=[0]
    return c

def cost(y):
    num_nonzeros = np.count_nonzero(y)/len(y)
    num_zeros = 1-num_nonzeros
    return 1-(num_nonzeros**2+num_zeros**2)

def majority(y):
    num_nonzeros = np.count_nonzero(y)
    num_zeros = len(y)-num_nonzeros
    if num_nonzeros>num_zeros:
        return 1
    else:
        return 0

class Tree:

    def __init__(self, rand=None,
                 get_candidate_columns=all_columns,
                 min_samples=2):
        self.rand = rand  # for replicability
        self.get_candidate_columns = get_candidate_columns  # needed for random forests
        self.min_samples = min_samples

    def build(self, X, y):
        return TreeNode(X, y, self.min_samples, self.get_candidate_columns, self.rand) # dummy output


class TreeNode:

    def __init__(self, X, y, min_samples, get_candidate_columns, rand):
        self.X=X
        self.y=y
        self.min_samples=min_samples
        self.get_candidate_columns=get_candidate_columns
        self.rand = rand
        self.fit_tree()
        #The following are initialized later:
        #self.max_gain_attr_idx
        #self.max_gain_attr_val
        #self.leaf

    def fit_tree(self):
        if self.X.shape[0]<self.min_samples:
            self.leaf=True
            num_nonzeros = np.count_nonzero(self.y)
            num_zeros = len(self.y)-num_nonzeros
            if num_nonzeros<=num_zeros:
                self.majority=1
            else:
                self.majority=0
        elif np.count_nonzero(self.y)==self.X.shape[0]:
            self.leaf=True
            self.majority=1
        elif len(self.y)-np.count_nonzero(self.y)==self.X.shape[0]:
            self.leaf=True
            self.majority=0
        else:
            max_gain=0
            self.max_gain_attr_idx=0
            self.max_gain_attr_val=0
            self.leaf=False
            max_logical_indices=[]

            features=self.get_candidate_columns(self.X, self.rand)
            for col in features:  #recall cols start with 0 aka this is not matlab :)
                #print(self.X[:, col]) is the feature column
                possible_values=np.unique(self.X[:,col])
                for value in possible_values:
                    logical_indices = self.X[:,col] <= value
                    D_l = self.y[logical_indices]
                    D_r = self.y[~logical_indices]
                    
                    if len(D_l)==0 or len(D_r)==0: #check if one partition is empty
                        continue
                    
                    gain = cost(self.y)-( cost(D_l)*len(D_l)/len(self.y) + cost(D_r)*len(D_r)/len(self.y) )

                    if gain>max_gain:
                        max_gain=gain
                        self.max_gain_attr_idx=col
                        self.max_gain_attr_val=value
                        max_logical_indices=logical_indices

            self.L = TreeNode(self.X[max_logical_indices, :], self.y[max_logical_indices], self.min_samples, self.get_candidate_columns, self.rand)
            self.R = TreeNode(self.X[~max_logical_indices, :], self.y[~max_logical_indices], self.min_samples, self.get_candidate_columns, self.rand)

    def predict(self, X):
        predictions=np.zeros(X.shape[0])
        for i, instance in enumerate(X):
            predictions[i]=self.predict_single(instance)

        return predictions  

    def predict_single(self, x):
        if self.leaf==True:
            return self.majority
        elif x[self.max_gain_attr_idx]<=self.max_gain_attr_val: #go left
            return self.L.predict_single(x)
        else: 
            return self.R.predict_single(x)


class RandomForest:

    def __init__(self, rand=None, n=50): 
        self.n = n #n is number of bootstrap samples
        self.rand = rand
        self.rftree = Tree(...)  # initialize the tree properly

    def build(self, X, y):
        # ...
        return RFModel(X,y, self.rand, self.n)


class RFModel:

    def __init__(self, X, y, rand, n):
        self.rand = rand
        self.X = X
        self.y = y
        self.n = n
        self.trees=[] #all n trees => list of type TreeNode object
        self.out_of_bag=[] #unused samples for every tree
        self.fit_rfmodel()

    def fit_rfmodel(self):
        all_instances=[i for i in range(self.X.shape[0])]
        
        for i in range(self.n):
            boostrap_sample_indices=self.rand.choices(all_instances, k=self.X.shape[0])
            bootstrap_sample_X=self.X[boostrap_sample_indices,:]
            bootstrap_sample_y=self.y[boostrap_sample_indices]
            
            t = Tree(rand=self.rand,
                 get_candidate_columns=random_sqrt_columns,
                 min_samples=2)
            p = t.build(bootstrap_sample_X, bootstrap_sample_y)

            self.out_of_bag.append(np.setdiff1d(all_instances, boostrap_sample_indices))
            self.trees.append(p)


    def predict(self, X):
        predictions=np.zeros(X.shape[0])

        all_predictions=np.zeros((self.n, X.shape[0]))
        for i,treenode in enumerate(self.trees):
            all_predictions[i,:]=treenode.predict(X)

        for col in range(all_predictions.shape[1]):
            num_nonzeros = np.count_nonzero(all_predictions[:,col])
            num_zeros = all_predictions.shape[0]-num_nonzeros
            if num_nonzeros>num_zeros:
                predictions[col]=1
            else:
                predictions[col]=0

        return predictions

    def importance(self):
        imps = np.zeros(self.X.shape[1])

        mc_scores=np.zeros((self.X.shape[1]+1, self.n))

        tree_counter=0
        for out_of_bag_samps,treenode in zip(self.out_of_bag, self.trees):
            if(len(out_of_bag_samps))==0:
                continue
            oob_sample_X=self.X[out_of_bag_samps,:] 
            oob_sample_y=self.y[out_of_bag_samps]

            for col in range(self.X.shape[1]):
                oob_sample_X_shuff = np.copy(oob_sample_X)
                #oob_sample_X_shuff[:,col] =  np.random.permutation(oob_sample_X[:,col])
                np.random.shuffle(oob_sample_X_shuff[:,col]) 
                preds = treenode.predict(oob_sample_X_shuff)
                mc=1-(np.count_nonzero(np.equal(preds,oob_sample_y))/len(preds)) 
                mc_scores[col,tree_counter]=mc

            #UNSHUFFLED
            preds = treenode.predict(oob_sample_X)
            mc=1-(np.count_nonzero(np.equal(preds,oob_sample_y))/len(preds)) 
            mc_scores[self.X.shape[1],tree_counter]=mc
            tree_counter+=1

        ms_averaged=mc_scores.sum(axis = 1)/self.n
        imps=ms_averaged[:self.X.shape[1]]-np.array([ms_averaged[self.X.shape[1]] for i in range(self.X.shape[1])])

        #plt.scatter([i for i in range(1,len(imps)+1)], imps)
        #plt.show()
        
        return imps

def tki(): 
    df = pd.read_csv('tki-resistance.csv', delimiter=',')
    df['Class'] = df['Class'].map({'Bcr-abl': 1,
                             'Wild type': 0},
                             na_action=None)
    data = df.to_numpy()                         

    return data[0:130,:], data[130:,:]

def std_err(predictions, ground):
    mc=1-(np.count_nonzero(np.equal(predictions,ground))/len(predictions)) 
    sample_score=~np.equal(predictions,ground)
    return math.sqrt(sum([(i-mc)**2 for i in sample_score])/(len(predictions)))/math.sqrt(len(predictions))


def dt_bootstrap_estimated_variance(dataset, test, rand, RF=False):
    features = dataset.shape[1]
    m=100
    n=dataset.shape[0]
    all_instances=[i for i in range(n)]

    estimators=[]

    if not RF:
        t = Tree(rand=rand,
                min_samples=2)

        p = t.build(dataset[:,:features-1], dataset[:,features-1])
    else:
        rf = RandomForest(rand=rand,
                        n=100)
        p = rf.build(dataset[:,:features-1], dataset[:,features-1])

    for i in range(m):
        boostrap_sample_indices=rand.choices(all_instances, k=n)
        X=dataset[boostrap_sample_indices,:features-1]
        y=dataset[boostrap_sample_indices,features-1]

        pred = p.predict(X)

        true_predictions=0
        for i in range(len(pred)):
            if pred[i]==y[i]:
                true_predictions+=1

        estimators.append(1-true_predictions/n)
        
    estimator_avg=sum(estimators)/len(estimators)
    sd1=1/(m-1)*sum([(i-estimator_avg)**2 for i in estimators])


    n=test.shape[0]
    all_instances=[i for i in range(n)]
    estimators=[]
    for i in range(m):
        boostrap_sample_indices=rand.choices(all_instances, k=n)
        X=test[boostrap_sample_indices,:features-1]
        y=test[boostrap_sample_indices,features-1]

        pred = p.predict(X)

        true_predictions=0
        for i in range(len(pred)):
            if pred[i]==y[i]:
                true_predictions+=1

        estimators.append(1-true_predictions/n)
        
    estimator_avg=sum(estimators)/len(estimators)
    sd2=1/(m-1)*sum([(i-estimator_avg)**2 for i in estimators])

    return (math.sqrt(sd1),math.sqrt(sd2))


def hw_tree_full(learn, test):
    if type(learn) is tuple:
        learn_x, learn_y = learn
        test_x, test_y = test
        learn=np.hstack((learn_x, np.atleast_2d(learn_y).T ))
        test=np.hstack((test_x, np.atleast_2d(test_y).T ))

    features = learn.shape[1]

    ## ON LEARN
    t = Tree(rand=random.Random(1),
                min_samples=2)

    p = t.build(learn[:,:features-1], learn[:,features-1])
    pred = p.predict(learn[:,:features-1])
    ground = learn[:,features-1]

    mcl=1-(np.count_nonzero(np.equal(pred,ground))/len(pred))
    stdl = std_err(pred, ground)

    ## ON TEST
    pred = p.predict(test[:,:features-1])
    ground = test[:,features-1]

    mct=1-(np.count_nonzero(np.equal(pred,ground))/len(pred))
    stdt = std_err(pred, ground)

    #stdl, stdt = dt_bootstrap_estimated_variance(learn,test, random.Random(1), False)
    return (( mcl, stdl) ,  (mct, stdt))

def hw_randomforests(learn, test):
    if type(learn) is tuple:
        learn_x, learn_y = learn
        test_x, test_y = test
        learn=np.hstack((learn_x, np.atleast_2d(learn_y).T ))
        test=np.hstack((test_x, np.atleast_2d(test_y).T ))

    features = learn.shape[1]

    ## ON LEARN
    rf = RandomForest(rand=random.Random(0),
                        n=100)
    p = rf.build(learn[:,:features-1], learn[:,features-1])
    pred = p.predict(learn[:,:features-1])
    ground = learn[:,features-1]

    mcl=1-(np.count_nonzero(np.equal(pred,ground))/len(pred))
    stdl = std_err(pred, ground)

    #ON TEST
    pred = p.predict(test[:,:features-1])
    ground = test[:,features-1]

    mct=1-(np.count_nonzero(np.equal(pred,ground))/len(pred))
    stdt = std_err(pred, ground)

    #print(p.importance())
    
    #stdl, stdt = dt_bootstrap_estimated_variance(learn,test, random.Random(1), True)
    return ((mcl, stdl), (mct, stdt))
    #return ((round(mcl*100,2), round(stdl*100,2)), (round(mct*100,2), round(stdt*100,2)))

def plot_mc_trees(learn, test):
    features = learn.shape[1]
    
    num_trees=[]
    mcs=[]
    stds=[]

    for i in range(1,101):
        rf = RandomForest(rand=random.Random(0),
                            n=i)
        p = rf.build(learn[:,:features-1], learn[:,features-1])

        pred = p.predict(test[:,:features-1])
        ground = test[:,features-1]

        mct=1-(np.count_nonzero(np.equal(pred,ground))/len(pred))
        stdt = std_err(pred, ground)
        num_trees.append(i)
        mcs.append(mct)
        stds.append(stdt)
        
    print(num_trees)
    print(mcs)
    print(stds)

    #plt.plot(num_trees,mcs)
    #plt.show()

def vars_in_roots(learn,rand):
    features = learn.shape[1]

    all_instances=[i for i in range(learn.shape[0])]
    roots=[]
    for i in range(100):
        boostrap_sample_indices=rand.choices(all_instances, k=learn.shape[0])
        bootstrap_sample=learn[boostrap_sample_indices,:]
        t = Tree(rand=random.Random(0),
                    min_samples=2)

        p = t.build(bootstrap_sample[:,:features-1], bootstrap_sample[:,features-1])
        roots.append(p.max_gain_attr_idx+1)

    roots_counted=Counter(roots)
    roots_counted=dict(sorted(roots_counted.items(), key=lambda x:x[1]))
    print(roots_counted)


if __name__ == "__main__":
    start = time.time()

    #learn, test, legend = tki()
    learn, test = tki()

    print("full", hw_tree_full(learn, test))
    print("random forests", hw_randomforests(learn, test))
    #plot_mc_trees(learn,test)
    #vars_in_roots(learn,random.Random(0))

    print("--- %s seconds ---" % (time.time() - start))