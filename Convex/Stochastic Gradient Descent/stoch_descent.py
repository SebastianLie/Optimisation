import numpy as np
import math
from scipy.io import loadmat
import time
import matplotlib.pyplot as plt

### DSA3102 CONVEX OPTIMISATION HW2 StochGD  ###
### Author: Sebastian Lie ###

mat = loadmat('HW1data.mat')

# by default 
testx = mat["Xtest"] 
trainx = mat["Xtrain"]  # (3065,57)
testy = mat["ytest"]  # (3065,1)
trainy = mat["ytrain"]

## line search method  ##

def armijo(alpha_bar,w,d,beta,sigma):
    fx0 = loglikelihood(w,trainx,trainy)
    alpha = alpha_bar
    delta = np.dot(ll_grad(w,trainx,trainy),d)
    while loglikelihood(w+alpha*d,trainx,trainy) >  fx0 + alpha*sigma*delta:
        alpha = beta * alpha
    return alpha

######################################

## Objective function and its gradient ##

def loglikelihood(w, X, y):
    result = 0
    for i in range(len(X)):
        exp_part = np.exp(-y[i] * np.dot(np.transpose(w),X[i]))
        result += np.log(1+exp_part)[0]
    return result/len(X)

def ll_grad(w, X, y):
    grad = np.zeros(57)
    for i in range(len(X)):
        exp_part = np.exp(y[i] * np.dot(np.transpose(w),X[i]))
        numerator = -y[i]*X[i]
        grad = grad +(numerator/(1+exp_part))
    return grad/len(X)

def ll_grad_stoch(w, X, y, indices):
    grad = np.zeros(57)
    for i in indices:
        exp_part = np.exp(y[i] * np.dot(np.transpose(w),X[i]))
        numerator = -y[i]*X[i]
        grad += (numerator/(1+exp_part))
    return grad/len(indices)

######################################

## Stochastic gradient descent ##

def stoch_descent(x0, X, y, batch, maxit, tol):
    
    step_size = 1
    w = x0  # inital guess
    obj_value_list = list()
    num_iter = 0
    m = len(X)
    p = m//batch
    maxit = maxit//(m//batch) #  quotient so get whole number 
    for j in range(1,maxit+1):
        
        for i in range(p):
            indices = np.random.randint(0, len(X)-1, size=batch)
            grad = ll_grad_stoch(w,trainx,trainy,indices) # choose random part of d and minus?
            d = -grad
            w_prev = w
            step_size = 1/(j**2) # fixed
            #step_size = armijo(step_size,w,d,0.7,0.1)  # default use armijo step size
            # update 
            w = w +  step_size * d
            #print(loglikelihood(w,trainx,trainy))
            
        norm_d = np.linalg.norm(d)
        obj_value = loglikelihood(w,trainx,trainy)
        obj_value_list.append(obj_value)

        print("Iteration {0}: obj = {1:9.3f}, err = {2:9.3f}".format(j,obj_value, norm_d))
        num_iter += 1
        if norm_d < tol:
            break
        
            
    return w, obj_value_list, num_iter 

######################################

## Helper functions that produce useful things ##


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def predict(w, xtest, ytest):
    if len(w) != len(xtest):
        w = np.transpose(w)
    yhat = sigmoid(np.dot(xtest,w))
    yhat = np.fromiter(map(lambda x: 1 if x > 0.5 else -1,yhat),dtype=np.double)
    correct = 0
    for i in range(len(yhat)):
	    if yhat[i] == ytest[i]:
		    correct += 1
    return correct/len(yhat)

def testing():
    total_time = 0
    total_train = 0
    total_test = 0
    total_iter = 0
    for i in range(1000):
        print(i)
        np.random.seed(i)
        start = time.time()
        w, obj_values, num_iter = stoch_descent(np.ones(57),trainx,trainy,10,10000000,0.1)
        end = time.time()
        total_train += predict(w,trainx,trainy)
        total_time += end-start
        total_test += predict(w,testx,testy)
        total_iter += num_iter
    return {"avg_time":total_time/1000, "avg_train_rate": total_train/1000, "avg_test_rate":total_test/1000, "avg iterations": total_iter/1000}

print(testing())
#{'avg_time': 0.2990842673778534, 'avg_train_rate': 0.9312143556280577, 'avg_test_rate': 0.9249720052083341, 'avg iterations': 2.507}
'''
start = time.time()
w, obj_values, num_iter = stoch_descent(np.ones(57),trainx,trainy,10,10000000,0.1)
end = time.time()
print(predict(w,trainx,trainy))
print(predict(w,testx,testy))
'''
