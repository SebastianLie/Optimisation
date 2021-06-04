import numpy as np
import math
from scipy.io import loadmat
import time
import matplotlib.pyplot as plt

### DSA3102 CONVEX OPTIMISATION HW1  ###
### Author: Sebastian Lie ###

mat = loadmat('HW1data.mat')

# by default 
testx = mat["Xtest"] 
trainx = mat["Xtrain"]  # (3065,57)
testy = mat["ytest"]  # (3065,1)
trainy = mat["ytrain"]

## line search methods  ##

def gprime(w,d,t,X,y):
    res = 0
    for i in range(len(X)):
        exp_part = np.exp(-y[i] * np.dot(w,X[i])) * np.exp(-y[i] * t * np.dot(d,X[i]))
        numerator = -y[i]*np.dot(np.transpose(d),X[i])*exp_part
        res = res +(numerator/(1+exp_part))
    print(res)
    return res[0]

def gprimeprime(w,d,t,X,y):
    res = 0 
    for i in range(len(X)):
        exp_part = np.exp(-y[i] * np.dot(np.transpose(w),X[i])) * np.exp(-y[i] *t* np.dot(np.transpose(d),X[i]))
        numerator = (-y[i]*np.dot(np.transpose(d),X[i])**2)*exp_part
        res = res +(numerator/(1+exp_part)**2)
    print(res[0])
    return res[0]
    
def newtons(w, d, tol):
    t = 1
    while abs(gprime(w,d,t,trainx,trainy)) > tol:
        
        t = t - (gprime(w,d,t,trainx,trainy)/gprimeprime(w,d,t,trainx,trainy))
    return t

def golden_search(w, d, a, b, maxit, tol):

    phi = (sqrt(5.0) - 1)/2.0
    lam = b - phi*(b - a)
    mu = a + phi *(b-a)
    flam = loglikelihood(w+lam*d,trainx,trainy)
    fmu = loglikelihood(w+mu*d,trainx,trainy)
    for i in range(maxit):
        if flam > fmu:
            a = lam
            lam = mu
            mu = a + phi*(b-a)
            fmu = loglikelihood(w+mu*d,trainx,trainy)
        else:
            b = mu
            mu = lam
            fmu = flam
            lam = b - phi*(b-a)
            flam = loglikelihood(w+lam*d,trainx,trainy)

        if (b-a) <= tol:
            break
    return (b-a)/2

def bisection(a,b,tol):

    maxit = 10000
    la = loglikelihood(a, trainx, trainy)
    lb = loglikelihood(b, trainx, trainy)
    
    for i in range(maxit):
        x = (a+b)/2
        lx = loglikelihood(x, trainx,trainy)
        if (lx * lb <= 0):
            a = x
            la = lx
        else:
            b = x
            lb = lx
        if (b-a) < tol:
            break
    return x


def armijo(alpha_bar,w,d,beta,sigma):
    fx0 = loglikelihood(w,trainx,trainy)
    alpha = alpha_bar
    delta = np.dot(loglikelihood_grad(w,trainx,trainy),d)
    while loglikelihood(w+alpha*d,trainx,trainy) >  fx0 + alpha*sigma*delta:
        alpha = beta * alpha
    return alpha

######################################

## Objective function and its gradient ##

def loglikelihood(w, X, y):
    result = sum(np.log(1+np.exp(-y * np.dot(w,np.transpose(X)))))[0]
    return result/len(X)

def loglikelihood_grad(w, X, y):
    grad = np.zeros(57)
    for i in range(len(X)):
        exp_part = np.exp(-y[i] * np.dot(np.transpose(w),X[i]))
        numerator = -y[i]*exp_part*X[i]
        grad = grad +(numerator/(1+exp_part))
    return grad/len(X)

def ll_grad(w, X,y):
    exp_part = sum(np.exp(-y * np.dot(w,np.transpose(X))))[0]
    result = (-y*exp_part*X)/(1+exp_part)
    return result/len(X)
    

######################################

## Steepest descent ##


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def steepest_descent(X,y,w0,maxit,tol,*line_search_params):  # use varargs
    step_size = 1
    w = w0  # inital guess
    line_search_method = line_search_params[0]
    obj_value_list = list()
    num_iter = 0
    for i in range(1,maxit):
        
        grad = loglikelihood_grad(w,trainx,trainy)
        d = -grad
        norm_d = np.linalg.norm(d)
        if norm_d < tol:
            break
        obj_value = loglikelihood(w,trainx,trainy)[0]
        obj_value_list.append(obj_value)
        print("Iteration {0}: obj = {1:9.3f}, err = {2:.4f}".format(i,obj_value,norm_d))
        if norm_d < tol:
            break
        else:
            
            if line_search_method == "armijo":
                beta = line_search_params[1]
                sigma = line_search_params[2]
                w_prev = w
                step_size = armijo(step_size,w,d,beta,sigma)
                
            elif line_search_method == "fixed":
                step_size = line_search_params[1]
                w_prev = w
                
            elif line_search_method == "exact":
                step_size = newtons(w,d,0.1)

            elif line_search_method == "diminishing":
                step_size = 1/(i**2)
                
            else:  # default is armijos
                w_prev = w
                step_size = armijo(step_size,w,d,0.7,0.2)

            # update 
            w = w +  step_size * d
            num_iter += 1
            
    return w, obj_value_list, num_iter 

######################################

## Helper functions that produce useful things ##

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


# Vimpt
def test_grad(): # proof that grad function works well enough
    alp = 1*10**(-8)
    x = np.random.rand(57)
    differences = list()
    for i in range(57):
        e0 = np.zeros(57)
        e0[i] = 1 # test 0th part
        diff = loglikelihood_grad(x,trainx,trainy)[i] - (loglikelihood(x+alp*e0,trainx,trainy)-loglikelihood(x, trainx,trainy))/alp
        differences.append(round(diff[0],5))
    return differences
'''
Produced:
[-0.01087, -0.00607, -0.00211, -0.00653, -0.00775, -0.00695, -0.01511, -0.00564, -0.01248, -0.00927, -0.01077, -0.00991, -0.00842, -0.00685, -0.00771, -0.007,
-0.01319, -0.01054, -0.0055, -0.00613, -0.01242, -0.00824, -0.00657, -0.00809, -0.00528, -0.00606, -0.0072, -0.00767, -0.00747, -0.0079, -0.00731, -0.00784,
-0.00672, -0.00775, -0.00796, -0.01021, -0.00604, -0.00764, -0.00567, -0.00743, -0.00753,
-0.00705, -0.00747, -0.00642, -0.00621, -0.00564, -0.00703, -0.00686, -0.005, -0.00376, -0.00468, 0.00207, -0.00642, -0.00411, -0.00089, -0.00049, -0.00891]
'''


def GridSearch():
    
    # find best armijo parameters and initial solution
    for b in np.arange(0.1,1,0.1): # use arange cos need to iterate through floats
        for s in np.arange(0.1,0.5,0.1):
            start =time.time()
            wA = steepest_descent(trainx,trainy,np.zeros(57),100000,0.5,"armijo",b,s)
            end = time.time()
            acc_dict["w0 = 0, Armijo beta = {0}, sigma = {1}".format(b,s)] = (predict(wA,testx,testy),end-start)
    for b in np.arange(0.1,1,0.1):
        for s in np.arange(0.1,0.5,0.1):
            start =time.time()
            wA = steepest_descent(trainx,trainy,np.ones(57),100000,0.5,"armijo",b,s)
            end =time.time()
            acc_dict["w0 = 1, Armijo beta = {0}, sigma = {1}".format(b,s)] = (predict(wA,testx,testy),end-start)
    for b in np.arange(0.1,1,0.1):
        for s in np.arange(0.1,0.5,0.1):
            start =time.time()
            wA = steepest_descent(trainx,trainy,-1*np.ones(57),100000,0.5,"armijo",b,s)
            end = time.time()
            acc_dict["w0 = -1, Armijo beta = {0}, sigma = {1}".format(b,s)] = (predict(wA,testx,testy),end-start)


def plot_results():
    
    wA, armijo_obj_values, num_iter1 = steepest_descent(trainx,trainy,np.ones(57),100000,100,"armijo",0.7,0.2)
    wA2, armijo_obj_values2, num_iter2 = steepest_descent(trainx,trainy,np.ones(57),100000,100,"armijo",0.7,0.1)

    fig, ax = plt.subplots()
    ax.plot(armijo_obj_values, range(1,num_iter1+1), 'r',label="beta = 0.7, sigma = 0.2") 
    ax.plot(armijo_obj_values2, range(1,num_iter2+1), 'b',label="beta = 0.7, sigma = 0.1") 
    plt.xlabel("Objective function values, Armijo's Step Size Strategy")
    plt.ylabel("Number of iterations")
    legend = ax.legend(loc='upper right',fontsize='small')
    plt.show()

    wf, fixed_obj_values, num_iter3 = steepest_descent(trainx,trainy,np.ones(57),100000,150,"fixed",0.001)
    wf2, fixed_obj_values2, num_iter4 = steepest_descent(trainx,trainy,np.ones(57),100000,150,"fixed",0.0005)

    fig, ax = plt.subplots()
    ax.plot(fixed_obj_values, range(1,num_iter3+1), 'r',label="Step Size = 0.001")  
    ax.plot(fixed_obj_values2, range(1,num_iter4+1), 'b',label="Step Size = 0.0005") 
    plt.xlabel("Objective function values, Fixed Step Size Strategy")
    plt.ylabel("Number of iterations")
    legend = ax.legend(loc='upper right',fontsize='small')
    plt.show()

#GridSearch()
#plot_results()
#start = time.time()
#wA1, armijo_obj_values, num_iter1 = steepest_descent(trainx,trainy,np.ones(57),100000,0.001,"armijo",0.7,0.1)
#end = time.time()
#wA2, armijo_obj_values, num_iter1 = steepest_descent(trainx,trainy,np.ones(57),100000,100,"armijo",0.7,0.4)

#wA3, armijo_obj_values, num_iter1 = steepest_descent(trainx,trainy,np.ones(57),100000,100,"armijo",0.6,0.1)
#print(predict(wA1,trainx,trainy))
#print(end-start)
#print(predict(wA1,testx,testy))
#print(predict(wA2,trainx,trainy))
#print(predict(wA3,trainx,trainy))

