import numpy as np

'''
General Algorithm for bisection search:

Searches for the minimum of a function 

X,y: inputs for the function
fn: function must produce scalar output
(it should be noted that fn is the diffrentiated
obj fn)

a: lower limit f(a) < 0 or f(a) > 0
b: upper limit  f(b) > 0 or f(b) < 0
tol: error tolerance, when to stop

smaller = longer = more accurate
'''

def bisection_search(a,b,tol,fn,x,y):

    maxit = 10000
    fa = fn(a,x,y)  # sign(fa) != sign(fb)
    fb = fn(b,x,y)
    
    for i in range(maxit):
        x = (a+b)/2
        fx = fn(x, trainx,trainy)
        if (fx * fb <= 0):
            a = x
            fa = fx
        else:
            b = x
            fb = fx
        if (b-a) < tol:
            break
    return x
