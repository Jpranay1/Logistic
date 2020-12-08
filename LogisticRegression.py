import numpy as np
import random

def gen(theta, n, m):
        beta = np.random.randint(0,9, (m+1,1))
        X = np.random.uniform(-1,1, (n, m+1))
        X[:,0] = 1
        z = np.dot(X, beta)
        y = np.ones(len(z))
        y = 1/(1 + np.exp(-z))
        y = np.where(y > 0.5, 1, 0)
        noise = np.random.binomial(1, theta, (n,1))
        y = y + noise
        y = np.where(y > 1, 0, y)
        b = int((2/3)*n)
        Xtrain = X[0:b,:]
        Ytrain = y[0:b]
        Xtest = X[b:,:]
        Ytest = y[b:]
        return beta, X, y, Xtrain, Ytrain, Xtest, Ytest

def costfun(x,y,beta):
        m = len(x[0,:])
        n = len(x[:,0])
        z = np.dot(x, beta)
        h = 1/(1 + np.exp(-z))
        h = np.transpose(h)
        return (1/n)*(-np.dot(np.log(h), y) - np.dot(np.log(1-h), (1-y)))
        
def grad_desc(x,y,beta,alpha,k,tou):
        m = len(x[0,:])-1 
        n = len(x[:,0])
        precost = 0
        for i in range(k+1):
            if(costfun(x,y,beta) - precost > tou):
                    z = np.dot(x, beta)
                    h = 1/(1 + np.exp(-z))
                    r = h - y
                    p = np.transpose(r)
                    precost = costfun(x,y,beta)
                    beta =  beta - (alpha/n)*np.transpose(np.dot(p,x))
            else:
                    break
        J = costfun(x,y,beta)
        return beta, J

def log_reg(X,y,k,alpha):
        m = len(X[0,:])-1 
        n = len(X[:,0])
        beta = np.random.randint(0,9, (m+1,1))
        beta, J = grad_desc(X,y,beta,alpha,k,tou)
        return beta, J

beta, X, y, Xa1, Ya1, Xa2, Ya2 = gen(0,500,10)
print('The value of beta :','\n', beta,'\n')
print('The array X : ','\n', X,'\n')
print('Binary array Y : ','\n', y,'\n')

k = 100
alpha = 0.5
tou = 0.001
betaN, J = log_reg(Xa1,Ya1,k,alpha)
print('The beta after gradient Descent : ', betaN,'\n')
print('The fnial cost value after GD : ', J,'\n')
def predict(beta,X):
        z = np.dot(X, beta)
        y = np.ones(len(z))
        y = 1/(1 + np.exp(-z))
        y = np.where(y > 0.5, 1, 0)
        return y
#yp1 = predict(betaN,Xa1)
#print(yp,'\n')
Yp2 = predict(betaN,Xa2)

def matrix(y1, y2):
        n = len(y1)
        mat = np.where(y1 == y2, 1,0)
        TP = np.sum(np.where((y1==1) & (y2==1), 1,0))
        FN = np.sum(np.where((y1==1) & (y2==0), 1,0))
        TN = np.sum(np.where((y1==0) & (y2==0), 1,0))
        FP = np.sum(np.where((y1==0) & (y2==1), 1,0))
        acc = np.sum(mat)/n
        TPR = TP/(TP + FN)
        FPR = FP/(FP + TN)
        TNR = TN/(TN + FP)
        FNR = FN/(FN + TN)
        print('accuracy : ',acc,'\n','TPR : ', TPR, '\n','FPR : ', FPR, '\n','TNR : ', TNR, '\n','FNR : ', FNR)

matrix(Ya2,Yp2)

################################################################################
#print('\n','\n','This is by using L1 regularisation : ','\n')

def costfuncL1(x,y,beta,lam):
        m = len(x[0,:])
        n = len(x[:,0])
        z = np.dot(x, beta)
        h = 1/(1 + np.exp(-z))
        h = np.transpose(h)
        l1 = np.sum.abs(beta[1:])
        return (1/n)*(-np.dot(np.log(h), y) - np.dot(np.log(1-h), (1-y))) + (lam/n)*l1

def grad_descL1(x,y,beta,alpha,k,tou,lam):
        m = len(x[0,:])-1 
        n = len(x[:,0])
        precost = 0
        for i in range(k+1):
            if(costfuncL1(x,y,beta,lam) - precost > tou):
                    z = np.dot(x, beta)
                    h = 1/(1 + np.exp(-z))
                    r = h - y
                    p = np.transpose(r)
                    precost = costfuncL1(x,y,beta,lam)
                    l1 = np.sum(np.abs(beta[1:])/beta[1:])
                    beta =  beta - (alpha/n)*np.transpose(np.dot(p,x)) + (lam/n)*l1
                    beta[0] = beta[0] - (lam/n)*l1
            else:
                    break
        J = costfuncL1(x,y,beta,lam)
        return beta, J
        
def log_regL1(X,y,k,alpha,lam):
        m = len(X[0,:])-1 
        n = len(X[:,0])
        beta = np.random.randint(0,9, (m+1,1))
        beta, J = grad_descL1(X,y,beta,alpha,k,tou,lam)
        return beta, J
        
        
###############################################################################

def costfuncL2(x,y,beta,lam):
        m = len(x[0,:])
        n = len(x[:,0])
        z = np.dot(x, beta)
        h = 1/(1 + np.exp(-z))
        h = np.transpose(h)
        l1 = np.dot(beta[1:],beta[1:])
        return (1/n)*(-np.dot(np.log(h), y) - np.dot(np.log(1-h), (1-y))) + (lam/n)*l1

def grad_descL2(x,y,beta,alpha,k,tou,lam):
        m = len(x[0,:])-1 
        n = len(x[:,0])
        precost = 0
        for i in range(k+1):
            if(costfuncL2(x,y,beta,lam) - precost > tou):
                    z = np.dot(x, beta)
                    h = 1/(1 + np.exp(-z))
                    r = h - y
                    p = np.transpose(r)
                    precost = costfuncL2(x,y,beta,lam)
                    l1 = np.sum(beta[1:])
                    beta =  beta - (alpha/n)*np.transpose(np.dot(p,x)) + (lam/n)*l1
                    beta[0] = beta[0] - (lam/n)*l1
            else:
                    break
        J = costfuncL2(x,y,beta,lam)
        return beta, J
        
def log_regL2(X,y,k,alpha,lam):
        m = len(X[0,:])-1 
        n = len(X[:,0])
        beta = np.random.randint(0,9, (m+1,1))
        beta, J = grad_descL2(X,y,beta,alpha,k,tou,lam)
        return beta, J
