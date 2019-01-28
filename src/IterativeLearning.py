# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:08:33 2018

@author: Sumit
"""


import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score


def GetBestModel(X,y,pen='l2',Cs = 10, cv=10, tol = 1e-2):
    L_Cv = LogisticRegressionCV(Cs = Cs, cv=cv, penalty=pen, solver = 'saga',
                                n_jobs = -1, tol = tol)
    L_Cv.fit(X, y)
    C_best = L_Cv.C_
    return C_best


def BalanceClasses(X,y):
    In = np.argwhere(y)
    In = In[:,0]
    X_new = X
    Reps = (len(y)-len(In))/len(In)
    Y_new = y
    for i in range(Reps):
        X_new = np.vstack((X_new,X[In]))
        Y_new = np.vstack((Y_new,np.ones((len(In),1))))

    return {'X':X_new, 'y':Y_new}


def ILB(X,y,C = 1.0,pen='l2',Iter = 10,
                      Thresh = 0.9):

    prev = y + 0.0
    y_orig = y + 0.0
    y_p = []

    NoPos = []
    for i in range(Iter):
        #creating training matrices at each iteration

        tmp = BalanceClasses(X + 0.0 ,y + 0.0)
        X_new, y_new = tmp['X'],tmp['y']


        LR = LogisticRegression(penalty = pen, solver = 'saga', C = C)
        LR.fit(X_new,y_new)

        y_pred = LR.predict_proba(X)
        y_pred = y_pred[:,1]

        y = y_orig + 0.0
        y[y_pred >=Thresh] = 1.0

        if sum(abs(y - prev)) == 0:
            break

        prev = y + 0.0
        NoPos += [sum(y)]

        y_p = np.reshape(LR.predict_proba(X)[:,1],(len(y),1))



    return {'LR':LR, 'y':y, 'y_p':y_p, 'NoPos':NoPos}


def CrossValScore(Mdl,X,y):
    c, r = y.shape
    y = y.reshape(c,)
    Prec = cross_val_score(Mdl, X, y, cv=10, scoring='precision', n_jobs = -1)
    Rec = cross_val_score(Mdl, X, y, cv=10, scoring='recall', n_jobs = -1)
    AUC = cross_val_score(Mdl, X, y, cv=10, scoring='roc_auc', n_jobs = -1)

    return {'Prec':Prec, 'Rec':Rec, 'AUC':AUC}



def ICCT(X,y, C = [1.0], Iter=10,rho=0.1,pen='l2', Thresh = 0.9, BC = True,Iter2=1):

    #X is a list of feature sets
    #y is the initial label in a numpy array
    #C is list of logistic regression penalties

    if len(C) < len(X):
        C = C*len(X)


    K = len(X)

    #balance feature classes

    X_new = []
    y_new = []

    y_t = []

    if BC:

        for X1 in X:
            tmp = BalanceClasses(X1 + 0.0 ,y + 0.0)
            X1_new, y1_new = tmp['X'],tmp['y']
            n = len(y1_new)
            n1 = len(y)
            m = n + 0.0
            X_new += [X1_new + 0.0 ]
            y_new += [np.reshape(y1_new,(n,1)) + 0.0]
            y_t += [y + 0.0]

    else:
        for X1 in X:
            n = len(y1_new)
            n1 = len(y)
            m = n + 0.0
            X_new += [X1 + 0.0]
            y_new += [np.reshape(y,(n,1)) + 0.0]
            y_t += [y + 0.0]



    y_orig = y_new[0] + 0.0

    y_p = []
    LR = []

    for i in range(Iter):

        LR = []

        y_r = 0.0

        for i in range(len(X_new)):
            X1_new = X_new[i]
            LR1 = LogisticRegression(penalty = pen, solver = 'saga', C = C[i])
            LR1.fit(X1_new,y_new[i])
            LR += [LR1]
            y_r += y_t[i] + 0.0


        y_p = []
        for cnt in range(Iter2):

            for i in range(len(X_new)):
                X1 = X[i]
                t = sum(y_t) - y[i] + 0.0
                a = (1.0/(rho))*(np.asscalar(LR[i].intercept_) + X1.dot(np.transpose(LR[i].coef_))) + (1.0/(K-1))*(t + 0.0)
                y_t[i] = np.maximum(np.zeros((n1,1)),np.minimum(np.ones((n1,1)),a)) + 0.0

                if cnt == Iter2-1:
                    y_t[i][y_t[i] >= Thresh] = 1.0
                    y_t[i][y_t[i]==1.0] = 1.0
                    y_t[i][y_t[i] < Thresh] = 0.0
                    y_p += [np.reshape(LR[i].predict_proba(X[i])[:,1],(n1,1))]


        X_new = []
        y_new = []

        if BC:

            for i in range(len(X)):
                X1 = X[i]
                tmp = BalanceClasses(X1 + 0.0 ,y_t[i] + 0.0)
                X1_new, y1_new = tmp['X'],tmp['y']
                X_new += [X1_new + 0.0 ]
                y_new += [y1_new + 0.0]

        else:
            for i in range(len(X)):
                X1 = X[i]
                X_new += [X1 + 0.0]
                y_new += [y_t[i] + 0.0]


    return {'LR':LR,'y_p':y_p,'y_t':y_t}
