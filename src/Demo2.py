import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score
import IterativeLearning as SF
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import scipy.io as scio


def TrainingCrossVal(X_normed,Y,Pen = 'l2'):
    #Balancing classes for better classification performance
    tmp = SF.BalanceClasses(X_normed + 0.0 ,Y + 0.0)
    X_train, y_train = tmp['X'],tmp['y']

    print 'Crossvalidation'
    Cs = np.logspace(-4,4,10)
    Cs = Cs.tolist()
    C_best = SF.GetBestModel(X_train,y_train,pen=Pen,Cs = Cs)
    C_best = C_best[0]
    print C_best
    return(C_best)


def MaskY(Y_act,I,fracCensor = 0.5):
    Y_new = Y_act + 0.0
    Y_new[Y_true == I] = 1.0
    Y_new[Y_true != I] = 0.0
    Y_new2 = Y_new + 0.0

    In1 = np.where(Y_new == 1.0)[0]
    In2 = np.random.choice(In1,size = (int(.5*len(In1)),1), replace = False)

    Y_new2[In2] = 0.0

    return([Y_new,Y_new2])




tmp = scio.loadmat('./Data/Caltech101-7.mat')
X1 = tmp['X'][0][0]
X2 = tmp['X'][0][1]
X3 = tmp['X'][0][2]

Y_true = tmp['Y']


Classes = np.ndarray.tolist(np.unique(tmp['Y']))
numClasses = len(Classes)

Mod1_acc = []
Mod2_acc = []
Mod3_acc = []

ILB_acc = []

IC_l_acc = []

IC_h_acc = []

for i in Classes:

    tmp = MaskY(Y_true,i)
    Yi_t = tmp[0] + 0.0
    Yi = tmp[1] + 0.0

    C1 = TrainingCrossVal(X1,Yi)
    C2 = TrainingCrossVal(X2,Yi)
    C3 = TrainingCrossVal(X3,Yi)

    C = [C1,C2,C3]

    #training non-iterative models
    tmp1 = SF.ILB(X1,Yi,Iter = 1, C = C1)
    tmp2 = SF.ILB(X2,Yi,Iter = 1, C = C2)
    tmp3 = SF.ILB(X3,Yi,Iter = 1, C = C3)


    Mod1_acc += [accuracy_score(Yi_t,tmp1['y'])]
    Mod2_acc += [accuracy_score(Yi_t,tmp2['y'])]
    Mod3_acc += [accuracy_score(Yi_t,tmp3['y'])]

    #training iterative models without co-training
    tmp1 = SF.ILB(X1,Yi,Iter = 10, C = C1)
    tmp2 = SF.ILB(X2,Yi,Iter = 10, C = C2)
    tmp3 = SF.ILB(X3,Yi,Iter = 10, C = C3)

    Y_temp = ((tmp1['y'] + tmp2['y'] + tmp3['y'])/3.0 > 0.5) + 0.0

    ILB_acc += [accuracy_score(Yi_t,Y_temp)]

    X_r = [X1,X2,X3]

    #training iterative models with co-training
    d1 = SF.ICCT(X_r,Yi,C=C, rho=1.0)
    d2 = SF.ICCT(X_r,Yi,C=C, rho=10.0)

    Yt1 = ((d1['y_p'][0]+d1['y_p'][1]+d1['y_p'][2])/3.0 > .5) + 0.0
    Yt2 = ((d2['y_p'][0]+d2['y_p'][1]+d2['y_p'][2])/3.0 > .5) + 0.0

    IC_l_acc += [accuracy_score(Yi_t,Yt1)]
    IC_h_acc += [accuracy_score(Yi_t,Yt2)]

    #print(IC_l_acc[i])
    #print(IC_h_acc[i])


d = {'Labels':Classes, 'Mod1': Mod1_acc, 'Mod2': Mod2_acc, 'Mod3': Mod3_acc, 'ILB': ILB_acc, 'IC_l': IC_l_acc, 'IC_h': IC_h_acc}

df = pd.DataFrame(d)

df.to_csv('./Results/Demo2_results.csv')
