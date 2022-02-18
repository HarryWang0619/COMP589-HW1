import sklearn.model_selection
import scipy
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from operator import itemgetter
from collections import Counter

#Load the Iris data file using python csv module

knn_file = open('iris.csv')
csvreader = csv.reader(knn_file)

knnd = []
for row in csvreader:
        knnd.append(row)

knndata = []
for row in knnd:
        c = []
        c.append(float(row[0]))
        c.append(float(row[1]))
        c.append(float(row[2]))
        c.append(float(row[3]))
        c.append(row[4])
        knndata.append(c)

# print(knndata)

# Implementing Helper functions

# Normalize module
def mini(col):
    min = col[0]
    for val in col:
        if val < min:
            min = val
    return min

def maxi(col):
    max = col[0]
    for val in col:
        if val > max:
            max = val
    return max

def normalizationall(col, max, min):
    newarr = []
    for val in col:
        newarr.append((val-min)/(max-min)) 
    return newarr

def normalization(col):
    min = mini(col)
    max = maxi(col)
    newarr = []
    for val in col:
        newarr.append((val-min)/(max-min)) 
    return newarr, min, max

def vote(arr):
    return max(set(arr), key=arr.count)

# Split the Training and Testing Data

def split(dat, ranumber):
    traknn, tesknn = sklearn.model_selection.train_test_split(dat, train_size=0.8, test_size=0.2, random_state=ranumber, shuffle=True)
    return traknn, tesknn

# trainknn, testknn = split(knndata, 589)

# Euclidean distance

def edistance(a, b):
    a = np.array(a)
    b = np.array(b)
    s = np.linalg.norm(a - b)
    return s

# print(edistance([1,1,1,4],[5,5,5,2]))

# KNN Helpers

# def seperate_d_c(data):
#     dat = []
#     cat = []
#     all = []
#     for row in data:
#         da = []
#         da.append(float(row[0]))
#         da.append(float(row[1]))
#         da.append(float(row[2]))
#         da.append(float(row[3]))
#         dat.append(da)
#         al = da.copy()
#         al.append(row[4])
#         all.append(al)
#         cat.append(row[4])
        
#     return dat, cat, all

# trainknndata, trainknncat, ktr = seperate_d_c(trainknn)
# testknndata, testknncat, kte = seperate_d_c(testknn)

def transpose(dat):
    a = []
    a.append([row[0] for row in dat])
    a.append([row[1] for row in dat])
    a.append([row[2] for row in dat])
    a.append([row[3] for row in dat])
    if len(dat[0]) > 4:
        a.append([row[4] for row in dat])
    return a

def normaltab(traindat, testdat):
    trainnom = []
    testnom = []
    i = 0
    for col in traindat:
        trarr = []
        tearr = []
        if i < 4:
            trarr, trmin, trmax = normalization(col)
            tearr = normalizationall(testdat[i], trmax, trmin)
            trainnom.append(trarr)
            testnom.append(tearr)
            i+=1
    if len(traindat) == 5:
        trainnom.append(traindat[4])
        testnom.append(testdat[4])
    return trainnom, testnom
    
def transback(dat):
    ret = []
    i = 0
    while i < len(dat[0]):
        row = []
        for col in dat:
            row.append(col[i])
        ret.append(row)
        i+=1
    return ret

def distarray(normpt, normeddat):
    pt1 = normpt[:-1]
    cat1 = normpt[-1]
    disarray = []
    for ins in normeddat:
        pt2 = ins[:-1]
        cat2 = ins[-1]
        dis = edistance(pt1, pt2)
        disarray.append([dis,cat2])
    return sorted(disarray, key=itemgetter(0))

def normflow(train, test):
    ttrainknn = transpose(train)
    ttestknn = transpose(test)
    normttrain, normttest = normaltab(ttrainknn,ttestknn)
    nrmtr, nrmte = transback(normttrain), transback(normttest)
    return nrmtr, nrmte

# we use normtr, normte.  stands for normal train & normal test.

#KNN

def knn(k, traindat, testdat):
    predict = []
    correct = [col[-1] for col in testdat]
    for datpt in testdat:
        distlist = distarray(datpt, traindat)
        catlist = [col[1] for col in distlist[:k]]
        predict.append(vote(catlist))

    return predict, correct

def knntrains(k, rand, dat):
    trainknn, testknn = split(dat, rand)
    normedtrain, normedtest =normflow(trainknn, testknn)
    predict, correct = knn(k, normedtrain, normedtest)
    return predict, correct

def knntraintrain(k, rand, dat):
    trainknn, testknn = split(dat, rand)
    normedtrain, normedtest =normflow(trainknn, testknn)
    predict, correct = knn(k, normedtrain, normedtrain)
    return predict, correct

def accuracy(pred, corr):
    i = 0
    blist = []
    while i < len(pred):
        blist.append(pred[i]==corr[i])
        i+=1
    return (Counter(blist)[True])/len(blist)

def kaccuracytest(k, r, data):
    p, c = knntrains(k, r, data)
    acc = accuracy(p, c)
    return acc

def kaccuracytrain(k, r, data):
    p, c = knntraintrain(k, r, data)
    acc = accuracy(p, c)
    return acc

# print(kaccuracytest(19, 589, knndata))
# print(kaccuracytrain(19, 589, knndata))

# The Statistical Process for the kNN
def statdatatest(data):
    k = 1
    result_list = []
    while k <= 51:
        random = 11589
        alist = []
        while random < 11689:
            alist.append(kaccuracytest(k, random, data))
            random += 5
        result_list.append(alist)
        k+=2
    
    return np.array(result_list)

def statdatatrain(data):
    k = 1
    result_list = []
    while k <= 51:
        random = 11589
        alist = []
        while random < 11689:
            alist.append(kaccuracytrain(k, random, data))
            random += 5
        result_list.append(alist)
        k+=2
    
    return np.array(result_list)

# narray = statdatatest(knndata)
# print(narray.std(axis=1))

k = np.arange(1,52,2)
narraytrain = statdatatrain(knndata)
narraytest = statdatatest(knndata)
acctrain = narraytrain.mean(axis=1)
# print(acctrain)
acctest = narraytest.mean(axis=1)
stdtrain = narraytrain.std(axis=1)
stdtest = narraytest.std(axis=1)
# print(stdtrain)

# Q1.1
plt.scatter(k, acctrain)
  
plt.errorbar(k, acctrain, yerr=stdtrain, fmt="-o")
plt.title("KNN using Training Data")
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.show()

# Q1.2
plt.scatter(k, acctest)
  
plt.errorbar(k, acctest, yerr=stdtest, fmt="-o")
plt.title("KNN using Testing Data")
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.show()