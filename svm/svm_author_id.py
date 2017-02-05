#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

#########################################################
### your code goes here ###
from sklearn import svm
import numpy as np
for C in [10000.]:    
    svc = svm.SVC(C=C, kernel='rbf')
    startTime = time()
    svc.fit(features_train, labels_train)
    print("Fitting took " + str(time() - startTime) + " seconds")

    startTime = time()
    accuracy = svc.score(features_test, labels_test)
    print("Accuracy: " + str(accuracy) + " took " + str(time() - startTime) + " seconds")
    
    pred = svc.predict(features_test)
    nonZero = np.count_nonzero(pred)
    print("Number of Chris (1) " + str(nonZero))
    print("Number of Sara (0) " + str(len(pred) - nonZero))
    
    for index in [10, 26, 50]:
        print("Index {} was label {}".format(index, pred[index]))

#########################################################
