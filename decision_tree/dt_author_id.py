#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print "Using 1/10 of the data."

split = 1

features_train = features_train[:(int)(len(features_train) * split)] 
labels_train = labels_train[:int(len(labels_train) * split)] 

print features_train.shape

#########################################################
### your code goes here ###
for max_depth in range(4,10):
	break
	clf = tree.DecisionTreeClassifier(max_depth=max_depth)
	clf.fit(features_train, labels_train)

	acc = clf.score(features_test, labels_test)
	print str(max_depth) + " : " + str(acc)

#########################################################
### Min sample split
### your code goes here ###
for min_split in range(20,60,20):
	clf = tree.DecisionTreeClassifier(min_samples_split=min_split)
	clf.fit(features_train, labels_train)

	acc = clf.score(features_test, labels_test)
	print str(min_split) + " : " + str(acc)
