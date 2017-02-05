#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

print ("DecisionTreeClassifier")
best = 0.0
args = ()
for max_depth in range(5, 50, 5):
	for min_samples_split in range(1, 10):
		clf = DecisionTreeClassifier(max_depth = max_depth, min_samples_split=min_samples_split)
		clf.fit(features_train, labels_train)
		acc = clf.score(features_test, labels_test)
		if (acc > best):
			args = (max_depth, min_samples_split)
			best = acc

print "Best DecisionTreeClassifier accuracy:{} args:{} ", best, args

clf = DecisionTreeClassifier(max_depth = args[0], min_samples_split=args[1])
clf.fit(features_train, labels_train)
acc = clf.score(features_test, labels_test)

try:
    prettyPicture(clf, features_test, labels_test, "DT.png")
except NameError:
    pass

print ("AdaBoostClassifier")
best = 0.0
args = ()
for estimators in range(5, 50, 5):
	for weight in range(1, 10):
		clf = ensemble.AdaBoostClassifier(n_estimators = estimators, learning_rate=weight/10.0)
		clf.fit(features_train, labels_train)
		acc = clf.score(features_test, labels_test)
		if (acc > best):
			args = (estimators, weight)
			best = acc

print "Best Ada accuracy:{} args:{} ", best, args

clf = KNeighborsClassifier(n_neighbors = args[0], leaf_size = args[1], n_jobs = 8)
clf.fit(features_train, labels_train)
acc = clf.score(features_test, labels_test)

try:
    prettyPicture(clf, features_test, labels_test, "adaboost.png")
except NameError:
    pass
	
best = 0
print ("Random forrest")
for estimators in range(1, 20, 1):
	for min_sample in range(1, 10):
		clf = ensemble.RandomForestClassifier(n_estimators = estimators, min_samples_split = min_sample)
		clf.fit(features_train, labels_train)
		acc = clf.score(features_test, labels_test)
		if (acc > best):
			args = (estimators, min_sample)
			best = acc
print "Best RF accuracy:{} args:{} ", best, args

clf = KNeighborsClassifier(n_neighbors = args[0], leaf_size = args[1], n_jobs = 8)
clf.fit(features_train, labels_train)
acc = clf.score(features_test, labels_test)

try:
    prettyPicture(clf, features_test, labels_test, "RF.png")
except NameError:
    pass
	
best = 0
print ("sklearn.neighbors.KNeighborsClassifier")
for weights in ['uniform', 'distance']:
	for n_neighbors in range(1, 20, 1):
		for leaf_size in range(10, 40, 5):
			clf = KNeighborsClassifier(n_neighbors = n_neighbors, leaf_size = leaf_size, weights=weights, n_jobs = 8)
			clf.fit(features_train, labels_train)
			acc = clf.score(features_test, labels_test)
			if (acc > best):
				args = (n_neighbors, leaf_size, weights)
				best = acc
print "Best KNN accuracy:{} args:{} ", best, args
clf = KNeighborsClassifier(n_neighbors = args[0], leaf_size = args[1], weights=args[2], n_jobs = 8)
clf.fit(features_train, labels_train)
acc = clf.score(features_test, labels_test)

try:
    prettyPicture(clf, features_test, labels_test, "KNN.png")
except NameError:
    pass
