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
t2 = time()
features_train, features_test, labels_train, labels_test = preprocess()
print(f"Your preprocessing time was: {round(time()-t2, 3)} s")

#tr -d '\015' <word_data.pkl >word2_data.plk

#########################################################
#import packages
import sklearn as sk
import numpy as np
#import time
from sklearn.svm import SVC

penalty = 10000.0 #C value for fit 4500 is really good

divide = 1 #reduce training data set by this denominator
features_train = features_train[:len(features_train)//divide]
labels_train = labels_train[:len(labels_train)//divide]

#Initialise functions
t0 = time()
#clf = SVC(C = penalty, kernel='linear') #default linear fit
clf = SVC(C = penalty, kernel='rbf') #rbf fit (slow)

#Create fit function
clf.fit(features_train, labels_train)
print(f"Your fitting time was: {round(time()-t0, 3)} s")

#Test against test data and time
t1 = time()
score = clf.score(features_test, labels_test)
print(f"Your testing time was: {round(time()-t1, 3)} s")
print(f"the score is: {score:.4f}")

a = clf.predict([features_test[10]])
b = clf.predict([features_test[26]])
c = clf.predict([features_test[50]])

def name(x):
    if x == 0:
        name = "Sara's"
    elif x == 1:
        name = "Chris'"
    else:
        name = "unkown's"
    return name

print(f"the tenth is predicted to be {name(a)} email.")
print(f"the twenty-sixth is predicted to be {name(b)} email.")
print(f"the fiftyeth is predicted to be {name(c)} email.")

x = sum(clf.predict(features_test)==1)
y = sum(clf.predict(features_test)==0)

print(f"There are {x} features in the 'Chris' class and {y} in the 'Sara' class.")
#########################################################


