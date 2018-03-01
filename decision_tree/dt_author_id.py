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


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# a = len(features_train[0])
# print(a)
# b = len(features_test)
# c = len(labels_train)
# d = len(labels_test)

# print(f"a:{a}+b:{b}={a+b}")
# print(f"a:{a}+c:{c}={a+c}")
# print(f"c:{c}+d:{d}={c+d}")
# print(f"a+b+c+d={a+b+c+d}")
# print(f".9*(a+b+c+d)={.9*(a+b+c+d)}")


#########################################################
### your code goes here ###

from sklearn import tree
from sklearn.metrics import accuracy_score as a_s

clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = a_s(labels_test, pred)

print(f"The accuracy of this classifier is {acc*100:.2f}%.")

#########################################################


