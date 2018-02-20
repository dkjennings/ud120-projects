#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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


#tr -d '\015' <word_data.pkl >word2_data.plk

#########################################################
import sklearn as sk
import numpy as np
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
#clf_pf = GaussianNB()
#clf_pf.partial_fit(features_train, labels_train, np.unique(labels_train))
#testlabels = clf.predict(features_test)
score = clf.score(features_test, labels_test)
print(f"the score is: {score:.4f}")
#########################################################


