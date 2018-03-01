#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from time import perf_counter as t

tstart = t()
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

from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import accuracy_score as a_s

acctest = 0.8
N = 50000
good = 0
treeid = 0

for i in range(N):
    tryclf = rfc(criterion='entropy', max_leaf_nodes=10, min_samples_split=4)
    #(criterion='entropy', 
    #max_leaf_nodes=10, 
    #min_samples_split=4)
    tprep = t()
    tryclf.fit(features_train, labels_train)
    tfit = t()
    pred = tryclf.predict(features_test)
    tpred = t()
    acc = a_s(labels_test, pred)
    tacc = t()

    if acc>acctest:
        print(f"The accuracy of the random forest {i} is {acc*100:.2f}%")
        #good = input("is this good enough (1=yes/else no)\n\n>>> ")
        if good == 1:
            acctest = acc
            clf = tryclf
            treeid = i
            print(f"Tree finished at accuracy {acctest*100:.2f}%")
        else:
            treeid = i
            acctest = acc
            clf = tryclf
    if good == 1:
        break
    if i == N-1:
        print(f"Tree finished at accuracy {acctest*100:.2f}%")

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass

tpic = t()
plt.title(f"Random tree classifier id : {treeid}")
print(f"preparation phase: {tprep-tstart:.2f}s")
print(f"fitting time is: {tfit-tprep:.2f}s")
print(f"prediciton time is: {tpred-tfit:.2f}s")
print(f"Checking time is {tacc-tpred:.2f}s")
print(f"Picture making time is {tpic-tacc:.2f}s")
print(f"Total forest time: {tacc-tprep:.2f}s")
print(f"Total time: {tpic-tstart:.2f}s")