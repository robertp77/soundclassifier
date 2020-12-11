# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:55:18 2020

@author: arobu
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import yfinance as yf
# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV 
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.metrics import plot_confusion_matrix, roc_curve, roc_auc_score

import time
start_time=time.time()

#Load the Data
#f = open('store.pckl', 'rb')
#f = open('store_100_JuanAvg.pckl', 'rb')
#f = open('store500.pckl', 'rb')
#f = open('store1000_20of40mfcc.pckl', 'rb')
#f = open('store_1000_20of100mfcc.pckl', 'rb')
#f = open('store_200_100_of_500_decibels.pckl', 'rb')
#f = open('store_raw.pckl', 'rb')
#f = open('store_200_20_of_100_decibels.pckl', 'rb')
f = open('store_100_50_of_100.pckl', 'rb') #<-- USE THIS FOR 92%

[X,y, key] = pickle.load(f)
f.close()

#Create a Pipeline
#Code adapted from DataCamp

# Setup the pipeline
pca=PCA()
steps = [('scaler', StandardScaler()),
         ('PCA',  pca),
         ('SVM', SVC(random_state=21, probability=True))]

pipeline = Pipeline(steps)
#'linear', 'rbf', 'poly', 'sigmoid'
# Specify the hyperparameter space
c_space = np.logspace(0, 2, 15) #From Datacamp Scikit Learn Hyperparameter Tuning with GridSearchCV
parameters = {'SVM__kernel':[ 'rbf'],
              'SVM__C':c_space,
              #'SVM__gamma': ['scale', 'auto'],
              #'SVM__gamma': np.logspace(-3, -1, 10),
              'SVM__gamma': [0.021544],
              #'SVM__decision_function_shape': ['ovo', 'ovr'],
              #'PCA__n_components':np.linspace(0.7, 0.99,10)}
              'PCA__n_components':[0.99]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=21, stratify=y)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, parameters, cv=10, n_jobs=-1)
#cv = RandomizedSearchCV(pipeline, parameters, cv=5, n_jobs=20, random_state=20)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))
#print("Tuned Model Score: {}".format(cv.best_scores_))

print("--- %s seconds ---" % (time.time() - start_time))

#Save the Classifier
f = open('Classifier_SVM_100_50_of_100_final.pckl', 'wb')
pickle.dump(cv, f)
f.close() 


#Confusion matrix

title="Normalized confusion matrix \n SVM Classifier with RBF Kernel"
normalize="true"
disp = plot_confusion_matrix(cv, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
disp.ax_.set_title(title)
plt.show()


## ROC Curve
def plot_roc_curve(fpr, tpr, label=None): 
    plt.plot(fpr, tpr, c='red', linewidth=2, label=label) 
    plt.plot([0, 1], [0, 1], 'k--') 
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis('square')
    plt.title('Receiver Operating Characteristic Curve \n SVM Classifier with RBF Kernel')



y_score = cv.predict_proba(X_test)
#y_score = cross_val_predict(knn, X_train,y_train,cv=10, method='predict_proba')
fpr, tpr, thresholds = roc_curve(y_test, y_score[:,1], pos_label=1)

#Compute the AUC
auc = roc_auc_score(y_test, y_score[:,:], multi_class='ovr')
print(auc)

#Plot the ROC Curve
plot_roc_curve(fpr, tpr)
plt.show()