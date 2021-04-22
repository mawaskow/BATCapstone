#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:10:43 2021

@author: rollympoyi
"""
## The variables are the one used in categorization program, need to add them here once finished
# Creating some predictions.
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

"""
Testing will be done first on categorization program variables 
"""

# Constructing the error matrix.
from sklearn.metrics import confusion_matrix
confusion_matrix(y_tree_5, y_train_pred)



# Precision: precision = (TP) / (TP+FP)
# Recall: recall = (TP) / (TP+FN)

# Finding precision and recall
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)


