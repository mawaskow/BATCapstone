#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:07:14 2021

@author: rollympoyi
"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Importing the templates.
from PIL import Image 

from skimage import data 
from skimage.feature import match_template
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mni', version=1)


# Creating independent and dependent variables.
X, y = mnist['data'], mnist['target']

# Splitting the data into training set and test set.
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

"""

"""

# Training a binary classifier.
y_train_5 = (y_train == 5) # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

"""
Building a  classifier that just classifies every single image in the “not-5” class for now will be changed by characteristic of trees later.
"""

from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
	def fit(self, X, y=None):
		pass
	def predict(self, X):
		return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

