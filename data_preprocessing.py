# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 16:20:58 2018

@author: JWalmsley
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Print full array
np.set_printoptions(threshold=np.inf)

# Importing the dataset
dataset = pd.read_csv('gender_submission.csv')
testset = pd.read_csv('test.csv')
trainset = pd.read_csv('train.csv')
combine = [trainset, testset]

print(trainset.columns.values)
print(testset.columns.values)

# preview the data
trainset.head()

trainset.tail()

trainset.info()
print('_'*40)
testset.info()



