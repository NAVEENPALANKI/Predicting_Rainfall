import os

# Scientific Python Libraries
import pandas as pd
import numpy as np
import math

# Essential libraries for visualization
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import missingno as mn
from matplotlib.pyplot import show

import warnings
warnings.filterwarnings("ignore")

# Essential for splitting into train and test data
from sklearn.model_selection import train_test_split   

#Importing the models for classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
# !pip install xgboost
import xgboost as xgb

# Essential libraries for evaluating performance metrics
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

data=pd.read_csv('weather.csv')      # read dataset
data.head(5)

data.shape

data.isnull().sum()

data.dtypes

data.describe()

# separating numerical and categorical features

numerical_feature = [f for f in data.columns if data[f].dtypes != 'object']
categorical_feature = [f for f in data.columns if f not in numerical_feature]
print("Numerical Features Count {}".format(len(numerical_feature)))
print(numerical_feature)
print("Categorical feature Count {}".format(len(categorical_feature)))
print(categorical_feature)


# heatmap for corrleation

plt.figure(figsize = (20,20))
ax= sns.heatmap(data.corr(), annot = True, cmap="RdYlGn",linewidth =2,  square= True)


for feature in numerical_feature:
    df=data.copy()
    plt.figure(figsize=(10,10))
    sns.distplot(data[feature])
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()



