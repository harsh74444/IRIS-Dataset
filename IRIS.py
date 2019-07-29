# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:48:01 2019

@author: HarshGB
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math as math
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn import metrics

iris = pd.read_csv("D:\Random\iris\iris.csv")
#print(iris)
#print('\n')
iris = pd.DataFrame(iris)
'''
print(iris)
print('\n')
print(type(iris))
'''

'''

                        Data Visualization


# Scatter Plot of Species vs Sepal Width

plt.scatter(iris['species'], iris['sepal.width'])
plt.xlabel("Species")
plt.ylabel("Sepal Width")
plt.title("Species vs Sepal Width")
plt.legend(["Sepal Width"], loc = 'upper right', edgecolor = 'red')
plt.grid(axis = 'y')

# Scatter Plot of Species vs Sepal Length

plt.scatter(iris['species'], iris['sepal.length'], color = 'magenta')
plt.xlabel("Species", )
plt.ylabel("Sepal Length")
plt.title("Species vs Sepal Length")
plt.legend(["Sepal Length"], loc = 'best', edgecolor = 'red')
plt.grid(axis = 'y', color = 'black')

# Scatter Plot of Species vs Petal Width

plt.scatter(iris['species'], iris['petal.width'])
plt.xlabel("Species")
plt.ylabel("Petal Width")
plt.title("Species vs Petal Width")
plt.legend(["Petal Width"], loc = 'best', edgecolor = 'darkblue')
plt.grid(axis = 'y')

# Scatter Plot of Species vs Petal Length
plt.scatter(iris['species'], iris['petal.length'], color = 'teal',
            edgecolor = 'black')
plt.xlabel("Species", )
plt.ylabel("Petal Length")
plt.title("Species vs Petal Length")
plt.legend(["Petal Length"], loc = 'best', edgecolor = 'mediumseagreen')
plt.grid(axis = 'y', color = 'black')


# https://www.rapidtables.com/web/color/RGB_Color.html#color-table
[ 
## RGP COLOR CODE WEBSITE FOR EASY CONVERSION
]

# Shuffling our data set to have better visualization with reduced biases

iris = iris.sample(frac = 1).reset_index(drop= True)
print(iris)
print('\n')
# Ploting Density curve of the shuffled dataset
#plt.hist(iris['sepal.width'], color = 'lime', edgecolor = 'black')
sns.distplot(iris['sepal.width'], hist = True, color = 'red', bins = 25)

# Pairplotting using seaborn library
sns.pairplot(iris, hue = 'species', size = 2.5)

'''

'''

            TRAINING & TESTING MODEL USING RANDOM FOREST

'''
X = iris[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

clf = rfc(n_estimators = 100)
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)
print(y_hat)
print('\n')
accuracy = metrics.accuracy_score(y_test, y_hat)
print(accuracy)
print('\n')
matrix = metrics.confusion_matrix(y_test, y_hat)
print(matrix)
print('\n')
metric = metrics.classification_report(y_test, y_hat)
print(metric)
print('\n')
table = pd.crosstab(y_test, y_hat, colnames = ['Predicted'], rownames= ['Actual'],
                margins = True)
print(table)