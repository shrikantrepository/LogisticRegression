# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:26:02 2020

@author: Shrikant Agrawal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x= dataset.iloc[:,2:4].values     # or x= dataset.iloc[:,[2,3]] - it will show column labels
y = dataset.iloc[:,4].values

# Divide the dataset between training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0) #Random state =0 to ensure we will get the same output


# Feature scalling - their is a huge gap between salary column hence preprocessing is required
# Basically it transform existing values between 0 to 1 or 2 to -2 or anything

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# Create the model for Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)
                                     
# Predict the test set results
# Now model is ready we need to apply it on our test dataset and check it is corrrect
y_pred = classifier.predict(x_test)

""" Now in order to find out how model is performed?  Is it predicted correctly - we will use 
confusion matrix. It helps us to determine how many predictions are correct"""

# Making confusion Matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)        # y_test is real training data and y_pred is predicted o/p
cm

""" In output is has provided two dimentional array 
[63,5]
[7,25]
 
means 65+24 is the right prediction and 7+5=12 is the wrong prediction 
Our model is 88% performed good"""


""" Visualise our dataset using matplotlib - Dont bother about below code, it is readily
available on matplotlib or github"""


# Visualising the Training set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

""" Our output shows 12 green dots in red field which means these are the wrong predicted values and same numbers
reflects in our confusion matrix"