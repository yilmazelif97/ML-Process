import sys

import color as color
import dp as dp
import pandas as pd
import scipy
import numpy
import matplotlib
import pandas
import sklearn

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

url =("https://raw.githubusercontent.com/meauxt/credit-card-default/master/credit_cards_dataset.csv")  #load data
# it is not necesseray
# name = ["X1", "X2","X3","X4","X5","X6","X7", "X8","X9","X10","X11","X12","X13", "X14","X15","X16","X17","X17","X19", "X20","X21","X22","X23","X24"] #Feature names
dataset = read_csv(url) #access data

#print(dataset.head(5))
print(dataset.shape) # number of inputs and features
print(dataset.size) #number of elements in dataset


#Preprocessing(removing unnecessary features deletion)
print(dataset.dtypes) #İf there were any object, I have to should delete. Because it could not apply on mathematical operations


X=dataset.iloc[:,:-1].values #Splitting dataset into independent Feature, use -1 for last row
#print(X)

Y=dataset.iloc[:,-1].values #Extracting dataset to get dependent feature, It is as target
#print(Y)


#siplittin data into training testing. For 100 rows; 80 rows training & 20 rows testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#print((y_train))


# Create Scatter

fig, ax = plt.subplots()
myscatterplot = ax.scatter(dataset["LIMIT_BAL"], dataset["BILL_AMT1"])
ax.set_xlabel("Limit")
ax.set_ylabel("Bill_ATM1")

plt.show()

#Compare ML algorithms -select bigger one for implement on dataset

"""LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train,y_train)
LR.predict(X_test)
round(LR.score(X_test,y_test), 4)""" #It could not work, dataset came big for this algorithm

"""svclassifier = SVC(kernel='poly', degree=4).fit(X_train, y_train)
svclassifier.predict(X_test)
res4= round(SVC.score(X_test, y_test),4)"""

SVM = SVC(decision_function_shape="ovo").fit(X_train, y_train)
SVM.predict(X_test)
res=round(SVM.score(X_test, y_test), 4)
print(res)
print("\n")


NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1).fit(X_train, y_train)
NN.predict(X_test)
res3=round(NN.score(X_test, y_test), 4)
print(res3)



#Training
trainData = SVC()
trainData.fit(X_train,y_train)

print(trainData.score(X_test,y_test))

y_predict = trainData.predict(X_test)
print(y_predict)


#Evaluation -making prediction
print(classification_report(y_test,y_predict))


dataset.describe().T.plot(kind='scatter', x='min', y='max')

