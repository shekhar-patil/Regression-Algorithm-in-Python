
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import cross_validation,datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("crime.csv")

X = df[['Year']]
y = df[['Cruelty by Husband or his Relatives']]




X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)

regressor = LinearRegression()

regressor.fit(X_train,y_train)

print "Your Test Set is:- \n",X_test
#X_tst = np.array([2015, 0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0])
accuracy = regressor.score(X_test,y_test)
print "\nYour Prediction has the accuracy of-",accuracy*100,"%"

X_test1 = [[2015]]

y_prediction = regressor.predict(X_test)
print "\nPredicted Total Crime for above given years is:-\n",y_prediction

y_prediction = regressor.predict(X_test1)
print "\nPredicted Total Crime for above given years is:-\n",y_prediction
