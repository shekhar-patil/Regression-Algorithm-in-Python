
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import cross_validation,datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("crime.csv")
X = df.iloc[:,:-1].values
y = df.iloc[:,7].values
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)

regressor = LinearRegression().0

regressor.fit(X_train,y_train)

print "Your Test Set is:- \n",X_test
#X_tst = np.array([2015, 0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0])
accuracy = regressor.score(X_test,y_test)
print "\nYour Prediction has the accuracy of-",accuracy*100,"%"


y_prediction = regressor.predict(X_test)
print "\nPredicted Total Crime for above given years is:-\n",y_prediction
