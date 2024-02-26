import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("Data/student_mat_2173a47420.csv", sep=";")

#print(data.head())
#print(data.describe())
#paid_converted = data.paid.map(dict(yes=1, no=0)) #did not work because of impossible to use changeable attribute in data[attribute selection]
#print(paid_converted)
#adding column with data.insert() did not work because 'method' object is not subscriptable
data_cat = data.select_dtypes(np.object_) # all non numerical
data_num = data.select_dtypes(np.number) # all numerical
focus_non_numerical_attr = "paid", "schoolsup", "famsup", "activities"
data_filtered = data_cat.filter(focus_non_numerical_attr)
#print(data_filtered) #works now only "paid", "schoolsup", "famsup", "activities" are kept, lets turn them into numerical values Yes=1, No =0
#data_paid = data_filtered.paid.map(dict(yes=1,no=0))
#print(tuple(data_paid))
#data_paid = data_paid.shape(395,6)
#print(data_paid)
#data_paid_F = tuple(data_paid)
data = data[["G1","G2","G3","studytime","failures", "age"]]
#data = data.add(data_paid_F)

predict = "G3"  #label = what you are trying to get

x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train,y_train)

acc = linear.score(x_test,y_test)
print(acc)

print("Co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range (len(predictions)):
                print(predictions[x], x_test[x], y_test[x])
