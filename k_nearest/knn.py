import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv('car.data')
#   buying  maint door persons lug_boot safety  class
# 0  vhigh  vhigh    2       2    small    low  unacc
# 1  vhigh  vhigh    2       2    small    med  unacc
# 2  vhigh  vhigh    2       2    small   high  unacc
# 3  vhigh  vhigh    2       2      med    low  unacc
# 4  vhigh  vhigh    2       2      med    med  unacc

# Transform text fields into numbers
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
# [3 3 3 ... 1 1 1] vhigh=3, low = 1
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

predict = 'class'

# Put all features into one list
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Don't use too much neighbours
model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)

predicted = model.predict(x_test)
names = ['unacc', 'acc', 'good', 'vgood']

for x in range(len(predicted)):
    print('Predicted: ', names[predicted[x]], 'Data: ', x_test[x], 'Actual: ', names[y_test[x]])
# Predicted:  unacc Data:  (3, 2, 3, 1, 1, 2) Actual:  unacc
# Predicted:  good Data:  (3, 3, 3, 0, 1, 1) Actual:  good