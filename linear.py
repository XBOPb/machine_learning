import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv('student-mat.csv', sep=';')

# Take only necessary 6 columns. 
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
#   G1  G2  G3  studytime  failures  absences
# 0   5   6   6          2         0         6
# 1   5   5   6          2         0         4
# 2   7   8  10          2         3        10
# 3  15  14  15          3         0         2
# 4   6  10  10          2         0         4

# Label is the data we want to predict
predict = 'G3'

# Data with no predicted column, x_train, x_test
x = np.array(data.drop([predict], axis=1))

# Column with values to predict, y_train, y_test
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# y = mx + b
linear = linear_model.LinearRegression()

# fit data into y = mx + b. 5D space since there are 5 columns with data.
linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)
# 0.8600075675355316 - 86% prediction accuracy

print('Coefficient: ', linear.coef_)    # the 'm' coefficient in y = mx + b. Bigger coefficient = bigger weight of attribute.
# [ 0.17862391  0.96858463 -0.232331   -0.25380448  0.03897997] - 5 valiues for 5D space

predictions = linear.predict(x_test)
# Use model to show prediction
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
# 10.721597905204973 [11 11  2  0  2] 10 -> predicted: 10.72, actual: 10
# 14.09247565025509 [13 14  1  0  0] 14 -> predicted: 14.09, actual: 14