import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv('student-mat.csv', sep=';')

# Take only necessary 6 columns. 
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

predict = 'G3'

x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

# Save the best model into file
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy > best:
        best = accuracy
        # Save model into file
        with open('studentmodel.pickle', 'wb') as f:
            pickle.dump(linear, f)

pickle_in = open('studentmodel.pickle', 'rb')
# Loading the model
linear = pickle.load(pickle_in)

predictions = linear.predict(x_test)
# Use model to show prediction
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# Show graph Grade/G1 correlation
p = 'G1'
style.use('ggplot')
pyplot.scatter(data[p], data['G3'])
pyplot.xlabel(p)
pyplot.ylabel('Final Grade')
pyplot.show()