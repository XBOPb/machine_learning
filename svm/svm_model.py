import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

x = cancer.data
# [[1.799e+01 1.038e+01 1.228e+02 ... 2.654e-01 4.601e-01 1.189e-01]]
y = cancer.target
# [0 0 0 1 0]; 0 = malignant, 1 = benign

# Split the data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
classes = ['malignant', 'benign']

# Support vector classification. Parameters are optional. Kernel is a multiplying function. C is a soft margin. C=0 for hard margin
clf = svm.SVC(kernel='linear', degree=3, C=2)
# Compare with k nearest alghoritm for the test purposes
# clf = KNeighborsClassifier(n_neighbours=9)

clf.fit(x_train, y_train)

y_prediction = clf.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_prediction)
# 0.9649122807017544