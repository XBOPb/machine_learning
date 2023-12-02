import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()

# Scale features down to (-1; 1) to make alghoritm faster
data = scale(digits.data)

# Labels
y = digits.target

# Amount of clusters(centroids)
k = len(np.unique(y))
# or k = 10 (10 numbers)

# Amount of features
samples, features = data.shape

def bench_k_means(estimator, name, data):
    # Fit data into classifier
    estimator.fit(data)
    # Score classifiers by calling this function
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))
    
clf = KMeans(n_clusters=k, init='random', n_init=10)
bench_k_means(clf, '1', data)
# 1               69688   0.678   0.716   0.696   0.569   0.693   0.147
