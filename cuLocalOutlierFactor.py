#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)

n_samples = 1000
n_features = 2 
random_state = 42

# make the test data
X, _ = make_blobs(n_samples=n_samples,
                    n_features=n_features,
                    centers=5,
                    random_state=random_state)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# LOF from cuml 
from utils import LocalOutlierFactor as cuLocalOutlierFactor
clf = cuLocalOutlierFactor(n_neighbors=20)
clf.fit(X)
cu_scores = clf.negative_outlier_factor_

# LOF from sklearn 
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit(X)
sk_scores = clf.negative_outlier_factor_

# plotting the result 
scores = {'cuml': cu_scores, 'sklearn': sk_scores}
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for score, ax in zip(scores, axes): 
    ax.scatter(X[:,0], X[:,1], c=scores[score])
    ax.set_title(score)
fig.savefig("cuml_sklearn_lof.pdf", bbox_inches='tight')
