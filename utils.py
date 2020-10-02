import cupy as cp 
import numpy as np
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors


class LocalOutlierFactor: 
    """
    Unsupervised Outlier Detection using Local Outlier Factor (LOF)

    Parameters
    ----------

    n_neighbors : int, default=20 
        Number of neighbors to use for `kneighbors` quesries. 

    Attributes
    ----------
    negative_outlier_factor_ : ndarray of shape (n_samples,)
        The opposite LOF of the training samples. The higher, the more normal.
        Inliers tend to have a LOF score close to 1
        (``negative_outlier_factor_`` close to -1), while outliers tend to have
        a larger LOF score.
        
        The local outlier factor (LOF) of a sample captures its
        supposed 'degree of abnormality'.
        It is the average of the ratio of the local reachability density of
        a sample and those of its k-nearest neighbors.

        Supposed to be identical with sklearn version 
    """ 

    def __init__(self, n_neighbors=20): 
        self.n_neighbors = n_neighbors 

    def fit(self, X): 
        X = cp.array(X)
        knn_cuml = cuNearestNeighbors() 
        knn_cuml.fit(X) 

        D_cuml, I_cuml = knn_cuml.kneighbors(X, self.n_neighbors)

        lrd = self._local_reachability_density(D_cuml, I_cuml) 
        X_lrd = lrd[I_cuml] / lrd[:,cp.newaxis] 

        self.negative_outlier_factor_ = -cp.asnumpy(cp.mean(X_lrd, axis=1))

    def _local_reachability_density(self, distance_X, neighbors_indices): 
        dist_k = distance_X[neighbors_indices, self.n_neighbors - 1] 
        reach_dist_array = np.maximum(distance_X, dist_k) 
        return 1. / (cp.mean(reach_dist_array, axis=1) + 1e-10) 
