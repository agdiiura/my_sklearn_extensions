"""
================================
hierarchical_cluster_selector.py

A module for feature selection based on hierarchical clustering
"""
from collections import defaultdict

import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin


class HierarchicalClusterSelector(SelectorMixin, BaseEstimator):
    """
    The HierarchicalClusterSelector class implements a feature selection
    algorithm based on hierarchical clustering to identify and retain a
    reduced set of non-collinear features.
    The algorithm begins by computing the Spearman rank correlation matrix
    of the input data.
    This matrix is then transformed into a distance matrix, which is used as
    input for hierarchical clustering.
    The clustering is performed using either Ward's linkage criterion or
    single linkage criterion, depending on the user's choice.

    The hierarchical clustering results in a dendrogram, and a specified
    number of clusters (k) are formed by cutting the dendrogram.
    The algorithm selects one feature from each cluster,
    and these selected features constitute the final set of
    non-collinear features.

    The algorithm provides a flexible and interpretable approach
    for feature selection, leveraging the insights from hierarchical clustering
    to retain a diverse set of features while minimizing redundancy.
    The user can control the number of clusters and the linkage criterion,
    offering adaptability to different data characteristics and
    analysis requirements.
    """

    def __init__(self, k=2, criterion='ward', random_state=None):
        """
        Initialize the class

        Parameters
        ----------
        k : int, default=2
            The number of clusters to form. It determines the number of
            selected features.
        criterion : {'ward', 'single'}, default='ward'
            The linkage criterion to use for hierarchical clustering.
            'ward' uses Ward's linkage, and 'single' uses single linkage.
        random_state : int or None, default=None
            Seed for random number generation to ensure reproducibility.
        """

        self.k = k
        self.criterion = criterion

        if self.criterion == 'ward':
            self._f_clustering = hierarchy.ward
        elif self.criterion == 'single':
            self._f_clustering = hierarchy.single
        else:
            raise Exception()

        self.random_state = random_state
        self._rng = np.random.default_rng(self.random_state)

    def fit(self, X, y=None):
        """Learn empirical clustering from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data from which to compute variances, where `n_samples` is
            the number of samples and `n_features` is the number
            of features.
        y : any, default=None
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        X = self._validate_data(X)

        corr = spearmanr(X).correlation
        # Ensure the correlation matrix is symmetric
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1)

        # We convert the correlation matrix to a distance matrix
        # before performing hierarchical clustering.
        distance_matrix = 1. - np.abs(corr)

        dist_linkage = self._f_clustering(squareform(distance_matrix))

        cluster_ids = \
            hierarchy.fcluster(dist_linkage, self.k, criterion='maxclust')
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)

        self.selected_features_ = [
            self._rng.choice(v) for v in cluster_id_to_feature_ids.values()
        ]

        return self

    def _get_support_mask(self):

        return np.array([
            x in self.selected_features_ for x in range(self.n_features_in_)
        ])
