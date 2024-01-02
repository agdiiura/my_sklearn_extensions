"""
==============================
correlated_feature_selector.py

A module for feature selection based on hierarchical clustering
"""
from collections import defaultdict

import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin


def compute_correlation(X, kind):
    """
    Compute the correlation matrix of a 2D array X.

    Parameters
    ----------
    X : Input data with shape (n_samples, n_features).
    kind : Type of correlation coefficient to compute.
         - 'pearson': Pearson correlation coefficient.
         - 'spearman': Spearman rank-order correlation coefficient.

    Returns
    -------
    corr : Correlation matrix of shape (n_features, n_features).

    Raises
    ------
    ValueError: If the specified 'kind' is not allowed.
        Allowed values: 'pearson', 'spearman'.
    """

    if kind == 'pearson':
        corr = np.corrcoef(X.T)
    elif kind == 'spearman':
        corr = spearmanr(X).correlation
    else:
        raise ValueError('kind not allowed. Allowed values: `pearson`, `spearman`')
    # Ensure the correlation matrix is symmetric
    corr = 0.5 * (corr + corr.T)
    # Set diagonal elements to 1
    np.fill_diagonal(corr, 1)
    return corr


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

    def __init__(self, k=2, criterion='ward', correlation='pearson', random_state=None):
        """
        Initialize the class

        Parameters
        ----------
        k : int, default=2
            The number of clusters to form. It determines the number of
            selected features.
        criterion : {'ward', 'single'}, default='ward'
            The linkage criterion to use for hierarchical clustering.
            'ward' uses Ward's linkage, 'single' uses single linkage
            and 'complete' uses complete linkage.
        correlation : correlation , 'pearson' or 'spearman'
        random_state : int or None, default=None
            Seed for random number generation to ensure reproducibility.
        """

        self.k = k
        self.criterion = criterion

        if self.criterion == 'ward':
            self._f_clustering = hierarchy.ward
        elif self.criterion == 'single':
            self._f_clustering = hierarchy.single
        elif self.criterion == 'complete':
            self._f_clustering = hierarchy.complete
        else:
            raise Exception()

        self.correlation = correlation

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

        corr = compute_correlation(X, self.correlation)

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


class CorrelationBasedSelector(SelectorMixin, BaseEstimator):
    """
    A feature selection algorithm based on empirical clustering
    using a genetic algorithm with correlation-based fitness.

    Notes
    -----
    - The genetic algorithm is used to evolve binary vectors indicating the
        selected features.
    - Fitness is determined by the correlation-based objective function.

    The overall goal of this genetic algorithm is to explore and evolve
    binary vectors representing feature selection solutions.
    The fitness function, based on correlation, guides the evolution
    towards solutions that capture relevant information in the data.
    The algorithm uses a combination of selection, crossover, and mutation
    to iteratively improve the solutions over generations.

    1. Initialization:

        The algorithm starts by generating an initial population of
        binary vectors, where each element represents the
        inclusion or exclusion of a feature.
        This population is represented by the populations array.

    2. Fitness Evaluation:

        The fitness of each solution in the population is evaluated using
        the objective function, which computes a correlation-based
        fitness score.
        The lower the score, the better the solution.
        Solutions with fewer than `k_min` selected features are
        assigned a fitness of infinity (np.inf) to penalize
        insufficient feature selection.

    3. Selection:

        The top `n_parents` solutions with the lowest fitness scores are
        selected as parents for the crossover operation.
        The probability of selecting each parent is proportional to
        their exponential fitness.

    4. Crossover:

        The crossover function performs crossover between pairs of parents
        to create a new set of offspring (children). Crossover involves
        randomly selecting elements from the parents,
        creating a child solution. Additionally, a mutation may occur
        with a probability determined by the mutation rate.

    5. Mutation:

        Mutation is introduced to add variability to the population.
        The mutation rate starts high and decreases over generations.
        Mutation involves flipping the value of random elements in the
        binary vectors.

    6. Survivor Selection:

        The new population consists of the best solution from the previous
         generation (best), the offspring (children), and randomly generated
         solutions. The selection is based on the fitness scores, and
         the process continues for the specified number of generations.

    7. Convergence Handling:

        If there is no improvement in the best fitness score for a
        certain number of consecutive generations
        (controlled by `max_patience`), the mutation rate is set
        to its maximum value to increase exploration.

    8. Termination:

        The algorithm terminates after a predefined number of generations
        (`n_generations`). The final selected features are determined
        by the best solution found throughout the evolution.


    See Also
    --------
    - `compute_correlation`: Function used to compute the correlation matrix.

    Example
    -------
    ```python
    selector = CorrelationBasedSelector(
        correlation='pearson', n_generations=50
    )
    selector.fit(X_train)
    selected_features = selector.transform(X_train)
    ```
    """

    def __init__(self,
                 correlation='pearson',
                 k_min=5,
                 n_populations=25,
                 n_generations=100,
                 n_parents=8,
                 n_children=15,
                 min_mutation_rate=0.01,
                 max_mutation_rate=0.25,
                 max_patience=5, random_state=None):
        """
        Initialize the class

        Parameters
        ----------
        correlation : Type of correlation coefficient to use for fitness
            evaluation.
            'pearson': Pearson correlation coefficient.
            'spearman': Spearman rank-order correlation coefficient.
        k_min  : Minimum number of selected features in each solution.
        n_populations : Number of solutions in each generation.
        n_generations : Number of generations in the genetic algorithm.
        n_parents : Number of parents selected for crossover.
        n_children  : Number of offspring produced in each generation.
        min_mutation_rate  : Minimum mutation rate in the genetic algorithm.
        max_mutation_rate  : Maximum mutation rate in the genetic algorithm.
        max_patience  : Maximum number of consecutive generations with
            no improvement.
        random_state : Seed for the random number generator.
        """

        self.correlation = correlation
        self.k_min = k_min
        self.n_populations = n_populations
        self.n_generations = n_generations
        self.n_parents = n_parents
        self.n_children = n_children
        self.min_mutation_rate = min_mutation_rate
        self.max_mutation_rate = max_mutation_rate
        self.max_patience = max_patience

        self.random_state = random_state
        self._rng = np.random.default_rng(self.random_state)

    def _crossover(self, a, b, mutation_rate: float = 0.1):
        size = a.size
        choice = self._rng.choice([True, False], size=size)
        child = np.where(choice, a, b)

        if self._rng.uniform() <= mutation_rate:
            n_mutations = int(mutation_rate * size)
            idx = self._rng.integers(0, high=size, size=n_mutations)
            child[idx] = np.logical_not(child[idx])

        return child

    def _objective(self, vect, corr):

        if sum(vect) < self.k_min:
            return np.inf
        return 0.5 * vect.T @ corr @ vect + sum(vect)

    def _evaluate(self, populations, **kwargs):
        scores = np.array([self._objective(p, **kwargs) for p in populations])
        order = np.argsort(scores)
        populations = populations[order]
        scores = scores[order]

        return populations, scores

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
        size = X.shape[1]

        corr = compute_correlation(X, self.correlation)
        corr = np.abs(corr)

        counter = 0
        best_score = np.inf
        best = None
        children = []

        self.best_scores_ = []
        self.median_scores_ = []

        for generation in range(self.n_generations):

            if generation > 0:
                populations = \
                    children + [best] + \
                    [self._rng.choice([True, False], size=size) for _
                     in range(self.n_populations - self.n_children - 1)]
                populations = np.array(populations)
            else:
                populations = np.array([
                    self._rng.choice([True, False], size=size)
                    for _ in range(self.n_populations)
                ])

            populations, scores = self._evaluate(populations, corr=corr)

            if scores[0] < best_score:
                best = populations[0]
                best_score = scores[0]
                counter = 0
            else:
                counter += 1

            self.best_scores_.append(best_score)
            self.median_scores_.append(np.nanmedian(scores))

            parents = populations[:self.n_parents]
            p = np.exp(1 / scores[:self.n_parents])

            p /= np.sum(p)

            mr = self.max_mutation_rate - (generation + 1) * (
                    self.max_mutation_rate - self.min_mutation_rate) / self.n_generations
            if counter > self.max_patience:
                mr = self.max_mutation_rate
                counter -= 2

            children = [self._crossover(
                self._rng.choice(parents, p=p),
                self._rng.choice(parents, p=p),
                mutation_rate=mr
            ) for _ in range(self.n_children)]

        populations, scores = self._evaluate(populations, corr=corr)

        self.selected_features_ = populations[0]

        return self

    def _get_support_mask(self):
        return self.selected_features_
