"""
==============================
frequency_time_series_split.py

A module for time-series splitting based on frequency
"""

import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold


class FrequencyTimeSeriesSplit(_BaseKFold):
    """
    The FreqTimeSeriesSplit class is a time series cross-validator designed
    for splitting time-ordered data into training and test sets.
    This algorithm is particularly suited for time series analysis,
    where maintaining temporal order is crucial.
    The class takes into account the frequency of the time series data,
    allowing users to define the time interval for creating training sets.

    The key feature of this algorithm is its ability to generate splits
    by considering fixed-size training sets within specified time intervals.
    The training frequency is defined using a string format following the
    pandas frequency strings convention.
    Additionally, users can set a maximum size for the training set in each
    split.

    The algorithm's split method utilizes these parameters to yield training
    and test set indices, ensuring that the temporal order is preserved.
    It works by iteratively moving through the time series data,
    creating training sets of a fixed size at defined intervals,
    and generating corresponding test sets.
    The class is designed to be compatible with scikit-learn's
    cross-validation framework, making it seamlessly integrable into machine
    learning workflows.
    """

    def __init__(self, train_freq='W-MON', max_train_size=100):
        """
        Initialize the class.

        Parameters
        ----------
        train_freq : str, default='W-MON'
            Frequency string defining the time interval for training set
            creation. It follows the pandas frequency strings format.

        max_train_size : int, default=100
            Maximum size of the training set for each split.
        """
        super().__init__(3, shuffle=False, random_state=None)
        self.train_freq = train_freq
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError('index is pd.DatetimeIndex object')

        if not X.index.is_monotonic_increasing:
            raise ValueError('index are not sorted')

        tmp = X.resample(self.train_freq).first()
        markers = tmp.index

        max_train_size = self.max_train_size

        indices = np.arange(len(X))

        for m_start, m_end in zip(markers, markers[1:]):
            if m_start > X.index.min() and m_end < X.index.max():
                m_start_idx = X.index.get_slice_bound(m_start, 'left')
                m_end_idx = X.index.get_slice_bound(m_end, 'left')

                idx_train = \
                    X.iloc[m_start_idx - max_train_size:m_start_idx].index
                if len(idx_train) >= max_train_size:
                    yield (
                        indices[m_start_idx - max_train_size:m_start_idx],
                        indices[m_start_idx:m_end_idx]
                    )
