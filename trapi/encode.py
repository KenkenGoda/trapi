from collections import Counter

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def label_encode(values, return_encoder=False):
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder
    encoder = LabelEncoder()
    encoded_values = encoder.fit_transform(values)
    if return_encoder:
        return encoded_values, encoder
    else:
        return encoded_values


def ordinal_encode(values, return_encoder=False):
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder
    encoder = OrdinalEncoder()
    encoded_values = encoder.fit_transform(values)
    if return_encoder:
        return encoded_values, encoder
    else:
        return encoded_values


def count_encode(values, return_encoder=False):
    # https://docs.python.org/ja/3/library/collections.html?highlight=collections%20counter#collections.Counter
    counter = Counter(values)
    count_dict = dict(counter.most_common())
    encoded_values = np.array([count_dict[value] for value in values])
    if return_encoder:
        return encoded_values, count_dict
    else:
        return encoded_values


def hash_encode(values, n_features=10, return_encoder=False):
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher
    encoder = FeatureHasher(n_features=n_features)
    encoded_values = encoder.fit_transform(values).toarray()
    if return_encoder:
        return encoded_values, encoder
    else:
        return encoded_values
