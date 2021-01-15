from collections import Counter
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def label_encode(df, column, append_column=False, return_encoder=False):
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(df[column].values)
    if append_column:
        column += "_encoded"
    df[column] = encoded
    if return_encoder:
        return df, encoder
    else:
        return df


def ordinal_encode(df, columns, append_columns=False, return_encoder=False):
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder
    encoder = OrdinalEncoder()
    encoded = encoder.fit_transform(df[columns].values)
    if append_columns:
        columns = [col + "_encoded" for col in columns]
    df[columns] = encoded
    if return_encoder:
        return df, encoder
    else:
        return df


def count_encode(df, column, append_column=False, return_encoder=False):
    # https://docs.python.org/ja/3/library/collections.html?highlight=collections%20counter#collections.Counter
    counter = Counter(df[column])
    count_dict = dict(counter.most_common())
    encoded = df[column].map(count_dict).values
    if append_column:
        column += "_count"
    df[column] = encoded
    if return_encoder:
        return df, count_dict
    else:
        return df


def hash_encode(df, column, n_features=10, return_encoder=False):
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher
    encoder = FeatureHasher(n_features=n_features)
    encoded = encoder.fit_transform(df[column].values).toarray()
    columns = [column + f"_{i}" for i in range(n_features)]
    df[columns] = encoded
    if return_encoder:
        return df, encoder
    else:
        return df
