"""
Class to manage all features.
"""

# Authors: Hongwei Zhang
# License: MIT


from collections import OrderedDict


class Features(object):
    """Class to manage all features.

    Parameters
    ----------
    number_features : array-like
        NumberFeature array.

    category_features : array-like
        CategoryFeature array.

    sequence_features : array-like
        SequenceFeature array.

    Attributes
    ----------
    number_features : array-like
        NumberFeature array.

    category_features : array-like
        CategoryFeature array.

    sequence_features : array-like
        SequenceFeature array.

    """
    def __init__(
            self,
            number_features=[],
            category_features=[],
            sequence_features=[]):
        self.number_features = number_features
        self.category_features = category_features
        self.sequence_features = sequence_features

    def fit(self, df):
        """Fit all transformers.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        ----------
        self : Features
        """
        for feature in (
                self.number_features +
                self.category_features +
                self.sequence_features):
            if feature.column_flow:
                feature.column_flow.fit(df[feature.name].values)

        return self

    def transform(self, df):
        """Transform df using fitted transformers.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        ----------
        transformed_X: dict

        {'feature1': numpy.array([...]), 'feature2': numpy.array([...])}
        """
        transformed_X = OrderedDict()

        for feature in (
                self.number_features +
                self.category_features +
                self.sequence_features):
            if feature.column_flow:
                transformed_X[feature.name] = feature.column_flow.transform(
                    df[feature.name].values)
            else:
                transformed_X[feature.name] = df[feature.name].values

        return transformed_X

    def number_feature_names(self):
        return [feature.name for feature in self.number_features]

    def category_feature_names(self):
        return [feature.name for feature in self.category_features]

    def sequence_feature_names(self):
        return [feature.name for feature in self.sequence_features]
