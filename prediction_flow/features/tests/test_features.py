from prediction_flow.features import Number, Category, Sequence, Features
from prediction_flow.transformers.column import (
    StandardScaler, CategoryEncoder, SequenceEncoder, ColumnFlow)

import numpy as np
import pandas as pd


__TEST_DATA = pd.DataFrame(
    [
        [23, 173, 'male', 'fish vegetable', 1],
        [43, 181, 'male', 'fish pork', 0],
        [35, 161, 'female', 'beef vegetable', 0],
        [41, 171, 'male', 'fish vegetable', 1],
        [16, 153, 'female', 'pork chicken vegetable', 0],
        [32, 168, 'female', 'fish beef', 1],
        [26, 177, 'male', 'chicken vegetable', 0],
        [76, 190, 'male', 'fish pork vegetable', 0]
    ],
    columns=['age', 'height', 'gender', 'likes', 'label'])


def test_simple_column_transformer_define():
    number_features = [
        Number('age', None),
        Number('height', StandardScaler())]

    category_features = [
        Category('gender', CategoryEncoder(min_cnt=1))]

    sequence_features = [
        Sequence('likes', SequenceEncoder(sep=' ', min_cnt=1))]

    features = Features(
        number_features, category_features, sequence_features)

    features.fit(__TEST_DATA)

    actual = features.transform(__TEST_DATA)

    expected_age = np.array([23, 43, 35, 41, 16, 32, 26, 76])
    expected_gender = np.array([2, 2, 1, 2, 1, 1, 2, 2])
    expected_height = np.array(
        [0.1159659, 0.85814767, -0.99730676, -0.06957954, -1.73948853,
         -0.34789771, 0.48705679, 1.69310217])

    assert len(actual) == 4
    assert features.number_feature_names() == ['age', 'height']
    assert features.category_feature_names() == ['gender']
    assert features.sequence_feature_names() == ['likes']
    np.testing.assert_array_equal(actual['age'], expected_age)
    np.testing.assert_array_equal(actual['gender'], expected_gender)
    np.testing.assert_array_almost_equal(actual['height'], expected_height)


def test_column_flow_define():
    number_features = [
        Number('age', None),
        Number('height', ColumnFlow([StandardScaler()]))]

    category_features = [
        Category('gender', ColumnFlow([CategoryEncoder(min_cnt=1)]))
    ]

    sequence_features = [
        Sequence('likes', ColumnFlow([SequenceEncoder(sep=' ', min_cnt=1)]))
    ]

    features = Features(
        number_features, category_features, sequence_features)

    features.fit(__TEST_DATA)

    actual = features.transform(__TEST_DATA)

    expected_age = np.array([23, 43, 35, 41, 16, 32, 26, 76])
    expected_gender = np.array([2, 2, 1, 2, 1, 1, 2, 2])
    expected_height = np.array(
        [0.1159659, 0.85814767, -0.99730676, -0.06957954, -1.73948853,
         -0.34789771, 0.48705679, 1.69310217])

    assert len(actual) == 4
    assert features.number_feature_names() == ['age', 'height']
    assert features.category_feature_names() == ['gender']
    assert features.sequence_feature_names() == ['likes']
    np.testing.assert_array_equal(actual['age'], expected_age)
    np.testing.assert_array_equal(actual['gender'], expected_gender)
    np.testing.assert_array_almost_equal(actual['height'], expected_height)
