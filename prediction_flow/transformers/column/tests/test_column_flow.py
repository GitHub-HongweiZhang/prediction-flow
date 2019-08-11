import pytest

from prediction_flow.transformers.column import (
    LogTransformer, CategoryEncoder, ColumnFlow)


def test_wrong_type_transformers():
    with pytest.raises(TypeError):
        ColumnFlow({CategoryEncoder()})


def test_multi_type_transformers():
    with pytest.raises(ValueError):
        ColumnFlow([LogTransformer(), CategoryEncoder()])


def test_transformers():
    column_flow = ColumnFlow([CategoryEncoder(min_cnt=1)])

    input_terms = ['this', 'is', 'a', 'simple', 'test']

    column_flow.fit(input_terms)

    transformed = column_flow.transform(input_terms)

    assert set(transformed) == {1, 2, 3, 4, 5}
    assert column_flow.transformers[-1].dimension() == 7
    assert isinstance(input_terms, list) == True
