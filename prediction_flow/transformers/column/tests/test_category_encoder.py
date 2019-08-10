from prediction_flow.transformers.column import CategoryEncoder


def test_str_inputs():
    category_encoder = CategoryEncoder(min_cnt=1)

    input_terms = ['this', 'is', 'a', 'simple', 'test']

    category_encoder.fit(input_terms)

    transformed = category_encoder.transform(input_terms)

    assert set(transformed) == {1, 2, 3, 4, 5}
    assert category_encoder.dimension() == 7


def test_int_inputs():
    category_encoder = CategoryEncoder(min_cnt=1)

    input_terms = [345, 3434, 23, 88, 4]

    category_encoder.fit(input_terms)

    transformed = category_encoder.transform(input_terms)

    assert set(transformed) == {1, 2, 3, 4, 5}
    assert category_encoder.dimension() == 7


def test_unseen_inputs():
    category_encoder = CategoryEncoder(min_cnt=1)

    input_terms = [345, 3434, 23, 88, 4]

    category_encoder.fit(input_terms)

    transformed = category_encoder.transform([345, 5343])

    assert set(transformed) == {4, 6}
