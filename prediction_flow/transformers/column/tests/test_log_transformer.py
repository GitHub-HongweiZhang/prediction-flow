import numpy as np

from prediction_flow.transformers.column import LogTransformer


def test_normal():
    log_transformer = LogTransformer()

    x = np.array([100, 10, 32])
    log_transformer.fit(x)

    np.testing.assert_array_almost_equal(
        log_transformer.transform(x), np.array([4.615121, 2.397895, 3.496508]))
