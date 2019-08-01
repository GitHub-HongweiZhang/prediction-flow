from prediction_flow.transformers.column import StandardScaler

import numpy as np


def test_normal():
    scaler = StandardScaler()

    x = np.array([3, 4, 2, 24, 2], dtype=np.float)

    scaler.fit(x)

    actual = scaler.transform(x)
    expected = np.array([
        -0.46880723, -0.35160542, -0.58600904,  1.99243073, -0.58600904])

    np.testing.assert_array_almost_equal(actual, expected)
