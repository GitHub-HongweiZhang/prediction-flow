from prediction_flow.pytorch.nn import FM

import numpy as np
import torch


def test_fm():
    fm = FM()

    x = torch.as_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    actual = fm(x)

    # 11.0 = 1 * 2 + 1 * 3 + 2 * 3
    # 77.0 = 4 * 5 + 4 * 6 + 5 * 6
    np.testing.assert_array_almost_equal(
        actual.numpy(), np.array([[11.0], [74.0]], dtype=np.float))
