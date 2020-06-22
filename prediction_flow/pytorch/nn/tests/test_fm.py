from prediction_flow.pytorch.nn import FM

import numpy as np
import torch


def test_fm():
    fm = FM()

    x = torch.as_tensor(
        [[[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]],
         [[1.0, 1.0, 1.0], [4.0, 5.0, 6.0]]])
    actual = fm(x)

    # 6.0 = 1 * 1 + 1 * 2 + 1 * 3
    # 15.0 = 1 * 4 + 1 * 5 + 1 * 6
    np.testing.assert_array_almost_equal(
        actual.numpy(), np.array([[6.0], [15.0]], dtype=np.float))
