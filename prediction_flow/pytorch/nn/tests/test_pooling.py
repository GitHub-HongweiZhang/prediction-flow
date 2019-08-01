from prediction_flow.pytorch.nn import MaxPooling

import torch


def test_max_pooling():
    x = torch.tensor(
        [[[1, 2, 1, 1],
          [1, 1, 3, 1]],
         [[10, 1, 1, 1],
          [1, 1, 4, 1]],
         [[2, 8, 9, 0],
          [1, 1, 1, 1]]])

    max_pooling = MaxPooling(dim=1)

    actual = max_pooling(x)

    assert actual.numpy().tolist() == [
        [1, 2, 3, 1], [10, 1, 4, 1], [2, 8, 9, 1]]
