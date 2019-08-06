from prediction_flow.pytorch.nn import Attention


import numpy as np
import torch
import torch.nn.init as init


def test_din_attention():
    attention = Attention(3, [8], batchnorm=False, activation=0.0)

    query = torch.tensor([[1, 1, 1], [0.1, 0.2, 0.3]], dtype=torch.float)

    keys = torch.tensor([
        [[0.1, 0.2, 0.3], [1, 2, 3], [0.4, 0.2, 1], [0.0, 0.0, 0.0]],
        [[0.1, 0.2, 0.3], [1, 2, 3], [0.4, 0.2, 1], [0.5, 0.5, 0.5]]
    ], dtype=torch.float)

    keys_length = torch.tensor([3, 4])

    for param in attention.mlp.parameters():
        init.constant_(param, 1)

    for param in attention.fc.parameters():
        init.constant_(param, 1)

    output = attention(query, keys, keys_length)

    actual = output.detach().numpy()
    assert output.size()[0] == 2
    assert output.size()[1] == 3
    np.testing.assert_array_almost_equal(
        actual, np.array([[1.0, 2.0, 3.0],
                          [0.989024, 1.969694, 2.959199]], dtype=np.float))
