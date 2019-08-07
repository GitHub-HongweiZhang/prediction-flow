from prediction_flow.pytorch.nn import Interest


import torch


def test_interest_evolution():
    interests = Interest(
        input_size=3,
        gru_type='GRU',
        gru_dropout=0,
        att_hidden_layers=[8],
        att_dropout=0,
        att_batchnorm=False,
        att_activation=None)

    query = torch.tensor([[1, 1, 1], [0.1, 0.2, 0.3]], dtype=torch.float)

    keys = torch.tensor([
        [[0.1, 0.2, 0.3], [1, 2, 3], [0.4, 0.2, 1], [0.0, 0.0, 0.0]],
        [[0.1, 0.2, 0.3], [1, 2, 3], [0.4, 0.2, 1], [0.5, 0.5, 0.5]]
    ], dtype=torch.float)

    keys_length = torch.tensor([3, 4])

    output = interests(query, keys, keys_length)

    assert output.size()[0] == 2
    assert output.size()[1] == 3
