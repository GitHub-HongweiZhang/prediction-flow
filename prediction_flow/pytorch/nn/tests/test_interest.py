from prediction_flow.pytorch.nn import Interest


import torch


def test_gru_interest_evolution():
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

    output, _ = interests(query, keys, keys_length)

    assert output.size()[0] == 2
    assert output.size()[1] == 3


def test_aigru_interest_evolution():
    interests = Interest(
        input_size=3,
        gru_type='AIGRU',
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

    output, _ = interests(query, keys, keys_length)

    assert output.size()[0] == 2
    assert output.size()[1] == 3


def test_agru_interest_evolution():
    interests = Interest(
        input_size=3,
        gru_type='AGRU',
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

    output, _ = interests(query, keys, keys_length)

    assert output.size()[0] == 2
    assert output.size()[1] == 3


def test_augru_interest_evolution():
    interests = Interest(
        input_size=3,
        gru_type='AUGRU',
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

    output, _ = interests(query, keys, keys_length)

    assert output.size()[0] == 2
    assert output.size()[1] == 3


def test_neg_sampling():
    interests = Interest(
        input_size=3,
        gru_type='AUGRU',
        gru_dropout=0,
        att_hidden_layers=[8],
        att_dropout=0,
        att_batchnorm=False,
        att_activation=None,
        use_negsampling=True)

    query = torch.tensor(
        [[1, 1, 1], [0.1, 0.2, 0.3], [0.3, 0.4, 0.5]], dtype=torch.float)

    keys = torch.tensor([
        [[0.1, 0.2, 0.3], [1, 2, 3], [0.4, 0.2, 1], [0.0, 0.0, 0.0]],
        [[0.1, 0.2, 0.3], [1, 2, 3], [0.4, 0.2, 1], [0.5, 0.5, 0.5]],
        [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    ], dtype=torch.float)

    neg_keys = torch.tensor([
        [[0.3, 0.2, 0.1], [3, 2, 1], [1, 0.2, 0.4], [0.0, 0.0, 0.0]],
        [[0.3, 0.2, 0.1], [3, 2, 1], [1, 0.2, 0.4], [0.5, 0.5, 0.5]],
        [[0.3, 0.2, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    ], dtype=torch.float)

    keys_length = torch.tensor([3, 4, 1])

    output, _ = interests(query, keys, keys_length, neg_keys)

    assert output.size()[0] == 3
    assert output.size()[1] == 3
