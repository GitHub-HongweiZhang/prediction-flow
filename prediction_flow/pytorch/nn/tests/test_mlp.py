from prediction_flow.pytorch.nn import MLP


def test_simple_creation():
    mlp = MLP(input_size=10, hidden_layers=(16, 4),
              activation=None, dropout=0.0)

    assert len(mlp._sequential) == 4


def test_creation_with_dropout():
    mlp = MLP(input_size=10, hidden_layers=(16, 4),
              activation=None, dropout=0.1)

    assert len(mlp._sequential) == 6


def test_creation_with_activation_and_dropout():
    mlp = MLP(input_size=10, hidden_layers=(16, 4),
              activation='relu', dropout=0.1)

    assert len(mlp._sequential) == 8
