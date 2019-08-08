from prediction_flow.pytorch.nn import (
    AttentionGRUCell, AttentionUpdateGateGRUCell, DynamicGRU)

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def test_attention_gru_cell():
    gru_cell = AttentionGRUCell(10, 20)
    input = torch.randn(6, 3, 10)
    hx = torch.randn(3, 20)
    att_scores = torch.tensor([
        [0.1, 0.3, 0.6],
        [0.2, 0.2, 0.6],
        [0.1, 0.6, 0.3],
        [1.0, 0., 0.],
        [0.2, 0.3, 0.5],
        [0.1, 0.3, 0.6],
    ])

    output = []
    for i in range(6):
        hx = gru_cell(input[i], hx, att_scores[i])
        output.append(hx)

    assert len(output) == 6


def test_attention_update_gate_gru_cell():
    gru_cell = AttentionUpdateGateGRUCell(10, 20)
    input = torch.randn(6, 3, 10)
    hx = torch.randn(3, 20)
    att_scores = torch.tensor([
        [0.1, 0.3, 0.6],
        [0.2, 0.2, 0.6],
        [0.1, 0.6, 0.3],
        [1.0, 0., 0.],
        [0.2, 0.3, 0.5],
        [0.1, 0.3, 0.6],
    ])

    output = []
    for i in range(6):
        hx = gru_cell(input[i], hx, att_scores[i])
        output.append(hx)

    assert len(output) == 6


def test_dynamic_gru():
    rnn = DynamicGRU(3, 5)

    keys = torch.tensor([
        [[0.1, 0.2, 0.3], [1, 2, 3], [0.4, 0.2, 1], [0.0, 0.0, 0.0]],
        [[0.1, 0.2, 0.3], [1, 2, 3], [0.4, 0.2, 1], [0.5, 0.5, 0.5]],
        [[0.1, 0.2, 0.3], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        [[0.1, 0.2, 0.3], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        [[0.1, 0.2, 0.3], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    ], dtype=torch.float)

    att_scores = torch.tensor([
        [0.0330, 0.7252, 0.2459, 0.],
        [0.2952, 0.8721, 0.4468, 0.0904],
        [0.4598,  0.,  0.,  0.],
        [0.0286, 0.,  0.,  0.],
        [0.0561, 0.,  0.,  0.]])

    lengths = torch.tensor([3, 4, 1, 1, 1])

    packed_att_scores = pack_padded_sequence(
        att_scores,
        lengths,
        batch_first=True, enforce_sorted=False)

    packed_keys = pack_padded_sequence(
        keys,
        lengths,
        batch_first=True, enforce_sorted=False)

    actual, actual_lengths = pad_packed_sequence(
        rnn(packed_keys, packed_att_scores), batch_first=True)

    assert actual.size() == (5, 4, 5)
    assert actual_lengths.numpy().tolist() == [3, 4, 1, 1, 1]
