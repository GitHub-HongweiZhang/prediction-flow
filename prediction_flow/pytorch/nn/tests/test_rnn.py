from prediction_flow.pytorch.nn import AttentionGRUCell

import torch


def test_simple_creation():
    gru_cell = AttentionGRUCell(10, 20)
    input = torch.randn(6, 3, 10)
    hx = torch.randn(3, 20)

    output = []
    for i in range(6):
        hx = gru_cell(input[i], hx, 0.1)
        output.append(hx)

    assert len(output) == 6
