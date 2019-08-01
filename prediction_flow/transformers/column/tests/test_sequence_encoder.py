from prediction_flow.transformers.column import SequenceEncoder

import numpy as np


def test_normal():
    sequence_encoder = SequenceEncoder(sep=' ', min_cnt=1, max_len=3)

    x = [
        "this is a simple test",
        "this class is work"
    ]

    sequence_encoder.fit(x)

    actual = sequence_encoder.transform(x)
    assert sequence_encoder.dimension() == 9
    assert sequence_encoder.max_length() == 3
    assert actual.tolist() == [[6, 3, 1], [6, 2, 3]]
    assert isinstance(x, list)


def test_unseen_inputs():
    sequence_encoder = SequenceEncoder(sep=' ', min_cnt=1, max_len=10)

    x = [
        "this is a simple test",
        "this class is work"
    ]

    sequence_encoder.fit(x)

    actual = sequence_encoder.transform(["this is an unseen test"])
    assert actual.tolist() == [[6, 3, 8, 8, 5]]
