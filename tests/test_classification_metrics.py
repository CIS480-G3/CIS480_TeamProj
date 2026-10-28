import pytest
from classification_metrics import precision_f1

# we'll assume matrix values will only be positive integers
@pytest.mark.parametrize(
        "tp, fp, fn, expected_precision, expected_f1",
        [
            # perfect scores
            (10, 0, 0, 100.0 ,1.0),

            # cutoff at 0.66
            (49, 196, 19, 20.0, 0.3131),

            # cutoff at 0.5
            (56, 346, 12, 13.93, 0.2383),

            # baseline (0.0)
            (0, 0, 68, None, 0.0),
            
            # all zeroes (0.0)
            (0, 0, 0, None, 0.0),
        ]
)

def test_precision_f1(tp,fp,fn, expected_precision, expected_f1):
    precision, f1 = precision_f1(tp,fp,fn)
    assert precision == expected_precision
    assert f1 == expected_f1