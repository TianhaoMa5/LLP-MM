import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cluster_rerun import valid_result_rows


def row(**changes):
    return dict(dict(args={'epochs': 500}, epoch=499.9, step=23999,
                     loss=0.3, test_acc=0.8), **changes)


@pytest.mark.parametrize('metric', ['loss', 'test_acc', 'epoch', 'step', 'val_PM'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_rejects_nonfinite_history_even_when_final_row_is_finite(metric, value):
    assert not valid_result_rows([row(**{metric: value}), row()], 500)


def test_valid_and_incomplete_results():
    assert valid_result_rows([row()], 500)
    assert not valid_result_rows([], 500)
    assert not valid_result_rows([row(epoch=498)], 500)
    assert not valid_result_rows([row(test_acc=10)], 500)
    assert not valid_result_rows([row(args={'epochs': 1})], 500)
