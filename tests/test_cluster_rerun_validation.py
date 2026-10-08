import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cluster_rerun import valid_result_rows, result_status
import json


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


def test_divergence_is_a_verified_outcome_and_oom_is_not(tmp_path):
    log = tmp_path / 'results.jsonl'
    log.write_text(json.dumps(row(test_acc=0.4)) + '\n')
    assert result_status(tmp_path, 1, 500)['outcome'] == 'infrastructure_failure'
    failure = tmp_path / 'numerical_failure.json'
    failure.write_text(json.dumps({'reason': 'nonfinite_training_loss', 'metric': 'loss',
                                   'step': 1001, 'value': 'nan'}))
    status = result_status(tmp_path, 1, 500)
    assert status['validated'] and status['outcome'] == 'numerical_divergence'
    assert status['first_nonfinite_step'] == 1001 and status['best_test_acc'] == 0.4
    assert not result_status(tmp_path, 1, 500, smoke=True)['validated']
    failure.write_text(json.dumps({'reason': 'out_of_memory', 'metric': 'loss',
                                   'step': 1001, 'value': 'nan'}))
    assert not result_status(tmp_path, 1, 500)['validated']
