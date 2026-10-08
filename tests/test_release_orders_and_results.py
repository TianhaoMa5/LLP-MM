import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from reproduce_paper import build_plan
from reproduce_ku import make_config
from summarize_cluster import summarize


@pytest.mark.parametrize('order', [3, 5, 13])
def test_order_ablation_keeps_seed_count_weights_and_separate_outputs(tmp_path, order):
    plan = build_plan(datasets=['CIFAR10'], methods=['LLP-MM'], bag_modes=['cluster'],
                      seeds=[0], order=order, data_root=tmp_path, output_root=tmp_path)
    assert len(plan) == 4
    assert {r['seed'] for r in plan} == {0}
    for r in plan:
        assert r['hparams']['order'] == order
        assert r['hparams']['order_weights'] == [1 / order] * order
        assert f'LLP-MM-order{order}' in r['output_dir']
    if order in [3, 5]:
        configs = [make_config('LLP-MM', s, tmp_path, tmp_path, order) for s in range(3)]
        assert len({c['output_dir'] for c in configs}) == 3
        assert all(json.loads(c['hparams'])['order'] == order for c in configs)


def test_order_override_rejects_baseline_and_duplicate_scope(tmp_path):
    with pytest.raises(ValueError):
        build_plan(methods=['PM', 'LLP-MM'], order=3)
    with pytest.raises(ValueError):
        make_config('PM', 0, tmp_path, tmp_path, order=3)


def test_published_cluster_results_have_complete_grid_and_divergence_provenance():
    records = json.loads((ROOT / 'results/cluster_best_runs.json').read_text())
    cells = summarize(records)
    assert len(cells) == 132
    assert sum(c['normal_n'] for c in cells) == 330
    assert sum(c['divergence_n'] for c in cells) == 66
    duplicated = records + [records[0]]
    with pytest.raises(ValueError, match='396 unique'):
        summarize(duplicated)
    changed = json.loads(json.dumps(records))
    changed[0]['best_test_acc'] = 1.1
    with pytest.raises(ValueError):
        summarize(changed)
