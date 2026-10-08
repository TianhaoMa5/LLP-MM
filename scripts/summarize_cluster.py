#!/usr/bin/env python3
"""Recompute the Cluster table from per-run best logged test accuracy."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['CIFAR10', 'CIFAR100', 'miniImageNet']
METHODS = ['PM', 'DSQ', 'LLP-PVC', 'LLP-FC', 'ROT', 'EasyLLP', 'EasyLLP-ABS',
           'GeneralUPM', 'GeneralUPM-ABS', 'FlowLLP', 'LLP-MM']
BAGS = [16, 32, 64, 128]


def summarize(records):
    expected = {f'{d}/cluster/bag{b}/{m}/seed{s}'
                for d, b, m, s in itertools.product(DATASETS, BAGS, METHODS, range(3))}
    indexed = {r['run']: r for r in records}
    if len(indexed) != len(records) or set(indexed) != expected:
        raise ValueError('Expected exactly 396 unique Cluster runs')
    cells = []
    for d, m, b in itertools.product(DATASETS, METHODS, BAGS):
        runs = [indexed[f'{d}/cluster/bag{b}/{m}/seed{s}'] for s in range(3)]
        values = []
        for r in runs:
            score = r['best_test_acc']
            if not r['validated'] or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f'Invalid result: {r["run"]}')
            if r['outcome'] == 'normal':
                if not r['normal_completion'] or not r['metrics']:
                    raise ValueError('Normal result needs a complete metric history')
                if max(q['test_acc'] for q in r['metrics']) != score:
                    raise ValueError('Peak accuracy disagrees with metric history')
                if r['configuration']['epochs'] != 500 or r['metrics'][-1]['epoch'] < 499:
                    raise ValueError('Normal result did not finish 500 epochs')
                if any(not math.isfinite(v) for q in r['metrics'] for v in q.values()):
                    raise ValueError('Nonfinite history cannot be a normal result')
            elif r['outcome'] == 'divergence':
                if r['normal_completion'] or not r['first_step_precision']:
                    raise ValueError('Divergence must retain failure provenance')
            else:
                raise ValueError('Unknown outcome')
            values.append(100 * score)
        cells.append(dict(dataset=d, method=m, bag=b, best_mean=statistics.mean(values),
                          best_sample_std=statistics.stdev(values),
                          normal_n=sum(r['outcome'] == 'normal' for r in runs),
                          divergence_n=sum(r['outcome'] == 'divergence' for r in runs)))
    return cells


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true', help='Compare against every published cell')
    p.add_argument('--output', type=Path, default=ROOT / 'results/cluster_best_summary.csv')
    args = p.parse_args()
    records = json.loads((ROOT / 'results/cluster_best_runs.json').read_text())
    cells = summarize(records)
    if args.check:
        for d in DATASETS:
            reference = json.loads((ROOT / f'results/{d}_cluster_cells.json').read_text())
            indexed = {(r['dataset'], r['method'], r['bag']): r for r in reference}
            for c in (c for c in cells if c['dataset'] == d):
                r = indexed[(d, c['method'], c['bag'])]
                for key in ['best_mean', 'best_sample_std', 'normal_n', 'divergence_n']:
                    if not math.isclose(c[key], r[key], rel_tol=0, abs_tol=1e-10):
                        raise ValueError(f'Published cell mismatch: {d}, {c["method"]}, {c["bag"]}, {key}')
    else:
        with args.output.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cells[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(cells)
    print(f'Checked {len(records)} runs and {len(cells)} cells (per-run peak test accuracy).')


if __name__ == '__main__':
    main()
