#!/usr/bin/env python3
"""Execute one explicitly indexed Cluster rerun; never overwrite prior outputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from reproduce_paper import build_plan, load_protocol, REPO_ROOT


def inventory():
    paths = sorted(set(REPO_ROOT.glob('plench/**/*.py')) | set(REPO_ROOT.glob('src/**/*.py')) |
                   set(REPO_ROOT.glob('scripts/*.py')) | set(REPO_ROOT.glob('configs/*.json')))
    return {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def valid_result_rows(rows, expected_epochs):
    """Reject truncated training and nonfinite recorded numerical metrics."""
    if not rows or rows[-1].get('args', {}).get('epochs') != expected_epochs:
        return False
    if rows[-1].get('epoch', -1) < expected_epochs - 1:
        return False
    for row in rows:
        for required in ('loss', 'test_acc', 'epoch', 'step'):
            if required not in row or not math.isfinite(float(row[required])):
                return False
        for key, value in row.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                return False
        if not 0 <= row['test_acc'] <= 1:
            return False
    return True


def result_status(output, returncode, expected_epochs, smoke=False):
    """Keep numerical divergence distinct from infrastructure failure."""
    status = {'returncode': returncode, 'finished_unix': time.time(), 'validated': False,
              'kind': 'smoke' if smoke else 'formal', 'outcome': 'infrastructure_failure'}
    log = output / 'results.jsonl'
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()] if log.is_file() else []
    if returncode == 0 and (output / 'done').is_file() and valid_result_rows(rows, expected_epochs):
        best = max(rows, key=lambda row: row['test_acc'])
        status.update(validated=True, outcome='normal', final_test_acc=rows[-1]['test_acc'],
                      last_epoch=rows[-1]['epoch'], best_test_acc=best['test_acc'], best_step=best['step'])
        return status
    failure_path = output / 'numerical_failure.json'
    if returncode != 0 and not (output / 'done').exists() and failure_path.is_file():
        failure = json.loads(failure_path.read_text())
        value = float(failure.get('value', '0'))
        step = failure.get('step')
        if (failure.get('reason') == 'nonfinite_training_loss' and failure.get('metric') == 'loss'
                and isinstance(step, int) and step >= 0 and not math.isfinite(value)):
            finite = [row for row in rows if math.isfinite(row.get('test_acc', float('nan')))]
            status.update(validated=not smoke, outcome='numerical_divergence', first_nonfinite_step=step,
                          failure_evidence='numerical_failure.json')
            if finite:
                best = max(finite, key=lambda row: row['test_acc'])
                status.update(best_test_acc=best['test_acc'], best_step=best['step'])
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--index', type=int)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--prepare', choices=['CIFAR10', 'CIFAR100', 'miniImageNet'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    runs = build_plan(bag_modes=['cluster'], data_root=args.data_root,
                      output_root=args.output_root / ('smoke' if args.smoke else 'runs'),
                      num_workers=args.workers)
    if args.list:
        for index, run in enumerate(runs):
            print(json.dumps({'index': index, **run}))
        return
    if not (os.environ.get('PBS_JOBID') or os.environ.get('PJM_JOBID') or socket.gethostname().split('.')[0].startswith(('ke', 'kf', 'kd', 'kg', 'kc', 'kb'))):
        raise RuntimeError('Execute on an allocated compute node, not a login node')
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / 'src'))
    if args.prepare:
        from plench.data.LLP_load import load_data_train
        protocol = load_protocol()
        for size in protocol['bag_sizes']:
            for seed in protocol['seeds']:
                path = args.output_root / 'manifests' / f'{args.prepare}_bag{size}_seed{seed}.npz'
                load_data_train(1., 'cluster', protocol['datasets'][args.prepare]['num_classes'], 0.,
                                dataset=args.prepare, dspth=str(args.data_root), bagsize=size,
                                seed=seed, cluster_manifest=path)
        (args.output_root / f'prepared_{args.prepare}.json').write_text(json.dumps({'source_hashes':inventory()}))
        return
    if args.index is None or not 0 <= args.index < len(runs):
        parser.error('--index must identify one of the 396 planned runs')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the rerun')
    run = runs[args.index]
    manifest = args.output_root / 'manifests' / f"{run['dataset']}_bag{run['bag_size']}_seed{run['seed']}.npz"
    if not manifest.is_file():
        raise FileNotFoundError(f'Prepare and distribute the frozen manifest first: {manifest}')
    output = Path(run['output_dir'])
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True)
    command = run['command'] + ['--cluster-manifest', str(manifest), '--skip_model_save']
    if args.smoke:
        command[command.index('--epochs') + 1] = '2' if run['method'] == 'FlowLLP' else '1'
        if run['method'] == 'FlowLLP':
            hp = dict(run['hparams'], flow_anchors_per_class=2, flow_particle_steps=2)
            command[command.index('--hparams') + 1] = json.dumps(hp)
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT/'src')])
    environment['PYTHONUNBUFFERED'] = '1'
    versions = {}
    for package in ['torch', 'torchvision', 'numpy', 'scipy', 'Pillow', 'scikit-learn', 'POT', 'opencv-python']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    evidence = {'run':run, 'command':command, 'smoke':args.smoke, 'source_hashes':inventory(),
                'versions':versions, 'python':sys.version, 'host':socket.gethostname(),
                'job_id':os.environ.get('PBS_JOBID',os.environ.get('PJM_JOBID')),
                'gpu':torch.cuda.get_device_name(0), 'started_unix':time.time()}
    (output/'execution.json').write_text(json.dumps(evidence,indent=2)+'\n')
    with (output/'train.log').open('w') as log:
        result = subprocess.run(command, cwd=REPO_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    status = result_status(output, result.returncode, int(command[command.index('--epochs')+1]), args.smoke)
    (output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps({'run_id':run['run_id'], **status}), flush=True)
    if not status['validated']:
        raise RuntimeError(f'Run failed validation; inspect {output}/train.log')


if __name__ == '__main__':
    main()
