"""Enlarge figure typography without recomputing or changing source values."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
input_names = ['original_vector_data.json', 'cluster_bag_verified.csv']
digests = {n: hashlib.sha256((OUT / n).read_bytes()).hexdigest() for n in input_names}
data = json.loads((OUT / input_names[0]).read_text())
verified = {}
with (OUT / input_names[1]).open() as stream:
    for row in csv.DictReader(stream):
        key = f"{row['dataset']}-bag{row['bag']}"
        order = int(row['order'])
        assert order not in verified.setdefault(key, {}), (key, order)
        verified[key][order] = float(row['test_acc_percent'])
assert verified['CIFAR10-bag128'][2] == 71.91
assert [verified[f'CIFAR10-bag{b}'][13] for b in [16, 32, 64, 128]] == [94.24, 91.87, 89.05, 72.21]

colors = {'alpha': (0.1215686, 0.4666667, 0.7058824),
          'cluster': (1, .4980392, .054902),
          'random': (.172549, .627451, .172549)}
plt.rcParams.update({'font.size': 18, 'axes.titlesize': 16,
                     'axes.labelsize': 18, 'xtick.labelsize': 14,
                     'ytick.labelsize': 14, 'pdf.fonttype': 42})
fig, axes = plt.subplots(2, 4, figsize=(14.8, 7.0))
fig.subplots_adjust(left=.066, right=.944, top=.84, bottom=.17,
                    wspace=.48, hspace=.52)
for r, dataset in enumerate(['CIFAR10', 'CIFAR100']):
    for c, bag in enumerate([16, 32, 64, 128]):
        ax = axes[r, c]
        key = f'{dataset}-bag{bag}'
        panel = data[key]
        x = np.arange(1, len(panel['alpha']) + 1)
        assert sorted(verified[key]) == list(x)
        for name, marker in [('alpha', 'o'), ('random', 'D')]:
            ax.plot(x, panel[name], color=colors[name], marker=marker,
                    markersize=4.5, linewidth=1.4)
        ax.plot(x, [verified[key][int(i)] for i in x], color=colors['cluster'],
                marker='^', markersize=5.5, linewidth=1.6, zorder=5)
        ax.set_title(f'{dataset}, bag = {bag}', pad=7)
        ax.set_xlabel('Order $s$', labelpad=3)
        ax.set_xticks(x[::2] if r == 0 else x)
        if r == 0:
            ax.set_xticks(x, minor=True)
        ax.tick_params(axis='both', pad=2)
        ax.grid(alpha=.35, linestyle=':')
        if c == 0:
            ax.set_ylabel('Accuracy (%)', labelpad=4)
        runtime = ax.twinx()
        runtime.plot(x, panel['runtime'], 'k--s', linewidth=1.4, markersize=4.3)
        runtime.set_ylim(min(panel['runtime']) * .93, max(panel['runtime']) * 1.06)
        runtime.tick_params(axis='y', pad=2)
        if c == 3:
            runtime.set_ylabel('Time (s)', labelpad=4)

handles = [Line2D([], [], color=colors['alpha'], marker='o', label=r'$\alpha$-First Bag'),
           Line2D([], [], color=colors['cluster'], marker='^', label='Cluster Bag'),
           Line2D([], [], color=colors['random'], marker='D', label='Random Bag'),
           Line2D([], [], color='black', linestyle='--', marker='s', label='Runtime')]
fig.legend(handles=handles, loc='upper center', ncol=4,
           bbox_to_anchor=(.5, .99), frameon=True, fontsize=18)
fig.text(.066, .052, 'Cluster Bag order 1 uses the corresponding PM result. CIFAR10 order 13 is complete for all four bag sizes (best test accuracy).', fontsize=14)
fig.text(.066, .020, 'Other accuracy curves and runtime retain the original figure data; runtime was not remeasured in the new runs.', fontsize=14)
fig.savefig(OUT / 'order_runtime_dataset_rows_bigfonts.pdf')
fig.savefig(OUT / 'order_runtime_dataset_rows_bigfonts.png', dpi=180)
assert digests == {n: hashlib.sha256((OUT / n).read_bytes()).hexdigest() for n in input_names}
print(json.dumps({'unchanged_input_sha256': digests, 'panels': len(verified),
                  'cluster_points': sum(map(len, verified.values()))}, indent=2))
