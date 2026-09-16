import collections
import argparse
import functools
import glob
import pickle
import itertools
import json
import os
import random
import sys

# So we can run as `python collect_results.py` or `python plench/collect_results.py`;
# repo root must be on path so plench is a package and relative imports inside core/lib work.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
from tqdm import tqdm

from plench.data import datasets
from plench.core import algorithms, model_selection
from plench.lib import misc, reporting
from plench.lib.query import Q
import warnings

def format_mean(data, latex):
    """Given a list of datapoints, return a string describing their mean and
    standard error"""
    if len(data) == 0:
        return None, None, "X"
    data_list = list(data)
    mean = 100*np.mean(data_list)
    err = 100*np.std(data_list) / np.sqrt(len(data_list))
    if latex:
        return mean, err, "{:.2f}$\\pm${:.2f}".format(mean, err)
    else:
        return mean, err, "{:.2f} +/- {:.2f}".format(mean, err)

def print_table(table, header_text, row_labels, col_labels, colwidth=10,
    latex=True):
    """Pretty-print a 2D array of data, optionally with row/col labels"""
    print("")

    if latex:
        num_cols = len(table[0])
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * num_cols + "}")
        print("\\toprule")
    else:
        print("--------", header_text)

    for row, label in zip(table, row_labels):
        row.insert(0, label)

    if latex:
        col_labels = ["\\textbf{" + str(col_label).replace("%", "\\%") + "}"
            for col_label in col_labels]
    table.insert(0, col_labels)

    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth, latex=latex)
        if latex and r == 0:
            print("\\midrule")
    if latex:
        print("\\bottomrule")
        print("\\end{tabular}}")
        print("\\end{center}")

def print_results_tables(records, dataset, latex):
    """Given all records, print a results table for each dataset."""
    SELECTION_METHODS = [
        model_selection.DSQSelectionMethod,
        model_selection.EasySelectionMethod,
        model_selection.GeneralUPMSelectionMethod,
        model_selection.TestAccSelectionMethod,
        model_selection.PMSelectionMethod,
        model_selection.ErrorSelectionMethod,
    ]

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])


    # Print an "averages" table
    if latex:
        print()
        print("\\subsubsection{Averages}")

    selection_method_names = []
    for j, selection_method in enumerate(SELECTION_METHODS):
        selection_method_names.append(selection_method.name)

    table = [[None for _ in [*selection_method_names]] for _ in alg_names]
    for i, algorithm in enumerate(alg_names):
        for j, selection_method in enumerate(SELECTION_METHODS):
            grouped_records = reporting.get_grouped_records(records).map(lambda group:
                { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
            ).filter(lambda g: g["sweep_acc"] is not None)
            trial_averages = (grouped_records
                .filter_equals("algorithm, dataset", (algorithm, dataset))
                .group("trial_seed")
                .map(lambda trial_seed, group:
                    group.select("sweep_acc").mean()
                )
            )
            mean, err, table[i][j] = format_mean(trial_averages, latex)

    col_labels = ["Algorithm", *selection_method_names]
    header_text = f"Dataset: {dataset}"
    print_table(table, header_text, alg_names, col_labels, colwidth=25,
        latex=latex)

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(description="Deep learning from label proportions")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--dataset", type=str, default=None,
        help="Only collect results for this dataset (e.g. MNIST). If omitted, collect all.")
    args = parser.parse_args()

    results_file = "results.tex" if args.latex else "results.txt"
    sys.stdout = misc.Tee(os.path.join(args.input_dir, results_file), "w")
    records = reporting.load_records(args.input_dir)

    if args.latex:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\usepackage{adjustbox}")
        print("\\begin{document}")
        print("\\section{Full PLENCH results}")
        print("% Total records:", len(records))
    else:
        print("Total records:", len(records))


    # Normalize dataset name in records for grouping (dataset_hparams_seed)
    for i in tqdm(range(len(records))):
        ds=records[i]['args']['dataset']
        bagsize=records[i]['args']['bagsize']
        bag_build=records[i]['args']['bag_build']
        new_dataset_name='{}_{}_{}'.format(ds,bagsize,bag_build)
        records[i]['args']['dataset']=new_dataset_name

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    # dataset_names = [d for d in datasets.DATASETS if d in dataset_names]
    if args.dataset is not None:
        dataset_names = [d for d in dataset_names if d.startswith(args.dataset)]


    for dataset in dataset_names:
        if args.latex:
            print()
            print("\\subsection{{Dataset: {}}}".format(
                dataset))
        print_results_tables(records, dataset, args.latex)

    if args.latex:
        print("\\end{document}")


# python collect_results.py --input_dir ./output/