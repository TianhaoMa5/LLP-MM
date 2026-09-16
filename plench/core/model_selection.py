import itertools
import numpy as np

def filter_step0_records(records):
    """Filter step 0"""
    return records.filter(lambda r: r['step'] != 0)

class SelectionMethod:
    """Abstract class whose subclasses implement strategies for model
    selection across hparams and timesteps."""

    def __init__(self):
        raise TypeError

    @classmethod
    def run_acc(self, run_records):
        """
        Given records from a run, return a {val_acc, test_acc} dict representing
        the best val-acc and corresponding test-acc for that run.
        """
        raise NotImplementedError

    @classmethod
    def hparams_accs(self, records):
        """
        Given all records from a single (dataset, algorithm) pair,
        return a sorted list of (run_acc, records) tuples.
        """
        return (records.group('args.hparams_seed')
            .map(lambda _, run_records:
                (
                    self.run_acc(run_records),
                    run_records
                )
            ).filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0]['val_acc'])[::-1]
        )


    @classmethod
    def sweep_acc(self, records):
        """
        Given all records from a single (dataset, algorithm) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc']
        else:
            return None

class GeneralUPMSelectionMethod(SelectionMethod):
    """Picks last checkpoint (no early stopping), uses val_GeneralUPM."""
    name = "GeneralUPM"

    @classmethod
    def _step_acc(self, record):
        """Picks argmin(GeneralUPM), lower is better."""
        return {'val_acc': -record.get('val_GeneralUPM', float('inf')), 'test_acc': record['test_acc']}

    @classmethod
    def run_acc(self, run_records):
        test_records = filter_step0_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')

class DSQSelectionMethod(SelectionMethod):
    """Picks argmin(val_DSQ), lower is better."""
    name = "DSQ"

    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        return {'val_acc': -record.get('val_DSQ', float('inf')), 'test_acc': record['test_acc']}

    @classmethod
    def run_acc(self, run_records):
        test_records = filter_step0_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')

class EasySelectionMethod(SelectionMethod):
    """Picks argmin(val_easy), lower is better."""
    name = "Easy"

    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        return {'val_acc': -record.get('val_easy', float('inf')), 'test_acc': record['test_acc']}

    @classmethod
    def run_acc(self, run_records):
        test_records = filter_step0_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')


class PMSelectionMethod(SelectionMethod):
    """Picks argmin(val_PM), lower is better."""
    name = "PM"

    @classmethod
    def _step_acc(self, record):
        return {'val_acc': -record.get('val_PM', float('inf')), 'test_acc': record['test_acc']}

    @classmethod
    def run_acc(self, run_records):
        test_records = filter_step0_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')


class ErrorSelectionMethod(SelectionMethod):
    """Picks argmin(val_error), lower is better. Use -val_error so larger is better for selection."""
    name = "Error"

    @classmethod
    def _step_acc(self, record):
        return {'val_acc': -record.get('val_error', float('inf')), 'test_acc': record['test_acc']}

    @classmethod
    def run_acc(self, run_records):
        test_records = filter_step0_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')


class TestAccSelectionMethod(SelectionMethod):
    """Last checkpoint, report test_acc. Higher test_acc is better for ranking runs."""
    name = "TestAcc"

    @classmethod
    def _step_acc(self, record):
        return {'val_acc': record['test_acc'], 'test_acc': record['test_acc']}

    @classmethod
    def run_acc(self, run_records):
        run_records = filter_step0_records(run_records)
        if not len(run_records):
            return None
        return run_records.map(self._step_acc).argmax('val_acc')
