"""Deterministic split profiles for the CV and LEM centre catalogues.

The paper-aligned split remains the default preprocessing output.  This module
adds an explicit benchmark profile whose stronger requirement is that every
class occurs in train, validation, and test.  Classes represented by fewer
than three logical fields cannot satisfy that requirement with a strictly
field-disjoint split.  Only those fields are divided spatially. A full patch
guard is used whenever the field geometry permits it; an explicitly recorded
maximal partial guard is used when a single compact field must cover all three
splits.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .remote_sensing import SPLIT_NAMES, SPLIT_TO_CODE


DROPPED_SPLIT_CODE = np.uint8(255)
COVERAGE_SPLIT_STRATEGY = "deterministic_all_class_coverage_field_spatial_v1"
DEFAULT_COVERAGE_SPLIT_RATIOS = (0.70, 0.10, 0.20)


def nested_split_target_counts(
    num_fields: int,
    test_fraction: float,
    val_fraction_of_train: float = 0.20,
) -> Dict[str, int]:
    """Return the exact field-owner counts used by the paper split."""
    if num_fields < 3:
        raise ValueError("At least three logical fields are required")
    n_test = max(1, min(num_fields - 2, int(round(num_fields * test_fraction))))
    original_train = num_fields - n_test
    n_val = max(1, min(original_train - 1, int(round(original_train * val_fraction_of_train))))
    return {"train": original_train - n_val, "val": n_val, "test": n_test}


def _field_class_map(field_indices: np.ndarray, labels: np.ndarray) -> Dict[int, int]:
    result: Dict[int, int] = {}
    for field_index in np.unique(field_indices):
        field_labels = np.unique(labels[field_indices == field_index])
        if len(field_labels) != 1:
            raise ValueError(
                f"Logical field index {int(field_index)} has multiple centre labels: "
                f"{field_labels.tolist()}"
            )
        result[int(field_index)] = int(field_labels[0])
    return result


def deterministic_coverage_field_plan(
    field_indices: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    seed: int,
    test_fraction: float,
    val_fraction_of_train: float = 0.20,
) -> Tuple[Dict[int, str], Dict[int, Dict[str, Any]], Dict[str, int]]:
    """Assign whole fields while reserving class coverage deterministically.

    A two-field class reserves one whole test field and spatially divides the
    other between train/validation.  A one-field class spatially divides its
    only field three ways.  No repeated seed search or favourable-split search
    is performed.
    """
    field_indices = np.asarray(field_indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if field_indices.shape != labels.shape:
        raise ValueError("field_indices and labels must be aligned")
    field_class = _field_class_map(field_indices, labels)
    by_class: Dict[int, list[int]] = defaultdict(list)
    for field_index, class_index in field_class.items():
        by_class[class_index].append(field_index)
    missing = sorted(set(range(int(num_classes))) - set(by_class))
    if missing:
        raise ValueError(f"Classes with no valid centres cannot be covered: {missing}")

    rng = np.random.default_rng(int(seed))
    owner: Dict[int, str] = {}
    exceptions: Dict[int, Dict[str, Any]] = {}
    remaining: list[int] = []
    for class_index in range(int(num_classes)):
        fields = np.asarray(sorted(by_class[class_index]), dtype=np.int64)
        fields = fields[rng.permutation(len(fields))].tolist()
        if len(fields) == 1:
            owner[fields[0]] = "train"
            exceptions[fields[0]] = {
                "class_index": class_index,
                "total_class_fields": 1,
                "mode": "train_val_test",
            }
        elif len(fields) == 2:
            owner[fields[0]] = "train"
            owner[fields[1]] = "test"
            exceptions[fields[0]] = {
                "class_index": class_index,
                "total_class_fields": 2,
                "mode": "train_val",
            }
        else:
            owner[fields[0]] = "train"
            owner[fields[1]] = "val"
            owner[fields[2]] = "test"
            remaining.extend(fields[3:])

    targets = nested_split_target_counts(
        len(field_class), test_fraction, val_fraction_of_train
    )
    reserved = {name: sum(value == name for value in owner.values()) for name in SPLIT_NAMES}
    slots: list[str] = []
    for name in SPLIT_NAMES:
        count = targets[name] - reserved[name]
        if count < 0:
            raise ValueError(
                f"Class-coverage reservations require {reserved[name]} {name} fields, "
                f"but the target is {targets[name]}"
            )
        slots.extend([name] * count)
    if len(slots) != len(remaining):
        raise AssertionError("Field targets do not consume every unreserved field")
    if remaining:
        remaining_array = np.asarray(remaining, dtype=np.int64)
        remaining_array = remaining_array[rng.permutation(len(remaining_array))]
        slot_array = np.asarray(slots, dtype=object)
        slot_array = slot_array[rng.permutation(len(slot_array))]
        owner.update({int(field): str(split) for field, split in zip(remaining_array, slot_array)})
    if set(owner) != set(field_class):
        raise AssertionError("Every observed logical field must have exactly one owner")
    actual = {name: sum(value == name for value in owner.values()) for name in SPLIT_NAMES}
    if actual != targets:
        raise AssertionError(f"Owner counts {actual} do not match targets {targets}")
    return owner, exceptions, targets


def _partition_masks(
    values: np.ndarray,
    boundaries: Sequence[int],
    radius: int,
) -> list[np.ndarray]:
    if len(boundaries) == 1:
        boundary = int(boundaries[0])
        return [values <= boundary - radius, values >= boundary + radius + 1]
    if len(boundaries) == 2:
        first, second = (int(value) for value in boundaries)
        return [
            values <= first - radius,
            (values >= first + radius + 1) & (values <= second - radius),
            values >= second + radius + 1,
        ]
    raise ValueError("Spatial partition supports two or three blocks")


def spatially_partition_field(
    rows: np.ndarray,
    cols: np.ndarray,
    split_names: Sequence[str],
    target_fractions: Sequence[float],
    patch_size: int,
) -> Tuple[list[np.ndarray], Dict[str, Any]]:
    """Split one field along one axis and drop a patch-support guard band."""
    if len(split_names) not in (2, 3) or len(split_names) != len(target_fractions):
        raise ValueError("Expected aligned two- or three-way split definitions")
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer")
    target = np.asarray(target_fractions, dtype=np.float64)
    target = target / target.sum()
    requested_radius = patch_size // 2
    axes = (("row", np.asarray(rows, dtype=np.int64)),
            ("col", np.asarray(cols, dtype=np.int64)))
    best: tuple[float, float, int, tuple[int, ...], list[np.ndarray]] | None = None
    achieved_radius = requested_radius
    # Prefer a complete 21x21 support guard. If a compact one-field class makes
    # that geometrically impossible, retain the largest achievable guard and
    # expose the resulting patch-context overlap in the manifest.
    for candidate_radius in range(requested_radius, -1, -1):
        radius_best = None
        for axis_rank, (_, values) in enumerate(axes):
            if len(np.unique(values)) < len(split_names):
                continue
            if len(split_names) == 2:
                centre = float(target[0])
                quantile_sets = [(q,) for q in np.linspace(max(0.05, centre - 0.25),
                                                            min(0.95, centre + 0.25), 51)]
            else:
                first_centre = float(target[0])
                second_centre = float(target[0] + target[1])
                first_values = np.linspace(max(0.05, first_centre - 0.20),
                                           min(0.85, first_centre + 0.20), 25)
                second_values = np.linspace(max(0.15, second_centre - 0.20),
                                            min(0.95, second_centre + 0.20), 25)
                quantile_sets = [(first, second) for first in first_values
                                 for second in second_values if first < second]
            seen: set[tuple[int, ...]] = set()
            for quantiles in quantile_sets:
                boundaries = tuple(int(np.quantile(values, q, method="nearest")) for q in quantiles)
                if boundaries in seen or any(
                    later <= earlier for earlier, later in zip(boundaries[:-1], boundaries[1:])
                ):
                    continue
                seen.add(boundaries)
                masks = _partition_masks(values, boundaries, candidate_radius)
                counts = np.asarray([int(mask.sum()) for mask in masks], dtype=np.int64)
                if np.any(counts == 0):
                    continue
                assigned = int(counts.sum())
                observed = counts / assigned
                dropped_fraction = 1.0 - assigned / len(values)
                fraction_error = float(np.abs(observed - target).sum())
                score = fraction_error + 2.0 * dropped_fraction
                candidate = (score, dropped_fraction, axis_rank, boundaries, masks)
                if radius_best is None or candidate[:4] < radius_best[:4]:
                    radius_best = candidate
        if radius_best is not None:
            best = radius_best
            achieved_radius = candidate_radius
            break
    if best is None:
        raise ValueError(
            f"Could not spatially divide a {len(rows)}-centre field into "
            f"{len(split_names)} non-empty guarded blocks"
        )
    _, _, axis_rank, boundaries, masks = best
    axis_name, axis_values = axes[axis_rank]
    selected = [np.flatnonzero(mask).astype(np.int64) for mask in masks]
    counts = [len(indices) for indices in selected]
    assigned = sum(counts)
    minimum_separation = 2 * achieved_radius + 1
    for left, right in zip(selected[:-1], selected[1:]):
        if int(axis_values[right].min()) - int(axis_values[left].max()) < minimum_separation:
            raise AssertionError("Spatial split is narrower than its recorded guard")
    details = {
        "axis": axis_name,
        "boundaries": list(boundaries),
        "requested_patch_radius": requested_radius,
        "achieved_guard_radius": achieved_radius,
        "minimum_cross_split_center_separation": minimum_separation,
        "patch_support_disjoint": bool(minimum_separation >= patch_size),
        "split_names": list(split_names),
        "target_fractions": target.tolist(),
        "valid_centres_before_guard": int(len(rows)),
        "valid_centres_by_split": dict(zip(split_names, counts)),
        "dropped_guard_centres": int(len(rows) - assigned),
    }
    return selected, details


def build_all_class_coverage_split(
    field_indices: np.ndarray,
    labels: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    num_classes: int,
    seed: int,
    test_fraction: float,
    patch_size: int = 21,
    val_fraction_of_train: float = 0.20,
    class_names: Mapping[int, str] | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build fixed centre assignments with complete class coverage."""
    field_indices = np.asarray(field_indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if len({len(field_indices), len(labels), len(rows), len(cols)}) != 1:
        raise ValueError("Centre catalogue arrays must have equal lengths")
    owner, exceptions, targets = deterministic_coverage_field_plan(
        field_indices, labels, num_classes, seed, test_fraction,
        val_fraction_of_train,
    )
    split_codes = np.full(len(labels), DROPPED_SPLIT_CODE, dtype=np.uint8)
    for field_index, split_name in owner.items():
        split_codes[field_indices == field_index] = SPLIT_TO_CODE[split_name]

    exception_records: list[Dict[str, Any]] = []
    final_ratios = np.asarray([
        (1.0 - test_fraction) * (1.0 - val_fraction_of_train),
        (1.0 - test_fraction) * val_fraction_of_train,
        test_fraction,
    ], dtype=np.float64)
    for field_index in sorted(exceptions):
        definition = exceptions[field_index]
        catalogue_indices = np.flatnonzero(field_indices == field_index).astype(np.int64)
        if definition["mode"] == "train_val":
            split_names = ("train", "val")
            fractions = (1.0 - val_fraction_of_train, val_fraction_of_train)
        else:
            split_names = SPLIT_NAMES
            fractions = final_ratios
        local_indices, spatial_details = spatially_partition_field(
            rows[catalogue_indices], cols[catalogue_indices], split_names,
            fractions, patch_size,
        )
        split_codes[catalogue_indices] = DROPPED_SPLIT_CODE
        for split_name, selected in zip(split_names, local_indices):
            split_codes[catalogue_indices[selected]] = SPLIT_TO_CODE[split_name]
        exception_records.append({
            "field_index": int(field_index),
            **definition,
            **spatial_details,
        })

    valid_codes = set(np.unique(split_codes).tolist())
    if not valid_codes.issubset({0, 1, 2, int(DROPPED_SPLIT_CODE)}):
        raise AssertionError(f"Unexpected split codes: {sorted(valid_codes)}")
    instances_per_class: Dict[str, Dict[str, int]] = {}
    field_ids_by_split: Dict[str, list[int]] = {}
    coverage: Dict[str, list[int]] = {}
    actual_field_sets: Dict[str, set[int]] = {}
    for split_name in SPLIT_NAMES:
        selected = split_codes == SPLIT_TO_CODE[split_name]
        counts = np.bincount(labels[selected], minlength=num_classes)
        if np.any(counts == 0):
            raise AssertionError(
                f"{split_name} is missing classes {np.flatnonzero(counts == 0).tolist()}"
            )
        instances_per_class[split_name] = {
            str(class_names[index] if class_names else index): int(counts[index])
            for index in range(num_classes)
        }
        coverage[split_name] = np.flatnonzero(counts > 0).astype(int).tolist()
        actual_field_sets[split_name] = set(np.unique(field_indices[selected]).astype(int).tolist())
        field_ids_by_split[split_name] = sorted(actual_field_sets[split_name])

    allowed_overlap = set(exceptions)
    pairwise_overlap: Dict[str, list[int]] = {}
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1:]:
            overlap = actual_field_sets[left] & actual_field_sets[right]
            if not overlap.issubset(allowed_overlap):
                raise AssertionError(f"Unexpected whole-field leakage: {sorted(overlap - allowed_overlap)}")
            pairwise_overlap[f"{left}_{right}"] = sorted(overlap)

    owner_fields = {
        name: sorted(field for field, split_name in owner.items() if split_name == name)
        for name in SPLIT_NAMES
    }
    manifest: Dict[str, Any] = {
        "strategy": COVERAGE_SPLIT_STRATEGY,
        "seed": int(seed),
        "test_fraction": float(test_fraction),
        "val_fraction_of_original_train": float(val_fraction_of_train),
        "final_split_fractions": {
            "train": float((1.0 - test_fraction) * (1.0 - val_fraction_of_train)),
            "val": float((1.0 - test_fraction) * val_fraction_of_train),
            "test": float(test_fraction),
        },
        "patch_size": int(patch_size),
        "fully_field_disjoint": False,
        "warning": (
            "Rare classes with fewer than three logical fields use guarded within-field spatial "
            "splits. All other fields are strictly split-disjoint. Compact exception fields may "
            "not permit a complete patch-support guard; inspect spatial_exceptions."
        ),
        "field_owner_counts": targets,
        "field_owner_ids": owner_fields,
        "field_ids": field_ids_by_split,
        "field_counts": {name: len(fields) for name, fields in field_ids_by_split.items()},
        "spatial_exception_field_indices": sorted(exceptions),
        "spatial_exceptions": exception_records,
        "all_spatial_exception_patch_support_disjoint": bool(all(
            record["patch_support_disjoint"] for record in exception_records
        )),
        "pairwise_field_overlap": pairwise_overlap,
        "dropped_guard_centres": int(np.sum(split_codes == DROPPED_SPLIT_CODE)),
        "valid_center_counts": {
            name: int(np.sum(split_codes == SPLIT_TO_CODE[name])) for name in SPLIT_NAMES
        },
        "instances_per_class": instances_per_class,
        "class_coverage": coverage,
    }
    return split_codes, manifest
