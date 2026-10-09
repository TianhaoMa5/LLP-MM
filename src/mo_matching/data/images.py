import os
import os.path as osp
import pickle, json
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter, OrderedDict
from .augmentation.sampler import RandomSampler, BatchSampler
from .augmentation import tran as T
from .ku_optofil_pbc import (
    build_ku_optofil_loaders,
    build_ku_optofil_eval_loader,
    is_ku_optofil_dataset,
)

label_map = {}
class_mapping = {}
_IMAGE_AUGMENTATION_IMPORT_ERROR = None


def load_mini_imagenet_data(dspth, split="train"):
    if split == "train":
        pkl_file = osp.join(dspth, "miniimagenet/mini-imagenet-cache-train.pkl")
    elif split == "val":
        pkl_file = osp.join(dspth, "miniimagenet/mini-imagenet-cache-val.pkl")
    elif split == "test":
        pkl_file = osp.join(dspth, "miniimagenet/mini-imagenet-cache-test.pkl")
    else:
        raise ValueError("无效的 split 参数，应为 'train', 'val' 或 'test'")
    with open(pkl_file, "rb") as f:
        data_dict = pickle.load(f)
    data = data_dict["image_data"]
    labels = data_dict["class_dict"]
    return (data, labels)


def merge_train_val_test(dspth):
    """
    把 Mini-ImageNet 的 train/val/test 合并；
    支持 labels 为：
      1) dict: {class_name: [sample_indices_in_split, ...]}
      2) list/array: [class_name_or_id_per_sample, ...]
    返回：
      merged_data: 按 train→val→test 拼接后的数据
      merged_labels: 与 merged_data 一一对齐的稳定数值标签（相同输入→相同输出）
    """
    train_data, train_labels = load_mini_imagenet_data(dspth, split="train")
    val_data, val_labels = load_mini_imagenet_data(dspth, split="val")
    test_data, test_labels = load_mini_imagenet_data(dspth, split="test")

    def collect_classes(lbl):
        if isinstance(lbl, dict):
            return {str(k) for k in lbl.keys()}
        else:
            return {
                str(x) for x in (lbl.tolist() if hasattr(lbl, "tolist") else list(lbl))
            }

    all_classes = sorted(
        collect_classes(train_labels)
        | collect_classes(val_labels)
        | collect_classes(test_labels)
    )
    cls2id = {c: i for i, c in enumerate(all_classes)}

    def to_per_sample_labels(data_split, labels_split):
        n = len(data_split)
        if isinstance(labels_split, dict):
            y = np.empty(n, dtype=np.int64)
            for c, idxs in labels_split.items():
                idxs = np.asarray(idxs, dtype=np.int64)
                assert idxs.ndim == 1, f"indices dim error for class {c}"
                assert idxs.size > 0, f"empty indices for class {c}"
                assert idxs.min() >= 0 and idxs.max() < n, (
                    f"indices out of range for split: max={idxs.max()} n={n}"
                )
                y[idxs] = cls2id[str(c)]
            return y
        else:
            arr = (
                labels_split.tolist()
                if hasattr(labels_split, "tolist")
                else list(labels_split)
            )
            assert len(arr) == n, f"labels length {len(arr)} != data length {n}"
            return np.fromiter((cls2id[str(c)] for c in arr), dtype=np.int64, count=n)

    y_train = to_per_sample_labels(train_data, train_labels)
    y_val = to_per_sample_labels(val_data, val_labels)
    y_test = to_per_sample_labels(test_data, test_labels)
    merged_data = np.concatenate([train_data, val_data, test_data], axis=0)
    merged_labels = np.concatenate([y_train, y_val, y_test], axis=0)
    return (merged_data, merged_labels)


class OneCropsTransform:
    def __init__(self, trans_weak):
        self.trans_weak = trans_weak

    def __call__(self, x):
        x1 = self.trans_weak(x)
        return [x1]


def _cluster_ids_for_training_samples(
    clusters,
    *,
    dataset,
    dataset_length,
    shuffled_indices,
    metadata=None,
    map_indices=None,
):
    """Align a label-free, canonical cluster map with the training image rows.

    miniImageNet's archived map covers the merged 600-image class blocks;
    the loader trains on the first 500 rows of each block. Other maps must
    cover the complete, untruncated training dataset in original row order.
    """
    clusters = np.asarray(clusters)
    if clusters.ndim != 1 or not np.issubdtype(clusters.dtype, np.integer):
        raise ValueError("Cluster assignments must be a one-dimensional integer array")
    if np.any(clusters < 0):
        raise ValueError("Cluster assignments must be non-negative")
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, dict):
        raise ValueError("Cluster metadata must be an object")
    if metadata.get("dataset", dataset) != dataset:
        raise ValueError(
            "Cluster metadata dataset does not match the requested dataset"
        )
    if int(metadata.get("N", len(clusters))) != len(clusters):
        raise ValueError("Cluster metadata N does not match the assignment length")
    if map_indices is not None:
        map_indices = np.asarray(map_indices)
        if not np.array_equal(map_indices, np.arange(len(clusters))):
            raise ValueError(
                "Cluster map indices must be in canonical source row order"
            )
    if len(clusters) != dataset_length:
        is_merged_mini = (
            dataset == "miniImageNet"
            and len(clusters) % 600 == 0
            and (len(clusters) // 600 * 500 == dataset_length)
            and (metadata.get("split") == "train+val+test")
        )
        if not is_merged_mini:
            raise ValueError(
                f"Cluster map length {len(clusters)} does not match training length {dataset_length} or a declared miniImageNet merged split"
            )
        clusters = clusters.reshape(-1, 600)[:, :500].reshape(-1)
    elif dataset == "miniImageNet" and metadata.get("split") == "train+val+test":
        raise ValueError(
            "Merged miniImageNet cluster metadata requires the 600-to-500 projection"
        )
    shuffled_indices = np.asarray(shuffled_indices)
    if shuffled_indices.ndim != 1 or not np.issubdtype(
        shuffled_indices.dtype, np.integer
    ):
        raise ValueError("Training row indices must be one-dimensional integers")
    if shuffled_indices.size and (
        shuffled_indices.min() < 0 or shuffled_indices.max() >= dataset_length
    ):
        raise ValueError("Training row index lies outside the cluster map")
    return clusters[shuffled_indices]


def load_data_train(
    pi,
    bag_build,
    num_classes,
    holdout_fraction,
    dataset="CIFAR10",
    dspth="./data",
    bagsize=16,
    backbone=None,
    seed=0,
    cluster_manifest=None,
):
    if cluster_manifest is not None and (
        bag_build != "cluster" or dataset not in {"CIFAR10", "CIFAR100", "miniImageNet"}
    ):
        raise ValueError(
            "Frozen Cluster manifests support the three paper image datasets only"
        )
    cluster_map_digests = []
    input_dim = 1
    if dataset == "CIFAR10":
        datalist = [
            osp.join(dspth, "cifar-10-batches-py", "data_batch_{}".format(i + 1))
            for i in range(5)
        ]
        n_class = 10
    elif dataset == "CIFAR100":
        datalist = [osp.join(dspth, "cifar-100-python", "train")]
        n_class = 100
    elif dataset == "miniImageNet":
        train_data, train_labels = merge_train_val_test(dspth)
        subset_data_list = []
        subset_labels_list = []
        n_class = 100
        for i in range(0, len(train_data), 600):
            chunk_data = train_data[i : i + 600][:500]
            chunk_labels = np.array(train_labels[i : i + 600][:500])
            subset_data_list.append(chunk_data)
            subset_labels_list.append(chunk_labels)
        train_data = np.concatenate(subset_data_list, axis=0)
        train_labels = np.concatenate(subset_labels_list, axis=0)
    else:
        raise ValueError("Unsupported dataset")
    if dataset in ["CIFAR10", "CIFAR100"]:
        data, labels = ([], [])
        for data_batch in datalist:
            with open(data_batch, "rb") as fr:
                entry = pickle.load(fr, encoding="latin1")
                lbs = (
                    entry["labels"]
                    if "labels" in entry.keys()
                    else entry["fine_labels"]
                )
                data.append(entry["data"])
                labels.append(lbs)
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
    elif dataset in ["miniImageNet"]:
        data = train_data
        labels = train_labels
    dataset_length = data.shape[0] if hasattr(data, "shape") else len(data)
    num_bags_all = dataset_length // bagsize
    data_length = num_bags_all * bagsize
    cluster_rng = np.random.default_rng(seed) if bag_build == "cluster" else None
    shuffle_rng = cluster_rng if cluster_rng is not None else np.random
    random_indices = np.arange(data_length)
    shuffle_rng.shuffle(random_indices)
    data = data[random_indices]
    labels = labels[random_indices]
    indices = np.arange(data_length)
    shuffle_rng.shuffle(indices)
    num_bags_all = data_length // bagsize
    num_train_bags = int(num_bags_all * (1 - holdout_fraction))
    num_test_bags = num_bags_all - num_train_bags

    def build_bags(bag_id_list):
        data_u, label_prob = ([], [])
        labels_real, labels_idx = ([], [])
        indices_u = []
        for new_j, j in enumerate(bag_id_list):
            bag_indices = indices[j * bagsize : (j + 1) * bagsize]
            if dataset in ["miniImageNet"]:
                bag_data = [data[i] for i in bag_indices]
            else:
                bag_data = [
                    data[i].reshape(3, 32, 32).transpose(1, 2, 0) for i in bag_indices
                ]
            bag_labels = np.array([labels[i] for i in bag_indices])
            labels_real.append(bag_labels)
            labels_idx.append(bag_indices)
            label_counts = Counter(bag_labels)
            label_counts = OrderedDict(sorted(label_counts.items()))
            label_proportions = [
                label_counts.get(label, 0) / len(bag_labels)
                for label in range(0, num_classes)
            ]
            data_u.append(bag_data)
            label_prob.append(label_proportions)
            indices_u.append(new_j)
        return (data_u, label_prob, labels_real, labels_idx, indices_u)

    bag_ids = np.arange(num_bags_all)
    shuffle_rng.shuffle(bag_ids)
    var_bag_ids_1 = bag_ids[:num_train_bags]
    var_bag_ids_2 = bag_ids[num_train_bags:]

    def build_bags_cluster(
        bag_id_list,
        *,
        data,
        labels,
        indices,
        dataset,
        num_classes,
        bagsize,
        dspth,
        alpha0=1.0,
        rng=None,
    ):
        """
        按 cluster_map 构造 bag：每个 bag 从多个簇里按 Dirichlet 混合比例取样（严格无放回）。
        返回：data_u, label_prob, labels_real, labels_idx, indices_u
        """
        if rng is None:
            rng = np.random.default_rng(seed)
        B = len(bag_id_list)
        m = bagsize
        need_total = B * m
        chosen = []
        for j in bag_id_list:
            blk = indices[j * m : (j + 1) * m]
            chosen.append(np.asarray(blk, dtype=np.int64))
        chosen = np.concatenate(chosen, axis=0)
        assert len(chosen) == need_total
        local2global = chosen.copy()

        def _find_cluster_npz(_dspth, _dataset):
            base = osp.join(_dspth, "cluster_maps")
            if _dataset == "CIFAR100":
                candidates = [256, 128, 64]
            else:
                candidates = [32, 64, 256]
            for K in candidates:
                p = osp.join(base, f"{_dataset}_train_K{K}.npz")
                if osp.exists(p):
                    return p
            bundled = osp.join(osp.dirname(__file__), "cluster_maps")
            for K in candidates:
                p = osp.join(bundled, f"{_dataset}_train_K{K}.npz")
                if osp.exists(p):
                    return p
            raise FileNotFoundError(
                f"Cannot find cluster map for {_dataset} under {base}"
            )

        cluster_map_path = _find_cluster_npz(dspth, dataset)
        if cluster_manifest is not None:
            import hashlib
            from pathlib import Path

            cluster_map_digests.append(
                hashlib.sha256(Path(cluster_map_path).read_bytes()).hexdigest()
            )
        with np.load(cluster_map_path, allow_pickle=False) as z:
            clusters_all = z["clusters"]
            cluster_metadata = json.loads(z["meta"].item()) if "meta" in z else None
            cluster_map_indices = z["indices"] if "indices" in z else None
            if "labels" in z:
                map_labels = z["labels"]
                if dataset == "miniImageNet" and len(map_labels) != dataset_length:
                    map_labels = map_labels.reshape(-1, 600)[:, :500].reshape(-1)
                if len(map_labels) != dataset_length or not np.array_equal(
                    map_labels[random_indices], labels
                ):
                    raise ValueError("Cluster map/image label alignment failed")
        aligned_clusters = _cluster_ids_for_training_samples(
            clusters_all,
            dataset=dataset,
            dataset_length=dataset_length,
            shuffled_indices=random_indices,
            metadata=cluster_metadata,
            map_indices=cluster_map_indices,
        )
        clusters = aligned_clusters[local2global]
        pools = {}
        for local_i, c in enumerate(clusters):
            pools.setdefault(int(c), []).append(local_i)
        for c in list(pools.keys()):
            arr = np.array(pools[c], dtype=np.int64)
            rng.shuffle(arr)
            pools[c] = arr
        cluster_ids_all = np.array(sorted(pools.keys()), dtype=np.int64)
        K = len(cluster_ids_all)
        cluster_sizes = np.array(
            [len(pools[int(c)]) for c in cluster_ids_all], dtype=np.int64
        )
        assert cluster_sizes.sum() == B * m
        pi = cluster_sizes.astype(np.float64)
        pi = pi / pi.sum()

        def _sample_counts_matrix_dirichlet(B, m, alpha0, pi, rng):
            dir_param = np.maximum(alpha0 * pi, 1e-12)
            W = rng.dirichlet(dir_param, size=B)
            F = W * m
            C0 = np.floor(F).astype(np.int64)
            rem = (m - C0.sum(axis=1)).astype(np.int64)
            frac = F - np.floor(F)
            for b in range(B):
                r = int(rem[b])
                if r <= 0:
                    continue
                p = frac[b].copy()
                ssum = p.sum()
                if ssum <= 0:
                    p = W[b].copy()
                    p = p / p.sum()
                else:
                    p = p / ssum
                add = rng.choice(np.arange(K), size=r, replace=True, p=p)
                for k in add:
                    C0[b, int(k)] += 1
            assert np.all(C0.sum(axis=1) == m)
            return C0

        def _balance_columns_to_targets(C, targets, rng):
            C = C.copy()
            cur = C.sum(axis=0)
            diff = targets - cur
            deficit = np.where(diff > 0)[0].tolist()
            surplus = np.where(diff < 0)[0].tolist()
            max_moves = int(diff[diff > 0].sum())
            moves = 0
            while deficit:
                d = deficit[-1]
                if not surplus:
                    raise RuntimeError(
                        "Balancing failed: no surplus but still deficit."
                    )
                s = surplus[-1]
                candidates = np.where(C[:, s] > 0)[0]
                if len(candidates) == 0:
                    surplus.pop()
                    continue
                b = int(rng.choice(candidates))
                C[b, s] -= 1
                C[b, d] += 1
                diff[s] += 1
                diff[d] -= 1
                moves += 1
                if moves > max_moves + 10 * K:
                    raise RuntimeError("Balancing stuck; check inputs.")
                if diff[d] == 0:
                    deficit.pop()
                if diff[s] == 0:
                    surplus.pop()
            assert np.all(C.sum(axis=0) == targets)
            assert np.all(C.sum(axis=1) == m)
            return C

        C0 = _sample_counts_matrix_dirichlet(B, m, alpha0, pi, rng)
        C = _balance_columns_to_targets(C0, cluster_sizes, rng)
        ptr = {int(c): 0 for c in cluster_ids_all}
        data_u, label_prob = ([], [])
        labels_real, labels_idx = ([], [])
        indices_u = []
        for new_j in range(B):
            bag_local_list = []
            for k, cid in enumerate(cluster_ids_all):
                need = int(C[new_j, k])
                if need <= 0:
                    continue
                cid = int(cid)
                s0 = ptr[cid]
                e0 = s0 + need
                chosen_local = pools[cid][s0:e0]
                ptr[cid] = e0
                bag_local_list.extend(chosen_local.tolist())
            bag_local = np.array(bag_local_list, dtype=np.int64)
            rng.shuffle(bag_local)
            bag_indices = local2global[bag_local]
            if dataset in ["miniImageNet"]:
                bag_data = [data[i] for i in bag_indices]
            else:
                bag_data = [
                    data[i].reshape(3, 32, 32).transpose(1, 2, 0) for i in bag_indices
                ]
            bag_labels = np.array([labels[i] for i in bag_indices])
            labels_real.append(bag_labels)
            labels_idx.append(bag_indices)
            label_counts = Counter(bag_labels)
            label_counts = OrderedDict(sorted(label_counts.items()))
            label_proportions = [
                label_counts.get(label, 0) / len(bag_labels)
                for label in range(num_classes)
            ]
            data_u.append(bag_data)
            label_prob.append(label_proportions)
            indices_u.append(new_j)
        return (data_u, label_prob, labels_real, labels_idx, indices_u)

    def build_bags_alphafirst(
        bag_id_list,
        *,
        data,
        labels,
        indices,
        dataset,
        num_classes,
        bagsize,
        alpha0=10.0,
        seed=0,
    ):
        """
        AlphaFirst（按类别池 + Dirichlet + 列和配平 + 严格无放回）
        - 只使用 bag_id_list 对应的那 B*m 个样本（不会去动别的样本）
        - 返回接口与 build_bags / build_bags_cluster 一致：
          return data_u, label_prob, labels_real, labels_idx, indices_u
        """
        rng = np.random.default_rng(seed)
        labels = np.asarray(labels)
        if labels.ndim != 1:
            raise ValueError("AlphaFirst 仅支持单标签 labels 为一维类别ID。")
        B = len(bag_id_list)
        m = bagsize
        need_total = B * m
        chosen = []
        for j in bag_id_list:
            blk = indices[j * m : (j + 1) * m]
            chosen.append(np.asarray(blk, dtype=np.int64))
        chosen = np.concatenate(chosen, axis=0)
        if len(chosen) != need_total:
            raise ValueError(f"chosen size {len(chosen)} != B*m {need_total}")
        local2global = chosen.copy()
        sub_labels = labels[local2global].astype(np.int64)
        if sub_labels.min() < 0 or sub_labels.max() >= num_classes:
            raise ValueError("labels value out of range; check num_classes")
        pools = {}
        for c in range(num_classes):
            idx = np.where(sub_labels == c)[0].astype(np.int64)
            rng.shuffle(idx)
            pools[c] = idx
        class_sizes = np.array(
            [len(pools[c]) for c in range(num_classes)], dtype=np.int64
        )
        if class_sizes.sum() != need_total:
            raise RuntimeError("Internal error: class_sizes.sum != B*m")

        def _sample_counts_matrix(B, m, Cn, alpha0, rng):
            dir_param = np.maximum(alpha0 * np.ones(Cn, dtype=np.float64), 1e-12)
            W = rng.dirichlet(dir_param, size=B)
            F = W * m
            C = np.floor(F).astype(np.int64)
            rem = (m - C.sum(axis=1)).astype(np.int64)
            frac = F - np.floor(F)
            for b in range(B):
                r = int(rem[b])
                if r <= 0:
                    continue
                p = frac[b].copy()
                ssum = p.sum()
                if ssum <= 0:
                    p = W[b].copy()
                    p = p / p.sum()
                else:
                    p = p / ssum
                add = rng.choice(np.arange(Cn), size=r, replace=True, p=p)
                for k in add:
                    C[b, int(k)] += 1
            assert np.all(C.sum(axis=1) == m)
            return C

        def _balance_columns_to_targets(C, targets, rng):
            C = C.copy()
            cur = C.sum(axis=0)
            diff = targets - cur
            deficit = np.where(diff > 0)[0].tolist()
            surplus = np.where(diff < 0)[0].tolist()
            need_moves = int(diff[diff > 0].sum())
            moves = 0
            while deficit:
                d = deficit[-1]
                if not surplus:
                    raise RuntimeError(
                        "Balancing failed: no surplus but still deficit."
                    )
                s = surplus[-1]
                candidates = np.where(C[:, s] > 0)[0]
                if len(candidates) == 0:
                    surplus.pop()
                    continue
                b = int(rng.choice(candidates))
                C[b, s] -= 1
                C[b, d] += 1
                diff[s] += 1
                diff[d] -= 1
                moves += 1
                if moves > need_moves + 10 * C.shape[1]:
                    raise RuntimeError(
                        "Balancing seems stuck; please check data/params."
                    )
                if diff[d] == 0:
                    deficit.pop()
                if diff[s] == 0:
                    surplus.pop()
            assert np.all(C.sum(axis=0) == targets)
            assert np.all(C.sum(axis=1) == m)
            return C

        C0 = _sample_counts_matrix(B, m, num_classes, alpha0, rng)
        C = _balance_columns_to_targets(C0, class_sizes, rng)
        ptr = {c: 0 for c in range(num_classes)}
        data_u, label_prob = ([], [])
        labels_real, labels_idx = ([], [])
        indices_u = []
        for new_j in range(B):
            bag_local_list = []
            for c in range(num_classes):
                need = int(C[new_j, c])
                if need <= 0:
                    continue
                s0 = ptr[c]
                e0 = s0 + need
                chosen_local = pools[c][s0:e0]
                ptr[c] = e0
                bag_local_list.extend(chosen_local.tolist())
            bag_local = np.array(bag_local_list, dtype=np.int64)
            rng.shuffle(bag_local)
            bag_indices = local2global[bag_local]
            if dataset in ["miniImageNet"]:
                bag_data = [data[i] for i in bag_indices]
            else:
                bag_data = [
                    data[i].reshape(3, 32, 32).transpose(1, 2, 0) for i in bag_indices
                ]
            bag_labels = sub_labels[bag_local]
            labels_real.append(np.asarray(bag_labels))
            labels_idx.append(np.asarray(bag_indices))
            label_counts = Counter(bag_labels.tolist())
            label_counts = OrderedDict(sorted(label_counts.items()))
            label_proportions = [
                label_counts.get(k, 0) / len(bag_labels) for k in range(num_classes)
            ]
            data_u.append(bag_data)
            label_prob.append(label_proportions)
            indices_u.append(new_j)
        return (data_u, label_prob, labels_real, labels_idx, indices_u)

    _empty = ([], [], [], [], [])
    if bag_build == "random":
        (
            var_data_u_1,
            var_label_prob_1,
            var_labels_real_1,
            var_labels_idx_1,
            var_indices_u_1,
        ) = build_bags(var_bag_ids_1)
        (
            var_data_u_2,
            var_label_prob_2,
            var_labels_real_2,
            var_labels_idx_2,
            var_indices_u_2,
        ) = build_bags(var_bag_ids_2) if len(var_bag_ids_2) > 0 else _empty
    elif bag_build == "cluster":
        (
            var_data_u_1,
            var_label_prob_1,
            var_labels_real_1,
            var_labels_idx_1,
            var_indices_u_1,
        ) = build_bags_cluster(
            var_bag_ids_1,
            data=data,
            labels=labels,
            indices=indices,
            dataset=dataset,
            num_classes=num_classes,
            bagsize=bagsize,
            dspth=dspth,
            alpha0=pi,
            rng=cluster_rng,
        )
        (
            var_data_u_2,
            var_label_prob_2,
            var_labels_real_2,
            var_labels_idx_2,
            var_indices_u_2,
        ) = (
            build_bags_cluster(
                var_bag_ids_2,
                data=data,
                labels=labels,
                indices=indices,
                dataset=dataset,
                num_classes=num_classes,
                bagsize=bagsize,
                dspth=dspth,
                alpha0=pi,
                rng=cluster_rng,
            )
            if len(var_bag_ids_2) > 0
            else _empty
        )
    elif bag_build == "alphafirst":
        (
            var_data_u_1,
            var_label_prob_1,
            var_labels_real_1,
            var_labels_idx_1,
            var_indices_u_1,
        ) = build_bags_alphafirst(
            var_bag_ids_1,
            data=data,
            labels=labels,
            indices=indices,
            dataset=dataset,
            num_classes=num_classes,
            bagsize=bagsize,
            alpha0=pi,
            seed=0,
        )
        (
            var_data_u_2,
            var_label_prob_2,
            var_labels_real_2,
            var_labels_idx_2,
            var_indices_u_2,
        ) = (
            build_bags_alphafirst(
                var_bag_ids_2,
                data=data,
                labels=labels,
                indices=indices,
                dataset=dataset,
                num_classes=num_classes,
                bagsize=bagsize,
                alpha0=pi,
                seed=1,
            )
            if len(var_bag_ids_2) > 0
            else _empty
        )
    if dataset in ["CIFAR10", "CIFAR100"]:
        input_dim = (3, 32, 32)
    elif dataset in ["miniImageNet"]:
        input_dim = (3, 64, 64)
    if cluster_manifest is not None:
        from .cluster_manifest import freeze_cluster_manifest

        def source_bags(bags):
            compact = random_indices[
                np.asarray(bags, dtype=np.int64).reshape(-1, bagsize)
            ]
            return (
                compact // 500 * 600 + compact % 500
                if dataset == "miniImageNet"
                else compact
            )

        freeze_cluster_manifest(
            cluster_manifest,
            metadata={
                "schema_version": 1,
                "dataset": dataset,
                "seed": int(seed),
                "pi": float(pi),
                "bag_size": int(bagsize),
                "holdout_fraction": float(holdout_fraction),
                "num_classes": int(num_classes),
                "training_population": int(dataset_length),
                "cluster_map_sha256": sorted(set(cluster_map_digests)),
                "source_index_space": "merged_600_blocks"
                if dataset == "miniImageNet"
                else "original_train_rows",
            },
            train_indices=source_bags(var_labels_idx_1),
            val_indices=source_bags(var_labels_idx_2),
            train_proportions=np.asarray(var_label_prob_1).reshape(-1, num_classes),
            val_proportions=np.asarray(var_label_prob_2).reshape(-1, num_classes),
        )
    var_pack_1 = (
        var_data_u_1,
        var_label_prob_1,
        var_labels_real_1,
        var_labels_idx_1,
        dataset_length,
        var_indices_u_1,
        input_dim,
    )
    if holdout_fraction == 0.0:
        var_pack_2 = var_pack_1
    else:
        var_pack_2 = (
            var_data_u_2,
            var_label_prob_2,
            var_labels_real_2,
            var_labels_idx_2,
            dataset_length,
            var_indices_u_2,
            input_dim,
        )
    prior_train = (
        np.mean(np.asarray(var_label_prob_1, dtype=np.float32), axis=0)
        if len(var_label_prob_1) > 0
        else np.zeros((num_classes,), dtype=np.float32)
    )
    prior_val = (
        np.mean(np.asarray(var_label_prob_2, dtype=np.float32), axis=0)
        if len(var_label_prob_2) > 0
        else np.zeros((num_classes,), dtype=np.float32)
    )
    all_probs = np.asarray(var_label_prob_1 + var_label_prob_2, dtype=np.float32)
    prior_all = (
        np.mean(all_probs, axis=0)
        if all_probs.shape[0] > 0
        else np.zeros((num_classes,), dtype=np.float32)
    )
    return (var_pack_1, var_pack_2, prior_train, prior_val, prior_all)


def load_data_val(dataset, dspth="./data", n_classes=10, backbone=None):
    if dataset == "CIFAR10":
        datalist = [osp.join(dspth, "cifar-10-batches-py", "test_batch")]
    elif dataset == "CIFAR100":
        datalist = [osp.join(dspth, "cifar-100-python", "test")]
    elif dataset == "miniImageNet":
        train_data, train_labels = merge_train_val_test(dspth)
        test_data_list = []
        test_labels_list = []
        n_class = 100
        for i in range(0, len(train_data), 600):
            chunk_data = train_data[i : i + 600][-100:]
            chunk_labels = np.array(train_labels[i : i + 600][-100:])
            test_data_list.append(chunk_data)
            test_labels_list.append(chunk_labels)
        data = np.concatenate(test_data_list, axis=0)
        labels = np.concatenate(test_labels_list, axis=0)
    if dataset == "CIFAR10" or dataset == "CIFAR100":
        data, labels = ([], [])
        for data_batch in datalist:
            with open(data_batch, "rb") as fr:
                entry = pickle.load(fr, encoding="latin1")
                lbs = (
                    entry["labels"]
                    if "labels" in entry.keys()
                    else entry["fine_labels"]
                )
                data.append(entry["data"])
                labels.append(lbs)
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        data = [el.reshape(3, 32, 32).transpose(1, 2, 0) for el in data]
    return (data, labels)


def get_train_loader(
    pi,
    bag_build,
    classes,
    holdout_fraction,
    dataset,
    batch_size,
    bag_size,
    root="data",
    method="co",
    supervised=False,
    backbone=None,
    seed=0,
    num_bags=None,
    num_workers=0,
    instances_per_epoch=200000,
    train_instance_sample_size=None,
    num_reviewers=None,
    cluster_seed=0,
    target_avg_bag_size=None,
    ku_merge_validation_into_train=False,
    ku_unknown_bag_max_size=None,
    ku_unknown_bag_seed=0,
    cluster_manifest=None,
):
    if is_ku_optofil_dataset(dataset):
        if bag_build != "random":
            raise ValueError(
                "KU-Optofil bags are natural patients; bag_build must remain 'random' and does not construct random bags"
            )
        if num_bags is not None:
            raise ValueError(
                "KU-Optofil does not support truncating the patient population"
            )
        train_loader, val_loader, bundle, input_shape = build_ku_optofil_loaders(
            root=root,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
            train_instance_sample_size=train_instance_sample_size,
            paired_views=False,
            merge_validation_into_train=ku_merge_validation_into_train,
            unknown_bag_max_size=ku_unknown_bag_max_size,
            unknown_bag_seed=ku_unknown_bag_seed,
        )
        if classes is not None and int(classes) != bundle.num_classes:
            raise ValueError(
                f"--n-classes={classes} disagrees with KU-Optofil ({bundle.num_classes})"
            )
        train_probs = np.asarray(train_loader.dataset.label_prob, dtype=np.float32)
        train_sizes = np.asarray(
            [len(bag.indices) for bag in train_loader.dataset.bags], dtype=np.float64
        )
        prior_train = np.average(train_probs, axis=0, weights=train_sizes)
        if val_loader is None:
            val_sizes = np.empty(0, dtype=np.float64)
            prior_val = prior_train.copy()
            prior_all = prior_train.copy()
        else:
            val_probs = np.asarray(val_loader.dataset.label_prob, dtype=np.float32)
            val_sizes = np.asarray(
                [len(bag.indices) for bag in val_loader.dataset.bags], dtype=np.float64
            )
            prior_val = np.average(val_probs, axis=0, weights=val_sizes)
            prior_all = np.average(
                np.concatenate([train_probs, val_probs], axis=0),
                axis=0,
                weights=np.concatenate([train_sizes, val_sizes]),
            )
        return (
            train_loader,
            val_loader,
            train_loader.dataset.label_prob,
            int(train_sizes.sum()),
            int(val_sizes.sum()),
            input_shape,
            prior_train,
            prior_val,
            prior_all,
        )
    train_pack, var_pack, prior_train, prior_val, prior_all = load_data_train(
        pi,
        bag_build,
        classes,
        holdout_fraction,
        dataset=dataset,
        dspth=root,
        bagsize=bag_size,
        backbone=backbone,
        seed=seed,
        cluster_manifest=cluster_manifest,
    )
    data_u, label_prob, labels, label_idx, dataset_length, indices_u, input_dim = (
        train_pack
    )
    (
        data_u_var,
        label_prob_var,
        labels_var,
        label_idx_var,
        dataset_length_var,
        indices_u_var,
        input_dim_var,
    ) = var_pack
    ds_u = Cifar(
        dataset=dataset,
        data=data_u,
        labels=label_prob,
        labels_real=labels,
        labels_idx=label_idx,
        indices_u=indices_u,
        mode="train_u_%s" % method,
    )
    sampler_u = RandomSampler(ds_u, replacement=False)
    batch_sampler_u = BatchSampler(sampler_u, batch_size, drop_last=True)
    dl_u = torch.utils.data.DataLoader(
        ds_u, batch_sampler=batch_sampler_u, num_workers=num_workers, pin_memory=True
    )
    ds_u_var = Cifar(
        dataset=dataset,
        data=data_u_var,
        labels=label_prob_var,
        labels_real=labels_var,
        labels_idx=label_idx_var,
        indices_u=indices_u_var,
        mode="train_x",
    )
    sampler_u_var = RandomSampler(ds_u_var, replacement=False)
    batch_sampler_u_var = BatchSampler(sampler_u_var, batch_size, drop_last=True)
    dl_u_var = torch.utils.data.DataLoader(
        ds_u_var,
        batch_sampler=batch_sampler_u_var,
        num_workers=num_workers,
        pin_memory=True,
    )
    return (
        dl_u,
        dl_u_var,
        label_prob,
        dataset_length,
        dataset_length_var,
        input_dim,
        prior_train,
        prior_val,
        prior_all,
    )


def get_val_loader(
    dataset,
    batch_size,
    num_workers,
    pin_memory=True,
    root="data",
    n_classes=10,
    backbone=None,
    seed=0,
):
    if is_ku_optofil_dataset(dataset):
        return build_ku_optofil_eval_loader(
            root=root, split="test", batch_size=batch_size, num_workers=num_workers
        )
    data, labels = load_data_val(
        dataset, dspth=root, n_classes=n_classes, backbone=backbone
    )
    ds = Cifar2(dataset=dataset, data=data, labels=labels, mode="test")
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dl


class Cifar(Dataset):
    def __init__(self, dataset, data, labels, labels_real, labels_idx, indices_u, mode):
        super(Cifar, self).__init__()
        self.data, self.labels, self.labels_real, self.labels_idx, self.indices_u = (
            data,
            labels,
            labels_real,
            labels_idx,
            indices_u,
        )
        self.mode = mode
        assert len(self.data) == len(self.labels)
        if dataset == "CIFAR10":
            mean, std = ((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        elif dataset == "CIFAR100":
            mean, std = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        elif dataset == "miniImageNet":
            mean, std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if dataset == "CIFAR10" or dataset == "CIFAR100":
            trans_weak = T.Compose(
                [
                    T.Resize((32, 32)),
                    T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.Normalize(mean, std),
                    T.ToTensor(),
                ]
            )
            trans_weak_noaug = T.Compose(
                [T.Resize((32, 32)), T.Normalize(mean, std), T.ToTensor()]
            )
        elif dataset in ["miniImageNet"]:
            trans_weak = T.Compose(
                [
                    T.Resize((64, 64)),
                    T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.Normalize(mean, std),
                    T.ToTensor(),
                ]
            )
            trans_weak_noaug = T.Compose(
                [T.Resize((64, 64)), T.Normalize(mean, std), T.ToTensor()]
            )
        if self.mode == "train_x":
            self.trans = OneCropsTransform(trans_weak_noaug)
        elif self.mode == "train_u_DLLP":
            self.trans = OneCropsTransform(trans_weak)
        elif dataset in ["CIFAR10", "CIFAR100"]:
            self.trans = T.Compose(
                [T.Resize((32, 32)), T.Normalize(mean, std), T.ToTensor()]
            )
        else:
            self.trans = T.Compose(
                [T.Resize((64, 64)), T.Normalize(mean, std), T.ToTensor()]
            )

    def __getitem__(self, idx):
        ims, lb_prob, lb_idx, indices_u = (
            self.data[idx],
            self.labels[idx],
            self.labels_idx[idx],
            self.indices_u[idx],
        )
        labels = self.labels_real[idx]
        if self.mode == "train_u_DLLP":
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            ims_transformed = [x_weak]
            return (ims_transformed, lb_prob, lb_idx, indices_u, labels)
        elif self.mode == "train_x":
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            ims_transformed = [x_weak]
            return (ims_transformed, lb_prob, lb_idx, indices_u, labels)

    def __len__(self):
        leng = len(self.data)
        return leng


class Cifar2(Dataset):
    def __init__(self, dataset, data, labels, mode):
        super(Cifar2, self).__init__()
        self.data, self.labels = (data, labels)
        self.mode = mode
        assert len(self.data) == len(self.labels)
        if dataset == "CIFAR10":
            mean, std = ((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        elif dataset == "CIFAR100":
            mean, std = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        elif dataset == "miniImageNet":
            mean, std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if dataset == "CIFAR10" or dataset == "CIFAR100":
            trans_weak = T.Compose(
                [
                    T.Resize((32, 32)),
                    T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.Normalize(mean, std),
                    T.ToTensor(),
                ]
            )
        elif dataset in ["miniImageNet"]:
            trans_weak = T.Compose(
                [
                    T.Resize((64, 64)),
                    T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.Normalize(mean, std),
                    T.ToTensor(),
                ]
            )
        if self.mode == "train_x":
            self.trans = trans_weak
        elif dataset in ["CIFAR10", "CIFAR100"]:
            self.trans = T.Compose(
                [T.Resize((32, 32)), T.Normalize(mean, std), T.ToTensor()]
            )
        else:
            self.trans = T.Compose(
                [T.Resize((64, 64)), T.Normalize(mean, std), T.ToTensor()]
            )

    def __getitem__(self, idx):
        im, lb = (self.data[idx], self.labels[idx])
        return (self.trans(im), lb)

    def __len__(self):
        leng = len(self.data)
        return leng
