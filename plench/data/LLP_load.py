import os.path as osp
import pickle
import json
import scipy.io as sio

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .augmentation.sampler import RandomSampler, BatchSampler
try:
    from .augmentation import tran as T
    from .augmentation.rand import RandomAugment
    from .augmentation import transform as T1
    from .augmentation.randaugment_grey import RandomAugment as RandomAugment1
    _IMAGE_AUGMENTATION_IMPORT_ERROR = None
except ImportError as exc:
    # CV/LEM and text datasets do not use the legacy OpenCV augmentation stack.
    # Keep those paths runnable in minimal environments.
    T = T1 = RandomAugment = RandomAugment1 = None
    _IMAGE_AUGMENTATION_IMPORT_ERROR = exc
import os
from PIL import Image

import numpy as np
from collections import Counter, OrderedDict
from .remote_sensing import (
    build_remote_sensing_eval_loader,
    build_remote_sensing_loaders,
    is_remote_sensing_dataset,
)
from .twitter_ethnicity_2017 import (
    build_twitter_ethnicity_loaders,
    build_twitter_evaluation_loader,
    is_twitter_ethnicity_dataset,
)
from .ref2021 import (
    build_ref2021_eval_loader,
    build_ref2021_loaders,
    is_ref2021_dataset,
)
from .ku_optofil_pbc import (
    build_ku_optofil_eval_loader,
    build_ku_optofil_loaders,
    is_ku_optofil_dataset,
)
from .amazon_wilds import (
    build_amazon_wilds_eval_loader,
    build_amazon_wilds_loaders,
    is_amazon_wilds_dataset,
)
from .cct import (
    build_cct_eval_loader,
    build_cct_loaders,
    is_cct_dataset,
)
from .fed_isic2019 import (
    build_fed_isic2019_eval_loader,
    build_fed_isic2019_loaders,
    is_fed_isic2019_dataset,
)
label_map = {}
class_mapping = {}

# ─────────────────────────────────────────────────────────────────────────────
# Text dataset support  (AGNEWS | Yelp | Yahoo | 20News)
# ─────────────────────────────────────────────────────────────────────────────

TEXT_DATASETS = {"AGNEWS", "Yelp", "Yahoo", "20News"}

# BERT model key → HuggingFace model name
BERT_MODELS = {
    "BERT_base":  "bert-base-uncased",
    "BERT_small": "prajjwal1/bert-small",
}

# TF-IDF vectorizer cache: keyed by (dataset, dspth, max_features)
_VEC_CACHE: dict = {}

# BERT tokenizer cache: keyed by model_key
_TOK_CACHE: dict = {}


def _get_bert_tokenizer(model_key: str):
    if model_key not in _TOK_CACHE:
        from transformers import BertTokenizerFast
        _TOK_CACHE[model_key] = BertTokenizerFast.from_pretrained(BERT_MODELS[model_key])
    return _TOK_CACHE[model_key]


def _tokenize_texts(texts: list, model_key: str, max_length: int = 128) -> np.ndarray:
    """Return int64 numpy array of shape (N, max_length) — padded token IDs."""
    tok = _get_bert_tokenizer(model_key)
    enc = tok(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    return enc["input_ids"].astype(np.int64)   # (N, max_length)


def _get_vectorizer(dataset: str, dspth: str, max_features: int = 10000):
    """Return (and lazily fit) the TF-IDF vectorizer for this dataset."""
    key = (dataset, dspth, max_features)
    if key not in _VEC_CACHE:
        from sklearn.feature_extraction.text import TfidfVectorizer
        train_texts, _, _ = _load_raw_text(dataset, dspth, split="train")
        vec = TfidfVectorizer(
            max_features=max_features,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{1,}",
            ngram_range=(1, 2),
        )
        vec.fit(train_texts)
        _VEC_CACHE[key] = vec
    return _VEC_CACHE[key]


def _load_raw_text(dataset: str, dspth: str, split: str):
    """Return (texts: list[str], labels: np.ndarray[int64], n_class: int)."""
    if dataset == "AGNEWS":
        return _load_agnews(dspth, split)
    elif dataset == "Yelp":
        return _load_yelp(dspth, split)
    elif dataset == "Yahoo":
        return _load_yahoo(dspth, split)
    elif dataset == "20News":
        return _load_20news(dspth, split)
    raise ValueError(f"Unknown text dataset: {dataset!r}")


def _load_agnews(dspth: str, split: str):
    import pandas as pd
    fname = "train.csv" if split == "train" else "test.csv"
    # Server format: CSV with header row, columns: label(1-4), content, ...
    df = pd.read_csv(osp.join(dspth, "ag_news", fname))
    texts  = df["content"].fillna("").tolist()
    labels = (df["label"] - 1).to_numpy(dtype=np.int64)   # 1-4 → 0-3
    return texts, labels, 4


def _load_agnews_aug(dspth: str):
    """Return (original, back_translation, synonym_aug, labels) for train split.
    Used by BERT path to give three distinct text views per sample."""
    import pandas as pd
    df = pd.read_csv(osp.join(dspth, "ag_news", "train.csv"))
    original  = df["content"].fillna("").tolist()
    aug_bt    = df.get("back_translation",  df["content"]).fillna(df["content"]).tolist()
    aug_syn   = df.get("synonym_aug",       df["content"]).fillna(df["content"]).tolist()
    labels    = (df["label"] - 1).to_numpy(dtype=np.int64)
    return original, aug_bt, aug_syn, labels, 4


def _load_yelp(dspth: str, split: str):
    import json
    # Server format: JSON lines, {"label": 0-4, "text": "..."}
    fname = "yelp_train.json" if split == "train" else "yelp_test.json"
    fpath = osp.join(dspth, "yelp", fname)
    texts, labels = [], []
    with open(fpath) as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(int(obj["label"]))   # already 0-indexed
    return texts, np.array(labels, dtype=np.int64), 5


def _load_yahoo(dspth: str, split: str):
    import json
    # Server format: JSON lines, {"id": N, "topic": 0-9, "question_title": ...,
    #                              "question_content": ..., "best_answer": ...}
    fname = "yahoo_train.json" if split == "train" else "yahoo_test.json"
    fpath = osp.join(dspth, "yahoo", fname)
    texts, labels = [], []
    with open(fpath) as f:
        for line in f:
            obj = json.loads(line)
            t = " ".join(filter(None, [
                obj.get("question_title", ""),
                obj.get("question_content", ""),
                obj.get("best_answer", ""),
            ]))
            texts.append(t)
            labels.append(int(obj["topic"]))   # already 0-indexed
    return texts, np.array(labels, dtype=np.int64), 10


def _load_20news(dspth: str, split: str):
    # Server format: directory tree  20news_bydate/20news-bydate-{train,test}/<group>/<file>
    subdir = "20news-bydate-train" if split == "train" else "20news-bydate-test"
    root = osp.join(dspth, "20news_bydate", subdir)
    groups = sorted(d for d in os.listdir(root) if osp.isdir(osp.join(root, d)))
    label_to_idx = {g: i for i, g in enumerate(groups)}
    texts, labels = [], []
    for group in groups:
        gdir = osp.join(root, group)
        for fname in os.listdir(gdir):
            fpath = osp.join(gdir, fname)
            if not osp.isfile(fpath):
                continue
            try:
                with open(fpath, encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read())
            except Exception:
                texts.append("")
            labels.append(label_to_idx[group])
    return texts, np.array(labels, dtype=np.int64), 20


# ── Feature-space text augmentation ──────────────────────────────────────────
# Applied per-sample (shape (D,) float32) inside Dataset.__getitem__.
# Mirrors the weak / strong0 / strong1 philosophy of the image pipeline.

def _text_aug_weak(x: np.ndarray) -> np.ndarray:
    """10 % feature dropout + tiny Gaussian noise."""
    mask  = (np.random.rand(len(x)) > 0.10).astype(np.float32)
    noise = np.random.randn(len(x)).astype(np.float32) * 0.005
    return np.clip(x * mask + noise, 0.0, None)


def _text_aug_strong0(x: np.ndarray) -> np.ndarray:
    """30 % feature dropout + larger noise (RandAugment analogue)."""
    mask  = (np.random.rand(len(x)) > 0.30).astype(np.float32)
    noise = np.random.randn(len(x)).astype(np.float32) * 0.02
    return np.clip(x * mask + noise, 0.0, None)


def _text_aug_strong1(x: np.ndarray) -> np.ndarray:
    """Contiguous 30 % block masking + noise (SimCLR-style distortion)."""
    out   = x.copy()
    n     = len(out)
    block = max(1, int(n * 0.30))
    start = np.random.randint(0, max(1, n - block))
    out[start:start + block] = 0.0
    out  += np.random.randn(n).astype(np.float32) * 0.02
    return np.clip(out, 0.0, None)

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
    return data, labels


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
    val_data, val_labels     = load_mini_imagenet_data(dspth, split="val")
    test_data, test_labels   = load_mini_imagenet_data(dspth, split="test")
    # ---------- 1) 稳定的类别到ID映射 ----------
    def collect_classes(lbl):
        if isinstance(lbl, dict):
            return {str(k) for k in lbl.keys()}
        else:
            return {str(x) for x in (lbl.tolist() if hasattr(lbl, "tolist") else list(lbl))}

    all_classes = sorted(
        collect_classes(train_labels)
        | collect_classes(val_labels)
        | collect_classes(test_labels)
    )  # 排序=稳定
    cls2id = {c: i for i, c in enumerate(all_classes)}  # 稳定映射
    # ---------- 2) 把每个 split 还原成“样本级标签数组” ----------
    def to_per_sample_labels(data_split, labels_split):
        n = len(data_split)
        if isinstance(labels_split, dict):
            y = np.empty(n, dtype=np.int64)
            # 安全性：索引必须在 [0, n)
            for c, idxs in labels_split.items():
                idxs = np.asarray(idxs, dtype=np.int64)
                assert idxs.ndim == 1, f"indices dim error for class {c}"
                assert idxs.size > 0, f"empty indices for class {c}"
                assert (idxs.min() >= 0 and idxs.max() < n), f"indices out of range for split: max={idxs.max()} n={n}"
                y[idxs] = cls2id[str(c)]
            return y
        else:
            # 已是一一对齐的标签列表/数组
            arr = labels_split.tolist() if hasattr(labels_split, "tolist") else list(labels_split)
            assert len(arr) == n, f"labels length {len(arr)} != data length {n}"
            return np.fromiter((cls2id[str(c)] for c in arr), dtype=np.int64, count=n)

    y_train = to_per_sample_labels(train_data, train_labels)
    y_val = to_per_sample_labels(val_data, val_labels)
    y_test = to_per_sample_labels(test_data, test_labels)

    # ---------- 3) 按相同顺序拼接数据与标签 ----------
    merged_data = np.concatenate([train_data, val_data, test_data], axis=0)
    merged_labels = np.concatenate([y_train, y_val, y_test], axis=0)

    return merged_data, merged_labels


def load_tiny_imagenet_val(root, image_size=(64, 64)):
    datalist = []
    labels = []
    n_class = 0

    # 读取标签文件
    with open(os.path.join(root, "tiny-imagenet-200/val", "val_annotations.txt"), "r") as f:
        for line in f:
            parts = line.split("\t")
            image_name = parts[0]
            class_name = parts[1]
            bbox = list(map(int, parts[2:]))

            if class_name not in label_map:
                label_map[class_name] = n_class
                n_class += 1

            # 读取图像
            image_path = os.path.join(root, "tiny-imagenet-200/val", "images", image_name)
            image = Image.open(image_path)
            image = image.resize(image_size)
            image = np.array(image)

            # 确保图像具有3个通道
            if len(image.shape) != 3 or image.shape[2] != 3:
                continue

            # 添加到数据列表和标签列表
            datalist.append(image)
            labels.append(label_map[class_name])

    return np.array(datalist), np.array(labels), n_class


def load_tiny_imagenet_data(root, image_size=(64, 64)):
    datalist = []
    labels = []
    n_class = 0

    # Loop through each class folder
    for class_folder in os.listdir(os.path.join(root, "tiny-imagenet-200/train")):
        class_folder_path = os.path.join(root, "tiny-imagenet-200/train", class_folder)
        if os.path.isdir(class_folder_path):
            label_map[class_folder] = n_class
            n_class += 1
            for image_file in os.listdir(os.path.join(class_folder_path, "images")):
                image_path = os.path.join(class_folder_path, "images", image_file)
                # Load and resize image
                image = Image.open(image_path)
                image = image.resize(image_size)
                # Convert to numpy array
                image = np.array(image)
                # Ensure image has 3 channels
                if len(image.shape) != 3 or image.shape[2] != 3:
                    continue
                # Append to data list and label list
                datalist.append(image)
                labels.append(label_map[class_folder])
    labels = np.array(labels)
    return np.array(datalist), labels, n_class


class OneCropsTransform:
    def __init__(self, trans_weak):
        self.trans_weak = trans_weak
    def __call__(self, x):
        x1 = self.trans_weak(x)
        return [x1]


class TwoCropsTransform:
    """Take 2 random augmentations of one image."""
    def __init__(self, trans_weak, trans_strong):
        self.trans_weak = trans_weak
        self.trans_strong = trans_strong
    def __call__(self, x):
        x1 = self.trans_weak(x)
        x2 = self.trans_strong(x)
        return [x1, x2]


class ThreeCropsTransform:
    """Take 3 random augmentations of one image."""
    def __init__(self, trans_weak, trans_strong0, trans_strong1):
        self.trans_weak = trans_weak
        self.trans_strong0 = trans_strong0
        self.trans_strong1 = trans_strong1
    def __call__(self, x):
        x1 = self.trans_weak(x)
        x2 = self.trans_strong0(x)
        x3 = self.trans_strong1(x)
        return [x1, x2, x3]


def _cluster_ids_for_training_samples(
    clusters, *, dataset, dataset_length, shuffled_indices,
    metadata=None, map_indices=None,
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
        raise ValueError("Cluster metadata dataset does not match the requested dataset")
    if int(metadata.get("N", len(clusters))) != len(clusters):
        raise ValueError("Cluster metadata N does not match the assignment length")
    if map_indices is not None:
        map_indices = np.asarray(map_indices)
        if not np.array_equal(map_indices, np.arange(len(clusters))):
            raise ValueError("Cluster map indices must be in canonical source row order")
    if len(clusters) != dataset_length:
        is_merged_mini = (
            dataset == "miniImageNet"
            and len(clusters) % 600 == 0
            and len(clusters) // 600 * 500 == dataset_length
            and metadata.get("split") == "train+val+test"
        )
        if not is_merged_mini:
            raise ValueError(
                f"Cluster map length {len(clusters)} does not match training "
                f"length {dataset_length} or a declared miniImageNet merged split"
            )
        clusters = clusters.reshape(-1, 600)[:, :500].reshape(-1)
    elif dataset == "miniImageNet" and metadata.get("split") == "train+val+test":
        raise ValueError("Merged miniImageNet cluster metadata requires the 600-to-500 projection")
    shuffled_indices = np.asarray(shuffled_indices)
    if shuffled_indices.ndim != 1 or not np.issubdtype(shuffled_indices.dtype, np.integer):
        raise ValueError("Training row indices must be one-dimensional integers")
    if shuffled_indices.size and (
        shuffled_indices.min() < 0 or shuffled_indices.max() >= dataset_length
    ):
        raise ValueError("Training row index lies outside the cluster map")
    return clusters[shuffled_indices]


def load_data_train(pi,bag_build,num_classes, holdout_fraction, dataset="CIFAR10", dspth="./data", bagsize=16, backbone=None, seed=0, cluster_manifest=None):
    if cluster_manifest is not None and (bag_build != 'cluster' or dataset not in {'CIFAR10', 'CIFAR100', 'miniImageNet'}):
        raise ValueError('Frozen Cluster manifests support the three paper image datasets only')
    cluster_map_digests = []
    input_dim = 1
    if dataset == "CIFAR10":
        datalist = [osp.join(dspth, "cifar-10-batches-py", "data_batch_{}".format(i + 1)) for i in range(5)]
        n_class = 10
    elif dataset == "CIFAR100":
        datalist = [osp.join(dspth, "cifar-100-python", "train")]
        n_class = 100
    elif dataset == "SVHN":
        data, labels = load_svhn_data(dspth)
    elif dataset == "MNIST":
        data, labels = [], []
        datalist = [osp.join(dspth, "MNIST", "raw", "train-images-idx3-ubyte")]
        labelslist = [osp.join(dspth, "MNIST", "raw", "train-labels-idx1-ubyte")]
        n_class = 10
    elif dataset == "FashionMNIST":
        data, labels = [], []
        datalist = [osp.join(dspth, "FashionMNIST", "raw", "train-images-idx3-ubyte")]
        labelslist = [osp.join(dspth, "FashionMNIST", "raw", "train-labels-idx1-ubyte")]
        n_class = 10
    elif dataset == "KMNIST":
        data, labels = [], []
        datalist = [osp.join(dspth, "KMNIST", "raw", "train-images-idx3-ubyte")]
        labelslist = [osp.join(dspth, "KMNIST", "raw", "train-labels-idx1-ubyte")]
        n_class = 10
    elif dataset == "EMNISTBalanced":
        data, labels = [], []
        datalist = [
            osp.join(dspth, "EMNIST", "raw", "emnist-letters-train-images-idx3-ubyte")
        ]
        labelslist = [
            osp.join(dspth, "EMNIST", "raw", "emnist-letters-train-labels-idx1-ubyte")
        ]
        n_class = 26
    elif dataset == "TinyImageNet":
        train_data, train_labels, n_class = load_tiny_imagenet_data(dspth)
    elif dataset == "miniImageNet":
        train_data, train_labels = merge_train_val_test(dspth)
        subset_data_list = []
        subset_labels_list = []
        n_class = 100
        for i in range(0, len(train_data), 600):
            # Get the first 500 samples from the current chunk
            chunk_data = train_data[i : i + 600][:500]
            chunk_labels = np.array(train_labels[i : i + 600][:500])
            # Append the data and labels to the lists
            subset_data_list.append(chunk_data)
            subset_labels_list.append(chunk_labels)
        # Concatenate the subsets into final arrays
        train_data = np.concatenate(subset_data_list, axis=0)
        train_labels = np.concatenate(subset_labels_list, axis=0)
    elif dataset in TEXT_DATASETS:
        if backbone in BERT_MODELS:
            # ── BERT path: tokenise → dense int64 (N, max_length) ──────────
            BERT_MAX_LEN = 128
            if dataset == "AGNEWS":
                orig, aug_bt, aug_syn, train_labels, n_class = _load_agnews_aug(dspth)
                data_orig = _tokenize_texts(orig,    backbone, BERT_MAX_LEN)  # (N, L)
                data_aug0 = _tokenize_texts(aug_bt,  backbone, BERT_MAX_LEN)
                data_aug1 = _tokenize_texts(aug_syn, backbone, BERT_MAX_LEN)
            else:
                train_texts, train_labels, n_class = _load_raw_text(dataset, dspth, "train")
                data_orig = _tokenize_texts(train_texts, backbone, BERT_MAX_LEN)
                data_aug0 = data_orig   # no pre-computed aug; same text
                data_aug1 = data_orig
            # Stack all three views: (N, 3, max_length)
            data   = np.stack([data_orig, data_aug0, data_aug1], axis=1)
            labels = train_labels
        else:
            # ── TF-IDF path (default) ────────────────────────────────────────
            train_texts, train_labels, n_class = _load_raw_text(dataset, dspth, split="train")
            vec = _get_vectorizer(dataset, dspth)
            data   = vec.transform(train_texts)   # scipy CSR sparse
            labels = train_labels
    else:
        raise ValueError("Unsupported dataset")

    if dataset in ["CIFAR10", "CIFAR100"]:
        data, labels = [], []
        for data_batch in datalist:
            with open(data_batch, "rb") as fr:
                entry = pickle.load(fr, encoding="latin1")
                lbs = entry["labels"] if "labels" in entry.keys() else entry["fine_labels"]
                data.append(entry["data"])
                labels.append(lbs)
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)

    elif dataset in ["MNIST", "FashionMNIST", "KMNIST", "EMNISTBalanced"]:
        for data_path, label_path in zip(datalist, labelslist):
            with open(data_path, "rb") as fr_data, open(label_path, "rb") as fr_label:
                fr_data.read(16)  # Skip the header
                fr_label.read(8)  # Skip the header
                data.append(np.frombuffer(fr_data.read(), dtype=np.uint8).reshape(-1, 784))
                labels.append(np.frombuffer(fr_label.read(), dtype=np.uint8))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        if dataset == "EMNISTBalanced":
            labels = labels % 26

    elif dataset in ["TinyImageNet", "miniImageNet"]:
        data = train_data
        labels = train_labels

    dataset_length = data.shape[0] if hasattr(data, 'shape') else len(data)

    # 1) 截断到能整除 bagsize
    num_bags_all = dataset_length // bagsize
    data_length = num_bags_all * bagsize

    # Cluster construction owns one seeded stream, including row/split order.
    # Random and AlphaFirst retain their existing global-RNG behavior.
    cluster_rng = np.random.default_rng(seed) if bag_build == 'cluster' else None
    shuffle_rng = cluster_rng if cluster_rng is not None else np.random
    # 2) 全局打乱（样本级）
    random_indices = np.arange(data_length)
    shuffle_rng.shuffle(random_indices)
    data = data[random_indices]
    labels = labels[random_indices]

    # 3) 再生成一个索引用于分 bag（bag 内连续切片）
    indices = np.arange(data_length)
    shuffle_rng.shuffle(indices)

    num_bags_all = data_length // bagsize
    # 4) 按 bag 切 80/20（64/80 train, 16/80 test）
    num_train_bags = int(num_bags_all * (1 - holdout_fraction))
    num_test_bags = num_bags_all - num_train_bags

    def build_bags(bag_id_list):
        data_u, label_prob = [], []
        labels_real, labels_idx = [], []
        indices_u = []

        for new_j, j in enumerate(bag_id_list):
            bag_indices = indices[j * bagsize : (j + 1) * bagsize]

            if dataset in ["MNIST", "FashionMNIST", "EMNISTBalanced", "KMNIST"]:
                bag_data = [data[i].reshape(28, 28) for i in bag_indices]
            elif dataset in TEXT_DATASETS:
                if backbone in BERT_MODELS:
                    # data[i] is (3, max_length) dense int64 — store directly
                    bag_data = [data[i] for i in bag_indices]
                else:
                    # TF-IDF sparse: store row indices; TextBag densifies at __getitem__
                    bag_data = list(bag_indices)
            elif dataset in ["SVHN", "TinyImageNet", "miniImageNet", "Corel16k", "Corel5k", "Delicious", "Bookmarks", "Eurlex_DC", "Eurlex_SM", "Scene", "Yeast"]:
                bag_data = [data[i] for i in bag_indices]
            else:
                # CIFAR* 32x32
                bag_data = [data[i].reshape(3, 32, 32).transpose(1, 2, 0) for i in bag_indices]

            bag_labels = np.array([labels[i] for i in bag_indices])
            labels_real.append(bag_labels)
            labels_idx.append(bag_indices)

            label_counts = Counter(bag_labels)
            label_counts = OrderedDict(sorted(label_counts.items()))
            label_proportions = [label_counts.get(label, 0) / len(bag_labels) for label in range(0, num_classes)]

            data_u.append(bag_data)
            label_prob.append(label_proportions)
            indices_u.append(new_j)

        return data_u, label_prob, labels_real, labels_idx, indices_u

    # 5) bag id 列表
    bag_ids = np.arange(num_bags_all)
    # （可选）再打乱 bag 顺序，保证 train/test bag 随机
    shuffle_rng.shuffle(bag_ids)

    var_bag_ids_1 = bag_ids[:num_train_bags]
    var_bag_ids_2 = bag_ids[num_train_bags:]

    def build_bags_cluster(
            bag_id_list,
            *,
            data,
            labels,
            indices,  # 你现在用来切片的“可用样本索引序列”，长度应 >= len(bag_id_list)*bagsize
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

        # ---------- 0) 取出这批 bag 需要的样本子集（严格无放回） ----------
        B = len(bag_id_list)
        m = bagsize
        need_total = B * m

        # 这里沿用你原来 build_bags 的“bag_id_list -> indices切片”习惯：
        # 每个 bag_id 对应 indices 的一个块。为了 cluster 构造，我们先把这些块合并成一个可用集合。
        chosen = []
        for j in bag_id_list:
            blk = indices[j * m: (j + 1) * m]
            chosen.append(np.asarray(blk, dtype=np.int64))
        chosen = np.concatenate(chosen, axis=0)  # [B*m]
        assert len(chosen) == need_total

        # 在 chosen 里重新编号 0..need_total-1，方便 pools 用局部索引
        # 但 labels/data 仍然用原始索引去取
        # local->global 映射：
        local2global = chosen.copy()

        # ---------- 1) 读 cluster map，并取出 chosen 对应的 cluster id ----------

        # text dataset 名称 → cluster CSV 文件前缀
        _TEXT_CLUSTER_PREFIX = {
            "AGNEWS": "ag_news",
            "20News": "20news",
        }

        def _find_cluster_csv(_dspth, _dataset):
            """找文本数据集的 cluster assign CSV，返回路径。"""
            base = osp.join(_dspth, "cluster_maps")
            prefix = _TEXT_CLUSTER_PREFIX.get(_dataset)
            if prefix is None:
                raise FileNotFoundError(
                    f"No cluster CSV prefix defined for text dataset {_dataset!r}. "
                    f"Supported: {list(_TEXT_CLUSTER_PREFIX)}"
                )
            # 匹配 <prefix>_tfidf_k<K>_assign.csv，取 K 最大的那个
            import glob
            pattern = osp.join(base, f"{prefix}_tfidf_k*_assign.csv")
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise FileNotFoundError(
                    f"Cannot find cluster CSV for {_dataset} "
                    f"(pattern: {pattern})"
                )
            return matches[-1]  # 取 K 最大

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
            raise FileNotFoundError(f"Cannot find cluster map for {_dataset} under {base}")

        if dataset in TEXT_DATASETS:
            # CSV 格式: idx, label, cluster  (按 idx 升序即原始样本顺序)
            import csv as _csv
            csv_path = _find_cluster_csv(dspth, dataset)
            _cluster_list = [None] * dataset_length
            with open(csv_path, newline="") as _f:
                for row in _csv.DictReader(_f):
                    _idx = int(row["idx"])
                    if _idx < len(_cluster_list):
                        _cluster_list[_idx] = int(row["cluster"])
            if any(value is None for value in _cluster_list):
                raise ValueError("Cluster CSV does not cover every original training row")
            clusters_all = np.array(_cluster_list, dtype=np.int64)
            cluster_metadata, cluster_map_indices = None, None
        else:
            cluster_map_path = _find_cluster_npz(dspth, dataset)
            if cluster_manifest is not None:
                import hashlib
                from pathlib import Path
                cluster_map_digests.append(hashlib.sha256(Path(cluster_map_path).read_bytes()).hexdigest())
            with np.load(cluster_map_path, allow_pickle=False) as z:
                clusters_all = z["clusters"]
                cluster_metadata = json.loads(z["meta"].item()) if "meta" in z else None
                cluster_map_indices = z["indices"] if "indices" in z else None
                if "labels" in z:
                    map_labels = z["labels"]
                    if dataset == 'miniImageNet' and len(map_labels) != dataset_length:
                        map_labels = map_labels.reshape(-1, 600)[:, :500].reshape(-1)
                    if len(map_labels) != dataset_length or not np.array_equal(map_labels[random_indices], labels):
                        raise ValueError('Cluster map/image label alignment failed')

        aligned_clusters = _cluster_ids_for_training_samples(
            clusters_all, dataset=dataset, dataset_length=dataset_length,
            shuffled_indices=random_indices, metadata=cluster_metadata,
            map_indices=cluster_map_indices,
        )
        clusters = aligned_clusters[local2global]

        # ---------- 2) 建每簇 pool（局部索引 0..B*m-1），并 shuffle ----------
        pools = {}
        for local_i, c in enumerate(clusters):
            pools.setdefault(int(c), []).append(local_i)

        for c in list(pools.keys()):
            arr = np.array(pools[c], dtype=np.int64)
            rng.shuffle(arr)
            pools[c] = arr

        cluster_ids_all = np.array(sorted(pools.keys()), dtype=np.int64)
        K = len(cluster_ids_all)
        cluster_sizes = np.array([len(pools[int(c)]) for c in cluster_ids_all], dtype=np.int64)
        assert cluster_sizes.sum() == B * m

        # base measure pi：按簇大小
        pi = cluster_sizes.astype(np.float64)
        pi = pi / pi.sum()

        # ---------- 3) 抽 counts 矩阵 C（行和=m），再配平列和到 cluster_sizes ----------
        def _sample_counts_matrix_dirichlet(B, m, alpha0, pi, rng):
            dir_param = np.maximum(alpha0 * pi, 1e-12)
            W = rng.dirichlet(dir_param, size=B)  # [B,K]
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
                    raise RuntimeError("Balancing failed: no surplus but still deficit.")
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

        # ---------- 4) 按 C 从各簇 pool 严格无放回取样，组装每个 bag ----------
        ptr = {int(c): 0 for c in cluster_ids_all}

        data_u, label_prob = [], []
        labels_real, labels_idx = [], []
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

            bag_indices = local2global[bag_local]  # 转回原始索引（用来取 data/labels）
            # --- 组装 bag_data（保持你原来分支）---
            if dataset in ["MNIST", "FashionMNIST", "EMNISTBalanced", "KMNIST"]:
                bag_data = [data[i].reshape(28, 28) for i in bag_indices]
            elif dataset in TEXT_DATASETS:
                if backbone in BERT_MODELS:
                    bag_data = [data[i] for i in bag_indices]
                else:
                    bag_data = list(bag_indices)
            elif dataset in ["SVHN", "TinyImageNet", "miniImageNet",
                             "Corel16k", "Corel5k", "Delicious", "Bookmarks",
                             "Eurlex_DC", "Eurlex_SM", "Scene", "Yeast"]:
                bag_data = [data[i] for i in bag_indices]
            else:
                bag_data = [data[i].reshape(3, 32, 32).transpose(1, 2, 0) for i in bag_indices]

            bag_labels = np.array([labels[i] for i in bag_indices])
            labels_real.append(bag_labels)
            labels_idx.append(bag_indices)

            label_counts = Counter(bag_labels)
            label_counts = OrderedDict(sorted(label_counts.items()))
            label_proportions = [label_counts.get(label, 0) / len(bag_labels) for label in range(num_classes)]

            data_u.append(bag_data)
            label_prob.append(label_proportions)
            indices_u.append(new_j)

        return data_u, label_prob, labels_real, labels_idx, indices_u


    def build_bags_alphafirst(
            bag_id_list,
            *,
            data,
            labels,
            indices,  # 你现在用来切片的“可用样本索引序列”
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

        # -------- 0) 收集这批 bag_id_list 对应的全局样本索引（严格无放回集合）--------
        chosen = []
        for j in bag_id_list:
            blk = indices[j * m: (j + 1) * m]
            chosen.append(np.asarray(blk, dtype=np.int64))
        chosen = np.concatenate(chosen, axis=0)  # [B*m]
        if len(chosen) != need_total:
            raise ValueError(f"chosen size {len(chosen)} != B*m {need_total}")

        # local -> global
        local2global = chosen.copy()

        # 子集内的 labels（用于建 pool）
        sub_labels = labels[local2global].astype(np.int64)

        # sanity: labels 必须在 [0, num_classes-1]
        if sub_labels.min() < 0 or sub_labels.max() >= num_classes:
            raise ValueError("labels value out of range; check num_classes")

        # -------- 1) build class pools on LOCAL indices 0..B*m-1 --------
        pools = {}
        for c in range(num_classes):
            idx = np.where(sub_labels == c)[0].astype(np.int64)  # local indices
            rng.shuffle(idx)
            pools[c] = idx

        class_sizes = np.array([len(pools[c]) for c in range(num_classes)], dtype=np.int64)
        if class_sizes.sum() != need_total:
            raise RuntimeError("Internal error: class_sizes.sum != B*m")

        # -------- 2) sample counts matrix C0 with Dirichlet flavour (row-sum = m) --------
        def _sample_counts_matrix(B, m, Cn, alpha0, rng):
            dir_param = np.maximum(alpha0 * np.ones(Cn, dtype=np.float64), 1e-12)
            W = rng.dirichlet(dir_param, size=B)  # (B, Cn)

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

        # -------- 3) balance column sums to match class_sizes exactly --------
        def _balance_columns_to_targets(C, targets, rng):
            C = C.copy()
            cur = C.sum(axis=0)
            diff = targets - cur  # >0 缺, <0 多

            deficit = np.where(diff > 0)[0].tolist()
            surplus = np.where(diff < 0)[0].tolist()

            need_moves = int(diff[diff > 0].sum())
            moves = 0

            while deficit:
                d = deficit[-1]
                if not surplus:
                    raise RuntimeError("Balancing failed: no surplus but still deficit.")
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
                    raise RuntimeError("Balancing seems stuck; please check data/params.")

                if diff[d] == 0:
                    deficit.pop()
                if diff[s] == 0:
                    surplus.pop()

            assert np.all(C.sum(axis=0) == targets)
            assert np.all(C.sum(axis=1) == m)
            return C

        C0 = _sample_counts_matrix(B, m, num_classes, alpha0, rng)
        C = _balance_columns_to_targets(C0, class_sizes, rng)

        # -------- 4) allocate without replacement according to C --------
        ptr = {c: 0 for c in range(num_classes)}

        data_u, label_prob = [], []
        labels_real, labels_idx = [], []
        indices_u = []

        for new_j in range(B):
            bag_local_list = []
            for c in range(num_classes):
                need = int(C[new_j, c])
                if need <= 0:
                    continue
                s0 = ptr[c]
                e0 = s0 + need
                chosen_local = pools[c][s0:e0]  # 一定够
                ptr[c] = e0
                bag_local_list.extend(chosen_local.tolist())

            bag_local = np.array(bag_local_list, dtype=np.int64)
            rng.shuffle(bag_local)

            bag_indices = local2global[bag_local]  # global indices

            # --- construct bag_data（沿用你原来的分支）---
            if dataset in ["MNIST", "FashionMNIST", "EMNISTBalanced", "KMNIST"]:
                bag_data = [data[i].reshape(28, 28) for i in bag_indices]
            elif dataset in TEXT_DATASETS:
                if backbone in BERT_MODELS:
                    bag_data = [data[i] for i in bag_indices]
                else:
                    bag_data = list(bag_indices)
            elif dataset in ["SVHN", "TinyImageNet", "miniImageNet",
                             "Corel16k", "Corel5k", "Delicious", "Bookmarks",
                             "Eurlex_DC", "Eurlex_SM", "Scene", "Yeast"]:
                bag_data = [data[i] for i in bag_indices]
            else:
                bag_data = [data[i].reshape(3, 32, 32).transpose(1, 2, 0) for i in bag_indices]

            bag_labels = sub_labels[bag_local]  # local labels（等价于 labels[bag_indices]）
            labels_real.append(np.asarray(bag_labels))
            labels_idx.append(np.asarray(bag_indices))

            label_counts = Counter(bag_labels.tolist())
            label_counts = OrderedDict(sorted(label_counts.items()))
            label_proportions = [label_counts.get(k, 0) / len(bag_labels) for k in range(num_classes)]

            data_u.append(bag_data)
            label_prob.append(label_proportions)
            indices_u.append(new_j)

        return data_u, label_prob, labels_real, labels_idx, indices_u
    # 6) 构造两份链表式数据
    # holdout_fraction=0 时 var_bag_ids_2 为空，跳过 var_2 的构造，后面直接复用 var_1
    _empty = ([], [], [], [], [])
    if bag_build == 'random':
        (var_data_u_1, var_label_prob_1, var_labels_real_1, var_labels_idx_1, var_indices_u_1) = build_bags(var_bag_ids_1)
        (var_data_u_2, var_label_prob_2, var_labels_real_2, var_labels_idx_2, var_indices_u_2) = (
            build_bags(var_bag_ids_2) if len(var_bag_ids_2) > 0 else _empty
        )
    elif bag_build == 'cluster':
        (var_data_u_1, var_label_prob_1, var_labels_real_1, var_labels_idx_1, var_indices_u_1) = build_bags_cluster(
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
        (var_data_u_2, var_label_prob_2, var_labels_real_2, var_labels_idx_2, var_indices_u_2) = (
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
            ) if len(var_bag_ids_2) > 0 else _empty
        )
    elif bag_build == 'alphafirst':
        (var_data_u_1, var_label_prob_1, var_labels_real_1, var_labels_idx_1, var_indices_u_1) = build_bags_alphafirst(
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
        (var_data_u_2, var_label_prob_2, var_labels_real_2, var_labels_idx_2, var_indices_u_2) = (
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
            ) if len(var_bag_ids_2) > 0 else _empty
        )


    if dataset in ["CIFAR10", "CIFAR100", "SVHN"]:
        input_dim = (3, 32, 32)
    elif dataset in ["MNIST", "FashionMNIST", "KMNIST", "EMNISTBalanced"]:
        input_dim = (1, 28, 28)
    elif dataset in ["TinyImageNet", "miniImageNet"]:
        input_dim = (3, 64, 64)
    elif dataset in TEXT_DATASETS:
        if backbone in BERT_MODELS:
            # data shape: (N, 3, max_length); input_dim = max_length
            input_dim = data.shape[2]   # max_length (e.g. 128)
            # var_data_u already contains the stacked token-ID arrays directly
        else:
            input_dim = data.shape[1]   # TF-IDF feature dim
            # Embed the shuffled sparse matrix so TextBag can densify rows on demand
            var_data_u_1 = (data, var_data_u_1)
            var_data_u_2 = (data, var_data_u_2)
    # 7) 返回两组（每组结构跟你原来一致）
    if cluster_manifest is not None:
        from .cluster_manifest import freeze_cluster_manifest
        def source_bags(bags):
            compact = random_indices[np.asarray(bags, dtype=np.int64).reshape(-1, bagsize)]
            return (compact // 500) * 600 + compact % 500 if dataset == 'miniImageNet' else compact
        freeze_cluster_manifest(
            cluster_manifest,
            metadata={"schema_version": 1, "dataset": dataset, "seed": int(seed),
                      "pi": float(pi), "bag_size": int(bagsize),
                      "holdout_fraction": float(holdout_fraction),
                      "num_classes": int(num_classes), "training_population": int(dataset_length),
                      "cluster_map_sha256": sorted(set(cluster_map_digests)),
                      "source_index_space": 'merged_600_blocks' if dataset == 'miniImageNet' else 'original_train_rows'},
            train_indices=source_bags(var_labels_idx_1),
            val_indices=source_bags(var_labels_idx_2),
            train_proportions=np.asarray(var_label_prob_1).reshape(-1, num_classes),
            val_proportions=np.asarray(var_label_prob_2).reshape(-1, num_classes),
        )
    var_pack_1 = (var_data_u_1, var_label_prob_1, var_labels_real_1, var_labels_idx_1, dataset_length, var_indices_u_1, input_dim)
    # holdout_fraction=0 时 var_bag_ids_2 为空，val loader 无法构造；直接复用 train pack
    if holdout_fraction == 0.0:
        var_pack_2 = var_pack_1
    else:
        var_pack_2 = (var_data_u_2, var_label_prob_2, var_labels_real_2, var_labels_idx_2, dataset_length, var_indices_u_2, input_dim)
    prior_train = (
        np.mean(np.asarray(var_label_prob_1, dtype=np.float32), axis=0)
        if len(var_label_prob_1) > 0
        else np.zeros((num_classes,), dtype=np.float32))
    prior_val = (
        np.mean(np.asarray(var_label_prob_2, dtype=np.float32), axis=0)
        if len(var_label_prob_2) > 0
        else np.zeros((num_classes,), dtype=np.float32))
    all_probs = np.asarray(var_label_prob_1 + var_label_prob_2, dtype=np.float32)
    prior_all = (
        np.mean(all_probs, axis=0)
        if all_probs.shape[0] > 0
        else np.zeros((num_classes,), dtype=np.float32))
    # ================================================
    return var_pack_1, var_pack_2, prior_train, prior_val, prior_all


def load_data_val(dataset, dspth="./data", n_classes=10, backbone=None):
    if dataset == "CIFAR10":
        datalist = [osp.join(dspth, "cifar-10-batches-py", "test_batch")]
    elif dataset == "CIFAR100":
        datalist = [osp.join(dspth, "cifar-100-python", "test")]
    elif dataset == "SVHN":
        data, labels = load_svhn_val(dspth)
    elif dataset == "TinyImageNet":
        data, labels, n_class = load_tiny_imagenet_val(dspth)
    elif dataset == "miniImageNet":
        # 加载miniImageNet数据集的训练、验证和测试集
        train_data, train_labels = merge_train_val_test(dspth)
        test_data_list = []
        test_labels_list = []
        n_class = 100
        for i in range(0, len(train_data), 600):
            # Get the last 100 samples from the current chunk
            chunk_data = train_data[i : i + 600][-100:]
            chunk_labels = np.array(train_labels[i : i + 600][-100:])

            # Append the data and labels to the lists
            test_data_list.append(chunk_data)
            test_labels_list.append(chunk_labels)

        # Concatenate the subsets into final arrays
        data = np.concatenate(test_data_list, axis=0)
        labels = np.concatenate(test_labels_list, axis=0)
        # 使用和训练集相同的 class_label 映射
    elif dataset in TEXT_DATASETS:
        test_texts, test_labels, _ = _load_raw_text(dataset, dspth, split="test")
        if backbone in BERT_MODELS:
            data = _tokenize_texts(test_texts, backbone, max_length=128)  # (N, 128) int64
        else:
            vec  = _get_vectorizer(dataset, dspth)
            data = vec.transform(test_texts)   # scipy CSR sparse
        labels = test_labels
        return data, labels

    if dataset == "CIFAR10" or dataset == "CIFAR100":
        data, labels = [], []
        for data_batch in datalist:
            with open(data_batch, "rb") as fr:
                entry = pickle.load(fr, encoding="latin1")
                lbs = entry["labels"] if "labels" in entry.keys() else entry["fine_labels"]
                data.append(entry["data"])
                labels.append(lbs)
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        data = [el.reshape(3, 32, 32).transpose(1, 2, 0) for el in data]

        if n_classes == 2:
            # 机器类映射为 1，其余为 0
            machine_classes = np.array([0, 1, 8, 9])
            labels = np.isin(labels, machine_classes).astype(np.int64)
    elif dataset == "MNIST":
        data, labels = [], []
        datalist = [osp.join(dspth, "MNIST", "raw", "t10k-images-idx3-ubyte")]
        labelslist = [osp.join(dspth, "MNIST", "raw", "t10k-labels-idx1-ubyte")]
        n_class = 2
        for data_path, label_path in zip(datalist, labelslist):
            with open(data_path, "rb") as fr_data, open(label_path, "rb") as fr_label:
                fr_data.read(16)  # Skip the header
                fr_label.read(8)  # Skip the header
                data.append(np.frombuffer(fr_data.read(), dtype=np.uint8).reshape(-1, 784))
                labels.append(np.frombuffer(fr_label.read(), dtype=np.uint8))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        if n_class == 2:
            labels = np.where(np.isin(labels, [0, 2, 4, 6, 8]), 0, 1)
        data = [el.reshape(28, 28) for el in data]
    elif dataset == "FashionMNIST":
        data, labels = [], []
        datalist = [osp.join(dspth, "FashionMNIST", "raw", "t10k-images-idx3-ubyte")]
        labelslist = [osp.join(dspth, "FashionMNIST", "raw", "t10k-labels-idx1-ubyte")]
        n_class = 10
        for data_path, label_path in zip(datalist, labelslist):
            with open(data_path, "rb") as fr_data, open(label_path, "rb") as fr_label:
                fr_data.read(16)  # Skip the header
                fr_label.read(8)  # Skip the header
                data.append(np.frombuffer(fr_data.read(), dtype=np.uint8).reshape(-1, 784))
                labels.append(np.frombuffer(fr_label.read(), dtype=np.uint8))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        data = [el.reshape(28, 28) for el in data]
    elif dataset == "KMNIST":
        data, labels = [], []
        datalist = [osp.join(dspth, "KMNIST", "raw", "t10k-images-idx3-ubyte")]
        labelslist = [osp.join(dspth, "KMNIST", "raw", "t10k-labels-idx1-ubyte")]
        n_class = 10
        for data_path, label_path in zip(datalist, labelslist):
            with open(data_path, "rb") as fr_data, open(label_path, "rb") as fr_label:
                fr_data.read(16)  # Skip the header
                fr_label.read(8)  # Skip the header
                data.append(np.frombuffer(fr_data.read(), dtype=np.uint8).reshape(-1, 784))
                labels.append(np.frombuffer(fr_label.read(), dtype=np.uint8))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        data = [el.reshape(28, 28) for el in data]

    elif dataset == "EMNISTBalanced":
        data, labels = [], []
        # 更新为EMNIST Balanced数据集的文件路径
        datalist = [osp.join(dspth, "EMNIST", "raw", "emnist-letters-test-images-idx3-ubyte")]
        labelslist = [osp.join(dspth, "EMNIST", "raw", "emnist-letters-test-labels-idx1-ubyte")]
        n_class = 26  # EMNIST Balanced有47个类别

        for data_path, label_path in zip(datalist, labelslist):
            with open(data_path, "rb") as fr_data, open(label_path, "rb") as fr_label:
                fr_data.read(16)  # 跳过头部信息
                fr_label.read(8)  # 跳过头部信息
                data.append(np.frombuffer(fr_data.read(), dtype=np.uint8).reshape(-1, 28 * 28))
                labels.append(np.frombuffer(fr_label.read(), dtype=np.uint8))

        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        data = [el.reshape(28, 28) for el in data]  # 将每个样本重塑为28x28
        labels = labels % 26
    return data, labels


def load_svhn_val(dspth="./data/svhn"):
    svhn_path = osp.join(dspth, "svhn")
    with open(osp.join(svhn_path, "test_32x32.mat"), "rb") as fr:
        svhn_data = sio.loadmat(fr)
        data = svhn_data["X"]
        labels = svhn_data["y"]
    data = np.transpose(data, (3, 0, 1, 2))

    labels = labels % 10
    labels = labels.squeeze()

    return data, labels


def load_svhn_data(dspth):
    svhn_path = osp.join(dspth, "svhn")

    # 加载训练数据
    with open(osp.join(svhn_path, "train_32x32.mat"), "rb") as fr:
        svhn_train = sio.loadmat(fr)
        train_data = svhn_train["X"]
        train_labels = svhn_train["y"]

    # 转换数据维度
    train_data = np.transpose(train_data, (3, 0, 1, 2))

    # 调整标签（从1-10改为0-9）
    train_labels = (train_labels) % 10

    # 压缩标签数组
    train_labels = train_labels.squeeze()

    # 合并训练数据和额外数据
    return train_data, train_labels


def get_train_loader(pi,bag_build,classes, holdout_fraction, dataset, batch_size, bag_size,
                     root="data", method="co", supervised=False, backbone=None,
                     seed=0, num_bags=None, num_workers=0, instances_per_epoch=200000,
                     train_instance_sample_size=None, num_reviewers=None,
                     cluster_seed=0, target_avg_bag_size=None,
                     ku_merge_validation_into_train=False,
                     ku_unknown_bag_max_size=None, ku_unknown_bag_seed=0,
                     cluster_manifest=None):
    if is_fed_isic2019_dataset(dataset):
        if bag_build != "feature":
            raise ValueError(
                "Fed-ISIC2019 requires immutable feature-defined bags; "
                "bag_build must be 'feature'"
            )
        if num_bags is not None:
            raise ValueError("Fed-ISIC2019 does not support truncating the bag population")
        if train_instance_sample_size is not None:
            raise ValueError("Fed-ISIC2019 feature bags must not be truncated")
        train_loader, val_loader, bundle, input_shape = build_fed_isic2019_loaders(
            root=root,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
            paired_views=(method == "L^2P-AHIL"),
        )
        if classes is not None and int(classes) != bundle.num_classes:
            raise ValueError(
                f"--n-classes={classes} disagrees with Fed-ISIC2019 "
                f"({bundle.num_classes})"
            )
        cached_target = int(bundle.metadata["bag_manifest"]["target_bag_size"])
        if target_avg_bag_size is not None and int(target_avg_bag_size) != cached_target:
            raise ValueError(
                f"--target-avg-bag-size={target_avg_bag_size} disagrees with "
                f"prepared Fed-ISIC2019 bags ({cached_target})"
            )
        train_probs = np.asarray(train_loader.dataset.label_prob, dtype=np.float32)
        train_sizes = np.asarray(
            [len(bag.indices) for bag in train_loader.dataset.bags], dtype=np.float64
        )
        prior_train = np.average(train_probs, axis=0, weights=train_sizes)
        return (
            train_loader,
            val_loader,
            train_loader.dataset.label_prob,
            int(train_sizes.sum()),
            0,
            input_shape,
            prior_train,
            prior_train.copy(),
            prior_train.copy(),
        )
    if is_cct_dataset(dataset):
        if bag_build != "feature":
            raise ValueError(
                "CCT bags are immutable feature-defined bags; bag_build must be "
                "'feature' and never 'random'"
            )
        if num_bags is not None:
            raise ValueError("CCT does not support truncating the feature-bag population")
        if train_instance_sample_size is not None:
            raise ValueError("CCT feature bags must not be truncated during training")
        train_loader, val_loader, bundle, input_shape = build_cct_loaders(
            root=root,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
            paired_views=(method == "L^2P-AHIL"),
        )
        if classes is not None and int(classes) != bundle.num_classes:
            raise ValueError(
                f"--n-classes={classes} disagrees with CCT ({bundle.num_classes})"
            )
        cached_target = int(bundle.metadata["bag_manifest"]["target_avg_bag_size"])
        if (
            target_avg_bag_size is not None
            and int(target_avg_bag_size) != cached_target
        ):
            raise ValueError(
                f"--target-avg-bag-size={target_avg_bag_size} disagrees with "
                f"prepared CCT bags ({cached_target})"
            )
        train_probs = np.asarray(train_loader.dataset.label_prob, dtype=np.float32)
        train_sizes = np.asarray(
            [len(bag.indices) for bag in train_loader.dataset.bags], dtype=np.float64
        )
        prior_train = np.average(train_probs, axis=0, weights=train_sizes)
        # CCT deliberately has no validation bags: official train/cis-val/
        # trans-val images all belong to the training population.
        prior_val = prior_train.copy()
        prior_all = prior_train.copy()
        return (
            train_loader,
            val_loader,
            train_loader.dataset.label_prob,
            int(train_sizes.sum()),
            0,
            input_shape,
            prior_train,
            prior_val,
            prior_all,
        )
    if is_amazon_wilds_dataset(dataset):
        if bag_build != "random":
            raise ValueError(
                "Amazon-WILDS bags are natural reviewers; bag_build must remain "
                "'random' and does not construct random bags"
            )
        if num_bags is not None:
            raise ValueError(
                "Amazon-WILDS uses --num-reviewers for complete-bag selection; "
                "--num-bags is not supported"
            )
        if train_instance_sample_size is not None:
            raise ValueError(
                "Amazon-WILDS never truncates reviewer bags; "
                "--train-instance-sample-size is not supported"
            )
        train_loader, val_loader, bundle, input_shape = build_amazon_wilds_loaders(
            root=root,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
            num_reviewers=num_reviewers,
        )
        if classes is not None and int(classes) != bundle.num_classes:
            raise ValueError(
                f"--n-classes={classes} disagrees with Amazon-WILDS "
                f"({bundle.num_classes})"
            )
        train_probs = np.asarray(train_loader.dataset.label_prob, dtype=np.float32)
        train_sizes = np.asarray(
            [len(bag.indices) for bag in train_loader.dataset.bags], dtype=np.float64
        )
        val_indices = val_loader.dataset.indices
        val_labels = bundle.labels[val_indices]
        val_counts = np.bincount(val_labels, minlength=bundle.num_classes).astype(np.float64)
        prior_train = np.average(train_probs, axis=0, weights=train_sizes)
        prior_val = val_counts / val_counts.sum()
        prior_all = (
            prior_train * train_sizes.sum() + val_counts
        ) / (train_sizes.sum() + len(val_labels))
        return (
            train_loader,
            val_loader,
            train_loader.dataset.label_prob,
            int(train_sizes.sum()),
            int(len(val_labels)),
            input_shape,
            prior_train,
            prior_val,
            prior_all,
        )
    if is_ku_optofil_dataset(dataset):
        if bag_build != "random":
            raise ValueError(
                "KU-Optofil bags are natural patients; bag_build must remain 'random' "
                "and does not construct random bags"
            )
        if num_bags is not None:
            raise ValueError("KU-Optofil does not support truncating the patient population")
        train_loader, val_loader, bundle, input_shape = build_ku_optofil_loaders(
            root=root,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
            train_instance_sample_size=train_instance_sample_size,
            paired_views=(method == "L^2P-AHIL"),
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
                np.concatenate([train_probs, val_probs], axis=0), axis=0,
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
    if is_ref2021_dataset(dataset):
        if bag_build != "random":
            raise ValueError(
                "REF2021 bags are official natural submissions; bag_build must remain 'random'"
            )
        if num_bags is not None:
            raise ValueError("REF2021 does not support truncating the official bag population")
        train_loader, val_loader, bundle, input_shape = build_ref2021_loaders(
            root=root,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
            train_instance_sample_size=train_instance_sample_size,
        )
        if classes is not None and int(classes) != bundle.num_classes:
            raise ValueError(
                f"--n-classes={classes} disagrees with REF2021 ({bundle.num_classes})"
            )
        train_probs = np.asarray(train_loader.dataset.label_prob, dtype=np.float32)
        val_probs = np.asarray(val_loader.dataset.label_prob, dtype=np.float32)
        train_sizes = np.asarray(
            [len(bag.indices) for bag in train_loader.dataset.bags], dtype=np.float64
        )
        val_sizes = np.asarray(
            [len(bag.indices) for bag in val_loader.dataset.bags], dtype=np.float64
        )
        prior_train = np.average(train_probs, axis=0, weights=train_sizes)
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
    if is_twitter_ethnicity_dataset(dataset):
        if bag_build != "random":
            raise ValueError(
                f"{dataset} supports bag_build='random' only. Its bags are natural "
                "counties; 'random' controls only reproducible within-county chunking."
            )
        representation = "end_to_end" if backbone == "Xception" else "precomputed_feature"
        train_loader, val_loader, bundle, input_shape = build_twitter_ethnicity_loaders(
            root=root,
            max_bag_size=bag_size,
            batch_size=batch_size,
            seed=seed,
            num_bags=num_bags,
            num_workers=num_workers,
            method=method,
            holdout_fraction=holdout_fraction,
            representation=representation,
        )
        if classes is not None and int(classes) != bundle.num_classes:
            raise ValueError(
                f"--n-classes={classes} disagrees with Twitter Race3 ({bundle.num_classes})"
            )
        train_probs = np.asarray(train_loader.dataset.label_prob, dtype=np.float32)
        val_probs = np.asarray(val_loader.dataset.label_prob, dtype=np.float32)
        train_sizes = np.asarray(
            [len(bag.indices) for bag in train_loader.dataset.bags], dtype=np.float64
        )
        val_sizes = np.asarray(
            [len(bag.indices) for bag in val_loader.dataset.bags], dtype=np.float64
        )
        prior_train = np.average(train_probs, axis=0, weights=train_sizes)
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
    if is_remote_sensing_dataset(dataset):
        if bag_build not in {"random", "cluster", "alphafirst"}:
            raise ValueError(
                f"{dataset} supports bag_build random, cluster, or alphafirst; "
                f"got {bag_build!r}."
            )
        train_loader, val_loader, bundle = build_remote_sensing_loaders(
            dataset=dataset,
            root=root,
            bag_size=bag_size,
            batch_size=batch_size,
            seed=seed,
            instances_per_epoch=instances_per_epoch,
            num_bags=num_bags,
            num_workers=num_workers,
            method=method,
            bag_build=bag_build,
            alpha0=pi,
            cluster_seed=cluster_seed,
        )
        if classes is not None and int(classes) != bundle.num_classes:
            raise ValueError(
                f"--n-classes={classes} disagrees with {dataset} metadata "
                f"({bundle.num_classes} valid classes)"
            )
        train_probs = np.asarray(train_loader.dataset.label_prob, dtype=np.float32)
        val_probs = np.asarray(val_loader.dataset.label_prob, dtype=np.float32)
        prior_train = train_probs.mean(axis=0)
        prior_val = val_probs.mean(axis=0)
        prior_all = np.concatenate([train_probs, val_probs], axis=0).mean(axis=0)
        return (
            train_loader,
            val_loader,
            train_loader.dataset.label_prob,
            len(bundle.split_indices["train"]),
            len(bundle.split_indices["val"]),
            bundle.input_shape,
            prior_train,
            prior_val,
            prior_all,
        )
    if dataset not in TEXT_DATASETS and _IMAGE_AUGMENTATION_IMPORT_ERROR is not None:
        raise ImportError(
            "Legacy image datasets require the OpenCV augmentation dependency"
        ) from _IMAGE_AUGMENTATION_IMPORT_ERROR

    train_pack, var_pack, prior_train, prior_val, prior_all = load_data_train(
        pi, bag_build, classes, holdout_fraction,
        dataset=dataset, dspth=root, bagsize=bag_size, backbone=backbone,
        seed=seed, cluster_manifest=cluster_manifest,
    )

    data_u, label_prob, labels, label_idx, dataset_length, indices_u, input_dim = train_pack
    (data_u_var, label_prob_var, labels_var, label_idx_var, dataset_length_var, indices_u_var, input_dim_var) = var_pack

    # --- build train ds ---
    if dataset in TEXT_DATASETS and backbone in BERT_MODELS:
        ds_u = TextBagBERT(
            data=data_u, labels=label_prob,
            labels_real=labels, labels_idx=label_idx,
            indices_u=indices_u, mode='train_u_%s' % method,
        )
    elif dataset in TEXT_DATASETS:
        sparse_X, bag_idx_lists = data_u
        ds_u = TextBag(
            sparse_X=sparse_X, bag_indices=bag_idx_lists, labels=label_prob,
            labels_real=labels, labels_idx=label_idx,
            indices_u=indices_u, mode='train_u_%s' % method,
        )
    elif dataset != 'SVHN':
        ds_u = Cifar(
            dataset=dataset,
            data=data_u,
            labels=label_prob,
            labels_real=labels,
            labels_idx=label_idx,
            indices_u=indices_u,
            mode='train_u_%s' % method
        )
    else:
        ds_u = SVHN(
            dataset=dataset,
            data=data_u,
            labels=label_prob,
            labels_real=labels,
            labels_idx=label_idx,
            indices_u=indices_u,
            mode='train_u_%s' % method
        )

    sampler_u = RandomSampler(ds_u, replacement=False)
    batch_sampler_u = BatchSampler(sampler_u, batch_size, drop_last=True)
    dl_u = torch.utils.data.DataLoader(
        ds_u, batch_sampler=batch_sampler_u, num_workers=num_workers, pin_memory=True
    )

    # --- build var ds ---
    if dataset in TEXT_DATASETS and backbone in BERT_MODELS:
        ds_u_var = TextBagBERT(
            data=data_u_var, labels=label_prob_var,
            labels_real=labels_var, labels_idx=label_idx_var,
            indices_u=indices_u_var, mode='train_x',
        )
    elif dataset in TEXT_DATASETS:
        sparse_X_var, bag_idx_lists_var = data_u_var
        ds_u_var = TextBag(
            sparse_X=sparse_X_var, bag_indices=bag_idx_lists_var, labels=label_prob_var,
            labels_real=labels_var, labels_idx=label_idx_var,
            indices_u=indices_u_var, mode='train_x',
        )
    elif dataset != 'SVHN':
        ds_u_var = Cifar(
            dataset=dataset,
            data=data_u_var,
            labels=label_prob_var,
            labels_real=labels_var,
            labels_idx=label_idx_var,
            indices_u=indices_u_var,
            mode='train_x'
        )
    else:
        ds_u_var = SVHN(
            dataset=dataset,
            data=data_u_var,
            labels=label_prob_var,
            labels_real=labels_var,
            labels_idx=label_idx_var,
            indices_u=indices_u_var,
            mode='train_x'
        )

    sampler_u_var = RandomSampler(ds_u_var, replacement=False)
    batch_sampler_u_var = BatchSampler(sampler_u_var, batch_size, drop_last=True)
    dl_u_var = torch.utils.data.DataLoader(
        ds_u_var, batch_sampler=batch_sampler_u_var, num_workers=num_workers, pin_memory=True
    )

    # dataset_length / input_dim 两边一般一样；想严谨就返回两份
    return (dl_u, dl_u_var, label_prob, dataset_length, dataset_length_var, input_dim, prior_train, prior_val, prior_all)


def get_val_loader(dataset, batch_size, num_workers, pin_memory=True, root='data',
                   n_classes=10, backbone=None, seed=0):
    if is_fed_isic2019_dataset(dataset):
        return build_fed_isic2019_eval_loader(
            root=root,
            split="test",
            batch_size=batch_size,
            num_workers=num_workers,
        )
    if is_cct_dataset(dataset):
        return build_cct_eval_loader(
            root=root,
            split="test",
            batch_size=batch_size,
            num_workers=num_workers,
        )
    if is_amazon_wilds_dataset(dataset):
        return build_amazon_wilds_eval_loader(
            root=root,
            split="test",
            batch_size=batch_size,
            num_workers=num_workers,
        )
    if is_ku_optofil_dataset(dataset):
        return build_ku_optofil_eval_loader(
            root=root,
            split="test",
            batch_size=batch_size,
            num_workers=num_workers,
        )
    if is_ref2021_dataset(dataset):
        return build_ref2021_eval_loader(
            root=root,
            split="test",
            batch_size=batch_size,
            num_workers=num_workers,
        )
    if is_twitter_ethnicity_dataset(dataset):
        representation = "end_to_end" if backbone == "Xception" else "precomputed_feature"
        return build_twitter_evaluation_loader(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            representation=representation,
        )
    if is_remote_sensing_dataset(dataset):
        return build_remote_sensing_eval_loader(
            dataset=dataset,
            root=root,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
        )
    if dataset not in TEXT_DATASETS and _IMAGE_AUGMENTATION_IMPORT_ERROR is not None:
        raise ImportError(
            "Legacy image datasets require the OpenCV augmentation dependency"
        ) from _IMAGE_AUGMENTATION_IMPORT_ERROR
    data, labels = load_data_val(dataset, dspth=root, n_classes=n_classes, backbone=backbone)
    if dataset in TEXT_DATASETS and backbone in BERT_MODELS:
        ds = TextValBERT(data=data, labels=labels)
    elif dataset in TEXT_DATASETS:
        ds = TextVal(
            dataset=dataset,
            data=data,
            labels=labels,
            mode='test'
        )
    elif dataset !='SVHN':
        ds = Cifar2(
            dataset=dataset,
            data=data,
            labels=labels,
            mode='test'
        )
    else:
        ds = SVHN2(
            dataset=dataset,
            data=data,
            labels=labels,
            mode='test'
        )
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dl


class SVHN(Dataset):
    def __init__(self, dataset, data, labels, labels_real, labels_idx, indices_u, mode):
        super(SVHN, self).__init__()
        self.data, self.labels, self.labels_real, self.labels_idx, self.indices_u = (data, labels, labels_real, labels_idx, indices_u)
        self.mode = mode
        assert len(self.data) == len(self.labels)

        mean, std = (0.4380, 0.4440, 0.4730), (0.1751, 0.1771, 0.1744)  # SVHN uses different mean and std

        trans_weak = T.Compose([
                T.Resize((32, 32)),  # 调整图像大小为 32x32 像素
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),  # 添加填充并随机裁剪，用于数据增强
                T.RandomAffine(degrees=15, translate=(0.125, 0.125)),
                T.Normalize(mean, std),  # 标准化图像（mean和std是均值和标准差）
                T.ToTensor(),])

        trans_strong0 = T.Compose([
                T.Resize((32, 32)),  # 调整图像大小为 32x32 像素
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),  # 添加填充并随机裁剪，用于数据增强
                RandomAugment(3, 5),
                T.Normalize(mean, std),  # 标准化图像（mean和std是均值和标准差）
                T.ToTensor(),])
        trans_weak_noaug = T.Compose([
                T.Resize((32, 32)),
                T.Normalize(mean, std),
                T.ToTensor(),])
        trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),  # 将张量转换为图像
                transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # 80%的概率应用颜色调整
                transforms.RandomGrayscale(p=0.2),  # 20%的概率转换为灰度图像
                transforms.ToTensor(),  # 将图像转换为张量
                transforms.Normalize(mean, std),])

        if self.mode == "train_x":
            self.trans = OneCropsTransform(trans_weak_noaug)
        elif self.mode == "train_u_co":
            self.trans = ThreeCropsTransform(trans_weak, trans_strong0, trans_strong1)
        elif self.mode == "train_u_L^2P-AHIL":
            self.trans = TwoCropsTransform(trans_weak, trans_strong0)
        elif self.mode == "train_u_DLLP":
            self.trans = OneCropsTransform(trans_weak)
        else:
            if dataset in ["MNIST", "EMNISTBalanced", "FashionMNIST"]:
                self.trans = T.Compose([
                        T1.Resize((64, 64)),
                        T1.Normalize(mean, std),
                        T1.ToTensor(),])
            else:
                self.trans = T.Compose([
                        T.Resize((64, 64)),
                        T.Normalize(mean, std),
                        T.ToTensor(),])

    def __getitem__(self, idx):
        # 获取一组图片和对应的标签
        ims, lb_prob, lb_idx, indices_u = self.data[idx], self.labels[idx], self.labels_idx[idx], self.indices_u[idx]
        labels = self.labels_real[idx]
        # 对图片进行变换，这里假设使用了名为 self.trans 的图像变换函数
        if self.mode == "train_u_co":
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            x_strong0 = torch.stack([self.trans(im)[1] for im in ims])
            x_strong1 = torch.stack([self.trans(im)[2] for im in ims])
            ims_transformed = [x_weak, x_strong0, x_strong1]
            return ims_transformed, lb_prob, labels, lb_idx, indices_u
        elif self.mode == "train_u_L^2P-AHIL":
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            x_strong0 = torch.stack([self.trans(im)[1] for im in ims])
            ims_transformed = [x_weak, x_strong0]
            return ims_transformed, lb_prob, labels, lb_idx, indices_u
        elif self.mode == "train_u_DLLP":
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            ims_transformed = [x_weak]
            return ims_transformed, lb_prob, labels, lb_idx, indices_u
        elif self.mode == "train_x":
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            ims_transformed = [x_weak]
            return ims_transformed, lb_prob, labels, lb_idx, indices_u

    def __len__(self):
        leng = len(self.data)
        return leng


class SVHN2(Dataset):
    def __init__(self, dataset, data, labels, mode):
        super(SVHN2, self).__init__()
        self.data, self.labels = data, labels
        self.mode = mode
        assert len(self.data) == len(self.labels)

        # 根据 SVHN 数据集的均值和标准差进行设置
        mean, std = (0.4380, 0.4440, 0.4730), (0.1751, 0.1771, 0.1744)
        trans_weak = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.Normalize(mean, std),
                T.ToTensor(),])
        trans_strong0 = T.Compose([
                T.Resize((32, 32)),  # 调整图像大小为 32x32 像素
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),  # 添加填充并随机裁剪，用于数据增强
                RandomAugment(2, 10),
                T.Normalize(mean, std),  # 标准化图像（mean和std是均值和标准差）
                T.ToTensor(),])
        trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),])

        if self.mode == "train_x":
            self.trans = trans_weak
        elif self.mode == "train_u_co":
            self.trans = ThreeCropsTransform(trans_weak, trans_strong0, trans_strong1)
        elif self.mode == "train_u_L^2P-AHIL":
            self.trans = TwoCropsTransform(trans_weak, trans_strong0)
        else:
            if dataset in ["MNIST", "EMNISTBalanced", "FashionMNIST"]:
                self.trans = T.Compose([
                        T1.Resize((64, 64)),
                        T1.Normalize(mean, std),
                        T1.ToTensor(),])
            else:
                self.trans = T.Compose([
                        T.Resize((32, 32)),
                        T.Normalize(mean, std),
                        T.ToTensor(),])

    def __getitem__(self, idx):
        im, lb = self.data[idx], self.labels[idx]
        return self.trans(im), lb

    def __len__(self):
        leng = len(self.data)
        return leng


class Cifar(Dataset):
    def __init__(self, dataset, data, labels, labels_real, labels_idx, indices_u, mode):
        super(Cifar, self).__init__()
        self.data, self.labels, self.labels_real, self.labels_idx, self.indices_u = (data, labels, labels_real, labels_idx, indices_u)
        self.mode = mode
        assert len(self.data) == len(self.labels)
        if dataset == "CIFAR10":
            mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        elif dataset == "CIFAR100":
            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        elif dataset == "FashionMNIST":
            mean, std = (0.1307), (0.3081)
        elif dataset == "EMNISTBalanced":
            mean, std = (0.1307), (0.3081)
        elif dataset == "MNIST":
            mean, std = (0.1307), (0.3081)
        elif dataset == "KMNIST":
            mean, std = (0.1307), (0.3081)
        elif dataset == "miniImageNet":
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        if dataset == "CIFAR10" or dataset == "CIFAR100":
            trans_weak = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),])
            trans_weak_noaug = T.Compose([
                T.Resize((32, 32)),
                T.Normalize(mean, std),
                T.ToTensor(),])
            trans_strong0 = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),])

        elif dataset in ["FashionMNIST", "KMNIST"]:
            trans_weak = T1.Compose([
                    T1.Resize((28, 28)),
                    T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                    T1.RandomHorizontalFlip(p=0.5),
                    T1.Normalize(mean, std),
                    transforms.ToTensor(),])
            trans_weak_noaug = T.Compose([
                    T.Resize((28, 28)),
                    T.Normalize(mean, std),
                    T.ToTensor(),])
            trans_strong0 = T.Compose([
                    T1.Resize((28, 28)),
                    T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                    T1.RandomHorizontalFlip(p=0.5),
                    RandomAugment1(2, 10),
                    T1.Normalize(mean, std),
                    transforms.ToTensor(),])
            trans_strong1 = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(28, scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),])

        elif dataset in ["MNIST", "EMNISTBalanced"]:
            trans_weak = T.Compose([
                    T1.Resize((28, 28)),
                    T1.Normalize(mean, std),
                    transforms.ToTensor(),])
            trans_weak_noaug = T.Compose([
                    T.Resize((28, 28)),
                    T.Normalize(mean, std),
                    T.ToTensor(),])
            trans_strong0 = T.Compose([
                    T1.Resize((28, 28)),
                    T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                    T1.RandomAffine(degrees=15, translate=(0.1, 0.1), scale_range=(0.9, 1.1)),
                    RandomAugment1(2, 10),
                    T1.Normalize(mean, std),
                    transforms.ToTensor(),])
            trans_strong1 = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(28, scale=(0.2, 1.0)),
                    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale_range=(0.9, 1.1)),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=0.2),
                    transforms.Normalize(mean, std),])

        elif dataset in ["TinyImageNet"]:
            trans_weak = T.Compose([
                    T.Resize((64, 64)),
                    T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.Normalize(mean, std),
                    T.ToTensor(),])
            trans_weak_noaug = T.Compose([
                    T.Resize((64, 64)),
                    T.Normalize(mean, std),
                    T.ToTensor(),])
            trans_strong0 = T.Compose([
                    T.Resize((64, 64)),
                    T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                    T.RandomHorizontalFlip(p=0.5),
                    RandomAugment(2, 10),
                    T.Normalize(mean, std),
                    T.ToTensor(),])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(64, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),])

        elif dataset in ["miniImageNet"]:
            trans_weak = T.Compose([
                    T.Resize((64, 64)),
                    T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.Normalize(mean, std),
                    T.ToTensor(),])
            trans_weak_noaug = T.Compose([
                    T.Resize((64, 64)),
                    T.Normalize(mean, std),
                    T.ToTensor(),])
            trans_strong0 = T.Compose([
                    T.Resize((64, 64)),
                    T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                    T.RandomHorizontalFlip(p=0.5),
                    RandomAugment(2, 10),
                    T.Normalize(mean, std),
                    T.ToTensor(),])
            trans_strong1 = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(64, scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),])

        if self.mode == "train_x":
            self.trans = OneCropsTransform(trans_weak_noaug)
        elif self.mode == "train_u_DLLP":
            self.trans = OneCropsTransform(trans_weak)
        elif self.mode == "train_u_co":
            self.trans = ThreeCropsTransform(trans_weak, trans_strong0, trans_strong1)
        elif self.mode == "train_u_L^2P-AHIL":

            self.trans = TwoCropsTransform(trans_weak, trans_strong0)
        else:
            if dataset in ["MNIST", "EMNISTBalanced", "FashionMNIST", "KMNIST"]:
                self.trans = T.Compose([
                        T1.Resize((28, 28)),
                        T1.Normalize(mean, std),
                        T1.ToTensor(),])
            elif dataset in ["CIFAR10", "CIFAR100"]:
                self.trans = T.Compose([
                        T.Resize((32, 32)),
                        T.Normalize(mean, std),
                        T.ToTensor(),])
            else:
                self.trans = T.Compose([
                        T.Resize((64, 64)),
                        T.Normalize(mean, std),
                        T.ToTensor(),])

    def __getitem__(self, idx):
        # 获取一组图片和对应的标签
        ims, lb_prob,lb_idx,indices_u = self.data[idx], self.labels[idx],self.labels_idx[idx],self.indices_u[idx]
        labels = self.labels_real[idx]
        # 对图片进行变换，这里假设使用了名为 self.trans 的图像变换函数
        if self.mode == "train_u_co":
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            x_strong0 = torch.stack([self.trans(im)[1] for im in ims])
            x_strong1 = torch.stack([self.trans(im)[2] for im in ims])
            ims_transformed = [x_weak, x_strong0, x_strong1]
            return ims_transformed, lb_prob, lb_idx, indices_u, labels
        elif self.mode == "train_u_L^2P-AHIL":
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            x_strong0 = torch.stack([self.trans(im)[1] for im in ims])

            ims_transformed = [x_weak, x_strong0]
            return ims_transformed, lb_prob, lb_idx, indices_u, labels
        elif self.mode == "train_u_DLLP":
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            ims_transformed = [x_weak]
            return ims_transformed, lb_prob, lb_idx, indices_u, labels
        elif self.mode == "train_x":
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            ims_transformed = [x_weak]
            return ims_transformed, lb_prob, lb_idx, indices_u, labels

    def __len__(self):
        leng = len(self.data)
        return leng


class Cifar2(Dataset):
    def __init__(self, dataset, data, labels, mode):
        super(Cifar2, self).__init__()
        self.data, self.labels = data, labels
        self.mode = mode
        assert len(self.data) == len(self.labels)
        if dataset == "CIFAR10":
            mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        elif dataset == "CIFAR100":
            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        elif dataset == "FashionMNIST":
            mean, std = (0.1307), (0.3081)
        elif dataset == "EMNISTBalanced":
            mean, std = (0.1307), (0.3081)
        elif dataset == "MNIST":
            mean, std = (0.1307), (0.3081)
        elif dataset == "KMNIST":
            mean, std = (0.1307), (0.3081)
        elif dataset == "miniImageNet":
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        if dataset == "CIFAR10" or dataset == "CIFAR100":
            trans_weak = T.Compose([
                    T.Resize((32, 32)),
                    T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.Normalize(mean, std),
                    T.ToTensor(),])
            trans_strong0 = T.Compose([
                    T.Resize((32, 32)),
                    T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                    T.RandomHorizontalFlip(p=0.5),
                    RandomAugment(2, 10),
                    T.Normalize(mean, std),
                    T.ToTensor(),])
            trans_strong1 = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),])

        elif dataset in ["FashionMNIST", "KMNIST"]:
            trans_weak = T.Compose([
                    T1.Resize((28, 28)),
                    T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                    T1.RandomHorizontalFlip(p=0.5),
                    T1.Normalize(mean, std),
                    transforms.ToTensor(),])
            trans_strong0 = T.Compose([
                    T1.Resize((28, 28)),
                    T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                    T1.RandomHorizontalFlip(p=0.5),
                    RandomAugment1(2, 10),
                    T1.Normalize(mean, std),
                    transforms.ToTensor(),])
            trans_strong1 = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(28, scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),])

        elif dataset in ["MNIST", "EMNISTBalanced"]:
            trans_weak = T.Compose([
                    T1.Resize((28, 28)),
                    T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                    T1.RandomAffine(degrees=15, translate=(0.1, 0.1), scale_range=(0.9, 1.1)),
                    T1.Normalize(mean, std),
                    transforms.ToTensor(),])
            trans_strong0 = T.Compose([
                    T1.Resize((28, 28)),
                    T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                    T1.RandomAffine(degrees=15, translate=(0.1, 0.1), scale_range=(0.9, 1.1)),
                    RandomAugment1(2, 10),
                    T1.Normalize(mean, std),
                    transforms.ToTensor(),])
            trans_strong1 = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(28, scale=(0.2, 1.0)),
                    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale_range=(0.9, 1.1)),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=0.2),
                    transforms.Normalize(mean, std),])

        elif dataset in ["TinyImageNet"]:
            trans_weak = T.Compose([
                    T.Resize((64, 64)),
                    T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.Normalize(mean, std),
                    T.ToTensor(),])
            trans_strong0 = T.Compose([
                    T.Resize((64, 64)),
                    T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                    T.RandomHorizontalFlip(p=0.5),
                    RandomAugment(2, 10),
                    T.Normalize(mean, std),
                    T.ToTensor(),])
            trans_strong1 = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(64, scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),])

        elif dataset in ["miniImageNet"]:
            trans_weak = T.Compose([
                    T.Resize((64, 64)),
                    T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.Normalize(mean, std),
                    T.ToTensor(),])
            trans_strong0 = T.Compose([
                    T.Resize((64, 64)),
                    T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                    T.RandomHorizontalFlip(p=0.5),
                    RandomAugment(2, 10),
                    T.Normalize(mean, std),
                    T.ToTensor(),])
            trans_strong1 = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(64, scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),])

        if self.mode == "train_x":
            self.trans = trans_weak
        elif self.mode == "train_u_co":
            self.trans = ThreeCropsTransform(trans_weak, trans_strong0, trans_strong1)
        elif self.mode == "train_u_L^2P-AHIL":
            self.trans = TwoCropsTransform(trans_weak, trans_strong0)
        else:
            if dataset in ["MNIST", "EMNISTBalanced", "FashionMNIST", "KMNIST"]:
                self.trans = T.Compose([
                        T1.Resize((28, 28)),
                        T1.Normalize(mean, std),
                        T1.ToTensor(),])
            elif dataset in ["CIFAR10", "CIFAR100", "SVHN"]:
                self.trans = T.Compose([
                        T.Resize((32, 32)),
                        T.Normalize(mean, std),
                        T.ToTensor(),])
            else:
                self.trans = T.Compose([
                        T.Resize((64, 64)),
                        T.Normalize(mean, std),
                        T.ToTensor(),])

    def __getitem__(self, idx):
        im, lb = self.data[idx], self.labels[idx]
        return self.trans(im), lb

    def __len__(self):
        leng = len(self.data)
        return leng


# ─────────────────────────────────────────────────────────────────────────────
# Text Dataset classes
# ─────────────────────────────────────────────────────────────────────────────

class TextBag(Dataset):
    """
    Training dataset for text — each item is one bag of TF-IDF feature vectors.

    Accepts a scipy CSR sparse matrix (sparse_X) and per-bag row-index lists
    (bag_indices).  Rows are densified only at __getitem__ time so memory stays
    proportional to the batch rather than the full dataset.

    __getitem__ return format mirrors Cifar exactly:
        (ims_transformed, lb_prob, lb_idx, indices_u, labels)

    ims_transformed is a list of FloatTensors each shape (bagsize, D):
        'train_u_co'          → [x_weak, x_strong0, x_strong1]
        'train_u_L^2P-AHIL'   → [x_weak, x_strong0]
        'train_u_DLLP'        → [x_weak]
        'train_x'             → [x_noaug]   (no dropout — holdout val bags)

    Augmentation is feature-space:
        weak    — 10 % feature dropout + tiny noise
        strong0 — 30 % feature dropout + larger noise   (RandAugment analogue)
        strong1 — contiguous 30 % block masking + noise (SimCLR analogue)
    """

    def __init__(self, sparse_X, bag_indices, labels, labels_real, labels_idx, indices_u, mode):
        super().__init__()
        self.sparse_X   = sparse_X       # scipy CSR, shape (N_train, D)
        self.bag_indices = bag_indices   # list[list[int]] — row indices per bag
        self.labels, self.labels_real    = labels, labels_real
        self.labels_idx, self.indices_u  = labels_idx, indices_u
        self.mode = mode
        assert len(self.bag_indices) == len(self.labels)

    @staticmethod
    def _t(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).float()

    def _bag_dense(self, idx):
        """Return (bagsize, D) float32 dense array for bag at idx."""
        row_ids = self.bag_indices[idx]          # list of int indices
        rows = self.sparse_X[row_ids]            # (bagsize, D) sparse
        return np.asarray(rows.todense(), dtype=np.float32)  # (bagsize, D) dense

    def __getitem__(self, idx):
        bag_arr   = self._bag_dense(idx)         # (bagsize, D) float32
        ims       = [bag_arr[i] for i in range(len(bag_arr))]  # list of (D,)
        lb_prob   = self.labels[idx]
        lb_idx    = self.labels_idx[idx]
        indices_u = self.indices_u[idx]
        labels    = self.labels_real[idx]

        if self.mode == "train_u_co":
            x_w  = torch.stack([self._t(_text_aug_weak(im))    for im in ims])
            x_s0 = torch.stack([self._t(_text_aug_strong0(im)) for im in ims])
            x_s1 = torch.stack([self._t(_text_aug_strong1(im)) for im in ims])
            ims_transformed = [x_w, x_s0, x_s1]
        elif self.mode == "train_u_L^2P-AHIL":
            x_w  = torch.stack([self._t(_text_aug_weak(im))    for im in ims])
            x_s0 = torch.stack([self._t(_text_aug_strong0(im)) for im in ims])
            ims_transformed = [x_w, x_s0]
        elif self.mode == "train_u_DLLP":
            x_w  = torch.stack([self._t(_text_aug_weak(im)) for im in ims])
            ims_transformed = [x_w]
        else:   # train_x — no augmentation (holdout val bags)
            x_w  = torch.stack([self._t(im.copy()) for im in ims])
            ims_transformed = [x_w]

        return ims_transformed, lb_prob, lb_idx, indices_u, labels

    def __len__(self):
        return len(self.bag_indices)


class TextVal(Dataset):
    """Evaluation dataset for text — each item is one individual sample.

    Accepts a scipy CSR sparse matrix; rows are densified one at a time.
    """

    def __init__(self, dataset, data, labels, mode):
        super().__init__()
        self.data, self.labels = data, labels  # data is scipy CSR
        self.mode = mode
        assert data.shape[0] == len(labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(
            np.asarray(self.data[idx].todense(), dtype=np.float32)[0]
        ).float()
        y = int(self.labels[idx])
        return x, y

    def __len__(self):
        return self.data.shape[0]


# ─────────────────────────────────────────────────────────────────────────────
# BERT-mode Dataset classes
# ─────────────────────────────────────────────────────────────────────────────

class TextBagBERT(Dataset):
    """Training dataset for text with BERT backbone.

    Each bag stores a list of (3, max_length) int64 arrays:
        axis 0 → view index: 0=original, 1=back_translation, 2=synonym_aug
        axis 1 → token ID sequence

    __getitem__ returns the same tuple structure as TextBag so train.py needs
    no changes:
        (ims_transformed, lb_prob, lb_idx, indices_u, labels)

    ims_transformed:
        'train_u_co'         → [x_w, x_s0, x_s1]  LongTensor (bagsize, L)
        'train_u_L^2P-AHIL'  → [x_w, x_s0]
        'train_u_DLLP'       → [x_w]
        'train_x'            → [x_w]   (no aug)
    """

    def __init__(self, data, labels, labels_real, labels_idx, indices_u, mode):
        super().__init__()
        # data: list of bags; each bag is a list of (3, L) int64 np.ndarray
        self.data        = data
        self.labels      = labels
        self.labels_real = labels_real
        self.labels_idx  = labels_idx
        self.indices_u   = indices_u
        self.mode        = mode
        assert len(self.data) == len(self.labels)

    def _bag_tensor(self, idx, view: int) -> torch.Tensor:
        """Stack token-ID view `view` across all samples in bag → (bagsize, L)."""
        samples = self.data[idx]   # list of (3, L) arrays
        return torch.from_numpy(
            np.stack([s[view] for s in samples], axis=0)
        ).long()   # (bagsize, L)

    def __getitem__(self, idx):
        lb_prob   = self.labels[idx]
        lb_idx    = self.labels_idx[idx]
        indices_u = self.indices_u[idx]
        labels    = self.labels_real[idx]

        x_w  = self._bag_tensor(idx, 0)   # original
        x_s0 = self._bag_tensor(idx, 1)   # back_translation (or same as original)
        x_s1 = self._bag_tensor(idx, 2)   # synonym_aug      (or same as original)

        if self.mode == "train_u_co":
            ims_transformed = [x_w, x_s0, x_s1]
        elif self.mode == "train_u_L^2P-AHIL":
            ims_transformed = [x_w, x_s0]
        elif self.mode == "train_u_DLLP":
            ims_transformed = [x_w]
        else:
            ims_transformed = [x_w]

        return ims_transformed, lb_prob, lb_idx, indices_u, labels

    def __len__(self):
        return len(self.data)


class TextValBERT(Dataset):
    """Evaluation dataset for text with BERT backbone.

    data: int64 numpy array of shape (N, max_length) — padded token IDs.
    Returns (LongTensor(L,), int) per item.
    """

    def __init__(self, data: np.ndarray, labels):
        super().__init__()
        self.data   = data    # (N, L) int64
        self.labels = labels

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).long(), int(self.labels[idx])

    def __len__(self):
        return len(self.data)
