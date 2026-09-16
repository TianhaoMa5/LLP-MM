import torch
import torch.nn as nn
import torch.nn.functional as F
from ..lib import resnet
from ..lib.WideResNet import WideResnet
from torchvision import models

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.n_outputs = hidden_dim

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        return out


class IdentityFeatures(nn.Module):
    """Pass flattened frozen features directly to the shared classifier."""

    def __init__(self, input_dim):
        super().__init__()
        self.n_outputs = int(input_dim)

    def forward(self, x):
        return x.view(x.size(0), -1)


class XceptionFeaturizer(nn.Module):
    """ImageNet Xception with global average pooling for 299x299 RGB users."""

    def __init__(self, freeze_first_two_blocks=True, pretrained=True):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "The optional end-to-end Twitter Xception mode requires timm. "
                "Install with: pip install -e '.[twitter]'"
            ) from exc
        self.backbone = timm.create_model(
            "legacy_xception",
            pretrained=bool(pretrained),
            num_classes=0,
            global_pool="avg",
        )
        self.n_outputs = int(getattr(self.backbone, "num_features", 2048))
        if self.n_outputs != 2048:
            raise ValueError(f"Xception feature dimension must be 2048, got {self.n_outputs}")
        if freeze_first_two_blocks:
            frozen_prefixes = (
                "conv1.", "bn1.", "conv2.", "bn2.", "block1.", "block2."
            )
            for name, parameter in self.backbone.named_parameters():
                if name.startswith(frozen_prefixes):
                    parameter.requires_grad_(False)

    def forward(self, x):
        return self.backbone(x)


# ─────────────────────────────────────────────────────────────────────────────
# BERT featurizers (bert-base-uncased / prajjwal1/bert-small)
#
# Input:  either
#   (a) a dict  {"input_ids": LongTensor(N, L),
#                "attention_mask": LongTensor(N, L)}   — preferred
#   (b) a plain LongTensor(N, L) of input_ids          — attention_mask inferred
#
# Output: FloatTensor(N, hidden_size)   — [CLS] pooled representation
#
# n_outputs: 768 for BERT-base, 512 for BERT-small
# ─────────────────────────────────────────────────────────────────────────────

class BERTFeaturizer(nn.Module):
    """Generic BERT backbone with [CLS] pooling.

    Args:
        model_name: HuggingFace model identifier, e.g.
                    'bert-base-uncased'  or  'prajjwal1/bert-small'
        freeze_layers: number of transformer layers to freeze from the bottom
                       (0 = fine-tune everything, -1 = freeze all except pooler)
    """

    def __init__(self, model_name: str, freeze_layers: int = 0):
        super().__init__()
        from transformers import BertModel, BertConfig
        self.bert = BertModel.from_pretrained(model_name)
        self.n_outputs = self.bert.config.hidden_size

        if freeze_layers == -1:
            # Freeze everything except the pooler
            for name, param in self.bert.named_parameters():
                if "pooler" not in name:
                    param.requires_grad_(False)
        elif freeze_layers > 0:
            # Freeze embeddings + bottom-k encoder layers
            for param in self.bert.embeddings.parameters():
                param.requires_grad_(False)
            for layer in self.bert.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad_(False)

    def forward(self, x):
        """
        x: dict with 'input_ids' (and optionally 'attention_mask')
           OR a plain LongTensor of shape (N, seq_len)
        Returns: (N, hidden_size) float tensor — [CLS] token embedding
        """
        if isinstance(x, dict):
            input_ids      = x["input_ids"]
            attention_mask = x.get("attention_mask", None)
        else:
            input_ids      = x
            attention_mask = (x != 0).long()  # mask padding (token_id 0)

        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]   # [CLS] embedding


def BERTBase(freeze_layers: int = 0) -> BERTFeaturizer:
    """BERT-base-uncased: 12 layers, hidden=768, n_outputs=768."""
    return BERTFeaturizer("bert-base-uncased", freeze_layers=freeze_layers)


def BERTSmall(freeze_layers: int = 0) -> BERTFeaturizer:
    """BERT-small (prajjwal1/bert-small): 4 layers, hidden=512, n_outputs=512."""
    return BERTFeaturizer("prajjwal1/bert-small", freeze_layers=freeze_layers)


# ─────────────────────────────────────────────────────────────────────────────

def calc_dim(input_shape):
    if isinstance(input_shape, int):
        return input_shape
    input_shape = input_shape[1:]
    num_features = 1
    for s in input_shape:
        num_features *= s
    return num_features


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    model = hparams["model"]

    if model == "MLP":
        dim = calc_dim(input_shape)
        return MLP(dim, 500)
    elif model == "Linear":
        dim = calc_dim(input_shape)
        return IdentityFeatures(dim)
    elif model == "LeNet":
        return LeNet()
    elif model == "ResNet":
        m = models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.n_outputs = m.fc.in_features
        m.fc = nn.Identity()
        m.pretrained_weights = None
        m.stem_variant = "modified_3x3_stride1_no_maxpool"
        if tuple(input_shape) == (3, 112, 112):
            print(
                "CCT classifier backbone: ResNet-18 modified stem "
                "(3x3/stride-1, no max-pool); pretrained_weights=None; "
                "input_resolution=112",
                flush=True,
            )
        return m
    elif model == "CCTResNet18":
        if tuple(input_shape) != (3, 112, 112):
            raise ValueError(f"CCTResNet18 expects (3,112,112), got {input_shape!r}")
        pretrained = bool(hparams.get("pretrained", True))
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        # Keep the canonical ImageNet 7x7/stride-2 stem and max-pool. The old
        # CCT path incorrectly inherited the CIFAR-specific 3x3 stem.
        m = models.resnet18(weights=weights)
        m.n_outputs = m.fc.in_features
        m.fc = nn.Identity()
        m.pretrained_weights = "IMAGENET1K_V1" if pretrained else None
        m.stem_variant = "standard_imagenet_7x7_stride2_maxpool"
        print(
            "CCT classifier backbone: ResNet-18 standard ImageNet stem; "
            f"pretrained_weights={m.pretrained_weights}; input_resolution=112",
            flush=True,
        )
        return m
    elif model == "ImageNetResNet18":
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 3:
            raise ValueError(f"ImageNetResNet18 expects (C,H,W), got {input_shape!r}")
        if int(input_shape[0]) != 3:
            raise ValueError(f"ImageNetResNet18 expects RGB input, got {input_shape!r}")
        pretrained = bool(hparams.get("pretrained", True))
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=weights)
        m.n_outputs = m.fc.in_features
        m.fc = nn.Identity()
        m.pretrained_weights = "IMAGENET1K_V1" if pretrained else None
        m.stem_variant = "standard_imagenet_7x7_stride2_maxpool"
        return m
    elif model == "ResNet_miniImageNet":
        m = models.resnet34(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.n_outputs = m.fc.in_features
        m.fc = nn.Identity()
        return m
    elif model == "RemoteResNet18":
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 3:
            raise ValueError(f"RemoteResNet18 expects (C,H,W), got {input_shape!r}")
        m = models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(
            int(input_shape[0]), 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        m.maxpool = nn.Identity()
        m.n_outputs = m.fc.in_features
        m.fc = nn.Identity()
        return m
    elif model == "Xception":
        if tuple(input_shape) != (3, 299, 299):
            raise ValueError(f"Xception expects (3,299,299), got {input_shape!r}")
        return XceptionFeaturizer(
            freeze_first_two_blocks=hparams.get(
                "xception_freeze_first_two_blocks", True
            ),
            pretrained=hparams.get("xception_pretrained", True),
        )
    elif model == "BERT_base":
        freeze = int(hparams.get("bert_freeze_layers", 0))
        return BERTBase(freeze_layers=freeze)
    elif model == "BERT_small":
        freeze = int(hparams.get("bert_freeze_layers", 0))
        return BERTSmall(freeze_layers=freeze)
    else:
        raise NotImplementedError(f"Unknown model: {model!r}")


def Classifier(in_features, out_features):
    return torch.nn.Linear(in_features, out_features)
