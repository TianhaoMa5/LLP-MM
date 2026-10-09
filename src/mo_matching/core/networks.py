from torch import nn
from torchvision import models


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    model = hparams["model"]
    if model == "Linear":
        dimension = (
            input_shape
            if isinstance(input_shape, int)
            else __import__("math").prod(input_shape[1:])
        )
        module = nn.Flatten(start_dim=1)
        module.n_outputs = dimension
        return module
    if model == "ResNet":
        m = models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.n_outputs = m.fc.in_features
        m.fc = nn.Identity()
        m.pretrained_weights = None
        m.stem_variant = "modified_3x3_stride1_no_maxpool"
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
    else:
        raise NotImplementedError(f"Unknown model: {model!r}")


def Classifier(in_features, out_features):
    return nn.Linear(in_features, out_features)
