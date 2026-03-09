#!/usr/bin/env python3

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms

MODEL_CONFIGS = [
    {
        "name": "ResNet18",
        "type": "torchvision",
        "weights_path": "/mnt/d/skin_cancer_project/models/torchvision/resnet18.pth",
        "feat_dim": 512,
        "loader": "resnet18",
        "feat_dir": "features_4class_resnet18",
        "batch_size": 64,
    },
    {
        "name": "ResNet50",
        "type": "torchvision",
        "weights_path": "/mnt/d/skin_cancer_project/models/torchvision/resnet50.pth",
        "feat_dim": 2048,
        "loader": "resnet50",
        "feat_dir": "features_4class_resnet50",
        "batch_size": 48,
    },
    {
        "name": "ConvNeXt-Small",
        "type": "torchvision",
        "weights_path": "/mnt/d/skin_cancer_project/models/torchvision/convnext_small.pth",
        "feat_dim": 768,
        "loader": "convnext_small",
        "feat_dir": "features_4class_convnext_small",
        "batch_size": 32,
    },
    {
        "name": "ConvNeXt-Base",
        "type": "torchvision",
        "weights_path": "/mnt/d/skin_cancer_project/models/torchvision/convnext_base.pth",
        "feat_dim": 1024,
        "loader": "convnext_base",
        "feat_dir": "features_4class_convnext_base",
        "batch_size": 24,
    },
    {
        "name": "DINOv2-base",
        "type": "dinov2",
        "weights_path": "/mnt/d/skin_cancer_project/models/vision/dinov2-base",
        "feat_dim": 768,
        "loader": "dinov2",
        "feat_dir": "features_4class_dinov2_base",
        "batch_size": 32,
    },
    {
        "name": "Phikon",
        "type": "phikon",
        "weights_path": "/mnt/d/skin_cancer_project/models/pathology/phikon",
        "feat_dim": 768,
        "loader": "phikon",
        "feat_dir": "features_4class_phikon",
        "batch_size": 32,
    },
    {
        "name": "UNI",
        "type": "uni",
        "weights_path": "/mnt/d/skin_cancer_project/models/pathology/uni/pytorch_model.bin",
        "feat_dim": 1024,
        "loader": "uni",
        "feat_dir": "features_4class_uni",
        "batch_size": 16,
    },
    {
        "name": "CONCH",
        "type": "conch",
        "weights_path": "/mnt/d/skin_cancer_project/models/pathology/conch/pytorch_model.bin",
        "feat_dim": 512,
        "loader": "conch",
        "feat_dir": "features_4class_conch",
        "batch_size": 24,
    },
]


def safe_model_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def feature_dir_name(model_cfg: dict) -> str:
    feat_dir = model_cfg.get("feat_dir")
    if feat_dir:
        return feat_dir
    return "features_4class_" + safe_model_name(model_cfg["name"])


def _load_torch_weights(path: str, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_feature_extractor(model_cfg: dict, device: str):
    default_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model_type = model_cfg["type"]
    weights_path = model_cfg["weights_path"]
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    if model_type == "torchvision":
        loader_name = model_cfg["loader"]
        model_fn = getattr(models, loader_name)
        model = model_fn()
        model.load_state_dict(_load_torch_weights(weights_path, device))
        if "resnet" in loader_name:
            model.fc = nn.Identity()
        elif "convnext" in loader_name:
            model.classifier = nn.Sequential(
                model.classifier[0],
                nn.Flatten(1),
            )
        transform = default_transform
    elif model_type in ("dinov2", "phikon"):
        from transformers import AutoModel
        model = AutoModel.from_pretrained(weights_path, local_files_only=True)
        transform = default_transform
    elif model_type == "uni":
        import timm
        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        model.load_state_dict(_load_torch_weights(weights_path, "cpu"), strict=True)
        transform = default_transform
    elif model_type == "conch":
        from conch.open_clip_custom import create_model_from_pretrained
        model, transform = create_model_from_pretrained("conch_ViT-B-16", weights_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = model.to(device)
    model.eval()
    return model, transform


def extract_batch_features(model, model_cfg: dict, batch):
    model_type = model_cfg["type"]
    with torch.no_grad():
        if model_type in ("dinov2", "phikon"):
            out = model(batch)
            feats = out.last_hidden_state[:, 0, :]
        elif model_type == "conch":
            feats = model.encode_image(batch, proj_contrast=False, normalize=False)
        else:
            feats = model(batch)
    return feats