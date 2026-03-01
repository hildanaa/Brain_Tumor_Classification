import torch
import torch.nn as nn
from torchvision import models

def build_resnet18(num_classes: int = 4, pretrained: bool = True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Para Grad-CAM en ResNet: target típico = model.layer4
    target_layer = model.layer4

    return model, target_layer

def set_trainable_head_only(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

def unfreeze_layer4_and_head(model: torch.nn.Module):
    for name, p in model.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            p.requires_grad = True