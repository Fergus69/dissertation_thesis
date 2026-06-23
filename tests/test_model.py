"""Unit tests for model.py.

The pretrained weights for ConvNeXt-Tiny are ~110 MB, so these tests build the
architecture with `pretrained=False` and monkeypatch timm so that build_model()
does not download anything. They verify the classifier head (2 classes) and the
input/output contract of a forward pass.
"""
import torch
import timm

from model import build_model

MODEL_NAME = "convnext_tiny.fb_in22k_ft_in1k"


def test_architecture_forward_shape():
    net = timm.create_model(MODEL_NAME, pretrained=False, num_classes=2)
    net.eval()
    with torch.no_grad():
        out = net(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 2)            # binary head: real vs. AI


def test_build_model_contract(monkeypatch):
    # Force pretrained=False inside build_model so CI never hits the network.
    real_create = timm.create_model

    def offline_create(name, *args, **kwargs):
        kwargs["pretrained"] = False
        return real_create(name, *args, **kwargs)

    monkeypatch.setattr(timm, "create_model", offline_create)

    net, criterion, optimizer, device = build_model()

    assert isinstance(net, torch.nn.Module)
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(device, torch.device)

    with torch.no_grad():
        out = net(torch.randn(1, 3, 224, 224).to(device))
    assert out.shape == (1, 2)
