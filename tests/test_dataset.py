"""Unit tests for the data pipeline in dataset.py.

These exercise the augmentation transforms and both Dataset classes on small
synthetic images.
"""
import numpy as np
import cv2
import torch

from dataset import (
    train_transform,
    val_transform,
    AIDetectionDataset,
    TTADetectionDataset,
)


def _write_jpeg(path, h=480, w=640):
    """Write a random RGB image to `path` as JPEG and return the path."""
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    assert cv2.imwrite(path, img), "cv2 failed to write the test JPEG"
    return path


def test_train_transform_outputs_normalized_tensor():
    img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    out = train_transform(image=img)["image"]
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)        # random 224x224 crop, CHW tensor
    assert out.dtype == torch.float32


def test_val_transform_outputs_center_crop():
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    out = val_transform(image=img)["image"]
    assert out.shape == (3, 224, 224)


def test_ai_detection_dataset_item(tmp_path):
    p = _write_jpeg(str(tmp_path / "sample.jpg"))
    ds = AIDetectionDataset([p], [1], transform=train_transform)

    assert len(ds) == 1
    image, label, path = ds[0]
    assert image.shape == (3, 224, 224)
    assert label == 1
    assert path == p


def test_tta_dataset_stacks_five_crops(tmp_path):
    p = _write_jpeg(str(tmp_path / "big.jpg"), h=800, w=800)
    ds = TTADetectionDataset([p], [0], crop_size=384)

    crops, label, path = ds[0]
    assert crops.shape == (5, 3, 384, 384)   # 4 corners + centre crop
    assert label == 0
    assert path == p


def test_tta_dataset_pads_small_images(tmp_path):
    # Image smaller than the crop must be padded up first, not crash.
    p = _write_jpeg(str(tmp_path / "small.jpg"), h=100, w=120)
    ds = TTADetectionDataset([p], [1], crop_size=384)

    crops, _, _ = ds[0]
    assert crops.shape == (5, 3, 384, 384)
