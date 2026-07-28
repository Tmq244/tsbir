import random

import numpy as np
import torch
from PIL import Image, ImageDraw

from tsbir.data import (
    DEFAULT_HUMAN_STYLE_CONFIG,
    human_style_augment,
    paired_train_sketch_transform,
)


def _dense_sketch() -> Image.Image:
    image = Image.new("L", (128, 128), 255)
    draw = ImageDraw.Draw(image)
    for offset in range(10, 120, 18):
        draw.line([(5, offset), (122, 128 - offset)], fill=0, width=4)
    draw.ellipse((24, 20, 105, 105), outline=20, width=5)
    draw.rectangle((42, 38, 83, 86), outline=0, width=4)
    return image


def _all_operations_config() -> dict:
    config = dict(DEFAULT_HUMAN_STYLE_CONFIG)
    for key in (
        "width_probability",
        "elastic_probability",
        "gap_probability",
        "simplify_probability",
        "small_structure_probability",
        "segment_probability",
    ):
        config[key] = 1.0
    return config


def test_human_style_is_deterministic_and_respects_shared_keep_budget():
    image = _dense_sketch()
    config = _all_operations_config()
    random.seed(2026)
    np.random.seed(2026)
    first, first_keep = human_style_augment(image, config, target_keep=0.6)
    random.seed(2026)
    np.random.seed(2026)
    second, second_keep = human_style_augment(image, config, target_keep=0.6)

    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert first_keep == second_keep
    assert 0.6 <= first_keep < 0.61
    assert np.count_nonzero(np.asarray(first) < 245) > 0


def test_two_human_style_views_are_independent_but_equally_complete():
    image = _dense_sketch()
    config = _all_operations_config()
    random.seed(17)
    np.random.seed(17)
    first, first_keep = human_style_augment(image, config, target_keep=0.75)
    second, second_keep = human_style_augment(image, config, target_keep=0.75)

    assert not np.array_equal(np.asarray(first), np.asarray(second))
    assert abs(first_keep - 0.75) < 0.01
    assert abs(second_keep - 0.75) < 0.01


def test_paired_transform_uses_identical_affine_and_crop_parameters():
    image = _dense_sketch()
    torch.manual_seed(99)
    first, second = paired_train_sketch_transform(image, image.copy(), 224)
    assert first.shape == (3, 224, 224)
    assert torch.equal(first, second)
