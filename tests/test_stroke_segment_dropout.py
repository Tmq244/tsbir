import random

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from tsbir.data import stroke_dropout, stroke_segment_dropout


def _test_sketch() -> Image.Image:
    image = Image.new("L", (128, 128), 255)
    draw = ImageDraw.Draw(image)
    draw.line([(8, 20), (118, 20), (70, 70), (118, 110)], fill=0, width=5)
    draw.line([(10, 105), (55, 55), (105, 105)], fill=32, width=4)
    draw.ellipse((42, 35, 88, 82), outline=0, width=4)
    return image


def _numpy_states_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_segment_mode_consumes_exactly_the_pixel_mode_rng_calls():
    image = _test_sketch()
    random.seed(2026)
    np.random.seed(2026)
    stroke_dropout(image)
    pixel_python_state = random.getstate()
    pixel_numpy_state = np.random.get_state()
    pixel_next = (random.random(), np.random.random())

    random.seed(2026)
    np.random.seed(2026)
    stroke_segment_dropout(image)
    segment_python_state = random.getstate()
    segment_numpy_state = np.random.get_state()
    segment_next = (random.random(), np.random.random())

    assert pixel_python_state == segment_python_state
    assert _numpy_states_equal(pixel_numpy_state, segment_numpy_state)
    assert pixel_next == segment_next


def test_segment_dropout_is_deterministic_coherent_and_keeps_at_least_60_percent():
    image = _test_sketch()
    before = np.asarray(image) < 245
    random.seed(11)
    np.random.seed(11)
    first = np.asarray(stroke_segment_dropout(image, maximum_segment_length=24))
    random.seed(11)
    np.random.seed(11)
    second = np.asarray(stroke_segment_dropout(image, maximum_segment_length=24))

    assert np.array_equal(first, second)
    after = first < 245
    assert 0.6 <= after.sum() / before.sum() <= 1.0
    removed = before & ~after
    assert removed.any()
    component_labels, component_count = ndimage.label(
        removed, structure=np.ones((3, 3), dtype=np.uint8),
    )
    component_areas = np.bincount(component_labels.ravel())
    assert component_count < int(removed.sum())
    assert component_areas[1:].max() >= 8


def test_blank_sketch_stays_blank_and_preserves_rng_contract():
    image = Image.new("L", (32, 32), 255)
    random.seed(7)
    np.random.seed(7)
    output = stroke_segment_dropout(image)
    assert np.all(np.asarray(output) == 255)
    assert random.random() == 0.15084917392450192
    assert np.isclose(np.random.random(), 0.037259982082638365)
