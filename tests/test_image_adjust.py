"""core/image_mesh.py（画像の調整）のテスト。"""

import numpy as np
import pytest

from core.image_mesh import (
    ImageAdjust,
    ImageMeshError,
    load_brightness,
    make_lithophane,
    make_relief,
    top_heights,
)


def _gray(value, rows=40, cols=80):
    return np.full((rows, cols), float(value))


def _gradient(rows=40, cols=80):
    """左が黒（0）、右が白（255）の横グラデーション。"""
    return np.tile(np.linspace(0, 255, cols), (rows, 1))


def _load(img, **adjust):
    return load_brightness(img, width=20, resolution=0.5, adjust=ImageAdjust(**adjust))


# ---- 明るさ・コントラスト ----

def test_default_is_unchanged():
    assert np.allclose(_load(_gradient()), load_brightness(_gradient(), 20, 0.5))


@pytest.mark.parametrize("brightness, expected", [(20, 0.7), (-20, 0.3), (100, 1.0), (-100, 0.0)])
def test_brightness(brightness, expected):
    assert _load(_gray(127.5), brightness=brightness).mean() == pytest.approx(expected)


def test_contrast():
    v = _load(_gray(191.25), contrast=200)  # 0.75 → 0.5 + 0.25 x 2 = 1.0
    assert v.mean() == pytest.approx(1.0)
    v = _load(_gray(191.25), contrast=50)   # 0.75 → 0.5 + 0.25 x 0.5 = 0.625
    assert v.mean() == pytest.approx(0.625)
    v = _load(_gradient(), contrast=0)      # 0% なら全部まん中の灰色
    assert np.allclose(v, 0.5)


def test_auto_levels():
    """暗く眠い画像（0.3〜0.5）が、ほぼ黒〜白いっぱいに広がる。"""
    dull = np.tile(np.linspace(0.3 * 255, 0.5 * 255, 80), (40, 1))
    v = _load(dull, auto_levels=True)
    assert v.min() < 0.02 and v.max() > 0.98


def test_auto_levels_flat_image():
    """一色だけの画像は、自動補正しても変わらない（0 で割らない）。"""
    assert np.allclose(_load(_gray(100), auto_levels=True), 100 / 255)


# ---- ぼかし ----

def test_blur_smooths():
    rng = np.random.default_rng(0)
    noisy = rng.uniform(0, 255, (80, 160))
    plain = load_brightness(noisy, 40, 0.25)
    blurred = load_brightness(noisy, 40, 0.25, ImageAdjust(blur=1.0))
    assert blurred.std() < plain.std() * 0.5
    assert blurred.mean() == pytest.approx(plain.mean(), abs=0.02)  # 全体の明るさは変わらない


def test_blur_is_in_mm():
    """ぼかしは mm で指定するので、解像度を変えても効き方がほぼ同じ。"""
    rng = np.random.default_rng(1)
    noisy = rng.uniform(0, 255, (400, 800))
    fine = load_brightness(noisy, 40, 0.2, ImageAdjust(blur=1.0))
    coarse = load_brightness(noisy, 40, 0.4, ImageAdjust(blur=1.0))
    assert fine.std() == pytest.approx(coarse.std(), rel=0.5)


# ---- 切り抜き ----

def test_crop_keeps_part():
    """左半分を切り取ると、右半分（明るい側）だけが残る。"""
    v = _load(_gradient(), crop_left=50)
    assert v.min() > 0.45


def test_crop_changes_aspect():
    """切り抜くと縦横比が変わり、奥行きが変わる（幅は指定どおり）。"""
    full = make_relief(_gradient(80, 80), width=20, resolution=0.5)
    cropped = make_relief(_gradient(80, 80), width=20, resolution=0.5, crop_top=25, crop_bottom=25)
    assert full.extents[0] == pytest.approx(cropped.extents[0])
    assert cropped.extents[1] == pytest.approx(full.extents[1] / 2, abs=0.5)


def test_crop_all_sides():
    img = np.zeros((100, 100))
    img[40:60, 40:60] = 255  # まん中だけ白
    v = _load(img, crop_left=40, crop_right=40, crop_top=40, crop_bottom=40)
    assert v.min() > 0.9  # 白い所だけが残る


# ---- レリーフ・リトファンへの反映 ----

def test_relief_with_adjust():
    plain = make_relief(_gray(100), width=20, resolution=0.5, min_height=1, max_height=5)
    bright = make_relief(_gray(100), width=20, resolution=0.5, min_height=1, max_height=5,
                         brightness=30)
    assert bright.extents[2] > plain.extents[2]  # 明るくすると高くなる
    assert bright.is_watertight


def test_lithophane_with_adjust():
    t = top_heights(make_lithophane(_gradient(), width=20, resolution=0.5,
                                    auto_levels=True, contrast=150, blur=0.5,
                                    crop_left=10, crop_right=10))
    assert t.min() == pytest.approx(0.8) and t.max() == pytest.approx(3.0)


# ---- エラー ----

@pytest.mark.parametrize("adjust, message", [
    ({"brightness": 101}, "明るさ"),
    ({"contrast": -1}, "コントラスト"),
    ({"blur": -0.5}, "ぼかし"),
    ({"crop_left": 100}, "切り抜き（左）"),
    ({"crop_top": -1}, "切り抜き（上）"),
    ({"crop_left": 60, "crop_right": 40}, "左と右"),
    ({"crop_top": 50, "crop_bottom": 50}, "上と下"),
    ({"brightness": float("nan")}, "明るさ"),
])
def test_invalid(adjust, message):
    with pytest.raises(ImageMeshError, match=message):
        _load(_gradient(), **adjust)


def test_crop_too_small():
    with pytest.raises(ImageMeshError, match="小さすぎます"):
        _load(np.zeros((10, 10)), crop_left=49, crop_right=49)
