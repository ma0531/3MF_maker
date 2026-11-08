"""core/image_mesh.py（共通部分・レリーフ）のテスト。"""

import numpy as np
import pytest
import trimesh
from PIL import Image

from core.exporter import export_mesh
from core.image_mesh import (
    ImageMeshError,
    heightmap_to_mesh,
    load_brightness,
    make_relief,
)


def _gradient(rows=50, cols=100):
    """左が黒（0）、右が白（255）の横グラデーション。"""
    return np.tile(np.linspace(0, 255, cols), (rows, 1))


def _top_height_at(mesh, x, y):
    """上面の頂点のうち (x, y) に一番近い点の高さ。"""
    v = mesh.vertices
    upper = v[v[:, 2] > 0]
    i = np.argmin((upper[:, 0] - x) ** 2 + (upper[:, 1] - y) ** 2)
    return upper[i, 2]


# ---- 高さの表 → 立体 ----

def test_flat_heightmap_is_box():
    """高さが一定なら、ただの直方体になる。"""
    mesh = heightmap_to_mesh(np.full((5, 11), 2.0), width=10)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.extents == pytest.approx([10, 4, 2])
    assert mesh.volume == pytest.approx(10 * 4 * 2)


def test_random_heightmap_is_closed():
    rng = np.random.default_rng(0)
    heights = rng.uniform(0.5, 3.0, size=(30, 40))
    mesh = heightmap_to_mesh(heights, width=39)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0  # 法線が外向き（内向きだと体積が負になる）
    # 体積は「平均の高さ x 面積」の近くになる
    assert mesh.volume == pytest.approx(heights.mean() * 39 * 29, rel=0.05)


def test_heightmap_errors():
    with pytest.raises(ImageMeshError, match="2 x 2 以上"):
        heightmap_to_mesh(np.ones((1, 5)), width=10)
    with pytest.raises(ImageMeshError, match="0 より大きい"):
        heightmap_to_mesh(np.zeros((3, 3)), width=10)


# ---- 画像の読み込み ----

def test_resize_to_resolution():
    """幅 / 解像度 + 1 列に縮小され、縦横比が保たれる。"""
    values = load_brightness(_gradient(500, 1000), width=50, resolution=0.5)
    assert values.shape == (51, 101)
    assert values.min() >= 0 and values.max() <= 1


def test_load_from_file_with_japanese_name(tmp_path):
    """日本語のファイル名でも読める（OpenCV の imread は読めないため Pillow で読む）。"""
    path = tmp_path / "テスト画像.png"
    Image.fromarray(_gradient().astype(np.uint8)).save(path)
    values = load_brightness(path, width=20, resolution=0.2)
    assert values[:, 0].mean() < 0.05 and values[:, -1].mean() > 0.95


def test_transparent_is_white(tmp_path):
    path = tmp_path / "clear.png"
    Image.new("RGBA", (40, 40), (0, 0, 0, 0)).save(path)
    values = load_brightness(path, width=10, resolution=0.5)
    assert values.min() == pytest.approx(1.0)


def test_color_image():
    img = Image.new("RGB", (40, 20), (255, 0, 0))  # 赤 → 少し暗い灰色
    values = load_brightness(img, width=10, resolution=0.5)
    assert 0.2 < values.mean() < 0.4


# ---- レリーフ ----

def test_relief_size_and_heights():
    mesh = make_relief(_gradient(), width=40, min_height=1, max_height=4, resolution=0.4)
    assert mesh.is_watertight
    # 幅 40mm、奥行きは縦横比（50:100）で 20mm、高さは最大 4mm
    assert mesh.extents == pytest.approx([40, 20, 4], abs=0.41)
    # 左（暗い）は低く、右（明るい）は高い
    assert _top_height_at(mesh, 0, 10) == pytest.approx(1, abs=0.1)
    assert _top_height_at(mesh, 40, 10) == pytest.approx(4, abs=0.1)


def test_relief_invert():
    mesh = make_relief(_gradient(), width=40, min_height=1, max_height=4,
                       invert=True, resolution=0.4)
    assert _top_height_at(mesh, 0, 10) == pytest.approx(4, abs=0.1)
    assert _top_height_at(mesh, 40, 10) == pytest.approx(1, abs=0.1)


def test_relief_image_top_is_plus_y():
    """画像の上側が +Y（奥）になる（上から見て画像が反転しない）。"""
    img = np.zeros((40, 40))
    img[:10, :] = 255  # 上の 1/4 を白に
    mesh = make_relief(img, width=20, min_height=1, max_height=3, resolution=0.5)
    assert _top_height_at(mesh, 10, 19) == pytest.approx(3)  # 奥 → 高い
    assert _top_height_at(mesh, 10, 1) == pytest.approx(1)   # 手前 → 低い


def test_relief_export(tmp_path):
    mesh = make_relief(_gradient(), width=30, resolution=0.5)
    path = export_mesh(mesh, tmp_path / "relief.stl")
    loaded = trimesh.load_mesh(path)
    assert loaded.is_watertight
    assert loaded.volume == pytest.approx(mesh.volume, rel=1e-4)


# ---- エラー ----

def test_relief_min_max():
    with pytest.raises(ImageMeshError, match="最小高さ.*より大きく"):
        make_relief(_gradient(), min_height=3, max_height=3)


@pytest.mark.parametrize("kwargs, message", [
    ({"width": 0}, "幅"),
    ({"min_height": 0}, "最小高さ"),
    ({"max_height": -1}, "最大高さ"),
    ({"resolution": 0}, "解像度"),
])
def test_relief_invalid_numbers(kwargs, message):
    with pytest.raises(ImageMeshError, match=message):
        make_relief(_gradient(), **kwargs)


def test_file_not_found():
    with pytest.raises(ImageMeshError, match="見つかりません"):
        make_relief("C:/no/such/image.png")


def test_broken_file(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ImageMeshError, match="開けませんでした"):
        make_relief(path)


def test_too_many_points():
    with pytest.raises(ImageMeshError, match="点の数が多すぎます"):
        make_relief(_gradient(1000, 1000), width=1000, resolution=0.2)


def test_too_coarse():
    with pytest.raises(ImageMeshError, match="粗すぎます"):
        make_relief(_gradient(), width=1, resolution=5)
