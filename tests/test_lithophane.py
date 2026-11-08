"""core/image_mesh.py（リトファン）のテスト。"""

import numpy as np
import pytest
import trimesh
from PIL import Image

from core.exporter import export_mesh
from core.image_mesh import ImageMeshError, make_lithophane, make_relief, top_heights
from gui.preview import backlit_image, save_preview_png


def _gradient(rows=50, cols=100):
    """左が黒（0）、右が白（255）の横グラデーション。"""
    return np.tile(np.linspace(0, 255, cols), (rows, 1))


# ---- 厚みの変換 ----

def test_defaults_follow_spec():
    """既定値は仕様どおり（最小 0.8mm・最大 3.0mm・0.2mm/画素）。"""
    mesh = make_lithophane(_gradient(), width=50)
    t = top_heights(mesh)
    assert t.min() == pytest.approx(0.8)
    assert t.max() == pytest.approx(3.0)
    assert mesh.metadata["grid_size"][0] == 50 / 0.2 + 1  # 251 列


def test_dark_is_thick_bright_is_thin():
    t = top_heights(make_lithophane(_gradient(), width=50))
    assert t[:, 0].mean() == pytest.approx(3.0, abs=0.05)   # 左（黒）→ 厚い
    assert t[:, -1].mean() == pytest.approx(0.8, abs=0.05)  # 右（白）→ 薄い
    assert np.all(np.diff(t[0]) <= 1e-9)  # 右へ行くほど薄くなる


@pytest.mark.parametrize("value, expected", [(0, 3.0), (255, 0.8), (128, 3.0 - 2.2 * 128 / 255)])
def test_uniform_image(value, expected):
    mesh = make_lithophane(np.full((20, 20), value), width=10, resolution=0.5)
    assert mesh.extents[2] == pytest.approx(expected, abs=1e-6)
    assert mesh.volume == pytest.approx(10 * 10 * expected, rel=1e-6)


def test_custom_thickness():
    t = top_heights(make_lithophane(_gradient(), width=30, min_thickness=0.6, max_thickness=4))
    assert (t.min(), t.max()) == pytest.approx((0.6, 4.0))


# ---- 形 ----

def test_shape():
    mesh = make_lithophane(_gradient(), width=100)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0
    assert mesh.extents[:2] == pytest.approx([100, 50], abs=0.21)
    assert mesh.metadata["kind"] == "lithophane"


def test_back_is_flat():
    """裏面（上面以外の頂点）はすべて Z=0。上面は最小厚み以上。"""
    mesh = make_lithophane(_gradient(), width=30)
    cols, rows = mesh.metadata["grid_size"]
    top = mesh.vertices[: cols * rows]
    back = mesh.vertices[cols * rows:]
    assert np.all(back[:, 2] == 0)
    assert top[:, 2].min() >= 0.8 - 1e-9


def test_export(tmp_path):
    mesh = make_lithophane(_gradient(), width=40)
    for ext in ("stl", "3mf"):
        loaded = trimesh.load_mesh(export_mesh(mesh, tmp_path / f"l.{ext}"))
        assert loaded.is_watertight
        assert loaded.volume == pytest.approx(mesh.volume, rel=1e-4)


# ---- 光を当てたときの見え方 ----

def test_backlit_image():
    """元の画像で明るい（＝薄い）部分ほど、光を当てた絵でも明るい。"""
    mesh = make_lithophane(_gradient(), width=50, resolution=0.5)
    light = backlit_image(mesh).mean(axis=2)  # RGB の平均 = 明るさ
    assert light.shape == top_heights(mesh).shape
    assert light[:, -1].mean() > 0.8          # 右（白・薄い）→ 明るい
    assert light[:, 0].mean() < 0.1           # 左（黒・厚い）→ 暗い
    assert np.all(np.diff(light[0]) >= -1e-9)  # 右へ行くほど明るくなる


def test_backlit_png(tmp_path):
    mesh = make_lithophane(_gradient(), width=50, resolution=0.5)
    path = save_preview_png(mesh, tmp_path / "b.png", backlit=True)
    assert Image.open(path).size == (600, 500)


def test_top_heights_needs_image_mesh():
    with pytest.raises(ImageMeshError, match="高さの表を取り出せません"):
        top_heights(trimesh.creation.box((1, 1, 1)))


def test_relief_kind():
    assert make_relief(_gradient(), width=20, resolution=0.5).metadata["kind"] == "relief"


# ---- エラー ----

def test_min_max():
    with pytest.raises(ImageMeshError, match="最小厚み.*より大きく"):
        make_lithophane(_gradient(), min_thickness=3, max_thickness=0.8)


@pytest.mark.parametrize("kwargs, message", [
    ({"min_thickness": 0}, "最小厚み"),
    ({"max_thickness": -1}, "最大厚み"),
    ({"width": -10}, "幅"),
    ({"resolution": float("inf")}, "解像度"),
])
def test_invalid_numbers(kwargs, message):
    with pytest.raises(ImageMeshError, match=message):
        make_lithophane(_gradient(), **kwargs)
