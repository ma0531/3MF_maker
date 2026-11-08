"""gui/preview.py（完成図の表示）のテスト。画面は開かず PNG に描く。"""

import numpy as np
import pytest
import trimesh
from PIL import Image

from core.image_mesh import make_relief
from core.text_mesh import make_nameplate
from gui.preview import PREVIEW_COLUMNS, preview_resolution, save_preview_png


@pytest.mark.parametrize("make", [
    lambda: trimesh.creation.box((10, 20, 5)),
    lambda: make_relief(np.random.default_rng(0).uniform(0, 255, (60, 80)), width=40, resolution=0.5),
])
def test_save_png(tmp_path, make):
    path = save_preview_png(make(), tmp_path / "p.png", title="テスト")
    img = Image.open(path)
    assert img.size == (600, 500)
    # 真っ白ではなく、何かが描かれている
    assert np.asarray(img.convert("L")).min() < 200


def test_nameplate_png(tmp_path, jp_font):
    mesh = make_nameplate("AB", jp_font, size=10)
    assert save_preview_png(mesh, tmp_path / "n.png").exists()


def test_preview_resolution():
    # 本番の解像度が細かい場合は、横 PREVIEW_COLUMNS 点ぶんまで粗くする
    assert preview_resolution(100, 0.2) == pytest.approx(100 / (PREVIEW_COLUMNS - 1))
    # 本番がもともと粗い場合は、それより細かくしない
    assert preview_resolution(10, 0.5) == 0.5


# ---- 真上から見た図 ----

def test_top_view_png(tmp_path):
    from core.holes import Hole, add_holes, hole_outline
    plate = trimesh.creation.box((40, 20, 3))
    plate.apply_translation(-plate.bounds[0])
    mesh = add_holes(plate, [Hole(10, 10, 6)])
    path = save_preview_png(mesh, tmp_path / "top.png", title="真上", top=True)
    assert Image.open(path).size == (600, 500)


def test_top_view_axes_are_mm():
    """真上から見た図は、目盛りが mm（クリックした座標がそのまま穴の位置になる）。"""
    from matplotlib.figure import Figure
    from gui.preview import draw_top
    mesh = trimesh.creation.box((40, 20, 3))
    mesh.apply_translation(-mesh.bounds[0])
    fig = Figure()
    draw_top(fig, mesh, marker=np.array([[0, 0], [1, 0], [1, 1]]))
    ax = fig.axes[0]
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert x0 < 0 < 40 < x1 and y0 < 0 < 20 < y1
    assert ax.get_aspect() == 1.0  # 縦横の縮尺が同じ
    assert len(ax.lines) == 2      # 赤い点線の輪郭と中心の印
