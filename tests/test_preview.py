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
