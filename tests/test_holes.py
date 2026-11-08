"""core/holes.py（穴あけ）のテスト。"""

import numpy as np
import pytest
import trimesh

from core.exporter import export_mesh
from core.holes import SHAPE_LABELS, Hole, HoleError, add_holes, hole_height, hole_outline
from core.image_mesh import make_lithophane, top_heights
from core.printer import printer_warnings
from core.text_mesh import make_nameplate
from gui.preview import backlit_image


@pytest.fixture
def plate():
    """40 x 20 x 4 mm の板（左下が原点）。"""
    mesh = trimesh.creation.box((40, 20, 4))
    mesh.apply_translation(-mesh.bounds[0])
    mesh.metadata["kind"] = "test"
    return mesh


def test_single_hole(plate):
    out = add_holes(plate, [Hole(10, 10, 4)])
    assert out.is_watertight
    removed = plate.volume - out.volume
    assert removed == pytest.approx(np.pi * 2 ** 2 * 4, rel=0.01)  # 円柱の体積
    assert out.euler_number == 0  # 穴が 1 つ（ドーナツと同じ形）
    assert out.metadata["warnings"] == []


def test_multiple_holes(plate):
    out = add_holes(plate, [(5, 10, 3), (20, 10, 3), (35, 10, 3)])  # タプルでも指定できる
    assert out.is_watertight
    assert out.euler_number == -4  # 穴が 3 つ（2 - 2 x 3）
    assert len(out.metadata["holes"]) == 3


def test_no_holes_returns_same(plate):
    assert add_holes(plate, []) is plate


def test_original_is_unchanged(plate):
    before = plate.volume
    add_holes(plate, [Hole(10, 10, 4)])
    assert plate.volume == before


def test_metadata_is_kept_and_grid_dropped():
    mesh = make_lithophane(np.full((50, 50), 128), width=20, resolution=0.5)
    out = add_holes(mesh, [Hole(10, 10, 3)])
    assert out.metadata["kind"] == "lithophane"
    assert "grid_size" not in out.metadata  # 頂点の並びが変わるので引き継がない
    assert out.is_watertight


def test_nameplate_keeps_text_size(jp_font):
    mesh = make_nameplate("AB", jp_font, width=60, depth=20, margin=8)
    out = add_holes(mesh, [Hole(4, 10, 3)])
    assert out.metadata["text_size"] == mesh.metadata["text_size"]
    assert out.metadata["kind"] == "nameplate"


def test_export_with_holes(plate, tmp_path):
    out = add_holes(plate, [Hole(10, 10, 4), Hole(30, 10, 4)])
    for ext in ("stl", "3mf", "amf"):
        assert export_mesh(out, tmp_path / f"h.{ext}").exists()


# ---- 警告 ----

def test_warning_near_edge(plate):
    out = add_holes(plate, [Hole(2.6, 10, 4)])  # 端まで 0.6mm
    assert any("割れやすく" in w for w in out.metadata["warnings"])
    # printer_warnings でもまとめて表示される
    assert any("割れやすく" in w for w in printer_warnings(out))


def test_warning_on_edge(plate):
    out = add_holes(plate, [Hole(0, 10, 4)])  # 半分が外 = 切り欠き
    assert out.is_watertight
    assert any("切り欠き" in w for w in out.metadata["warnings"])


def test_warning_nothing_cut(jp_font):
    """文字の間など、何もない所の穴は警告になる。"""
    mesh = make_text_mesh_with_gap(jp_font)
    x_gap = mesh.metadata["gap_x"]
    out = add_holes(mesh, [Hole(x_gap, mesh.extents[1] / 2, 0.5)])
    assert any("何もない" in w for w in out.metadata["warnings"])


def make_text_mesh_with_gap(jp_font):
    """「I  I」のように、間がはっきりあいた文字を作る。"""
    from core.text_mesh import make_text_mesh
    mesh = make_text_mesh("I    I", jp_font, size=10)
    mesh.metadata["gap_x"] = mesh.extents[0] / 2
    return mesh


# ---- エラー ----

@pytest.mark.parametrize("hole", [Hole(100, 10, 4), Hole(10, -10, 4), Hole(-5, 10, 2)])
def test_outside_raises(plate, hole):
    with pytest.raises(HoleError, match="形の外"):
        add_holes(plate, [hole])


@pytest.mark.parametrize("hole, message", [
    (Hole(10, 10, 0), "幅"),
    (Hole(10, 10, -1), "幅"),
    (Hole(float("nan"), 10, 3), "X"),
])
def test_invalid_hole(plate, hole, message):
    with pytest.raises(HoleError, match=message):
        add_holes(plate, [hole])


def test_hole_removes_everything(plate):
    with pytest.raises(HoleError, match="形がなくなって"):
        add_holes(plate, [Hole(20, 10, 100)])


def test_not_closed_mesh(plate):
    broken = trimesh.Trimesh(plate.vertices, plate.faces[1:], process=False)
    with pytest.raises(HoleError, match="閉じていない"):
        add_holes(broken, [Hole(10, 10, 3)])


# ---- リトファンの光の見え方 ----

def test_backlit_hole_is_white():
    mesh = make_lithophane(np.zeros((40, 40)), width=20, resolution=0.5)  # 全部黒 = 厚い = 暗い
    rgb = backlit_image(mesh, [Hole(10, 10, 4)])
    rows, cols = top_heights(mesh).shape
    assert rgb[rows // 2, cols // 2].tolist() == [1.0, 1.0, 1.0]  # 穴の中心は白
    assert rgb[0, 0].max() < 0.2                                    # 穴の外は暗いまま


def test_warning_hole_on_letters(jp_font):
    """ネームプレートで、穴が文字にかかると警告。台座だけなら警告なし。"""
    mesh = make_nameplate("山田", jp_font, width=60, depth=30, margin=8)
    on_text = add_holes(mesh, [Hole(30, 15, 4)])      # 中央 = 文字の上
    on_margin = add_holes(mesh, [Hole(4, 15, 4)])     # 左の余白
    assert any("文字にかかって" in w for w in on_text.metadata["warnings"])
    assert on_margin.metadata["warnings"] == []


# ---- 穴の形 ----

def _cut_area(plate, hole):
    """穴で削った体積 ÷ 板の厚み = 穴の断面積。"""
    out = add_holes(plate, [hole])
    assert out.is_watertight
    return (plate.volume - out.volume) / 4.0


W = 8.0  # テストに使う穴の幅（mm）
SQRT3 = np.sqrt(3)


@pytest.mark.parametrize("shape, expected_area", [
    ("circle", np.pi * (W / 2) ** 2),
    ("square", W * W),
    ("triangle", SQRT3 / 4 * W ** 2),            # 正三角形
    ("diamond", W * W / 2),
    ("hexagon", SQRT3 / 2 * W ** 2),             # 対辺 W の正六角形
    ("slot", (W - 0.4 * W) * 0.4 * W + np.pi * (0.2 * W) ** 2),  # 長方形 + 円
])
def test_shape_area(plate, shape, expected_area):
    assert _cut_area(plate, Hole(20, 10, W, shape=shape)) == pytest.approx(expected_area, rel=0.01)


@pytest.mark.parametrize("shape", ["star", "heart"])
def test_shape_star_heart(plate, shape):
    area = _cut_area(plate, Hole(20, 10, W, shape=shape))
    assert 0.2 * W * W < area < W * W  # 外枠の正方形より小さく、ある程度の大きさ


@pytest.mark.parametrize("shape", list(SHAPE_LABELS))
def test_shape_size_is_width(shape):
    """どの形も、幅の指定どおりの横幅になる。縦を省略すると自然な比率。"""
    outline = hole_outline(Hole(0, 0, 10, shape=shape))
    assert outline[:, 0].max() - outline[:, 0].min() == pytest.approx(10, rel=0.01)
    assert hole_height(Hole(0, 0, 10, shape=shape)) == pytest.approx(
        outline[:, 1].max() - outline[:, 1].min(), rel=0.01)


def test_natural_ratios():
    assert hole_height(Hole(0, 0, 10, shape="circle")) == pytest.approx(10)
    assert hole_height(Hole(0, 0, 10, shape="triangle")) == pytest.approx(10 * SQRT3 / 2)
    assert hole_height(Hole(0, 0, 10, shape="hexagon")) == pytest.approx(20 / SQRT3)  # 頂点間
    assert hole_height(Hole(0, 0, 10, shape="slot")) == pytest.approx(4)


def test_height_stretches(plate):
    """縦を指定すると伸び縮みする（丸 → 楕円）。"""
    area = _cut_area(plate, Hole(20, 10, 8, height=4, shape="circle"))
    assert area == pytest.approx(np.pi * 4 * 2, rel=0.01)


def test_vertical_slot():
    outline = hole_outline(Hole(0, 0, 4, height=12, shape="slot"))
    assert np.ptp(outline[:, 0]) == pytest.approx(4, rel=0.01)
    assert np.ptp(outline[:, 1]) == pytest.approx(12, rel=0.01)


def test_rotation():
    """四角を 45° 回すと、外枠は幅 x √2 になる。"""
    outline = hole_outline(Hole(0, 0, 10, shape="square", angle=45))
    assert np.ptp(outline[:, 0]) == pytest.approx(10 * np.sqrt(2), rel=1e-6)


def test_rotated_hole_near_edge_warns(plate):
    """回した形の実際の外枠で、端との距離を判断する。"""
    ok = add_holes(plate, [Hole(10, 10, 10, shape="square")])            # 端まで 5mm
    near = add_holes(plate, [Hole(10, 10, 10, shape="square", angle=45)])  # 端まで約 2.9mm
    thin = add_holes(plate, [Hole(10, 10, 14, shape="square", angle=45)])  # 端まで約 0.1mm
    edge = add_holes(plate, [Hole(10, 10, 15, shape="square", angle=45)])  # 端にかかる
    assert ok.metadata["warnings"] == [] and near.metadata["warnings"] == []
    assert any("割れやすく" in w for w in thin.metadata["warnings"])
    assert any("端にかかって" in w for w in edge.metadata["warnings"])


def test_invalid_shape(plate):
    with pytest.raises(HoleError, match="形の指定"):
        add_holes(plate, [Hole(10, 10, 4, shape="moon")])


def test_invalid_height(plate):
    with pytest.raises(HoleError, match="縦"):
        add_holes(plate, [Hole(10, 10, 4, height=0)])


def test_backlit_star_hole():
    mesh = make_lithophane(np.zeros((40, 40)), width=20, resolution=0.5)
    rgb = backlit_image(mesh, [Hole(10, 10, 8, shape="star")])
    rows, cols = top_heights(mesh).shape
    assert rgb[rows // 2, cols // 2].tolist() == [1.0, 1.0, 1.0]  # 星の中心は白
