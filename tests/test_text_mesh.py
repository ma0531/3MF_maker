"""core/text_mesh.py（立体文字）のテスト。"""

import numpy as np
import pytest
import trimesh

from core.exporter import export_mesh
from core.text_mesh import TextMeshError, make_text_mesh, text_to_cross_section


# ---- 基本の形 ----

@pytest.mark.parametrize("text", ["A", "O", "回", "あいう", "Hello 日本", "8%&@"])
def test_watertight(jp_font, text):
    mesh = make_text_mesh(text, jp_font, size=10, height=3)
    assert mesh.is_watertight
    assert mesh.volume > 0


def test_height_and_origin(jp_font):
    """押し出しの高さどおりで、最小値が原点に置かれている。"""
    mesh = make_text_mesh("ABC", jp_font, size=10, height=2.5)
    assert mesh.extents[2] == pytest.approx(2.5)
    assert np.allclose(mesh.bounds[0], 0)


def test_holes(jp_font):
    """穴のある文字は穴が開いている（オイラー数 0 = 穴が1つ）。"""
    assert make_text_mesh("I", jp_font).euler_number == 2  # 穴なし
    assert make_text_mesh("O", jp_font).euler_number == 0  # 穴1つ


def test_size_scales(jp_font):
    """大きさを2倍にすると、面積（＝同じ高さなら体積）は約4倍になる。"""
    small = make_text_mesh("日本", jp_font, size=10, height=2)
    large = make_text_mesh("日本", jp_font, size=20, height=2)
    assert large.volume / small.volume == pytest.approx(4, rel=0.02)
    assert large.extents[0] / small.extents[0] == pytest.approx(2, rel=0.02)


def test_size_is_reasonable(jp_font):
    """size=10 のとき、漢字の高さは 10mm 以下で、小さすぎない。"""
    mesh = make_text_mesh("国", jp_font, size=10)
    assert 7 < mesh.extents[1] <= 10


def test_dollar_is_not_math(jp_font):
    """"$" で囲んでも数式にならず、文字としてそのまま出る。"""
    a = text_to_cross_section("$5$", jp_font).area()
    b = text_to_cross_section("$", jp_font).area() * 2 + text_to_cross_section("5", jp_font).area()
    assert a == pytest.approx(b, rel=0.01)


# ---- 複数行 ----

def test_multiline(jp_font):
    one = make_text_mesh("AB", jp_font, size=10)
    two = make_text_mesh("AB\nCD", jp_font, size=10, line_spacing=1.5)
    assert two.is_watertight
    # 2行目の分（およそ 10 * 1.5 = 15mm）だけ縦に伸びる
    assert two.extents[1] - one.extents[1] == pytest.approx(15, abs=2)


@pytest.mark.parametrize("align", ["left", "center", "right"])
def test_align(jp_font, align):
    """揃え方に応じて、短い行の位置が変わる。"""
    section = text_to_cross_section("WWWW\nI", jp_font, size=10, align=align)
    # 2行目「I」の中心が、全体の幅のどの位置にあるか（左=0, 中央=0.5, 右=1）
    xmin, _, xmax, _ = _bounds(section)
    ratio = (_second_line_center_x(section) - xmin) / (xmax - xmin)
    expected = {"left": 0.0, "center": 0.5, "right": 1.0}[align]
    assert ratio == pytest.approx(expected, abs=0.1)


# ---- 書き出しとの連携 ----

def test_export(jp_font, tmp_path):
    mesh = make_text_mesh("テスト", jp_font)
    path = export_mesh(mesh, tmp_path / "text.stl")
    assert trimesh.load_mesh(path).is_watertight


# ---- エラー ----

@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_empty_text(jp_font, text):
    with pytest.raises(TextMeshError, match="入力されていません"):
        make_text_mesh(text, jp_font)


def test_font_not_found():
    with pytest.raises(TextMeshError, match="フォントファイルが見つかりません"):
        make_text_mesh("A", "C:/no/such/font.ttf")


def test_missing_glyph(jp_font):
    with pytest.raises(TextMeshError, match="このフォントにない文字"):
        make_text_mesh("A\U0001F600", jp_font)  # 絵文字はメイリオにない


@pytest.mark.parametrize("kwargs", [
    {"size": 0}, {"size": -1}, {"height": 0}, {"line_spacing": 0}, {"size": float("nan")},
])
def test_invalid_number(jp_font, kwargs):
    with pytest.raises(TextMeshError, match="0 より大きい"):
        make_text_mesh("A", jp_font, **kwargs)


def test_invalid_align(jp_font):
    with pytest.raises(TextMeshError, match="揃え方"):
        make_text_mesh("A", jp_font, align="middle")


# ---- テスト用の補助 ----

def _bounds(section):
    b = section.bounds()
    return b[0], b[1], b[2], b[3]


def _second_line_center_x(section):
    """2行目（ベースラインが y<0 の行）の x 方向の中心。"""
    pts = np.concatenate([np.asarray(p) for p in section.to_polygons()])
    lower = pts[pts[:, 1] < -3]
    return (lower[:, 0].min() + lower[:, 0].max()) / 2
