"""core/text_mesh.py（縦書き）のテスト。"""

import numpy as np
import pytest

from core.text_mesh import TextMeshError, make_nameplate, make_text_mesh, text_to_cross_section


def _points(section):
    return np.concatenate([np.asarray(p) for p in section.to_polygons()])


def _bounds(section):
    x0, y0, x1, y1 = section.bounds()
    return x0, y0, x1, y1


def test_vertical_is_tall(jp_font):
    """縦書きでは、1 行の文字列が縦に長くなる。"""
    h = make_text_mesh("日本語", jp_font, size=10, direction="horizontal")
    v = make_text_mesh("日本語", jp_font, size=10, direction="vertical")
    assert v.is_watertight
    assert h.extents[0] > h.extents[1]
    assert v.extents[1] > v.extents[0]
    # 3 文字ぶん（1 文字 = 10mm のマス）の高さに近い
    assert v.extents[1] == pytest.approx(30, abs=3)


def test_vertical_same_area(jp_font):
    """縦書きにしても文字の形（面積）は変わらない。"""
    h = text_to_cross_section("日本語", jp_font, size=10, direction="horizontal").area()
    v = text_to_cross_section("日本語", jp_font, size=10, direction="vertical").area()
    assert v == pytest.approx(h, rel=0.01)


def test_columns_right_to_left(jp_font):
    """2 行目は 1 行目の左に来る。"""
    section = text_to_cross_section("一\n二", jp_font, size=10, direction="vertical")
    first = text_to_cross_section("一", jp_font, size=10, direction="vertical")
    x0, _, x1, _ = _bounds(section)
    fx0, _, fx1, _ = _bounds(first)
    # 1 行目（一）は全体の右側にあり、全体は左へ広がっている
    assert fx1 == pytest.approx(x1, abs=0.01)
    assert x0 < fx0 - 5


def test_long_vowel_is_rotated(jp_font):
    """「ー」は縦書きで縦棒になる（横より縦が長い）。"""
    h = _bounds(text_to_cross_section("ー", jp_font, size=10, direction="horizontal"))
    v = _bounds(text_to_cross_section("ー", jp_font, size=10, direction="vertical"))
    assert (h[2] - h[0]) > (h[3] - h[1])
    assert (v[3] - v[1]) > (v[2] - v[0])


def test_punctuation_top_right(jp_font):
    """「、」は縦書きでマスの右上に置かれる。"""
    section = text_to_cross_section("、", jp_font, size=10, direction="vertical")
    x0, y0, x1, y1 = _bounds(section)
    # マスは x: -5〜5、y: -10〜0。中心が右上（x>0, y>-5）にある
    assert (x0 + x1) / 2 > 1
    assert (y0 + y1) / 2 > -5


@pytest.mark.parametrize("align, expected", [("left", "top"), ("center", "middle"), ("right", "bottom")])
def test_vertical_align(jp_font, align, expected):
    """縦書きの揃え方: 行頭=上、中央、行末=下。"""
    section = text_to_cross_section("一二三四\n五", jp_font, size=10, direction="vertical", align=align)
    pts = _points(section)
    x0, y0, x1, y1 = _bounds(section)
    second = pts[pts[:, 0] < x1 - 10]  # 2 行目（左の行）の点
    center = (second[:, 1].min() + second[:, 1].max()) / 2
    ratio = (center - y0) / (y1 - y0)  # 0 = 下、1 = 上
    target = {"top": 0.875, "middle": 0.5, "bottom": 0.125}[expected]
    assert ratio == pytest.approx(target, abs=0.08)


def test_spaces(jp_font):
    """全角スペースは 1 マス、半角スペースは半マスあける。"""
    base = make_text_mesh("一一", jp_font, size=10, direction="vertical").extents[1]
    full = make_text_mesh("一　一", jp_font, size=10, direction="vertical").extents[1]
    half = make_text_mesh("一 一", jp_font, size=10, direction="vertical").extents[1]
    assert full - base == pytest.approx(10, abs=0.01)
    assert half - base == pytest.approx(5, abs=0.01)


def test_vertical_nameplate_fit(jp_font):
    """縦書きのネームプレートでも、文字が板の内側に収まる。"""
    mesh = make_nameplate("山田太郎", jp_font, width=30, depth=100, margin=5, direction="vertical")
    assert mesh.is_watertight
    assert mesh.extents[:2] == pytest.approx([30, 100])
    top = mesh.vertices[mesh.vertices[:, 2] > 3 + 1e-6]  # 台座より上 = 文字
    text_w = top[:, 0].max() - top[:, 0].min()
    text_d = top[:, 1].max() - top[:, 1].min()
    # 余白を除いた内側（20 x 90mm）に収まり、幅か奥行きのどちらかがぴったり
    assert text_w <= 20 + 1e-6 and text_d <= 90 + 1e-6
    assert text_w == pytest.approx(20, abs=0.1) or text_d == pytest.approx(90, abs=0.1)
    assert text_d > text_w * 3  # 縦に長い


def test_invalid_direction(jp_font):
    with pytest.raises(TextMeshError, match="書き方"):
        make_text_mesh("A", jp_font, direction="diagonal")
