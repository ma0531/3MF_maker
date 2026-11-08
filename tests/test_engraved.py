"""core/text_mesh.py（彫り込みのネームプレート）のテスト。"""

import pytest

from core.holes import Hole, add_holes
from core.printer import printer_warnings
from core.text_mesh import TextMeshError, make_nameplate, make_text_mesh

KW = dict(width=80, depth=30, base_thickness=3, text_height=1.5, margin=5)


def test_engraved_shape(jp_font):
    """彫り込みは、板の体積から文字の体積（深さ分）を引いたものになる。"""
    mesh = make_nameplate("回O", jp_font, style="engraved", **KW)
    assert mesh.is_watertight
    assert mesh.extents == pytest.approx([80, 30, 3])  # 高さは台座の厚みのまま
    letters = make_text_mesh("回O", jp_font, size=mesh.metadata["text_size"], height=1.5)
    assert mesh.volume == pytest.approx(80 * 30 * 3 - letters.volume, rel=1e-4)
    assert mesh.metadata["text_style"] == "engraved"


def test_raised_is_default(jp_font):
    mesh = make_nameplate("AB", jp_font, **KW)
    assert mesh.metadata["text_style"] == "raised"
    assert mesh.extents[2] == pytest.approx(4.5)  # 台座 3 + 文字 1.5


def test_engraved_depth(jp_font):
    """彫った底は、上面から深さ分だけ下がった位置にある。"""
    mesh = make_nameplate("I", jp_font, style="engraved", **KW)
    zs = sorted({round(z, 4) for z in mesh.vertices[:, 2]})
    assert zs == [0.0, 1.5, 3.0]  # 底面・彫った底・上面


def test_engraved_too_deep(jp_font):
    with pytest.raises(TextMeshError, match="突き抜け"):
        make_nameplate("A", jp_font, style="engraved", **{**KW, "text_height": 3})


def test_engraved_thin_floor_warns(jp_font):
    mesh = make_nameplate("A", jp_font, style="engraved", **{**KW, "text_height": 2.5})
    assert any("底の厚み" in w for w in printer_warnings(mesh))


def test_invalid_style(jp_font):
    with pytest.raises(TextMeshError, match="文字の形"):
        make_nameplate("A", jp_font, style="emboss", **KW)


def test_engraved_hole_warnings(jp_font):
    """彫り込みでも、穴が文字にかかると警告。余白なら警告なし。"""
    mesh = make_nameplate("山田", jp_font, style="engraved", width=60, depth=30, margin=8)
    on_text = add_holes(mesh, [Hole(30, 15, 4)])
    on_margin = add_holes(mesh, [Hole(4, 15, 4)])
    assert any("文字にかかって" in w for w in on_text.metadata["warnings"])
    assert on_margin.metadata["warnings"] == []
    assert on_text.is_watertight
