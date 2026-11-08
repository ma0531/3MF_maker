"""core/printer.py（プリンターに合わせた警告）と、文字の細さ判定のテスト。"""

import numpy as np
import pytest
import trimesh

from core.image_mesh import make_lithophane
from core.printer import estimate_filament, printer_warnings
from core.text_mesh import make_nameplate, make_text_mesh, text_to_cross_section, thin_ratio


def _box(x, y, z, kind=None):
    mesh = trimesh.creation.box((x, y, z))
    if kind:
        mesh.metadata["kind"] = kind
    return mesh


# ---- 造形サイズ ----

@pytest.mark.parametrize("size", [(100, 100, 10), (220, 220, 240)])
def test_fits(size):
    assert printer_warnings(_box(*size)) == []


@pytest.mark.parametrize("size", [(221, 100, 10), (100, 221, 10), (100, 100, 241)])
def test_too_large(size):
    (message,) = printer_warnings(_box(*size))
    assert "造形サイズ" in message and "超えています" in message


def test_lithophane_standing_only():
    """縦 230mm のリトファン：寝かせると入らないが、立てれば入る。"""
    (message,) = printer_warnings(_box(150, 230, 3, "lithophane"))
    assert "立てて印刷してください" in message


def test_lithophane_flat_only():
    """横 200mm x 縦 215mm は寝かせれば入る。立てる場合も 240mm 以内なので警告なし。"""
    assert printer_warnings(_box(200, 215, 3, "lithophane")) == []


def test_lithophane_too_large():
    (message,) = printer_warnings(_box(230, 230, 3, "lithophane"))
    assert "超えています" in message


def test_real_lithophane_metadata():
    mesh = make_lithophane(np.zeros((300, 100)), width=80, resolution=0.5)  # 縦 240mm
    assert mesh.extents[1] == pytest.approx(240)
    assert "立てて印刷してください" in printer_warnings(mesh)[0]


# ---- 文字の細さ ----

def test_thin_ratio_large_text_is_ok(jp_font):
    assert thin_ratio(text_to_cross_section("山田", jp_font, size=10)) < 0.05


def test_thin_ratio_small_text(jp_font):
    assert thin_ratio(text_to_cross_section("山田", jp_font, size=4)) > 0.5


def test_complex_kanji_warns(jp_font):
    """画数の多い漢字は 8mm でも細い部分が多く、警告が出る。"""
    mesh = make_text_mesh("鬱", jp_font, size=8)
    warnings = printer_warnings(mesh)
    assert any("ノズル径" in w for w in warnings)


def test_nameplate_has_thin_ratio(jp_font):
    mesh = make_nameplate("ABC", jp_font, size=10)
    assert 0 <= mesh.metadata["thin_ratio"] < 0.05
    assert printer_warnings(mesh) == []


# ---- フィラメントの見積もり ----

def test_filament_solid():
    """インフィル 100% なら、体積そのもの（20mm 立方 = 8cm³）をフィラメントの断面積で割った長さ。"""
    est = estimate_filament(_box(20, 20, 20), infill=1.0)
    assert est.volume_cm3 == pytest.approx(8.0)
    assert est.length_m == pytest.approx(8000 / (np.pi * 0.875 ** 2) / 1000)  # 約 3.33m
    assert est.weight_g == pytest.approx(8.0 * 1.24)


def test_filament_shell_only():
    """インフィル 0% なら外側だけ（表面積 2400mm² x 0.8mm = 1.92cm³）。"""
    est = estimate_filament(_box(20, 20, 20), infill=0.0)
    assert est.volume_cm3 == pytest.approx(1.92)


def test_filament_thin_part_is_solid():
    """外側の厚みより薄い形は、インフィルに関係なく中まで詰まる。"""
    thin = _box(50, 50, 1)
    assert estimate_filament(thin, infill=0.0).volume_cm3 == pytest.approx(thin.volume / 1000)


def test_filament_increases_with_infill():
    box = _box(30, 30, 30)
    values = [estimate_filament(box, infill=i).length_m for i in (0.0, 0.15, 0.5, 1.0)]
    assert values == sorted(values)


@pytest.mark.parametrize("infill", [-0.1, 1.5])
def test_filament_invalid_infill(infill):
    with pytest.raises(ValueError, match="インフィル"):
        estimate_filament(_box(10, 10, 10), infill=infill)
