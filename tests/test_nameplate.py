"""core/text_mesh.py（ネームプレート）のテスト。"""

import numpy as np
import pytest
import trimesh

from core.exporter import export_mesh
from core.text_mesh import TextMeshError, make_nameplate, make_text_mesh


# ---- 大きさの決め方 3 パターン ----

def test_fit_to_plate(jp_font):
    """板の大きさを指定すると、文字が余白の内側にぴったり収まる。"""
    mesh = make_nameplate("山田 太郎", jp_font, width=100, depth=30,
                          base_thickness=3, text_height=2, margin=5)
    assert mesh.is_watertight
    assert mesh.extents == pytest.approx([100, 30, 5])
    assert mesh.metadata["plate_size"] == (100, 30)

    # 横長の文字なので、幅いっぱい（100 - 5x2 = 90mm）に合わせられている
    letters = _letters_bounds(mesh, base_thickness=3)
    assert letters[1][0] - letters[0][0] == pytest.approx(90, abs=0.2)
    # 中央に置かれている
    assert (letters[0][0] + letters[1][0]) / 2 == pytest.approx(50, abs=0.1)
    assert (letters[0][1] + letters[1][1]) / 2 == pytest.approx(15, abs=0.1)


def test_fit_limited_by_depth(jp_font):
    """縦長の板では、奥行きで文字の大きさが決まる。"""
    mesh = make_nameplate("AB", jp_font, width=200, depth=20, margin=3)
    letters = _letters_bounds(mesh, base_thickness=3)
    assert letters[1][1] - letters[0][1] == pytest.approx(14, abs=0.2)


def test_plate_from_text_size(jp_font):
    """文字の大きさを指定すると、板は「文字 + 余白」の大きさになる。"""
    text_only = make_text_mesh("ABC", jp_font, size=10)
    mesh = make_nameplate("ABC", jp_font, size=10, margin=4)
    w, d = mesh.metadata["plate_size"]
    assert w == pytest.approx(text_only.extents[0] + 8, abs=0.01)
    assert d == pytest.approx(text_only.extents[1] + 8, abs=0.01)
    assert mesh.metadata["text_size"] == 10


def test_width_only(jp_font):
    """幅だけ指定すると、奥行きは文字に合わせて決まる。"""
    mesh = make_nameplate("ABC", jp_font, width=80, margin=5)
    w, d = mesh.metadata["plate_size"]
    assert w == 80
    assert 10 < d < 80


def test_both_specified(jp_font):
    mesh = make_nameplate("A", jp_font, width=50, depth=50, size=10)
    assert mesh.metadata["text_size"] == 10
    assert mesh.extents[:2] == pytest.approx([50, 50])


# ---- 形の正しさ ----

def test_volume_is_plate_plus_letters(jp_font):
    """体積 = 板の体積 + 文字の体積（結合で欠けたり重なったりしていない）。"""
    mesh = make_nameplate("回O", jp_font, size=15, margin=5, base_thickness=3, text_height=2)
    letters = make_text_mesh("回O", jp_font, size=15, height=2)
    w, d = mesh.metadata["plate_size"]
    assert mesh.volume == pytest.approx(w * d * 3 + letters.volume, rel=1e-4)


def test_origin(jp_font):
    mesh = make_nameplate("A", jp_font, size=10)
    assert np.allclose(mesh.bounds[0], 0)


def test_rounded_corners(jp_font):
    """角を丸めると、その分だけ体積が減る（4 隅で r²(4 - π)）。"""
    kw = dict(width=60, depth=30, size=8, base_thickness=2, text_height=1)
    square = make_nameplate("A", jp_font, **kw)
    rounded = make_nameplate("A", jp_font, corner_radius=5, **kw)
    assert rounded.is_watertight
    removed = (5 ** 2) * (4 - np.pi) * 2  # 厚み 2mm
    assert square.volume - rounded.volume == pytest.approx(removed, rel=0.01)


def test_max_corner_radius(jp_font):
    """丸みが短い辺のちょうど半分（両端が半円）でも作れる。"""
    mesh = make_nameplate("A", jp_font, width=60, depth=20, size=8, margin=2, corner_radius=10)
    assert mesh.is_watertight
    assert mesh.extents[:2] == pytest.approx([60, 20], abs=0.01)


@pytest.mark.parametrize("text", ["山田 太郎", "Name\nTag", "あいうえお\nかきくけこ"])
def test_watertight_and_export(jp_font, tmp_path, text):
    mesh = make_nameplate(text, jp_font, width=90, depth=40, corner_radius=4)
    assert mesh.is_watertight
    path = export_mesh(mesh, tmp_path / "plate.3mf")
    assert trimesh.load_mesh(path).is_watertight


# ---- エラー ----

def test_no_size_given(jp_font):
    with pytest.raises(TextMeshError, match="どちらかを指定"):
        make_nameplate("A", jp_font)


def test_text_does_not_fit(jp_font):
    with pytest.raises(TextMeshError, match="板に収まりません"):
        make_nameplate("ABCDEFG", jp_font, width=30, depth=20, size=10)


def test_margin_too_large(jp_font):
    with pytest.raises(TextMeshError, match="文字を置く場所がありません"):
        make_nameplate("A", jp_font, width=20, depth=20, margin=10)


def test_corner_radius_too_large(jp_font):
    with pytest.raises(TextMeshError, match="角の丸み"):
        make_nameplate("A", jp_font, width=40, depth=20, size=5, corner_radius=11)


@pytest.mark.parametrize("kwargs, message", [
    ({"base_thickness": 0}, "台座の厚み"),
    ({"text_height": -1}, "文字の高さ"),
    ({"margin": -1}, "余白"),
    ({"corner_radius": -1}, "角の丸み"),
    ({"width": 0}, "板の幅"),
    ({"depth": -5}, "板の奥行き"),
])
def test_invalid_numbers(jp_font, kwargs, message):
    params = {"size": 10, **kwargs}
    with pytest.raises(TextMeshError, match=message):
        make_nameplate("A", jp_font, **params)


# ---- テスト用の補助 ----

def _letters_bounds(mesh, base_thickness):
    """台座より上（文字の部分）の頂点の範囲を返す。"""
    top = mesh.vertices[mesh.vertices[:, 2] > base_thickness + 1e-6]
    return top.min(axis=0), top.max(axis=0)
