"""core/exporter.py のテスト。"""

import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from core.exporter import (
    ExportError,
    NotWatertightError,
    export_amf,
    export_mesh,
    get_format,
)


@pytest.fixture
def box():
    """10 x 20 x 30 mm の箱（体積 6000 mm³）。"""
    return trimesh.creation.box((10, 20, 30))


@pytest.fixture
def open_box(box):
    """面を1枚取り除いて穴を開けた箱。"""
    return trimesh.Trimesh(vertices=box.vertices, faces=box.faces[1:], process=False)


# ---- 形式の判定 ----

@pytest.mark.parametrize("name, expected", [
    ("a.stl", "stl"), ("a.OBJ", "obj"), ("a.3mf", "3mf"), ("a.Amf", "amf"),
])
def test_get_format(name, expected):
    assert get_format(name) == expected


@pytest.mark.parametrize("name", ["a.ply", "a.txt", "a"])
def test_get_format_unsupported(name):
    with pytest.raises(ExportError, match="対応していない拡張子"):
        get_format(name)


# ---- STL / OBJ / 3MF ----

@pytest.mark.parametrize("ext", ["stl", "obj", "3mf"])
def test_export_and_reload(box, tmp_path, ext):
    """書き出したファイルを読み直して、形と大きさが保たれているか確認する。"""
    path = export_mesh(box, tmp_path / f"box.{ext}")
    assert path.exists()

    loaded = trimesh.load_mesh(path)
    assert loaded.is_watertight
    assert loaded.volume == pytest.approx(6000, rel=1e-6)
    assert np.allclose(loaded.extents, [10, 20, 30])


def test_creates_parent_folder(box, tmp_path):
    path = export_mesh(box, tmp_path / "sub" / "dir" / "box.stl")
    assert path.exists()


# ---- AMF ----

def test_export_amf(box, tmp_path):
    path = export_mesh(box, tmp_path / "box.amf")
    root = ET.parse(path).getroot()

    assert root.tag == "amf"
    assert root.get("unit") == "millimeter"

    # 頂点と三角形の数が元のメッシュと一致するか
    vertices = root.findall("./object/mesh/vertices/vertex")
    triangles = root.findall("./object/mesh/volume/triangle")
    assert len(vertices) == len(box.vertices)
    assert len(triangles) == len(box.faces)

    # 読み直した値から同じメッシュを組み立てられるか
    coords = np.array([
        [float(v.find(f"coordinates/{a}").text) for a in "xyz"] for v in vertices
    ])
    faces = np.array([
        [int(t.find(k).text) for k in ("v1", "v2", "v3")] for t in triangles
    ])
    assert np.allclose(coords, box.vertices)
    assert np.array_equal(faces, box.faces)

    rebuilt = trimesh.Trimesh(vertices=coords, faces=faces)
    assert rebuilt.is_watertight
    assert rebuilt.volume == pytest.approx(6000, rel=1e-6)


def test_amf_name_is_escaped(box, tmp_path):
    """名前に XML の特殊文字や日本語が入っても壊れないか。"""
    path = tmp_path / "box.amf"
    export_amf(box, path, name='<名前> & "テスト"')
    root = ET.parse(path).getroot()
    assert root.find("metadata").text == '<名前> & "テスト"'


# ---- エラー ----

@pytest.mark.parametrize("ext", ["stl", "obj", "3mf", "amf"])
def test_not_watertight_raises(open_box, tmp_path, ext):
    """穴のあるメッシュはエラーになり、ファイルも作られない。"""
    assert not open_box.is_watertight
    path = tmp_path / f"open.{ext}"
    with pytest.raises(NotWatertightError, match="watertight ではありません"):
        export_mesh(open_box, path)
    assert not path.exists()


def test_empty_mesh_raises(tmp_path):
    with pytest.raises(ExportError, match="空です"):
        export_mesh(trimesh.Trimesh(), tmp_path / "empty.stl")


def test_wrong_type_raises(tmp_path):
    with pytest.raises(ExportError, match="型が正しくありません"):
        export_mesh(trimesh.Scene(), tmp_path / "scene.stl")


def test_unsupported_extension_raises(box, tmp_path):
    with pytest.raises(ExportError, match="対応していない拡張子"):
        export_mesh(box, tmp_path / "box.ply")
