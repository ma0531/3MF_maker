"""メッシュを 3D プリンター用のファイルに書き出す処理。

対応形式: STL / OBJ / 3MF / AMF
- STL・OBJ・3MF は trimesh の書き出し機能を使う
- AMF は trimesh が対応していないため、XML を自分で組み立てて書き出す

単位はすべて mm。保存前に必ず watertight（穴のない閉じた形）かどうかを確認する。
"""

from pathlib import Path
from xml.sax.saxutils import quoteattr

import trimesh

# 対応している形式（拡張子、小文字・ドットなし）
SUPPORTED_FORMATS = ("stl", "obj", "3mf", "amf")


class ExportError(Exception):
    """書き出しに失敗したときのエラー。"""


class NotWatertightError(ExportError):
    """メッシュが watertight でないときのエラー。"""


def check_watertight(mesh: trimesh.Trimesh) -> None:
    """メッシュが書き出せる状態かを確認する。問題があれば例外を出す。"""
    if not isinstance(mesh, trimesh.Trimesh):
        raise ExportError(
            f"メッシュの型が正しくありません（{type(mesh).__name__}）。"
            "trimesh.Trimesh を渡してください。"
        )
    if len(mesh.faces) == 0:
        raise ExportError("メッシュが空です（三角形が1つもありません）。")
    if not mesh.is_watertight:
        raise NotWatertightError(
            "メッシュが watertight ではありません（穴や隙間があります）。"
            "このままでは 3D プリンターで正しく造形できないため、保存を中止しました。"
        )


def get_format(path: str | Path) -> str:
    """ファイルパスの拡張子から形式を判定する。"""
    ext = Path(path).suffix.lower().lstrip(".")
    if ext not in SUPPORTED_FORMATS:
        names = " / ".join(f.upper() for f in SUPPORTED_FORMATS)
        raise ExportError(
            f"対応していない拡張子です（.{ext or '（なし）'}）。{names} のいずれかを指定してください。"
        )
    return ext


def export_mesh(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    """メッシュを拡張子に応じた形式で保存し、保存先のパスを返す。

    watertight でない場合は NotWatertightError を出し、ファイルは作らない。
    保存先のフォルダがなければ作成する。
    """
    path = Path(path)
    file_format = get_format(path)
    check_watertight(mesh)

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if file_format == "amf":
            export_amf(mesh, path)
        else:
            mesh.export(path, file_type=file_format)
    except OSError as e:
        raise ExportError(f"ファイルを保存できませんでした: {path}（{e}）") from e
    return path


def export_amf(mesh: trimesh.Trimesh, path: str | Path, name: str = "mesh") -> None:
    """メッシュを AMF（XML 形式）で保存する。単位は mm。

    構成:
        <amf unit="millimeter">
          <object id="0">
            <mesh>
              <vertices> <vertex><coordinates><x/><y/><z/></coordinates></vertex> ... </vertices>
              <volume>   <triangle><v1/><v2/><v3/></triangle> ... </volume>
            </mesh>
          </object>
        </amf>
    """
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<amf unit="millimeter" version="1.1">\n')
        f.write(f"  <metadata type=\"name\">{_escape(name)}</metadata>\n")
        f.write('  <object id="0">\n')
        f.write("    <mesh>\n")

        # 頂点の一覧（小数点以下 6 桁 = 1nm 単位まで残す）
        f.write("      <vertices>\n")
        f.writelines(
            "        <vertex><coordinates>"
            f"<x>{x:.6f}</x><y>{y:.6f}</y><z>{z:.6f}</z>"
            "</coordinates></vertex>\n"
            for x, y, z in mesh.vertices.tolist()
        )
        f.write("      </vertices>\n")

        # 三角形の一覧（頂点番号で指定）
        f.write("      <volume>\n")
        f.writelines(
            f"        <triangle><v1>{a}</v1><v2>{b}</v2><v3>{c}</v3></triangle>\n"
            for a, b, c in mesh.faces.tolist()
        )
        f.write("      </volume>\n")

        f.write("    </mesh>\n")
        f.write("  </object>\n")
        f.write("</amf>\n")


def _escape(text: str) -> str:
    """XML の要素内に入れる文字列をエスケープする。"""
    # quoteattr は前後に引用符を付けるので取り除く
    return quoteattr(text)[1:-1]
