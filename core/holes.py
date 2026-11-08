"""作った形に、上から下まで貫通する穴を開ける処理。

穴の形は、丸・四角・三角・ひし形・星・六角形・ハート・長穴から選べる。
穴は manifold3d で「穴の形の柱」を削り取って作るので、結果も必ず閉じた立体（watertight）になる。

座標: 位置 X・Y は穴の中心で、完成図の左下（形の最小の X・Y）を 0 とした mm。
大きさ: 幅（X 方向）と縦（Y 方向）で指定する。縦を省略すると、その形の自然な比率になる
        （正三角形・正六角形・円など）。角度を指定すると、中心を軸に反時計回りに回す。
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import manifold3d as m3d
import numpy as np
import trimesh

from core.text_mesh import CURVE_STEP_MM, manifold_to_trimesh

# 穴の形（値 → 画面に出す名前）
SHAPE_LABELS = {
    "circle": "丸",
    "square": "四角",
    "triangle": "三角",
    "diamond": "ひし形",
    "star": "星",
    "hexagon": "六角形",
    "heart": "ハート",
    "slot": "長穴",
}

# 穴と形の端の間がこれより薄いと、割れやすいので警告する（mm）
MIN_WALL_MM = 1.0
# 「何も削っていない」と判断する体積（mm³）
EMPTY_VOLUME = 1e-6
# 長穴の縦を省略したときの、幅に対する比率
SLOT_RATIO = 0.4
# 星のへこみの深さ（内側の頂点の半径 / 外側の頂点の半径）。0.382 で正しい五芒星
STAR_INNER_RATIO = 0.382
# 穴をあけた後もそのまま引き継がない情報
# （grid_size は「頂点の並び順」を前提にしているため、穴あけ後は使えない）
DROP_METADATA = ("grid_size",)


class HoleError(Exception):
    """穴あけに失敗したときのエラー。"""


@dataclass(frozen=True)
class Hole:
    """穴 1 つ分（単位 mm・度）。"""
    x: float                    # 中心の X
    y: float                    # 中心の Y
    width: float                # 幅（X 方向）。丸なら直径、六角形なら対辺の長さ
    height: float | None = None  # 縦（Y 方向）。None なら形の自然な比率
    shape: str = "circle"       # SHAPE_LABELS のキー
    angle: float = 0.0          # 回転（度, 反時計回り）

    @property
    def label(self) -> str:
        return SHAPE_LABELS.get(self.shape, self.shape)


def add_holes(mesh: trimesh.Trimesh, holes: Iterable[Hole | tuple]) -> trimesh.Trimesh:
    """メッシュに貫通穴を開けた新しいメッシュを返す（元のメッシュは変えない）。

    穴が形の外にある場合はエラー。
    端に近すぎる・何も削っていない・文字にかかる場合は、mesh.metadata["warnings"] に警告を追加する。
    """
    holes = [h if isinstance(h, Hole) else Hole(*h) for h in holes]
    if not holes:
        return mesh

    (x0, y0, z0), (x1, y1, z1) = mesh.bounds
    warnings = list(mesh.metadata.get("warnings", []))
    solid = trimesh_to_manifold(mesh)
    cut_height = (z1 - z0) + 2.0  # 上下に 1mm ずつはみ出させて確実に貫通させる

    cutters = []
    for i, h in enumerate(holes, start=1):
        _check_hole(i, h)
        outline = hole_outline(h)
        hx0, hy0 = outline.min(axis=0)
        hx1, hy1 = outline.max(axis=0)

        # 形の範囲（外枠）と重なっていなければ、入力ミスとしてエラーにする
        if hx1 <= x0 or hx0 >= x1 or hy1 <= y0 or hy0 >= y1:
            raise HoleError(
                f"穴{i}（{h.label}, X={h.x:g}, Y={h.y:g}）が形の外にあります。"
                f"形の範囲は X {x0:.1f}〜{x1:.1f} mm、Y {y0:.1f}〜{y1:.1f} mm です。"
            )

        # 外枠の端との間の厚み（角を丸めた板や文字では目安）
        wall = min(hx0 - x0, x1 - hx1, hy0 - y0, y1 - hy1)
        if wall <= 0:
            warnings.append(f"穴{i}が形の端にかかっています（切り欠きになります）。")
        elif wall < MIN_WALL_MM:
            warnings.append(
                f"穴{i}と形の端の間が {wall:.1f} mm しかなく、割れやすくなります。"
                f"{MIN_WALL_MM:g} mm 以上離すのがおすすめです。"
            )

        cutter = (
            m3d.CrossSection([outline], m3d.FillRule.NonZero)
            .extrude(cut_height)
            .translate((0.0, 0.0, z0 - 1.0))
        )

        cut_volume = (solid ^ cutter).volume()  # この穴で削る体積
        if cut_volume <= EMPTY_VOLUME:
            warnings.append(f"穴{i}の位置には何もないため、穴になっていません。")
        elif "base_thickness" in mesh.metadata:
            # ネームプレート: 台座だけを削るなら「穴の断面積 x 台座の厚み」以下になる
            area = cutter.volume() / cut_height
            if cut_volume > area * mesh.metadata["base_thickness"] * 1.01 + EMPTY_VOLUME:
                warnings.append(
                    f"穴{i}が文字にかかっていて、文字の一部が削れます。"
                    "文字にかからない位置にするか、余白を広げてください。"
                )
        cutters.append(cutter)

    result = solid - m3d.Manifold.batch_boolean(cutters, m3d.OpType.Add)
    if result.is_empty():
        raise HoleError("穴を開けると形がなくなってしまいます。穴を小さくしてください。")

    out = manifold_to_trimesh(result)
    out.metadata.update(
        {k: v for k, v in mesh.metadata.items() if k not in DROP_METADATA}
    )
    out.metadata["holes"] = tuple(holes)
    out.metadata["warnings"] = warnings
    return out


def hole_outline(h: Hole) -> np.ndarray:
    """穴の輪郭の点列（N x 2, mm）。位置・大きさ・回転を反映した実際の座標。"""
    if h.shape == "slot":
        pts = _slot_outline(h.width, hole_height(h))
    else:
        unit = _unit_outline(h.shape, _segments(max(h.width, hole_height(h))))
        natural = _natural_ratio(h.shape)
        pts = unit * np.array([h.width, hole_height(h) / natural])

    if h.angle:
        a = math.radians(h.angle)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        pts = pts @ rot.T
    return pts + np.array([h.x, h.y])


def hole_height(h: Hole) -> float:
    """穴の縦の長さ（省略されていれば、形の自然な比率から求める）。"""
    if h.height is not None:
        return h.height
    if h.shape == "slot":
        return h.width * SLOT_RATIO
    return h.width * _natural_ratio(h.shape)


def trimesh_to_manifold(mesh: trimesh.Trimesh) -> m3d.Manifold:
    """trimesh のメッシュを manifold3d の立体に変換する。"""
    m = m3d.Mesh(
        vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
        tri_verts=np.asarray(mesh.faces, dtype=np.uint32),
    )
    solid = m3d.Manifold(m)
    if solid.status() != m3d.Error.NoError:
        raise HoleError("形が閉じていないため、穴を開けられません。")
    return solid


# ---------------------------------------------------------------------------
# 形の定義
# ---------------------------------------------------------------------------

def _raw_outline(shape: str, n: int) -> np.ndarray:
    """形の輪郭（大きさ・中心はまだそろえていない）。"""
    if shape == "circle":
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(t), np.sin(t)])
    if shape == "square":
        return np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    if shape == "triangle":
        # 正三角形（上向き）
        return np.array([[-1, 0], [1, 0], [0, math.sqrt(3)]], dtype=float)
    if shape == "diamond":
        return np.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=float)
    if shape == "star":
        # 五芒星（上向き）。外側と内側の頂点を交互に並べる
        angles = np.pi / 2 + np.arange(10) * np.pi / 5
        radius = np.where(np.arange(10) % 2 == 0, 1.0, STAR_INNER_RATIO)
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    if shape == "hexagon":
        # 正六角形（頂点が上下）。幅＝対辺の長さになる（ナットのサイズと同じ測り方）
        angles = np.pi / 6 + np.arange(6) * np.pi / 3
        return np.column_stack([np.cos(angles), np.sin(angles)])
    if shape == "heart":
        t = np.linspace(0, 2 * np.pi, max(n, 64), endpoint=False)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        return np.column_stack([x, y])
    raise HoleError(f"穴の形の指定が正しくありません（{shape}）。")


def _unit_outline(shape: str, n: int = 96) -> np.ndarray:
    """幅 1・中心 (0, 0) にそろえた輪郭（縦は形の自然な比率のまま）。"""
    pts = _raw_outline(shape, n)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return (pts - (lo + hi) / 2) / (hi[0] - lo[0])


def _natural_ratio(shape: str) -> float:
    """形の自然な「縦 / 幅」の比率（正三角形なら約 0.866）。"""
    pts = _unit_outline(shape, 96)
    return float(pts[:, 1].max() - pts[:, 1].min())


def _slot_outline(width: float, height: float) -> np.ndarray:
    """長穴（両端が半円の細長い形）。長い方向の両端を丸める。"""
    if width < height:
        # 縦長の長穴は、横長を作ってから 90° 回す
        return _slot_outline(height, width)[:, ::-1] * np.array([1, -1])
    r = height / 2
    c = width / 2 - r  # 半円の中心の位置
    n = max(8, _segments(height) // 2)
    t_right = np.linspace(-np.pi / 2, np.pi / 2, n)
    t_left = np.linspace(np.pi / 2, 3 * np.pi / 2, n)
    right = np.column_stack([c + r * np.cos(t_right), r * np.sin(t_right)])
    left = np.column_stack([-c + r * np.cos(t_left), r * np.sin(t_left)])
    return np.vstack([right, left])


def _segments(size: float) -> int:
    """曲線の分割数（大きいほど細かく）。"""
    return max(24, math.ceil(math.pi * size / CURVE_STEP_MM))


def _check_hole(i: int, h: Hole) -> None:
    if h.shape not in SHAPE_LABELS:
        names = "・".join(SHAPE_LABELS.values())
        raise HoleError(f"穴{i}の形の指定が正しくありません（{h.shape}）。{names} から選んでください。")
    values = [(h.x, "X"), (h.y, "Y"), (h.width, "幅"), (h.angle, "角度")]
    if h.height is not None:
        values.append((h.height, "縦"))
    for value, label in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise HoleError(f"穴{i}の{label}が数値ではありません（入力値: {value}）。")
    if h.width <= 0:
        raise HoleError(f"穴{i}の幅は 0 より大きくしてください（入力値: {h.width:g}）。")
    if h.height is not None and h.height <= 0:
        raise HoleError(f"穴{i}の縦は 0 より大きくしてください（入力値: {h.height:g}）。")
