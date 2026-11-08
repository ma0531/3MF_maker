"""完成図（3D プレビュー）を matplotlib で描く処理。

tkinter の画面には FigureCanvasTkAgg で Figure を埋め込んで使う。
画面なしで PNG に保存することもできる（動作確認用）。

matplotlib の 3D 表示は三角形が多いと重いので、
レリーフ・リトファンは解像度を粗くしたメッシュを渡すこと（preview_resolution を使う）。
"""

from pathlib import Path

import matplotlib
import numpy as np
import trimesh
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
# ファイルパスの Path と名前がぶつからないよう MplPath と呼ぶ
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from core.holes import hole_outline
from core.image_mesh import top_heights
from core.text_mesh import DEFAULT_FONT


def _setup_japanese_font() -> None:
    """グラフの文字（タイトルなど）を日本語で表示できるようにする。"""
    try:
        font_manager.fontManager.addfont(DEFAULT_FONT)
        name = font_manager.FontProperties(fname=DEFAULT_FONT).get_name()
        matplotlib.rcParams["font.family"] = name
    except (OSError, RuntimeError, ValueError):
        pass  # フォントがなくても表示自体はできる（日本語が□になるだけ）


_setup_japanese_font()

# プレビューで描く三角形の数の目安（これを超えると表示が遅くなる）
PREVIEW_MAX_FACES = 60_000
# プレビュー時の画像メッシュの列数（横方向の点の数）
PREVIEW_COLUMNS = 150

# 色と光の向き
BASE_COLOR = np.array([0.30, 0.55, 0.85])  # 明るめの青
LIGHT_DIR = np.array([-0.8, -0.3, 0.8])     # 左手前の斜め上から照らす（凹凸が見えやすい）

# リトファンの光の見え方の計算に使う値
LIGHT_ABSORPTION = 1.2                        # 厚み 1mm あたりの光の弱まり方（白い PLA の目安）
BACKLIGHT_COLOR = np.array([1.0, 0.88, 0.70])  # 電球色

# 視点（仰角・方位角, 度）
VIEW_ELEV = 35
VIEW_AZIM = -60


def _subdivide_for_preview(mesh: trimesh.Trimesh, size: float) -> tuple[np.ndarray, np.ndarray]:
    """大きな三角形を細かく分ける（三角形の数が上限を超えない範囲で）。

    matplotlib は三角形の中心の奥行きで描く順番を決めるため、
    大きな三角形があると手前・奥を間違えて表示が崩れる。それを防ぐ。
    """
    vertices, faces = mesh.vertices, mesh.faces
    if len(faces) >= PREVIEW_MAX_FACES // 2:
        return vertices, faces  # 元から細かいメッシュ（レリーフなど）はそのまま
    # 細かい分け方から順に試し、上限に収まった最初のものを使う
    for divisions in (40, 25, 15, 8):
        v, f = trimesh.remesh.subdivide_to_size(
            vertices, faces, max_edge=size / divisions, max_iter=8
        )
        if len(f) <= PREVIEW_MAX_FACES:
            return v, f
    return vertices, faces


def _face_normals(tris: np.ndarray) -> np.ndarray:
    """三角形の頂点座標から、単位法線を求める。"""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(length > 0, length, 1.0)


def preview_resolution(width: float, resolution: float) -> float:
    """プレビュー用の解像度（mm/画素）。本番より細かくはしない。"""
    return max(resolution, width / (PREVIEW_COLUMNS - 1))


def draw_mesh(
    fig: Figure,
    mesh: trimesh.Trimesh,
    title: str = "",
    size: np.ndarray | None = None,
) -> None:
    """Figure の中身を消して、メッシュの完成図を描く。

    size: タイトルに出す寸法 (X, Y, Z)。粗いメッシュを描くときに、
          本番のメッシュの寸法を出すために使う。None なら描くメッシュの寸法。
    """
    fig.clear()
    ax = fig.add_subplot(projection="3d")
    lo, hi = mesh.bounds
    ext = np.maximum(hi - lo, 1e-6)

    vertices, faces = _subdivide_for_preview(mesh, ext.max())
    tris = vertices[faces]
    normals = _face_normals(tris)

    # 視点から見て裏を向いている面（底面など）は描かない
    elev, azim = np.radians(VIEW_ELEV), np.radians(VIEW_AZIM)
    view = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    visible = normals @ view > -1e-3
    tris, normals = tris[visible], normals[visible]

    if len(tris) > PREVIEW_MAX_FACES:
        # 数が多すぎるときは間引いて描く（あくまで目安の表示）
        step = int(np.ceil(len(tris) / PREVIEW_MAX_FACES))
        tris, normals = tris[::step], normals[::step]

    # 面の向きと光の向きから明るさを決める（簡易的な陰影）
    light = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
    shade = 0.25 + 0.75 * np.clip(normals @ light, 0.0, 1.0)
    colors = np.clip(BASE_COLOR[None, :] * shade[:, None] + 0.08, 0, 1)

    # 縁を面と同じ色で細く塗り、三角形の境目に出る白い線を消す
    ax.add_collection3d(
        Poly3DCollection(tris, facecolors=colors, edgecolors=colors, linewidths=0.3)
    )

    # 縦横比を実寸どおりにする（薄い形がつぶれて見えないよう最小値を設ける）
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    # zoom を少し下げて、軸の名前が枠からはみ出さないようにする
    ax.set_box_aspect(np.maximum(ext, ext.max() * 0.05), zoom=0.85)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    # 薄い・細長い形でも目盛りが重ならないように数を減らす
    for axis, length in zip((ax.xaxis, ax.yaxis), ext[:2]):
        axis.set_major_locator(MaxNLocator(nbins=max(2, int(6 * length / ext.max()))))
    ax.set_zticks([round(lo[2], 1), round(hi[2], 1)])
    ax.tick_params(labelsize=7)

    sx, sy, sz = ext if size is None else size
    size_text = f"{sx:.1f} x {sy:.1f} x {sz:.1f} mm"
    ax.set_title(f"{title}\n{size_text}" if title else size_text, fontsize=10)
    fig.tight_layout()


def backlit_image(mesh: trimesh.Trimesh, holes=()) -> np.ndarray:
    """リトファンに光を当てたときの見え方を、RGB（0〜1）の配列で返す。

    mesh は穴を開ける前のリトファン。holes（core.holes.Hole の並び）を渡すと、
    穴の部分は光がそのまま通るので白く描く。
    """
    thickness = top_heights(mesh)
    t_min, _ = mesh.metadata.get("thickness_range", (thickness.min(), thickness.max()))
    light = np.exp(-LIGHT_ABSORPTION * (thickness - t_min))  # 1.0 = 一番明るい
    rgb = np.clip(light[..., None] * BACKLIGHT_COLOR[None, None, :], 0, 1)

    if holes:
        rows, cols = thickness.shape
        width, depth = mesh.metadata["size_mm"]
        step = width / (cols - 1)
        # 表の各点の位置（mm）。0 行目が奥（Y が大きい側）
        xs = np.arange(cols) * step
        ys = depth - np.arange(rows) * step
        gx, gy = np.meshgrid(xs, ys)
        points = np.column_stack([gx.ravel(), gy.ravel()])
        for h in holes:
            # 穴の輪郭の内側にある点を白くする（丸以外の形にも対応）
            inside = MplPath(hole_outline(h)).contains_points(points).reshape(gx.shape)
            rgb[inside] = 1.0
    return rgb


def draw_backlit(fig: Figure, mesh: trimesh.Trimesh, title: str = "", holes=(), marker=None) -> None:
    """リトファンに後ろから光を当てたときの見え方を、2D の画像で描く。

    光は厚みに応じて指数的に弱くなる（厚い＝暗い）として簡易的に計算する。
    白い PLA を想定した目安で、実物の明るさとは一致しない。
    """
    rgb = backlit_image(mesh, holes)
    width, depth = mesh.metadata["size_mm"]
    fig.clear()
    ax = fig.add_subplot()
    # 表の 0 行目が画像の上（奥 = +Y）なので origin="upper" で描く
    ax.imshow(rgb, extent=(0, width, 0, depth), origin="upper", interpolation="bilinear")
    ax.set_facecolor("black")
    _draw_marker(ax, marker)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.tick_params(labelsize=7)
    size_text = f"{width:.1f} x {depth:.1f} mm（光を当てたときの見え方・目安）"
    ax.set_title(f"{title}\n{size_text}" if title else size_text, fontsize=10)
    fig.tight_layout()


def draw_top(
    fig: Figure,
    mesh: trimesh.Trimesh,
    title: str = "",
    size: np.ndarray | None = None,
    marker: np.ndarray | None = None,
) -> None:
    """真上から見た図（平面図）を描く。高い所ほど明るい色にする。

    座標は mm で、クリックした位置をそのまま穴の位置に使える。
    marker（輪郭の点列, mm）を渡すと、赤い点線で重ねて描く（クリックした穴の位置の確認用）。
    """
    fig.clear()
    ax = fig.add_subplot()
    lo, hi = mesh.bounds

    # 上を向いている面だけを、低い順に描く（高い面が上に重なるように）
    normals = mesh.face_normals
    up = normals[:, 2] > 0.3
    tris = mesh.vertices[mesh.faces[up]]
    heights = tris[:, :, 2].mean(axis=1)
    order = np.argsort(heights)
    tris, heights = tris[order], heights[order]

    span = max(hi[2] - lo[2], 1e-6)
    level = (heights - lo[2]) / span  # 0 = 一番低い, 1 = 一番高い
    colors = np.clip(BASE_COLOR[None, :] * (0.45 + 0.55 * level[:, None]) + 0.1 * level[:, None], 0, 1)
    ax.add_collection(PolyCollection(tris[:, :, :2], facecolors=colors, edgecolors=colors,
                                     linewidths=0.3))

    _draw_marker(ax, marker)
    margin = 0.03 * max(hi[0] - lo[0], hi[1] - lo[1])
    ax.set_xlim(lo[0] - margin, hi[0] + margin)
    ax.set_ylim(lo[1] - margin, hi[1] + margin)
    ax.set_aspect("equal")
    ax.set_facecolor("#f4f4f4")
    ax.set_axisbelow(True)  # 目盛りの線は形の下に描く
    ax.grid(True, color="#cccccc", linewidth=0.5)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.tick_params(labelsize=7)

    sx, sy, sz = (hi - lo) if size is None else size
    size_text = f"{sx:.1f} x {sy:.1f} x {sz:.1f} mm（真上から・クリックで穴の位置）"
    ax.set_title(f"{title}\n{size_text}" if title else size_text, fontsize=10)
    fig.tight_layout()


def _draw_marker(ax, marker: np.ndarray | None) -> None:
    """クリックした穴の位置を、赤い点線の輪郭で描く。"""
    if marker is None or len(marker) < 2:
        return
    closed = np.vstack([marker, marker[:1]])
    ax.plot(closed[:, 0], closed[:, 1], linestyle="--", color="#d32f2f", linewidth=1.5)
    center = marker.mean(axis=0)
    ax.plot([center[0]], [center[1]], marker="+", color="#d32f2f", markersize=10)


def save_preview_png(
    mesh: trimesh.Trimesh,
    path: str | Path,
    title: str = "",
    size_inch: tuple[float, float] = (6, 5),
    backlit: bool = False,
    top: bool = False,
) -> Path:
    """完成図を PNG に保存する（画面なしで動く）。

    backlit=True なら、リトファンに光を当てたときの見え方を描く。
    top=True なら、真上から見た図を描く。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=size_inch, dpi=100)
    FigureCanvasAgg(fig)
    if backlit:
        draw_backlit(fig, mesh, title)
    elif top:
        draw_top(fig, mesh, title)
    else:
        draw_mesh(fig, mesh, title)
    fig.savefig(path)
    return path
