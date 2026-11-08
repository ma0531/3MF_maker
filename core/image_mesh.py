"""画像からレリーフ・リトファンのメッシュを作る処理。

流れ:
    1. 画像を読み込み、グレースケール（0=黒, 1=白）にする
    2. 解像度（mm/画素）に合わせて縮小する（大きすぎると重くなるため）
    3. 明るさを高さ（厚み）に変換する
    4. 高さの表から、上面・側面・底面をもつ閉じた立体を組み立てる

座標: 単位は mm。画像の上が +Y、左が X=0、底面が Z=0 になる。
      上（+Z）から見ると、画像が反転せずにそのまま見える向き。
"""

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageOps, UnidentifiedImageError

# 既定の解像度（mm/画素）
DEFAULT_RESOLUTION = 0.2
# 頂点数の上限（これを超えると処理・保存・スライスが極端に重くなる）
MAX_VERTICES = 4_000_000

ImageInput = str | Path | Image.Image | np.ndarray


class ImageMeshError(Exception):
    """画像メッシュの作成に失敗したときのエラー。"""


# ---------------------------------------------------------------------------
# 公開する関数
# ---------------------------------------------------------------------------

def make_relief(
    image: ImageInput,
    width: float = 100.0,
    min_height: float = 1.0,
    max_height: float = 5.0,
    invert: bool = False,
    resolution: float = DEFAULT_RESOLUTION,
    brightness: float = 0.0,
    contrast: float = 100.0,
    blur: float = 0.0,
    auto_levels: bool = False,
    crop_left: float = 0.0,
    crop_top: float = 0.0,
    crop_right: float = 0.0,
    crop_bottom: float = 0.0,
) -> trimesh.Trimesh:
    """画像の明るさを高さに変換したレリーフを作る（明るい＝高い）。

    Args:
        image: 画像ファイルのパス、PIL の画像、または 2 次元の配列（0〜255 か 0〜1）。
        width: 完成品の幅（X 方向, mm）。奥行きは画像の縦横比で決まる。
        min_height: 一番暗い部分の高さ（mm）。底板の厚みになるので 0 より大きくする。
        max_height: 一番明るい部分の高さ（mm）。
        invert: True なら反転（暗い＝高い）。
        resolution: 1 画素の大きさ（mm）。小さいほど細かいが重くなる。
        brightness, contrast, blur, auto_levels, crop_*: 画像の調整（ImageAdjust を参照）。

    Returns:
        watertight な trimesh.Trimesh。
    """
    _check_positive(min_height, "最小高さ")
    _check_positive(max_height, "最大高さ")
    if max_height <= min_height:
        raise ImageMeshError(
            f"最大高さ（{max_height} mm）は最小高さ（{min_height} mm）より大きくしてください。"
        )

    adjust = ImageAdjust(brightness, contrast, blur, auto_levels,
                         crop_left, crop_top, crop_right, crop_bottom)
    values = load_brightness(image, width, resolution, adjust)
    if invert:
        values = 1.0 - values
    heights = min_height + values * (max_height - min_height)
    mesh = heightmap_to_mesh(heights, width)
    mesh.metadata["kind"] = "relief"
    return mesh


def make_lithophane(
    image: ImageInput,
    width: float = 100.0,
    min_thickness: float = 0.8,
    max_thickness: float = 3.0,
    resolution: float = DEFAULT_RESOLUTION,
    brightness: float = 0.0,
    contrast: float = 100.0,
    blur: float = 0.0,
    auto_levels: bool = False,
    crop_left: float = 0.0,
    crop_top: float = 0.0,
    crop_right: float = 0.0,
    crop_bottom: float = 0.0,
) -> trimesh.Trimesh:
    """光に透かして見るリトファンを作る（暗い＝厚い、明るい＝薄い）。

    Args:
        image: 画像ファイルのパス、PIL の画像、または 2 次元の配列。
        width: 完成品の幅（X 方向, mm）。
        min_thickness: 一番明るい部分の厚み（mm）。
        max_thickness: 一番暗い部分の厚み（mm）。
        resolution: 1 画素の大きさ（mm）。
        brightness, contrast, blur, auto_levels, crop_*: 画像の調整（ImageAdjust を参照）。

    Returns:
        watertight な trimesh.Trimesh。裏面（Z=0）は平ら。
    """
    _check_positive(min_thickness, "最小厚み")
    _check_positive(max_thickness, "最大厚み")
    if max_thickness <= min_thickness:
        raise ImageMeshError(
            f"最大厚み（{max_thickness} mm）は最小厚み（{min_thickness} mm）より大きくしてください。"
        )

    adjust = ImageAdjust(brightness, contrast, blur, auto_levels,
                         crop_left, crop_top, crop_right, crop_bottom)
    values = load_brightness(image, width, resolution, adjust)
    thickness = max_thickness - values * (max_thickness - min_thickness)
    mesh = heightmap_to_mesh(thickness, width)
    # 完成図（光を当てたときの見え方）を描くときに使う情報
    mesh.metadata["kind"] = "lithophane"
    mesh.metadata["thickness_range"] = (float(min_thickness), float(max_thickness))
    return mesh


def top_heights(mesh: trimesh.Trimesh) -> np.ndarray:
    """heightmap_to_mesh で作ったメッシュから、上面の高さの表を取り出す。

    上面の頂点は「行ごと・左から順」に先頭に並んでいるので、その順で並べ直す。
    """
    if "grid_size" not in mesh.metadata:
        raise ImageMeshError("画像から作ったメッシュではないため、高さの表を取り出せません。")
    cols, rows = mesh.metadata["grid_size"]
    return np.asarray(mesh.vertices[: cols * rows, 2]).reshape(rows, cols)


def load_brightness(
    image: ImageInput,
    width: float,
    resolution: float,
    adjust: "ImageAdjust | None" = None,
) -> np.ndarray:
    """画像を読み込み、解像度に合わせて縮小した明るさの表（0.0=黒〜1.0=白）を返す。

    表の大きさは「幅 / 解像度 + 1」列。画素の中心を頂点として使うため +1 している。
    adjust を渡すと、切り抜き → 縮小 → ぼかし → 自動補正 → コントラスト → 明るさ の順に調整する。
    """
    _check_positive(width, "幅")
    _check_positive(resolution, "解像度")
    adjust = adjust or ImageAdjust()
    adjust.check()

    gray = _crop(_to_gray_array(image), adjust)
    src_h, src_w = gray.shape

    # 点と点の「間隔の数」で縦横比を合わせる（奥行き = 幅 x 縦 / 横 になるように）
    cols = round(width / resolution) + 1
    rows = round((cols - 1) * src_h / src_w) + 1
    if cols < 2 or rows < 2:
        raise ImageMeshError(
            "画像が小さすぎるか、解像度が粗すぎます。幅を大きくするか、解像度を細かくしてください。"
        )
    if cols * rows > MAX_VERTICES:
        raise ImageMeshError(
            f"点の数が多すぎます（{cols} x {rows} = {cols * rows:,} 点、上限 {MAX_VERTICES:,} 点）。"
            "幅を小さくするか、解像度の値を大きく（粗く）してください。"
        )

    # 縮小は INTER_AREA（平均をとるので細かい模様がギザギザになりにくい）、
    # 拡大になる場合は INTER_LINEAR を使う
    interp = cv2.INTER_AREA if cols < src_w else cv2.INTER_LINEAR
    resized = cv2.resize(gray, (cols, rows), interpolation=interp)
    # float32 のままだと最小厚みが 0.79999995mm のようにわずかに下回るため float64 にする
    values = np.clip(resized.astype(np.float64), 0.0, 1.0)
    return _apply_adjust(values, adjust, step=width / (cols - 1))


@dataclass(frozen=True)
class ImageAdjust:
    """画像の調整（既定値はすべて「調整しない」）。"""
    brightness: float = 0.0    # 明るさ（%, -100〜100）。足し算で明るく・暗くする
    contrast: float = 100.0    # コントラスト（%）。100 でそのまま、大きいほど明暗の差が強い
    blur: float = 0.0          # ぼかしの強さ（mm）。写真のざらつきを減らす
    auto_levels: bool = False  # 自動補正: 暗い所〜明るい所を、黒〜白いっぱいに広げる
    crop_left: float = 0.0     # 切り抜き（%）: 左から何 % 切り取るか
    crop_top: float = 0.0
    crop_right: float = 0.0
    crop_bottom: float = 0.0

    def check(self) -> None:
        """値が正しい範囲にあるか確認する。"""
        for value, label in ((self.brightness, "明るさ"), (self.contrast, "コントラスト"),
                             (self.blur, "ぼかし"), (self.crop_left, "切り抜き（左）"),
                             (self.crop_top, "切り抜き（上）"), (self.crop_right, "切り抜き（右）"),
                             (self.crop_bottom, "切り抜き（下）")):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ImageMeshError(f"{label}が数値ではありません（入力値: {value}）。")
        if not -100 <= self.brightness <= 100:
            raise ImageMeshError(f"明るさは -100〜100% にしてください（入力値: {self.brightness:g}）。")
        if self.contrast < 0:
            raise ImageMeshError(f"コントラストは 0% 以上にしてください（入力値: {self.contrast:g}）。")
        if self.blur < 0:
            raise ImageMeshError(f"ぼかしは 0 以上にしてください（入力値: {self.blur:g}）。")
        for value, label in ((self.crop_left, "左"), (self.crop_top, "上"),
                             (self.crop_right, "右"), (self.crop_bottom, "下")):
            if not 0 <= value < 100:
                raise ImageMeshError(f"切り抜き（{label}）は 0 以上 100% 未満にしてください（入力値: {value:g}）。")
        if self.crop_left + self.crop_right >= 100:
            raise ImageMeshError("切り抜きの左と右の合計が 100% 以上あり、画像が残りません。")
        if self.crop_top + self.crop_bottom >= 100:
            raise ImageMeshError("切り抜きの上と下の合計が 100% 以上あり、画像が残りません。")


def _crop(gray: np.ndarray, adjust: ImageAdjust) -> np.ndarray:
    """画像の上下左右を、指定した割合（%）だけ切り取る。"""
    h, w = gray.shape
    left = round(w * adjust.crop_left / 100)
    right = w - round(w * adjust.crop_right / 100)
    top = round(h * adjust.crop_top / 100)
    bottom = h - round(h * adjust.crop_bottom / 100)
    cropped = gray[top:bottom, left:right]
    if min(cropped.shape) < 2:
        raise ImageMeshError("切り抜いた後の画像が小さすぎます。切り抜きを減らしてください。")
    return cropped


def _apply_adjust(values: np.ndarray, adjust: ImageAdjust, step: float) -> np.ndarray:
    """縮小した明るさの表に、ぼかし → 自動補正 → コントラスト → 明るさ の順で調整をかける。

    step: 表の 1 マスの大きさ（mm）。ぼかしの強さ（mm）をマスの数に直すのに使う。
    """
    if adjust.blur > 0:
        sigma = adjust.blur / step  # mm → マスの数
        values = cv2.GaussianBlur(values, (0, 0), sigmaX=sigma, sigmaY=sigma,
                                  borderType=cv2.BORDER_REFLECT)
    if adjust.auto_levels:
        # ごく一部の極端に暗い・明るい点に引っ張られないよう、上下 1% は無視する
        lo, hi = np.percentile(values, [1, 99])
        if hi - lo > 1e-6:
            values = (values - lo) / (hi - lo)
    if adjust.contrast != 100:
        values = (values - 0.5) * (adjust.contrast / 100) + 0.5
    if adjust.brightness:
        values = values + adjust.brightness / 100
    return np.clip(values, 0.0, 1.0)


def heightmap_to_mesh(heights: np.ndarray, width: float) -> trimesh.Trimesh:
    """高さの表（行=画像の縦、列=画像の横）から、底が平らな閉じた立体を作る。

    構成:
        - 上面: 表の各点を頂点にして、隣り合う 4 点を 2 つの三角形で張る
        - 側面: 上面のふちの点と、その真下（Z=0）の点を結ぶ
        - 底面: Z=0 のふちの点を、中心の 1 点から扇形に結ぶ
                （底は平らなので、細かく分ける必要がない）
    """
    heights = np.asarray(heights, dtype=np.float64)
    if heights.ndim != 2 or min(heights.shape) < 2:
        raise ImageMeshError("高さの表は 2 x 2 以上の 2 次元配列にしてください。")
    if not np.all(np.isfinite(heights)) or heights.min() <= 0:
        raise ImageMeshError("高さはすべて 0 より大きい数値にしてください。")

    rows, cols = heights.shape
    step = width / (cols - 1)  # 頂点の間隔（mm）
    depth = step * (rows - 1)

    # --- 上面の頂点（i 行 j 列 → x = j*step, y = 上が大きい） ---
    jj, ii = np.meshgrid(np.arange(cols), np.arange(rows))
    top = np.column_stack([
        (jj * step).ravel(),
        (depth - ii * step).ravel(),
        heights.ravel(),
    ])
    idx = np.arange(rows * cols).reshape(rows, cols)

    # --- 上面の三角形（上から見て反時計回り＝法線が +Z） ---
    a = idx[:-1, :-1].ravel()  # 左上
    b = idx[:-1, 1:].ravel()   # 右上
    c = idx[1:, :-1].ravel()   # 左下
    d = idx[1:, 1:].ravel()    # 右下
    top_faces = np.concatenate([
        np.column_stack([c, d, b]),
        np.column_stack([c, b, a]),
    ])

    # --- ふちの点を、上から見て反時計回りに並べる ---
    # 手前（下の行, 左→右）→ 右の列（下→上）→ 奥（上の行, 右→左）→ 左の列（上→下）
    perimeter = np.concatenate([
        idx[-1, :-1],
        idx[::-1, -1][:-1],
        idx[0, ::-1][:-1],
        idx[:, 0][:-1],
    ])
    n_top = len(top)
    n_per = len(perimeter)

    # 底面のふちの点（上面のふちの真下, Z=0）と、底面の中心の点
    bottom = top[perimeter].copy()
    bottom[:, 2] = 0.0
    center = np.array([[width / 2, depth / 2, 0.0]])
    vertices = np.vstack([top, bottom, center])

    k = np.arange(n_per)
    k_next = (k + 1) % n_per
    t0, t1 = perimeter[k], perimeter[k_next]  # 上のふち
    b0, b1 = n_top + k, n_top + k_next        # 下のふち
    c_idx = np.full(n_per, n_top + n_per)     # 底面の中心

    # --- 側面（外から見て反時計回り＝法線が外向き） ---
    side_faces = np.concatenate([
        np.column_stack([b0, b1, t1]),
        np.column_stack([b0, t1, t0]),
    ])
    # --- 底面（法線が -Z） ---
    bottom_faces = np.column_stack([c_idx, b1, b0])

    faces = np.vstack([top_faces, side_faces, bottom_faces])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.metadata["grid_size"] = (cols, rows)
    mesh.metadata["size_mm"] = (float(width), float(depth))
    return mesh


# ---------------------------------------------------------------------------
# 内部の処理
# ---------------------------------------------------------------------------

def _check_positive(value: float, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ImageMeshError(f"{label}は 0 より大きい数値にしてください（入力値: {value}）。")


def _to_gray_array(image: ImageInput) -> np.ndarray:
    """いろいろな形の入力を、0.0〜1.0 のグレースケール配列（float32）にする。"""
    if isinstance(image, np.ndarray):
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim != 2:
            raise ImageMeshError("配列で渡す場合は 2 次元（グレースケール）にしてください。")
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ImageMeshError("配列が空か、数値でない値が含まれています。")
        # 0〜255 の値なら 0〜1 に直す
        return arr / 255.0 if arr.max() > 1.0 else arr

    if isinstance(image, Image.Image):
        img = image
    else:
        path = Path(image)
        if not path.is_file():
            raise ImageMeshError(f"画像ファイルが見つかりません: {path}")
        try:
            img = Image.open(path)
            img.load()
        except (UnidentifiedImageError, OSError) as e:
            raise ImageMeshError(f"画像ファイルを開けませんでした: {path}（{e}）") from e

    # スマホ写真などの回転情報（EXIF）を反映する
    img = ImageOps.exif_transpose(img)

    # 透明な部分は白（レリーフ＝高い / リトファン＝薄い）として扱う
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, rgba)

    gray = img.convert("L")
    return np.asarray(gray, dtype=np.float32) / 255.0
