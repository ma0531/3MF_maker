"""テキストから立体文字・ネームプレートのメッシュを作る処理。

流れ:
    1. matplotlib の TextPath で文字の輪郭（ベジェ曲線）を取り出す
    2. 曲線を折れ線に分割して、輪郭の点列（mm 単位）にする
    3. manifold3d の CrossSection に NonZero 規則で渡す
       （フォントと同じ塗り規則なので、「O」「回」の穴や字形の重なりが正しく処理される）
    4. 押し出して Manifold（必ず閉じた立体）にし、trimesh に変換する

座標: 単位は mm。完成した立体は X・Y・Z の最小値がすべて 0 になるように置く。
"""

import math
from pathlib import Path

import manifold3d as m3d
import numpy as np
import trimesh
from matplotlib.font_manager import FontProperties
from matplotlib.ft2font import FT2Font
from matplotlib.textpath import TextPath

from core.printer import NOZZLE_DIAMETER

# Windows の既定の日本語フォント
DEFAULT_FONT = "C:/Windows/Fonts/meiryo.ttc"

# 曲線を折れ線にするときの1区間の長さの目安（mm）。
# ノズル径 0.4mm に対して十分細かく、点が増えすぎない値にしている。
CURVE_STEP_MM = 0.2
# 1つの曲線を分割する数の上限・下限
CURVE_MIN_SEGMENTS = 2
CURVE_MAX_SEGMENTS = 32

ALIGNS = ("left", "center", "right")
DIRECTIONS = ("horizontal", "vertical")
STYLES = ("raised", "engraved")  # 浮き出し / 彫り込み
# 彫り込みで、彫った部分の底に残したい厚み（mm）。これより薄いと警告する
MIN_ENGRAVE_FLOOR = 1.0

# 縦書きで 90° 回す文字（長音・ダッシュ・括弧など）
VERTICAL_ROTATE = set("ー－―‐～〜…‥（）()「」『』【】〔〕［］[]｛｝{}〈〉《》<>＜＞-~=＝")
# 縦書きでマスの右上に置く句読点
VERTICAL_PUNCT = set("、。，．,.")
# 縦書きで少し右上に寄せる小さい仮名
VERTICAL_SMALL = set("ぁぃぅぇぉっゃゅょゎゕゖァィゥェォッャュョヮヵヶ")


class TextMeshError(Exception):
    """文字メッシュの作成に失敗したときのエラー。"""


# ---------------------------------------------------------------------------
# 公開する関数
# ---------------------------------------------------------------------------

def make_text_mesh(
    text: str,
    font_path: str | Path | None = None,
    size: float = 10.0,
    height: float = 3.0,
    line_spacing: float = 1.2,
    align: str = "center",
    direction: str = "horizontal",
) -> trimesh.Trimesh:
    """テキストをそのまま押し出した立体文字を作る。

    Args:
        text: 文字列。改行（\\n）で複数行にできる。
        font_path: フォントファイル（.ttf / .otf / .ttc）。None なら DEFAULT_FONT。
        size: 文字の大きさ（mm）。ワープロのフォントサイズと同じ考え方で、
              実際の文字の高さはこれより少し小さくなる（漢字で 8〜9 割ほど）。
        height: 押し出す高さ（Z 方向の厚み, mm）。
        line_spacing: 行の間隔（size の何倍か）。
        align: 複数行のときの揃え方（"left"=行頭 / "center" / "right"=行末）。
        direction: "horizontal"（横書き）または "vertical"（縦書き）。

    Returns:
        watertight な trimesh.Trimesh。
    """
    _check_positive(height, "文字の高さ")
    section = text_to_cross_section(text, font_path, size, line_spacing, align, direction)
    mesh = place_on_origin(manifold_to_trimesh(section.extrude(height)))
    mesh.metadata["kind"] = "text"
    mesh.metadata["thin_ratio"] = thin_ratio(section)
    return mesh


def text_to_cross_section(
    text: str,
    font_path: str | Path | None = None,
    size: float = 10.0,
    line_spacing: float = 1.2,
    align: str = "center",
    direction: str = "horizontal",
) -> m3d.CrossSection:
    """テキストを 2D の断面（CrossSection, mm 単位）に変換する。

    ネームプレートでも使うため、押し出し前の形として公開している。

    direction:
        "horizontal"（横書き）: 左から右へ書き、行は上から下へ進む。
        "vertical"（縦書き）  : 上から下へ書き、行は右から左へ進む。
    align（複数行のときの揃え方）:
        "left" = 行頭揃え（横書きは左、縦書きは上）
        "center" = 中央揃え
        "right" = 行末揃え（横書きは右、縦書きは下）
    """
    if not isinstance(text, str) or not text.strip():
        raise TextMeshError("文字が入力されていません。")
    _check_positive(size, "文字の大きさ")
    _check_positive(line_spacing, "行の間隔")
    if align not in ALIGNS:
        raise TextMeshError(
            f"揃え方の指定が正しくありません（{align}）。left / center / right のいずれかにしてください。"
        )
    if direction not in DIRECTIONS:
        raise TextMeshError(
            f"書き方の指定が正しくありません（{direction}）。horizontal / vertical のいずれかにしてください。"
        )

    font_path = resolve_font(font_path)
    _check_glyphs(text, font_path)
    prop = FontProperties(fname=str(font_path))

    lines = text.splitlines()
    if direction == "horizontal":
        all_rings = _layout_horizontal(lines, prop, size, line_spacing, align)
    else:
        all_rings = _layout_vertical(lines, prop, size, line_spacing, align)

    if not all_rings:
        raise TextMeshError("文字の形を作れませんでした（空白だけの可能性があります）。")

    section = m3d.CrossSection(all_rings, m3d.FillRule.NonZero)
    if section.is_empty() or section.area() <= 0:
        raise TextMeshError("文字の形を作れませんでした（空白だけの可能性があります）。")
    return section


def make_nameplate(
    text: str,
    font_path: str | Path | None = None,
    width: float | None = None,
    depth: float | None = None,
    base_thickness: float = 3.0,
    text_height: float = 2.0,
    margin: float = 5.0,
    size: float | None = None,
    corner_radius: float = 0.0,
    line_spacing: float = 1.2,
    align: str = "center",
    direction: str = "horizontal",
    style: str = "raised",
) -> trimesh.Trimesh:
    """台座（板）の上に立体文字を載せた（または板に文字を彫り込んだ）ネームプレートを作る。

    大きさの決め方:
        - width・depth を指定、size=None
            → 文字を余白の内側にぴったり収まる大きさに自動調整する
        - size を指定、width・depth=None
            → 板の大きさを「文字の大きさ + 余白 x 2」で自動で決める
        - 両方指定 → 指定どおりに作る（文字がはみ出す場合はエラー）
        width と depth は片方だけの指定もできる（もう片方は文字から決める）。

    Args:
        text: 文字列（改行で複数行）。
        font_path: フォントファイル。None なら DEFAULT_FONT。
        width: 板の幅（X 方向, mm）。
        depth: 板の奥行き（Y 方向, mm）。
        base_thickness: 台座の厚み（mm）。
        text_height: 台座の上に出る文字の高さ（mm）。
        margin: 板の端から文字までの余白（mm）。
        size: 文字の大きさ（mm, make_text_mesh と同じ意味）。
        corner_radius: 板の角の丸み（mm）。0 なら角ばった板。
        line_spacing: 行の間隔（size の何倍か）。
        align: 複数行のときの揃え方。
        direction: "horizontal"（横書き）または "vertical"（縦書き）。
        style: "raised"（浮き出し: 板の上に文字が出る）または
               "engraved"（彫り込み: 板に文字を彫る。text_height が彫りの深さになる）。

    Returns:
        watertight な trimesh.Trimesh。
        mesh.metadata["text_size"] に実際に使った文字の大きさ（mm）、
        mesh.metadata["plate_size"] に板の (幅, 奥行き)（mm）が入る。
    """
    _check_positive(base_thickness, "台座の厚み")
    _check_positive(text_height, "文字の高さ")
    if style not in STYLES:
        raise TextMeshError(
            f"文字の形の指定が正しくありません（{style}）。raised / engraved のいずれかにしてください。"
        )
    if style == "engraved" and text_height >= base_thickness:
        raise TextMeshError(
            f"彫りの深さ（{text_height:g} mm）が台座の厚み（{base_thickness:g} mm）以上あり、"
            "板を突き抜けてしまいます。彫りを浅くするか、台座を厚くしてください。"
        )
    _check_non_negative(margin, "余白")
    _check_non_negative(corner_radius, "角の丸み")
    if width is not None:
        _check_positive(width, "板の幅")
    if depth is not None:
        _check_positive(depth, "板の奥行き")
    if size is not None:
        _check_positive(size, "文字の大きさ")

    # --- 文字の大きさを決める ---
    if size is None:
        if width is None and depth is None:
            raise TextMeshError("板の幅・奥行きか、文字の大きさのどちらかを指定してください。")
        size = _fit_text_size(text, font_path, width, depth, margin, line_spacing, align, direction)

    section = text_to_cross_section(text, font_path, size, line_spacing, align, direction)
    tx0, ty0, tx1, ty1 = section.bounds()
    text_w, text_d = tx1 - tx0, ty1 - ty0

    # --- 板の大きさを決める（指定がなければ文字 + 余白） ---
    plate_w = width if width is not None else text_w + 2 * margin
    plate_d = depth if depth is not None else text_d + 2 * margin

    # 文字が板からはみ出さないか（丸めの誤差ぶん少しだけ許容する）
    tol = 1e-6
    if text_w > plate_w - 2 * margin + tol or text_d > plate_d - 2 * margin + tol:
        raise TextMeshError(
            f"文字が板に収まりません。文字の大きさ {text_w:.1f} x {text_d:.1f} mm に対し、"
            f"余白を除いた板の内側は {plate_w - 2 * margin:.1f} x {plate_d - 2 * margin:.1f} mm です。"
            "文字を小さくするか、板を大きく・余白を小さくしてください。"
        )

    if corner_radius > min(plate_w, plate_d) / 2:
        raise TextMeshError(
            f"角の丸み（{corner_radius} mm）が大きすぎます。"
            f"板の短い辺の半分（{min(plate_w, plate_d) / 2:.1f} mm）以下にしてください。"
        )

    # --- 立体を組み立てる ---
    plate = _rounded_rectangle(plate_w, plate_d, corner_radius).extrude(base_thickness)

    # 文字を板の中央へ移動する
    dx = plate_w / 2 - (tx0 + tx1) / 2
    dy = plate_d / 2 - (ty0 + ty1) / 2
    centered = section.translate((dx, dy))

    warnings = []
    if style == "raised":
        # 浮き出し: 台座の上面に文字を載せる
        letters = centered.extrude(text_height).translate((0, 0, base_thickness))
        solid = plate + letters
    else:
        # 彫り込み: 上面から text_height の深さまで文字の形を削る
        # （削る形は上に 1mm はみ出させて、上面にうすい膜が残らないようにする）
        cutter = centered.extrude(text_height + 1.0).translate((0, 0, base_thickness - text_height))
        solid = plate - cutter
        floor = base_thickness - text_height
        if floor < MIN_ENGRAVE_FLOOR:
            warnings.append(
                f"彫った部分の底の厚みが {floor:.1f} mm しかなく、割れたり透けたりしやすくなります。"
                f"{MIN_ENGRAVE_FLOOR:g} mm 以上残すのがおすすめです。"
            )

    mesh = place_on_origin(manifold_to_trimesh(solid))
    mesh.metadata["kind"] = "nameplate"
    mesh.metadata["text_style"] = style
    mesh.metadata["warnings"] = warnings
    mesh.metadata["base_thickness"] = float(base_thickness)  # 穴が文字にかかるかの判定に使う
    mesh.metadata["thin_ratio"] = thin_ratio(section)
    mesh.metadata["text_size"] = float(size)
    mesh.metadata["plate_size"] = (float(plate_w), float(plate_d))
    return mesh


def thin_ratio(section: m3d.CrossSection, min_width: float = NOZZLE_DIAMETER) -> float:
    """文字の形のうち、min_width（mm）より細い部分の面積の割合（0.0〜1.0）を返す。

    形を min_width / 2 だけ内側に縮めてから元に戻すと、
    min_width より細い線は消えて戻らない。その減った面積の割合を使う。
    """
    area = section.area()
    if area <= 0:
        return 0.0
    r = min_width / 2
    opened = (
        section.offset(-r, m3d.JoinType.Round, circular_segments=16)
        .offset(r, m3d.JoinType.Round, circular_segments=16)
    )
    return float(max(0.0, 1.0 - opened.area() / area))


def resolve_font(font_path: str | Path | None) -> Path:
    """フォントのパスを決めて、存在するか確認する。"""
    path = Path(font_path) if font_path else Path(DEFAULT_FONT)
    if not path.is_file():
        raise TextMeshError(f"フォントファイルが見つかりません: {path}")
    return path


def manifold_to_trimesh(manifold: m3d.Manifold) -> trimesh.Trimesh:
    """manifold3d の立体を trimesh のメッシュに変換する。"""
    mesh = manifold.to_mesh()
    vertices = np.asarray(mesh.vert_properties)[:, :3]
    faces = np.asarray(mesh.tri_verts)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def place_on_origin(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """X・Y・Z の最小値が 0 になるように移動する（スライサーで扱いやすくするため）。"""
    mesh.apply_translation(-mesh.bounds[0])
    return mesh


# ---------------------------------------------------------------------------
# 内部の処理
# ---------------------------------------------------------------------------

def _check_positive(value: float, label: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise TextMeshError(f"{label}は 0 より大きい数値にしてください（入力値: {value}）。")


def _check_non_negative(value: float, label: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise TextMeshError(f"{label}は 0 以上の数値にしてください（入力値: {value}）。")


def _fit_text_size(
    text: str,
    font_path: str | Path | None,
    width: float | None,
    depth: float | None,
    margin: float,
    line_spacing: float,
    align: str,
    direction: str,
) -> float:
    """板の内側（余白を除いた部分）にちょうど収まる文字の大きさを求める。

    文字の形は大きさに比例するので、基準の大きさで一度作って倍率を計算する。
    """
    reference = 10.0
    x0, y0, x1, y1 = text_to_cross_section(
        text, font_path, reference, line_spacing, align, direction
    ).bounds()

    scales = []
    for plate_len, text_len, label in ((width, x1 - x0, "幅"), (depth, y1 - y0, "奥行き")):
        if plate_len is None:
            continue
        inner = plate_len - 2 * margin
        if inner <= 0:
            raise TextMeshError(
                f"余白（{margin} mm x 2）が板の{label}（{plate_len} mm）以上あり、文字を置く場所がありません。"
            )
        scales.append(inner / text_len)

    # 曲線の分割の違いでわずかにはみ出さないよう、ほんの少しだけ小さくする
    return reference * min(scales) * 0.999


def _rounded_rectangle(width: float, depth: float, radius: float) -> m3d.CrossSection:
    """左下が原点の長方形（角を丸めることもできる）。"""
    if radius <= 0:
        return m3d.CrossSection.square((width, depth))
    # 小さい長方形を作ってから、丸い角で外側へ広げる
    # 丸みが短い辺のちょうど半分のとき幅が 0 になり形が消えるので、ごく小さい幅を残す
    min_len = 1e-3
    radius = min(radius, (min(width, depth) - min_len) / 2)
    inner = m3d.CrossSection.square((width - 2 * radius, depth - 2 * radius))
    circular_segments = max(16, math.ceil(2 * math.pi * radius / CURVE_STEP_MM))
    return inner.offset(radius, m3d.JoinType.Round, circular_segments=circular_segments).translate(
        (radius, radius)
    )


def _check_glyphs(text: str, font_path: Path) -> None:
    """フォントに含まれていない文字があればエラーにする。"""
    font = FT2Font(str(font_path))
    missing = sorted({
        ch for ch in text
        if not ch.isspace() and font.get_char_index(ord(ch)) == 0
    })
    if missing:
        raise TextMeshError(
            f"このフォントにない文字があります: {' '.join(missing)}"
            f"（フォント: {font_path.name}）。別のフォントを指定するか、文字を変えてください。"
        )


def _align_offset(free: float, align: str) -> float:
    """揃え方に応じて、余っている長さ free のうちどれだけずらすか。"""
    return {"left": 0.0, "center": free / 2, "right": free}[align]


def _layout_horizontal(
    lines: list[str], prop: FontProperties, size: float, line_spacing: float, align: str
) -> list[np.ndarray]:
    """横書き: 1行ずつ輪郭を作り、横方向の位置を揃えてから上から下へ並べる。"""
    line_rings: list[list[np.ndarray]] = []
    widths: list[float] = []
    for line in lines:
        rings = _line_to_rings(line, prop, size)
        line_rings.append(rings)
        if rings:
            xs = np.concatenate([r[:, 0] for r in rings])
            widths.append(float(xs.max()))
        else:
            widths.append(0.0)  # 空行

    max_width = max(widths)
    all_rings: list[np.ndarray] = []
    for i, (rings, width) in enumerate(zip(line_rings, widths)):
        dx = _align_offset(max_width - width, align)
        dy = -i * size * line_spacing
        all_rings.extend(r + (dx, dy) for r in rings)
    return all_rings


def _layout_vertical(
    lines: list[str], prop: FontProperties, size: float, line_spacing: float, align: str
) -> list[np.ndarray]:
    """縦書き: 1文字ずつ上から下へ置き、行は右から左へ並べる。

    1文字は「幅・高さとも size の正方形のマス」に入れる。
    マスの座標は、中心線が x=0、上端が y=0、下端が y=-size。
    """
    columns: list[tuple[list[np.ndarray], float]] = []  # (輪郭, 行の長さ)
    for line in lines:
        rings: list[np.ndarray] = []
        y = 0.0  # 次の文字のマスの上端
        for ch in line:
            if ch == " ":
                y -= size / 2  # 半角スペースは半マス
                continue
            if ch.isspace():
                y -= size  # 全角スペースなどは1マス
                continue
            glyph = _vertical_glyph(ch, prop, size)
            rings.extend(r + (0.0, y) for r in glyph)
            y -= size
        columns.append((rings, -y))

    max_length = max(length for _, length in columns)
    all_rings: list[np.ndarray] = []
    for i, (rings, length) in enumerate(columns):
        dx = -i * size * line_spacing  # 2行目以降は左へ
        dy = -_align_offset(max_length - length, align)
        all_rings.extend(r + (dx, dy) for r in rings)
    return all_rings


def _vertical_glyph(ch: str, prop: FontProperties, size: float) -> list[np.ndarray]:
    """縦書き用に、1文字の輪郭をマスの中に置く（マスの上端が y=0、中心線が x=0）。"""
    rings = _line_to_rings(ch, prop, size)
    if not rings:
        return []
    if ch in VERTICAL_ROTATE:
        # 「ー」や括弧は時計回りに 90° 回す（横棒が縦棒になる）
        rings = [np.column_stack([r[:, 1], -r[:, 0]]) for r in rings]

    # 文字の中心（インクの範囲の中心）を、置きたい位置に合わせる
    pts = np.concatenate(rings)
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2
    if ch in VERTICAL_PUNCT:
        target = np.array([0.25, -0.25]) * size   # 句読点はマスの右上
    elif ch in VERTICAL_SMALL:
        target = np.array([0.08, -0.42]) * size   # 小さい「っ」などは少し右上
    else:
        target = np.array([0.0, -0.5]) * size     # それ以外はマスの中央
    return [r - center + target for r in rings]


def _line_to_rings(line: str, prop: FontProperties, size: float) -> list[np.ndarray]:
    """1行分の文字を、輪郭の点列（閉じた多角形）のリストにする。"""
    if not line.strip():
        return []
    # "$" が2つ以上あると数式として解釈されるので、記号としてそのまま扱わせる
    escaped = line.replace("$", r"\$")
    path = TextPath((0, 0), escaped, size=size, prop=prop)

    rings: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for segment, code in path.iter_bezier():
        if code == TextPath.MOVETO:
            # 新しい輪郭の始まり
            if len(current) >= 3:
                rings.append(np.array(current))
            current = [segment.control_points[0]]
            continue
        n = _segment_count(segment.control_points, segment.degree)
        t = np.linspace(0.0, 1.0, n + 1)[1:]
        current.extend(segment(t))
    if len(current) >= 3:
        rings.append(np.array(current))
    return rings


def _segment_count(control_points: np.ndarray, degree: int) -> int:
    """曲線を何本の線分に分けるかを、曲線のおおよその長さから決める。"""
    if degree <= 1:
        return 1  # 直線はそのまま
    # 制御点を結んだ折れ線の長さ（曲線の長さ以上になる）を目安にする
    length = float(np.linalg.norm(np.diff(control_points, axis=0), axis=1).sum())
    n = math.ceil(length / CURVE_STEP_MM)
    return max(CURVE_MIN_SEGMENTS, min(CURVE_MAX_SEGMENTS, n))
