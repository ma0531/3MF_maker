"""使用する 3D プリンター（Creality Ender-3 V3 KE）に合わせた確認処理。

保存自体は止めずに、印刷で問題になりそうな点を「警告」の文章として返す。
"""

import math
from dataclasses import dataclass

import trimesh

# 造形サイズ（X, Y, Z, mm）。販売店の仕様表の多くが 220 x 220 x 240 mm。
BED_SIZE = (220.0, 220.0, 240.0)
# ノズル径（mm）
NOZZLE_DIAMETER = 0.4

# フィラメント使用量の見積もりに使う値
FILAMENT_DIAMETER = 1.75  # フィラメントの太さ（mm）
PLA_DENSITY = 1.24        # PLA の密度（g/cm³）
# 外側（壁 2 周 = 0.8mm、上下 4 層 x 0.2mm = 0.8mm）の厚み（mm）
SHELL_THICKNESS = 0.8
# ノズル径より細い部分がこの割合を超えたら警告する
THIN_RATIO_WARNING = 0.05

PRINTER_NAME = "Ender-3 V3 KE"


def printer_warnings(mesh: trimesh.Trimesh) -> list[str]:
    """メッシュを印刷するときに注意が必要な点を、日本語の文章のリストで返す。"""
    warnings: list[str] = []
    x, y, z = mesh.extents
    bx, by, bz = BED_SIZE

    fits_flat = x <= bx and y <= by and z <= bz
    # 立てた向き（Y 方向が高さ、厚みが奥行き）。リトファンはこの向きで印刷することが多い
    fits_standing = x <= bx and y <= bz and z <= by
    is_lithophane = mesh.metadata.get("kind") == "lithophane"

    if is_lithophane:
        if not fits_flat and not fits_standing:
            warnings.append(_too_large_message(x, y, z))
        elif not fits_standing:
            warnings.append(
                f"立てて印刷すると高さ {y:.1f} mm になり、造形できる高さ（{bz:.0f} mm）を超えます。"
                "寝かせて印刷してください。"
            )
        elif not fits_flat:
            warnings.append("寝かせると造形サイズを超えるため、立てて印刷してください。")
    elif not fits_flat:
        warnings.append(_too_large_message(x, y, z))

    thin = mesh.metadata.get("thin_ratio", 0.0)
    if thin > THIN_RATIO_WARNING:
        warnings.append(
            f"文字の {thin:.0%} がノズル径 {NOZZLE_DIAMETER} mm より細く、きれいに出ない可能性があります。"
            "文字を大きくするか、太いフォントを使ってください。"
        )

    # 穴あけなど、形を作る途中で出た警告
    warnings.extend(mesh.metadata.get("warnings", []))
    return warnings


@dataclass(frozen=True)
class FilamentEstimate:
    """フィラメント使用量の目安。"""
    length_m: float     # 長さ（m）
    weight_g: float     # 重さ（g）
    volume_cm3: float   # 使う樹脂の体積（cm³）
    infill: float       # 計算に使ったインフィル（0.0〜1.0）


def estimate_filament(
    mesh: trimesh.Trimesh,
    infill: float = 0.15,
    filament_diameter: float = FILAMENT_DIAMETER,
    density: float = PLA_DENSITY,
    shell: float = SHELL_THICKNESS,
) -> FilamentEstimate:
    """フィラメントの使用量を見積もる（スライサーの値の目安）。

    外側（壁・上下の面）は中まで詰まっていて、内側はインフィルの割合だけ樹脂が入る、
    として計算する:
        外側の体積 ≒ 表面積 x 外側の厚み（立体の体積を超えないようにする）
        使用量     = 外側の体積 + (立体の体積 - 外側の体積) x インフィル

    実際の量は、スライサーの設定（壁の数・上下の層数・サポート・ブリムなど）で変わる。
    """
    if not (0.0 <= infill <= 1.0):
        raise ValueError(f"インフィルは 0〜100% の範囲にしてください（入力値: {infill:.0%}）。")
    volume = float(mesh.volume)
    shell_volume = min(volume, float(mesh.area) * shell)
    used = shell_volume + (volume - shell_volume) * infill  # mm³

    cross_section = math.pi * (filament_diameter / 2) ** 2  # フィラメントの断面積（mm²）
    return FilamentEstimate(
        length_m=used / cross_section / 1000.0,
        weight_g=used / 1000.0 * density,
        volume_cm3=used / 1000.0,
        infill=infill,
    )


def _too_large_message(x: float, y: float, z: float) -> str:
    bx, by, bz = BED_SIZE
    return (
        f"{PRINTER_NAME} の造形サイズ（{bx:.0f} x {by:.0f} x {bz:.0f} mm）を超えています"
        f"（{x:.1f} x {y:.1f} x {z:.1f} mm）。小さくしてください。"
    )
