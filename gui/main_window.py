"""tkinter のメイン画面。

左: 形の種類ごとのタブでパラメータを入力し、穴を追加して、出力形式を選んで保存する
右: 完成図（ドラッグで回転できる）。リトファンは「光を当てたときの見え方」も表示できる

重い処理（メッシュ作成・保存）は別スレッドで行い、画面が固まらないようにする。
メッシュ作成は core の関数を呼ぶだけで、形を作る処理はここには書かない。
"""

import queue
import threading
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import trimesh
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core.exporter import ExportError, export_mesh
from core.holes import SHAPE_LABELS, Hole, HoleError, add_holes
from core.image_mesh import ImageMeshError, make_lithophane, make_relief
from core.printer import (
    BED_SIZE,
    FILAMENT_DIAMETER,
    PRINTER_NAME,
    estimate_filament,
    printer_warnings,
)
from core.text_mesh import DEFAULT_FONT, TextMeshError, make_nameplate, make_text_mesh
from gui.dnd import enable_file_drop
from gui.preview import draw_backlit, draw_mesh, preview_resolution

# 保存先の初期フォルダ
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

KIND_LABELS = {
    "text": "立体文字",
    "nameplate": "ネームプレート",
    "relief": "レリーフ",
    "lithophane": "リトファン",
}
FORMATS = ("STL", "OBJ", "3MF", "AMF")

# 選択肢（画面の表示 → core に渡す値）
ALIGN_CHOICES = {"中央": "center", "行頭（左・上）": "left", "行末（右・下）": "right"}
DIRECTION_CHOICES = {"横書き": "horizontal", "縦書き": "vertical"}
STYLE_CHOICES = {"浮き出し": "raised", "彫り込み": "engraved"}

# フィラメント見積もり用のインフィルの初期値（%）
DEFAULT_INFILL = {"text": 15, "nameplate": 15, "relief": 15, "lithophane": 100}
# 穴の幅の初期値（mm）
DEFAULT_HOLE_WIDTH = 4.0
# 穴の形（画面の表示 → core に渡す値）
SHAPE_CHOICES = {label: key for key, label in SHAPE_LABELS.items()}

# 入力が止まってから完成図を更新するまでの時間（ミリ秒）
PREVIEW_DELAY_MS = 600
# 別スレッドの結果を確認する間隔（ミリ秒）
POLL_MS = 50

# core が出す「入力や形の問題」のエラー（利用者に内容をそのまま見せる）
USER_ERRORS = (TextMeshError, ImageMeshError, ExportError, HoleError)

# ドラッグ＆ドロップで受け付けるファイルの拡張子
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")
FONT_EXTS = (".ttf", ".ttc", ".otf")


# ---------------------------------------------------------------------------
# 入力欄の定義
# ---------------------------------------------------------------------------

@dataclass
class Field:
    """入力欄 1 つ分の定義。"""
    key: str          # core の関数の引数名
    label: str        # 画面に出す名前
    default: object   # 初期値
    kind: str = "float"  # float / optional（空欄=自動）/ text / font / image / choice / bool
    unit: str = "mm"
    choices: dict | None = None  # kind="choice" のときの選択肢（表示 → 値）


FIELDS: dict[str, list[Field]] = {
    "text": [
        Field("text", "文字", "立体文字", "text"),
        Field("font_path", "フォント", DEFAULT_FONT, "font"),
        Field("direction", "書き方", "横書き", "choice", choices=DIRECTION_CHOICES),
        Field("size", "文字の大きさ", 15.0),
        Field("height", "文字の高さ（厚み）", 3.0),
        Field("line_spacing", "行の間隔", 1.2, unit="倍"),
        Field("align", "揃え方", "中央", "choice", choices=ALIGN_CHOICES),
    ],
    "nameplate": [
        Field("text", "文字", "山田 太郎", "text"),
        Field("font_path", "フォント", DEFAULT_FONT, "font"),
        Field("direction", "書き方", "横書き", "choice", choices=DIRECTION_CHOICES),
        Field("style", "文字の形", "浮き出し", "choice", choices=STYLE_CHOICES),
        Field("width", "板の幅", 100.0, "optional"),
        Field("depth", "板の奥行き", 30.0, "optional"),
        Field("size", "文字の大きさ", "", "optional"),
        Field("base_thickness", "台座の厚み", 3.0),
        Field("text_height", "文字の高さ・彫りの深さ", 2.0),
        Field("margin", "余白", 5.0),
        Field("corner_radius", "角の丸み", 0.0),
        Field("line_spacing", "行の間隔", 1.2, unit="倍"),
        Field("align", "揃え方", "中央", "choice", choices=ALIGN_CHOICES),
    ],
    "relief": [
        Field("image", "画像", "", "image"),
        Field("width", "幅", 100.0),
        Field("min_height", "最小高さ", 1.0),
        Field("max_height", "最大高さ", 5.0),
        Field("invert", "反転（暗い所を高く）", False, "bool", unit=""),
        Field("resolution", "解像度", 0.2, unit="mm/画素"),
    ],
    "lithophane": [
        Field("image", "画像", "", "image"),
        Field("width", "幅", 100.0),
        Field("min_thickness", "最小厚み", 0.8),
        Field("max_thickness", "最大厚み", 3.0),
        Field("resolution", "解像度", 0.2, unit="mm/画素"),
    ],
}

HINTS = {
    "text": "フォントはファイルをウィンドウにドラッグ＆ドロップしても変えられます。",
    "nameplate": "板の幅・奥行き・文字の大きさは、空欄にすると自動で決まります。",
    "relief": "明るい所が高くなります。奥行きは画像の縦横比で決まります。\n"
              "画像はウィンドウにドラッグ＆ドロップしても読み込めます。",
    "lithophane": "暗い所が厚く、明るい所が薄くなります。印刷はインフィル 100% で。\n"
                  "画像はウィンドウにドラッグ＆ドロップしても読み込めます。",
}


class InputError(Exception):
    """入力欄の値が正しくないときのエラー。"""


# ---------------------------------------------------------------------------
# メッシュの作成（画面に依存しない部分。テストからも呼べる）
# ---------------------------------------------------------------------------

@dataclass
class BuildResult:
    kind: str
    params: dict                      # 穴（"holes"）も含むパラメータ
    mesh: trimesh.Trimesh             # 保存用（本番の解像度・穴あけ済み）
    preview_mesh: trimesh.Trimesh     # 表示用（画像系は粗い解像度）
    base_mesh: trimesh.Trimesh        # 穴を開ける前（リトファンの光の見え方に使う）
    warnings: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple:
        return params_key(self.kind, self.params)

    @property
    def holes(self) -> tuple[Hole, ...]:
        return tuple(self.params.get("holes", ()))


def params_key(kind: str, params: dict) -> tuple:
    """パラメータが同じかどうかを比べるためのキー。"""
    return (kind, tuple(sorted(params.items())))


def build_meshes(kind: str, params: dict) -> BuildResult:
    """種類とパラメータからメッシュを作る。表示用の軽いメッシュも一緒に作る。

    params["holes"]（Hole の並び）があれば、形を作った後に穴を開ける。
    """
    core_params = {k: v for k, v in params.items() if k != "holes"}
    holes = tuple(params.get("holes", ()))

    if kind == "text":
        base = make_text_mesh(**core_params)
        coarse = base
    elif kind == "nameplate":
        base = make_nameplate(**core_params)
        coarse = base
    elif kind in ("relief", "lithophane"):
        if not core_params.get("image"):
            raise InputError("画像ファイルを選んでください。")
        make = make_relief if kind == "relief" else make_lithophane
        base = make(**core_params)
        res = preview_resolution(core_params["width"], core_params["resolution"])
        coarse = base if res == core_params["resolution"] else make(**{**core_params, "resolution": res})
    else:
        raise ValueError(f"不明な種類です: {kind}")

    mesh = add_holes(base, holes)
    preview = mesh if coarse is base else add_holes(coarse, holes)
    return BuildResult(kind, params, mesh, preview, base, printer_warnings(mesh))


def filament_text(mesh: trimesh.Trimesh, infill_percent: float) -> str:
    """フィラメントの使用量の目安を、画面に出す文章にする。"""
    est = estimate_filament(mesh, infill=infill_percent / 100.0)
    return (f"フィラメント 約 {est.length_m:.2f} m（{est.weight_g:.1f} g・"
            f"インフィル {infill_percent:g}%・{FILAMENT_DIAMETER} mm PLA の目安）")


# ---------------------------------------------------------------------------
# 画面
# ---------------------------------------------------------------------------

class MainWindow:
    def __init__(self, root: tk.Tk | tk.Toplevel):
        self.root = root
        root.title("stl_maker - 3D プリンター用ファイル作成")
        root.geometry("1200x880")
        root.minsize(900, 640)

        # 入力欄の値（種類 → 引数名 → tk 変数 または Text ウィジェット）
        self.inputs: dict[str, dict[str, object]] = {}
        # フォントと画像は、タブをまたいで同じものを使う
        self.font_var = tk.StringVar(value=DEFAULT_FONT)
        self.image_var = tk.StringVar(value="")
        # 穴とインフィルは、形の種類ごとに別々に覚えておく
        self.holes: dict[str, list[Hole]] = {kind: [] for kind in KIND_LABELS}
        self.infill_vars = {kind: tk.StringVar(value=str(v)) for kind, v in DEFAULT_INFILL.items()}
        self.hole_shape_var = tk.StringVar(value=SHAPE_LABELS["circle"])
        self.hole_x_var = tk.StringVar()
        self.hole_y_var = tk.StringVar()
        self.hole_w_var = tk.StringVar(value=str(DEFAULT_HOLE_WIDTH))
        self.hole_h_var = tk.StringVar()  # 空欄 = 形の自然な比率
        self.hole_angle_var = tk.StringVar(value="0")

        self.format_var = tk.StringVar(value="3MF")
        self.view_var = tk.StringVar(value="shape")  # shape / backlit（リトファン用）
        self.status_var = tk.StringVar(value="")
        self.info_var = tk.StringVar(value="")
        self.filament_var = tk.StringVar(value="")

        self.result: BuildResult | None = None  # 最後に作ったメッシュ
        self.results: queue.Queue = queue.Queue()  # 別スレッドからの結果
        self.job_id = 0           # 完成図の更新を数える（古い結果を捨てるため）
        self.pending_after = None  # 自動更新の予約
        self.busy = 0             # 処理中の数

        self._build_layout()
        self.font_var.trace_add("write", lambda *_: self.schedule_preview())
        self.image_var.trace_add("write", lambda *_: self.schedule_preview())
        for var in self.infill_vars.values():
            var.trace_add("write", lambda *_: self._update_filament())
        # エクスプローラーからのドラッグ＆ドロップ（Windows のみ）
        self.drop_enabled = enable_file_drop(self.root, self.on_files_dropped)
        self.root.after(POLL_MS, self._poll_results)
        self.schedule_preview(delay=100)

    # ---- 画面の組み立て ----

    def _build_layout(self) -> None:
        # 左は中身に必要な幅で固定、右（完成図）は残りの幅をすべて使う。
        # （PanedWindow は表示のタイミングで仕切りの位置が変わり、左が狭くなることがあったため使わない）
        left = ttk.Frame(self.root, padding=8)
        right = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        left.pack(side="left", fill="y")
        right.pack(side="left", fill="both", expand=True)

        # 種類ごとのタブ
        self.notebook = ttk.Notebook(left)
        self.notebook.pack(fill="x")
        self.tab_kinds: list[str] = []
        for kind, label in KIND_LABELS.items():
            frame = ttk.Frame(self.notebook, padding=8)
            self.notebook.add(frame, text=label)
            self.tab_kinds.append(kind)
            self._build_tab(frame, kind)
        self.notebook.bind("<<NotebookTabChanged>>", lambda _e: self._on_tab_changed())

        self._build_holes_panel(left)

        # 出力形式と保存
        out = ttk.LabelFrame(left, text="書き出し", padding=8)
        out.pack(fill="x", pady=(8, 0))
        ttk.Label(out, text="形式").grid(row=0, column=0, sticky="w")
        ttk.Combobox(out, textvariable=self.format_var, values=FORMATS,
                     state="readonly", width=8).grid(row=0, column=1, sticky="w", padx=6)
        self.save_button = ttk.Button(out, text="保存…", command=self.save)
        self.save_button.grid(row=0, column=2, sticky="e", padx=(12, 0))
        ttk.Button(out, text="完成図を更新", command=lambda: self.schedule_preview(delay=0)).grid(
            row=0, column=3, sticky="e", padx=(6, 0))
        ttk.Label(out, text="インフィル").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.infill_entry = ttk.Entry(out, width=6)
        self.infill_entry.grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(out, text="%（フィラメント量の目安に使用）", foreground="#555").grid(
            row=1, column=2, columnspan=2, sticky="w", pady=(6, 0))
        out.columnconfigure(1, weight=1)

        # 状態の表示
        self.progress = ttk.Progressbar(left, mode="determinate")
        self.progress.pack(fill="x", pady=(8, 0))
        self.status_label = tk.Label(left, textvariable=self.status_var, anchor="w",
                                     justify="left", wraplength=380, fg="#b00020")
        self.status_label.pack(fill="x", pady=(4, 0))

        # 右側: 完成図
        header = ttk.Frame(right)
        header.pack(fill="x")
        info = ttk.Frame(header)
        info.pack(side="left", fill="x")
        ttk.Label(info, textvariable=self.info_var).pack(anchor="w")
        ttk.Label(info, textvariable=self.filament_var, foreground="#1a4f8b").pack(anchor="w")
        self.view_frame = ttk.Frame(header)
        ttk.Radiobutton(self.view_frame, text="形", value="shape", variable=self.view_var,
                        command=self._redraw).pack(side="left")
        ttk.Radiobutton(self.view_frame, text="光を当てた見え方", value="backlit",
                        variable=self.view_var, command=self._redraw).pack(side="left")

        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        ttk.Label(right, text=f"ドラッグで回転できます。{PRINTER_NAME}（造形サイズ "
                              f"{BED_SIZE[0]:.0f} x {BED_SIZE[1]:.0f} x {BED_SIZE[2]:.0f} mm）"
                              "を基準に確認しています。",
                  foreground="#555").pack(fill="x")

        self._sync_tab_widgets()

    def _build_tab(self, frame: ttk.Frame, kind: str) -> None:
        values: dict[str, object] = {}
        row = 0
        for f in FIELDS[kind]:
            ttk.Label(frame, text=f.label).grid(row=row, column=0, sticky="nw", pady=2)

            if f.kind == "text":
                widget = tk.Text(frame, width=26, height=3, font=("Meiryo", 11), undo=True)
                widget.insert("1.0", f.default)
                widget.bind("<KeyRelease>", lambda _e: self.schedule_preview())
                widget.grid(row=row, column=1, columnspan=2, sticky="we", pady=2)
                values[f.key] = widget
            elif f.kind in ("font", "image"):
                var = self.font_var if f.kind == "font" else self.image_var
                entry = ttk.Entry(frame, textvariable=var, width=24)
                entry.grid(row=row, column=1, sticky="we", pady=2)
                # パスが長いときにファイル名（末尾）が見えるようにする
                # （タブが表示される前は効かないので、表示・大きさ変更のときにも合わせる）
                var.trace_add("write", lambda *_, e=entry: e.after_idle(e.xview_moveto, 1.0))
                entry.bind("<Configure>", lambda ev: ev.widget.xview_moveto(1.0))
                ttk.Button(frame, text="参照…", width=6,
                           command=(self._choose_font if f.kind == "font" else self._choose_image)
                           ).grid(row=row, column=2, padx=(4, 0))
                values[f.key] = var
            elif f.kind == "choice":
                var = tk.StringVar(value=f.default)
                ttk.Combobox(frame, textvariable=var, values=list(f.choices), state="readonly",
                             width=14).grid(row=row, column=1, sticky="w", pady=2)
                var.trace_add("write", lambda *_: self.schedule_preview())
                values[f.key] = var
            elif f.kind == "bool":
                var = tk.BooleanVar(value=f.default)
                ttk.Checkbutton(frame, variable=var, command=self.schedule_preview).grid(
                    row=row, column=1, sticky="w", pady=2)
                values[f.key] = var
            else:  # float / optional
                var = tk.StringVar(value=str(f.default))
                ttk.Entry(frame, textvariable=var, width=10).grid(row=row, column=1, sticky="w", pady=2)
                unit = f.unit + ("（空欄=自動）" if f.kind == "optional" else "")
                ttk.Label(frame, text=unit, foreground="#555").grid(row=row, column=2, sticky="w")
                var.trace_add("write", lambda *_: self.schedule_preview())
                values[f.key] = var
            row += 1

        if kind in HINTS:
            ttk.Label(frame, text=HINTS[kind], foreground="#555", wraplength=330,
                      justify="left").grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))
        frame.columnconfigure(1, weight=1)
        self.inputs[kind] = values

    def _build_holes_panel(self, parent: ttk.Frame) -> None:
        """穴の一覧と、追加・削除の欄。"""
        box = ttk.LabelFrame(parent, text="穴（上から下まで貫通）", padding=8)
        box.pack(fill="x", pady=(8, 0))

        # 入力欄（2 行）: 形・X・Y ／ 幅・縦・角度
        grid = ttk.Frame(box)
        grid.pack(fill="x")
        ttk.Label(grid, text="形").grid(row=0, column=0, sticky="w")
        ttk.Combobox(grid, textvariable=self.hole_shape_var, values=list(SHAPE_CHOICES),
                     state="readonly", width=7).grid(row=0, column=1, sticky="w", padx=(2, 8))
        entries = [
            (0, 2, "X", self.hole_x_var), (0, 4, "Y", self.hole_y_var),
            (1, 0, "幅", self.hole_w_var), (1, 2, "縦", self.hole_h_var),
            (1, 4, "角度", self.hole_angle_var),
        ]
        for row, col, label, var in entries:
            ttk.Label(grid, text=label).grid(row=row, column=col, sticky="w", pady=2)
            entry = ttk.Entry(grid, textvariable=var, width=7)
            entry.grid(row=row, column=col + 1, sticky="w", padx=(2, 8), pady=2)
            entry.bind("<Return>", lambda _e: self._on_hole_enter())

        buttons = ttk.Frame(box)
        buttons.pack(fill="x", pady=(4, 0))
        ttk.Button(buttons, text="追加", width=6, command=self.add_hole).pack(side="left")
        self.update_hole_button = ttk.Button(buttons, text="更新", width=6, command=self.update_hole)
        self.update_hole_button.pack(side="left", padx=(4, 0))
        ttk.Button(buttons, text="削除", width=6, command=self.remove_selected_holes).pack(
            side="left", padx=(4, 0))
        ttk.Button(buttons, text="すべて削除", command=self.clear_holes).pack(side="left", padx=(4, 0))

        columns = (("no", "No", 32), ("shape", "形", 56), ("x", "X", 52), ("y", "Y", 52),
                   ("w", "幅", 48), ("h", "縦", 48), ("angle", "角度", 44))
        # 一覧（穴が多いときはスクロールバーで見る）
        tree_frame = ttk.Frame(box)
        tree_frame.pack(fill="x", pady=(6, 4))
        self.hole_tree = ttk.Treeview(tree_frame, columns=[c[0] for c in columns], show="headings",
                                      height=3, selectmode="extended")
        for col, text, width in columns:
            self.hole_tree.heading(col, text=text)
            self.hole_tree.column(col, width=width, anchor="center")
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.hole_tree.yview)
        self.hole_tree.configure(yscrollcommand=scrollbar.set)
        self.hole_tree.pack(side="left", fill="x", expand=True)
        scrollbar.pack(side="left", fill="y")
        self.hole_tree.bind("<<TreeviewSelect>>", lambda _e: self._on_hole_selected())

        ttk.Label(box, text="位置は穴の中心（完成図の左下が 0, 0）。単位は mm・度。縦は空欄で自動。\n"
                            "一覧で選ぶと値が入り、直して「更新」（Enter）で変更できます。\n"
                            "六角形の幅は対辺の長さ（M3 ナットなら 5.5 + すき間 0.3 mm ほど）。",
                  foreground="#555", justify="left").pack(anchor="w")
        self.update_hole_button.state(["disabled"])

    # ---- 入力の読み取り ----

    def current_kind(self) -> str:
        return self.tab_kinds[self.notebook.index(self.notebook.select())]

    def collect_params(self, kind: str) -> dict:
        """入力欄の値を core の関数に渡す形（引数名 → 値）にする。穴は "holes" に入れる。"""
        params = {}
        for f in FIELDS[kind]:
            source = self.inputs[kind][f.key]
            if f.kind == "text":
                params[f.key] = source.get("1.0", "end-1c")
            elif f.kind == "choice":
                params[f.key] = f.choices[source.get()]
            elif f.kind == "bool":
                params[f.key] = bool(source.get())
            elif f.kind in ("font", "image"):
                params[f.key] = source.get().strip()
            else:
                params[f.key] = _parse_number(source.get(), f)
        params["holes"] = tuple(self.holes[kind])
        return params

    def collect_infill(self, kind: str) -> float:
        """インフィル（%）を読み取る。"""
        text = self.infill_vars[kind].get().strip()
        try:
            value = float(text)
        except ValueError:
            raise InputError(f"「インフィル」には 0〜100 の数値を入力してください（入力値: {text}）。") from None
        if not 0 <= value <= 100:
            raise InputError(f"「インフィル」は 0〜100 の範囲にしてください（入力値: {value:g}）。")
        return value

    # ---- 穴の追加・削除 ----

    def read_hole_inputs(self) -> Hole:
        """穴の入力欄の値から Hole を作る。おかしな値なら InputError。"""
        hole = Hole(
            x=_parse_number(self.hole_x_var.get(), Field("x", "穴の X", 0)),
            y=_parse_number(self.hole_y_var.get(), Field("y", "穴の Y", 0)),
            width=_parse_number(self.hole_w_var.get(), Field("w", "穴の幅", 0)),
            height=_parse_number(self.hole_h_var.get(), Field("h", "穴の縦", "", "optional")),
            shape=SHAPE_CHOICES[self.hole_shape_var.get()],
            angle=_parse_number(self.hole_angle_var.get() or "0", Field("a", "穴の角度", 0)),
        )
        if hole.width <= 0 or (hole.height is not None and hole.height <= 0):
            raise InputError("穴の幅・縦は 0 より大きくしてください。")
        return hole

    def add_hole(self) -> None:
        try:
            hole = self.read_hole_inputs()
        except InputError as e:
            self._show_status(str(e), error=True)
            return
        # 入力欄の値は残す（少し変えて続けて追加しやすいように）。
        # 追加後は何も選んでいない状態なので、Enter を押すと続けて「追加」になる
        self.holes[self.current_kind()].append(hole)
        self._refresh_hole_list()
        self.schedule_preview(delay=0)

    def update_hole(self) -> None:
        """一覧で選んでいる穴を、入力欄の値で置き換える。"""
        selected = self.hole_tree.selection()
        if len(selected) != 1:
            self._show_status("更新する穴を一覧から 1 つ選んでください。", error=True)
            return
        try:
            hole = self.read_hole_inputs()
        except InputError as e:
            self._show_status(str(e), error=True)
            return
        index = int(selected[0])
        self.holes[self.current_kind()][index] = hole
        self._refresh_hole_list()
        self._select_hole(index)
        self.schedule_preview(delay=0)

    def _on_hole_enter(self) -> None:
        """入力欄で Enter: 穴を 1 つ選んでいれば更新、そうでなければ追加。"""
        if len(self.hole_tree.selection()) == 1:
            self.update_hole()
        else:
            self.add_hole()

    def _on_hole_selected(self) -> None:
        """一覧で穴を 1 つ選んだら、その値を入力欄に入れる。"""
        selected = self.hole_tree.selection()
        if len(selected) != 1:
            self.update_hole_button.state(["disabled"])
            return
        self.update_hole_button.state(["!disabled"])
        h = self.holes[self.current_kind()][int(selected[0])]
        self.hole_shape_var.set(SHAPE_LABELS[h.shape])
        self.hole_x_var.set(f"{h.x:g}")
        self.hole_y_var.set(f"{h.y:g}")
        self.hole_w_var.set(f"{h.width:g}")
        self.hole_h_var.set("" if h.height is None else f"{h.height:g}")
        self.hole_angle_var.set(f"{h.angle:g}")

    def _select_hole(self, index: int) -> None:
        iid = str(index)
        if self.hole_tree.exists(iid):
            self.hole_tree.selection_set(iid)
            self.hole_tree.see(iid)

    def remove_selected_holes(self) -> None:
        selected = {int(iid) for iid in self.hole_tree.selection()}
        if not selected:
            self._show_status("削除する穴を一覧から選んでください。", error=True)
            return
        kind = self.current_kind()
        self.holes[kind] = [h for i, h in enumerate(self.holes[kind]) if i not in selected]
        self._refresh_hole_list()
        self.schedule_preview(delay=0)

    def clear_holes(self) -> None:
        self.holes[self.current_kind()].clear()
        self._refresh_hole_list()
        self.schedule_preview(delay=0)

    def _refresh_hole_list(self) -> None:
        self.hole_tree.delete(*self.hole_tree.get_children())
        for i, h in enumerate(self.holes[self.current_kind()]):
            height = "自動" if h.height is None else f"{h.height:g}"
            self.hole_tree.insert("", "end", iid=str(i), values=(
                i + 1, h.label, f"{h.x:g}", f"{h.y:g}", f"{h.width:g}", height, f"{h.angle:g}"))
        self.update_hole_button.state(["disabled"])

    # ---- 完成図の更新 ----

    def schedule_preview(self, delay: int = PREVIEW_DELAY_MS) -> None:
        """入力が止まってから少し待って完成図を更新する（打つたびに作り直さないため）。"""
        if self.pending_after is not None:
            self.root.after_cancel(self.pending_after)
        self.pending_after = self.root.after(delay, self.refresh_preview)

    def refresh_preview(self) -> None:
        self.pending_after = None
        kind = self.current_kind()
        try:
            params = self.collect_params(kind)
        except InputError as e:
            self._show_status(str(e), error=True)
            return
        if self.result is not None and self.result.key == params_key(kind, params):
            self._redraw()
            return
        self.job_id += 1
        self._run_in_thread(("preview", self.job_id), lambda: build_meshes(kind, params))

    def _redraw(self) -> None:
        """最後に作ったメッシュを描き直す。"""
        r = self.result
        if r is None or r.kind != self.current_kind():
            self.figure.clear()
            self.canvas.draw_idle()
            self.info_var.set("")
            self.filament_var.set("")
            return
        title = KIND_LABELS[r.kind]
        if r.kind == "lithophane" and self.view_var.get() == "backlit":
            draw_backlit(self.figure, r.base_mesh, title, holes=r.holes)
        else:
            # 寸法は表示用（粗い）メッシュではなく、保存する本番のメッシュの値を出す
            draw_mesh(self.figure, r.preview_mesh, title, size=r.mesh.extents)
        self.canvas.draw_idle()

        x, y, z = r.mesh.extents
        self.info_var.set(
            f"寸法 {x:.1f} x {y:.1f} x {z:.1f} mm ／ 三角形 {len(r.mesh.faces):,} 個"
            + _extra_info(r)
        )
        self._update_filament()
        if r.warnings:
            self._show_status("⚠ " + "\n⚠ ".join(r.warnings), error=True)
        else:
            self._show_status("問題は見つかりませんでした。", error=False)

    def _update_filament(self) -> None:
        """フィラメントの使用量の目安を表示し直す（インフィルを変えたときも呼ぶ）。"""
        r = self.result
        if r is None or r.kind != self.current_kind():
            self.filament_var.set("")
            return
        try:
            self.filament_var.set(filament_text(r.mesh, self.collect_infill(r.kind)))
        except InputError as e:
            self.filament_var.set(str(e))

    # ---- 保存 ----

    def save(self) -> None:
        kind = self.current_kind()
        try:
            params = self.collect_params(kind)
        except InputError as e:
            messagebox.showerror("入力エラー", str(e), parent=self.root)
            return

        cached = self.result if (self.result and self.result.key == params_key(kind, params)) else None
        if cached and cached.warnings:
            message = "\n\n".join(cached.warnings) + "\n\nこのまま保存しますか？"
            if not messagebox.askyesno("警告", message, icon="warning", parent=self.root):
                return

        ext = self.format_var.get().lower()
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="保存先を選ぶ",
            initialdir=OUTPUT_DIR,
            initialfile=f"{KIND_LABELS[kind]}.{ext}",
            defaultextension=f".{ext}",
            filetypes=[(f"{ext.upper()} ファイル", f"*.{ext}")]
            + [(f"{e} ファイル", f"*.{e.lower()}") for e in FORMATS if e.lower() != ext],
        )
        if not path:
            return

        def task():
            result = cached or build_meshes(kind, params)
            saved = export_mesh(result.mesh, path)
            return result, saved

        self._run_in_thread(("save", None), task)

    # ---- 別スレッドでの処理 ----

    def _run_in_thread(self, tag: tuple, func) -> None:
        self.busy += 1
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self.save_button.state(["disabled"])
        if tag[0] == "save":
            self._show_status("保存しています…", error=False)

        def worker():
            try:
                self.results.put((tag, func(), None))
            except Exception as e:  # 結果は画面側でまとめて表示する
                self.results.put((tag, None, e))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_results(self) -> None:
        try:
            while True:
                tag, value, error = self.results.get_nowait()
                self._handle_result(tag, value, error)
        except queue.Empty:
            pass
        self.root.after(POLL_MS, self._poll_results)

    def _handle_result(self, tag: tuple, value, error: Exception | None) -> None:
        self.busy -= 1
        if self.busy == 0:
            # 止めただけだとバーに四角が残るので、空の状態に戻す
            self.progress.stop()
            self.progress.configure(mode="determinate", value=0)
            self.save_button.state(["!disabled"])

        what, job = tag
        if what == "preview":
            if job != self.job_id:
                return  # もっと新しい更新が始まっているので、この結果は使わない
            if error is not None:
                self.result = None
                self._redraw()
                self._show_status(_error_text(error), error=True)
                return
            self.result = value
            self._redraw()
        else:  # save
            if error is not None:
                self._show_status(_error_text(error), error=True)
                messagebox.showerror("保存できませんでした", _error_text(error), parent=self.root)
                return
            result, saved = value
            self._show_status(f"保存しました: {saved}", error=False)
            lines = [str(saved), "",
                     "寸法 " + " x ".join(f"{v:.1f}" for v in result.mesh.extents) + " mm"]
            try:
                lines.append(filament_text(result.mesh, self.collect_infill(result.kind)))
            except InputError:
                pass
            messagebox.showinfo("保存しました", "\n".join(lines), parent=self.root)

    # ---- その他 ----

    def _on_tab_changed(self) -> None:
        self._sync_tab_widgets()
        self._redraw()
        self.schedule_preview(delay=0)

    def _sync_tab_widgets(self) -> None:
        """タブに合わせて、種類ごとに覚えている値（穴・インフィル）の表示を切り替える。"""
        kind = self.current_kind()
        # リトファンのときだけ「光を当てた見え方」を選べるようにする
        if kind == "lithophane":
            self.view_frame.pack(side="right")
        else:
            self.view_frame.pack_forget()
        self.infill_entry.configure(textvariable=self.infill_vars[kind])
        self._refresh_hole_list()

    def on_files_dropped(self, paths: list[str]) -> None:
        """ドロップされたファイルを読み込む。画像なら画像、フォントならフォントとして使う。

        複数のファイルが落とされたときは、使える最初の 1 つだけを使う。
        """
        for path in paths:
            ext = Path(path).suffix.lower()
            if ext in IMAGE_EXTS:
                self.image_var.set(path)
                # 文字のタブを開いていたら、画像を使うタブ（レリーフ）に切り替える
                if self.current_kind() not in ("relief", "lithophane"):
                    self.notebook.select(self.tab_kinds.index("relief"))
                self._show_status(f"画像を読み込みました: {Path(path).name}", error=False)
                return
            if ext in FONT_EXTS:
                self.font_var.set(path)
                # 画像のタブを開いていたら、文字のタブ（立体文字）に切り替える
                if self.current_kind() not in ("text", "nameplate"):
                    self.notebook.select(self.tab_kinds.index("text"))
                self._show_status(f"フォントを読み込みました: {Path(path).name}", error=False)
                return
        names = "、".join(Path(p).name for p in paths) or "（なし）"
        self._show_status(
            f"読み込めないファイルです: {names}\n"
            "画像（png・jpg・bmp など）かフォント（ttf・ttc・otf）をドロップしてください。",
            error=True,
        )

    def _choose_font(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root, title="フォントを選ぶ", initialdir="C:/Windows/Fonts",
            filetypes=[("フォント", "*.ttf *.ttc *.otf"), ("すべてのファイル", "*.*")],
        )
        if path:
            self.font_var.set(path)

    def _choose_image(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root, title="画像を選ぶ",
            filetypes=[("画像", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp"),
                       ("すべてのファイル", "*.*")],
        )
        if path:
            self.image_var.set(path)

    def _show_status(self, text: str, error: bool) -> None:
        self.status_var.set(text)
        self.status_label.configure(fg="#b00020" if error else "#1b5e20")


# ---------------------------------------------------------------------------
# 補助
# ---------------------------------------------------------------------------

def _parse_number(text: str, f: Field) -> float | None:
    text = text.strip()
    if text == "":
        if f.kind == "optional":
            return None  # 空欄 = 自動
        raise InputError(f"「{f.label}」を入力してください。")
    try:
        return float(text)  # 全角数字（１０など）もそのまま読める
    except ValueError:
        raise InputError(f"「{f.label}」には数値を入力してください（入力値: {text}）。") from None


def _extra_info(r: BuildResult) -> str:
    text = ""
    if r.kind == "nameplate":
        text += f" ／ 文字の大きさ {r.mesh.metadata['text_size']:.1f} mm"
    if r.holes:
        text += f" ／ 穴 {len(r.holes)} 個"
    return text


def _error_text(error: Exception) -> str:
    if isinstance(error, USER_ERRORS + (InputError,)):
        return str(error)
    return f"予期しないエラーが起きました（{type(error).__name__}: {error}）"
