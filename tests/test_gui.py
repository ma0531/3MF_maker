"""gui/main_window.py のテスト。

メッシュ作成部分（build_meshes）は画面なしで確認する。
画面そのものは実際に作って、完成図の更新と保存を自動で操作する。
"""

import sys
import time
import tkinter as tk

import numpy as np
import pytest
import trimesh
from PIL import Image

from gui import main_window
from gui.main_window import (
    FIELDS,
    Field,
    InputError,
    MainWindow,
    _parse_number,
    build_meshes,
)


@pytest.fixture
def gradient_png(tmp_path):
    path = tmp_path / "グラデーション.png"
    Image.fromarray(np.tile(np.linspace(0, 255, 200), (100, 1)).astype(np.uint8)).save(path)
    return str(path)


# ---- 入力値の読み取り ----

def test_parse_number():
    f = Field("w", "幅", 1.0)
    assert _parse_number(" 12.5 ", f) == 12.5
    assert _parse_number("１０", f) == 10  # 全角数字
    with pytest.raises(InputError, match="数値を入力"):
        _parse_number("abc", f)
    with pytest.raises(InputError, match="入力してください"):
        _parse_number("", f)
    assert _parse_number("", Field("w", "幅", "", "optional")) is None


# ---- メッシュ作成 ----

def test_build_text(jp_font):
    r = build_meshes("text", {"text": "AB", "font_path": jp_font, "size": 10,
                              "height": 2, "line_spacing": 1.2, "align": "center"})
    assert r.mesh.is_watertight and r.preview_mesh is r.mesh and r.warnings == []


def test_build_nameplate_auto(jp_font):
    r = build_meshes("nameplate", {"text": "AB", "font_path": jp_font, "width": 80, "depth": 30,
                                   "size": None, "base_thickness": 3, "text_height": 2,
                                   "margin": 5, "corner_radius": 3, "line_spacing": 1.2,
                                   "align": "center"})
    assert r.mesh.extents[:2] == pytest.approx([80, 30])


@pytest.mark.parametrize("kind, extra", [
    ("relief", {"min_height": 1, "max_height": 4, "invert": False}),
    ("lithophane", {"min_thickness": 0.8, "max_thickness": 3}),
])
def test_build_image_kinds(gradient_png, kind, extra):
    r = build_meshes(kind, {"image": gradient_png, "width": 100, "resolution": 0.2, **extra})
    assert r.mesh.is_watertight
    # 表示用は粗く、保存用は本番の解像度
    assert len(r.preview_mesh.faces) < len(r.mesh.faces)
    assert r.mesh.metadata["grid_size"][0] == 501


def test_build_image_requires_file():
    with pytest.raises(InputError, match="画像ファイルを選んで"):
        build_meshes("relief", {"image": "", "width": 100, "min_height": 1,
                                "max_height": 4, "invert": False, "resolution": 0.2})


def test_fields_match_core_arguments():
    """入力欄の引数名が core の関数の引数名と一致している（打ち間違い防止）。"""
    import inspect
    from core.image_mesh import make_lithophane, make_relief
    from core.text_mesh import make_nameplate, make_text_mesh
    funcs = {"text": make_text_mesh, "nameplate": make_nameplate,
             "relief": make_relief, "lithophane": make_lithophane}
    for kind, func in funcs.items():
        names = set(inspect.signature(func).parameters)
        assert {f.key for f in FIELDS[kind]} <= names, kind


# ---- 画面 ----

@pytest.fixture(scope="module")
def tk_root():
    """Tk はテストごとに作り直すと不安定になるため、1 つだけ作って使い回す。"""
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"画面を作れない環境です（{e}）")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def window(tk_root):
    top = tk.Toplevel(tk_root)
    top.withdraw()
    win = MainWindow(top)
    yield win
    # 別スレッドの処理が終わってから閉じる
    _wait(win, lambda: win.busy == 0)
    top.destroy()


def _wait(win, condition, timeout=20.0):
    """画面の処理を進めながら、条件を満たすまで待つ。"""
    end = time.time() + timeout
    while time.time() < end:
        win.root.update()
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError("時間内に処理が終わりませんでした")


def _select(win, kind):
    win.notebook.select(win.tab_kinds.index(kind))
    win.root.update()


def test_window_preview_text(window, jp_font):
    _wait(window, lambda: window.result is not None and window.busy == 0)
    assert window.result.kind == "text"
    assert "寸法" in window.info_var.get()


def test_window_preview_all_tabs(window, gradient_png):
    window.image_var.set(gradient_png)
    for kind in ("nameplate", "relief", "lithophane"):
        _select(window, kind)
        window.refresh_preview()
        _wait(window, lambda: window.busy == 0 and window.result is not None
              and window.result.kind == kind)
        assert window.result.mesh.is_watertight
    # リトファンは「光を当てた見え方」に切り替えられる
    window.view_var.set("backlit")
    window._redraw()


def test_window_input_error(window):
    window.inputs["text"]["size"].set("abc")
    window.refresh_preview()
    assert "数値を入力" in window.status_var.get()


def test_window_warning_shown(window):
    window.inputs["text"]["size"].set("300")  # 造形サイズを超える
    window.refresh_preview()
    _wait(window, lambda: window.busy == 0 and window.result is not None
          and window.result.params["size"] == 300)
    assert "造形サイズ" in window.status_var.get()


def test_window_save(window, tmp_path, monkeypatch):
    path = tmp_path / "保存テスト.stl"
    monkeypatch.setattr(main_window.filedialog, "asksaveasfilename", lambda **_: str(path))
    shown = []
    monkeypatch.setattr(main_window.messagebox, "showinfo", lambda *a, **k: shown.append(a))
    monkeypatch.setattr(main_window.messagebox, "showerror",
                        lambda *a, **k: pytest.fail(f"エラー表示: {a}"))

    _wait(window, lambda: window.result is not None and window.busy == 0)
    window.save()
    _wait(window, lambda: bool(shown))
    assert trimesh.load_mesh(path).is_watertight
    assert "保存しました" in window.status_var.get()


# ---- 穴・縦書き・フィラメント ----

def test_build_with_holes(gradient_png):
    from core.holes import Hole
    params = {"image": gradient_png, "width": 100, "resolution": 0.2,
              "min_thickness": 0.8, "max_thickness": 3, "holes": (Hole(50, 45, 5),)}
    r = build_meshes("lithophane", params)
    assert r.mesh.is_watertight and r.preview_mesh.is_watertight
    assert r.mesh.volume < r.base_mesh.volume  # 穴の分だけ減る
    assert "grid_size" in r.base_mesh.metadata  # 光の見え方は穴あけ前のメッシュで描く
    assert r.holes == (Hole(50, 45, 5),)


def test_filament_text(jp_font):
    r = build_meshes("text", {"text": "AB", "font_path": jp_font, "size": 10, "height": 2,
                              "line_spacing": 1.2, "align": "center", "direction": "horizontal"})
    text = main_window.filament_text(r.mesh, 15)
    assert "フィラメント 約" in text and "m" in text and "g" in text


def test_window_holes(window):
    _wait(window, lambda: window.result is not None and window.busy == 0)
    _select(window, "nameplate")
    window.hole_x_var.set("5")
    window.hole_y_var.set("15")
    window.hole_w_var.set("3")
    window.add_hole()
    window.hole_x_var.set("95")
    window.hole_y_var.set("15")
    window.add_hole()
    assert len(window.hole_tree.get_children()) == 2
    _wait(window, lambda: window.busy == 0 and window.result is not None
          and len(window.result.holes) == 2)
    assert "穴 2 個" in window.info_var.get()

    # 穴は種類ごとに別々: 立体文字のタブには穴がない
    _select(window, "text")
    assert len(window.hole_tree.get_children()) == 0
    _select(window, "nameplate")
    assert len(window.hole_tree.get_children()) == 2

    # 選んで削除
    window.hole_tree.selection_set("0")
    window.remove_selected_holes()
    assert [h.x for h in window.holes["nameplate"]] == [95]
    window.clear_holes()
    assert window.holes["nameplate"] == []


def test_window_hole_input_error(window):
    window.hole_x_var.set("abc")
    window.hole_y_var.set("1")
    window.add_hole()
    assert "数値を入力" in window.status_var.get()
    assert window.holes["text"] == []


def test_window_vertical_and_filament(window):
    window.inputs["text"]["direction"].set("縦書き")
    window.refresh_preview()
    _wait(window, lambda: window.busy == 0 and window.result is not None
          and window.result.params["direction"] == "vertical")
    x, y, _ = window.result.mesh.extents
    assert y > x  # 縦長
    assert "フィラメント 約" in window.filament_var.get()

    # インフィルを変えると、作り直さずに表示だけ変わる
    before = window.filament_var.get()
    window.infill_vars["text"].set("100")
    window.root.update()
    assert window.filament_var.get() != before
    window.infill_vars["text"].set("200")
    window.root.update()
    assert "0〜100" in window.filament_var.get()


def test_window_edit_hole(window):
    """一覧で選ぶと値が入力欄に入り、直して「更新」で変更できる。"""
    _wait(window, lambda: window.result is not None and window.busy == 0)
    _select(window, "nameplate")
    window.hole_shape_var.set("星")
    window.hole_x_var.set("6")
    window.hole_y_var.set("15")
    window.hole_w_var.set("5")
    window.add_hole()
    window.hole_shape_var.set("丸")
    window.hole_x_var.set("94")
    window.add_hole()
    holes = window.holes["nameplate"]
    assert [(h.shape, h.x) for h in holes] == [("star", 6), ("circle", 94)]
    assert window.hole_tree.set("0", "shape") == "星"

    # 1 つ目を選ぶ → 入力欄に値が入る
    window.hole_tree.selection_set("0")
    window.root.update()
    assert window.hole_shape_var.get() == "星"
    assert window.hole_x_var.get() == "6"
    assert window.hole_h_var.get() == ""  # 縦は自動

    # 値を直して更新（形をハートに・大きさと角度も変える）
    window.hole_shape_var.set("ハート")
    window.hole_w_var.set("6")
    window.hole_h_var.set("5")
    window.hole_angle_var.set("15")
    window.update_hole()
    h = window.holes["nameplate"][0]
    assert (h.shape, h.width, h.height, h.angle) == ("heart", 6, 5, 15)
    assert len(window.holes["nameplate"]) == 2  # 数は変わらない
    assert window.hole_tree.set("0", "shape") == "ハート"

    _wait(window, lambda: window.busy == 0 and window.result is not None
          and window.result.holes == tuple(window.holes["nameplate"]))
    assert window.result.mesh.is_watertight


def test_window_enter_adds_or_updates(window):
    _select(window, "nameplate")
    window.hole_x_var.set("6")
    window.hole_y_var.set("15")
    window._on_hole_enter()                # 何も選んでいない → 追加
    window._on_hole_enter()                # まだ選んでいない → もう 1 つ追加
    assert len(window.holes["nameplate"]) == 2
    window.hole_tree.selection_set("1")
    window.root.update()
    window.hole_x_var.set("94")
    window._on_hole_enter()                # 選んでいる → 更新
    assert [h.x for h in window.holes["nameplate"]] == [6, 94]


def test_window_update_needs_selection(window):
    window.update_hole()
    assert "選んでください" in window.status_var.get()


# ---- ドラッグ＆ドロップ ----

def test_drop_image_switches_tab(window, gradient_png):
    """文字のタブで画像を落とすと、画像が入り、レリーフのタブに切り替わる。"""
    assert window.current_kind() == "text"
    window.on_files_dropped([gradient_png])
    assert window.image_var.get() == gradient_png
    assert window.current_kind() == "relief"
    assert "画像を読み込みました" in window.status_var.get()


def test_drop_image_keeps_lithophane_tab(window, gradient_png):
    _select(window, "lithophane")
    window.on_files_dropped([gradient_png])
    assert window.current_kind() == "lithophane"


def test_drop_font(window, jp_font):
    _select(window, "relief")
    window.on_files_dropped(["C:/memo.txt", jp_font])  # 使えない物は飛ばして、最初に使える物を使う
    assert window.font_var.get() == jp_font
    assert window.current_kind() == "text"


def test_drop_unsupported(window):
    window.on_files_dropped(["C:/memo.txt", "C:/data.csv"])
    assert "読み込めないファイル" in window.status_var.get()
    assert "memo.txt" in window.status_var.get()


# 本物のアプリと同じく mainloop で動かし、ドロップのメッセージを 3 回送るスクリプト。
# 不具合があると Python ごと強制終了するので、別プロセスで動かして結果を確かめる。
DROP_SCRIPT = r"""
import sys
sys.path[:0] = [{root!r}, {tests!r}]
import tkinter as tk
from test_gui import _send_drop_message
from gui.main_window import MainWindow

root = tk.Tk()
root.withdraw()
win = MainWindow(root)
images = {images!r}
loaded = []
kinds = []

def drop(i=0):
    if i < len(images):
        _send_drop_message(root.winfo_id(), [images[i]])  # PostMessage（本物と同じ経路）
    root.after(400, check, i)

def check(i):
    loaded.append(win.image_var.get())
    kinds.append(win.current_kind())  # 閉じる前に記録する
    if i + 1 < len(images):
        drop(i + 1)
    else:
        root.after(300, root.destroy)

root.after(300, drop)
root.mainloop()
print("RESULT", loaded == images, kinds[-1])
"""


@pytest.mark.skipif(sys.platform != "win32", reason="Windows 専用の機能")
def test_drop_through_windows_message(tmp_path, gradient_png):
    """エクスプローラーと同じ形式・経路（PostMessage）でドロップを送り、
    mainloop 中に受け取っても強制終了せず、画像が読み込まれることを確認する。"""
    import subprocess
    from pathlib import Path

    project = Path(__file__).resolve().parent.parent
    images = []
    for i in range(3):
        path = tmp_path / f"画像{i}.png"
        path.write_bytes(Path(gradient_png).read_bytes())
        images.append(str(path))
    script = DROP_SCRIPT.format(root=str(project), tests=str(project / "tests"), images=images)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=120,
                          encoding="utf-8", errors="replace",
                          env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "RESULT True relief" in proc.stdout, proc.stdout + proc.stderr[-2000:]


def _send_drop_message(hwnd, paths, post=True):
    """WM_DROPFILES を作って送る（エクスプローラーがドロップのときに送るものと同じ形）。

    post=True（既定）: 本物のドロップと同じく PostMessage で送る。メッセージは tkinter の
    待機中（GIL を手放している間）に届くので、本物と同じ経路を確かめられる。
    post=False: SendMessage でその場で直接届ける。
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32")
    user32 = ctypes.WinDLL("user32")
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]

    class DROPFILES(ctypes.Structure):
        _fields_ = [("pFiles", wintypes.DWORD), ("pt", wintypes.POINT),
                    ("fNC", wintypes.BOOL), ("fWide", wintypes.BOOL)]

    # ファイル名を「\0」で区切り、最後に「\0」をもう 1 つ付けた UTF-16 の文字列
    names = ("\0".join(paths) + "\0\0").encode("utf-16-le")
    header = DROPFILES(pFiles=ctypes.sizeof(DROPFILES), fWide=True)
    size = ctypes.sizeof(header) + len(names)
    handle = kernel32.GlobalAlloc(0x0042, size)  # GHND: 動かせる・0 で埋める
    ptr = kernel32.GlobalLock(handle)
    ctypes.memmove(ptr, ctypes.byref(header), ctypes.sizeof(header))
    ctypes.memmove(ptr + ctypes.sizeof(header), names, len(names))
    kernel32.GlobalUnlock(handle)
    # 受け取った側が DragFinish で片付ける（メモリを解放する）
    send = user32.PostMessageW if post else user32.SendMessageW
    assert send(hwnd, 0x0233, handle, 0)
