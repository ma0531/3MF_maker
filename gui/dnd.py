"""エクスプローラーからのファイルのドラッグ＆ドロップを受け取る（Windows 専用）。

tkinter には標準でドラッグ＆ドロップの機能がないため、追加のライブラリを使わずに
Windows の仕組み（WM_DROPFILES メッセージ）を ctypes で直接受け取る。

しくみ:
    1. DragAcceptFiles でウィンドウを「ファイルを受け付ける」状態にする
    2. ウィンドウのメッセージ処理（ウィンドウプロシージャ）を差し替え、
       WM_DROPFILES が来たらファイルのパスを取り出してキューに入れる
    3. それ以外のメッセージは、元のメッセージ処理（tkinter）にそのまま渡す
    4. tkinter 側で定期的にキューを確認し、入っていれば callback を呼ぶ

重要: メッセージ処理（wndproc）の中では tkinter を絶対に呼ばない。
      ドロップのメッセージは tkinter の待機中（Python の GIL を手放している間）に届く。
      そこで tkinter を呼ぶと tkinter 内部の状態が壊れ、
      「Fatal Python error: PyEval_RestoreThread ... GIL is released」で強制終了する。
      そのため、パスをキューに入れるだけにして、tkinter の処理は after の定期確認で行う。

注意: 管理者として実行していると、Windows の制限で通常のエクスプローラーからはドロップできない。
"""

import queue
import sys
from collections.abc import Callable

WM_DROPFILES = 0x0233
# キューを確認する間隔（ミリ秒）
POLL_MS = 100


def enable_file_drop(widget, callback: Callable[[list[str]], None]) -> bool:
    """widget（とその中の部品）へのファイルのドロップを受け付ける。

    ドロップされると callback(パスのリスト) を呼ぶ。
    受け付けられるようになれば True、Windows 以外などでできなければ False を返す。
    """
    if sys.platform != "win32":
        return False
    try:
        return _enable_windows(widget, callback)
    except (OSError, AttributeError):
        return False


def _enable_windows(widget, callback) -> bool:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    shell32 = ctypes.WinDLL("shell32")

    lresult = ctypes.c_ssize_t
    wndproc_type = ctypes.WINFUNCTYPE(
        lresult, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
    )
    # 64 ビット版でポインタが切り詰められないよう、引数と戻り値の型をはっきり決める
    user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongPtrW.restype = ctypes.c_void_p
    user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
    user32.SetWindowLongPtrW.restype = ctypes.c_void_p
    user32.CallWindowProcW.argtypes = [
        ctypes.c_void_p, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
    ]
    user32.CallWindowProcW.restype = lresult
    shell32.DragAcceptFiles.argtypes = [wintypes.HWND, wintypes.BOOL]
    shell32.DragQueryFileW.argtypes = [ctypes.c_void_p, wintypes.UINT, wintypes.LPWSTR, wintypes.UINT]
    shell32.DragQueryFileW.restype = wintypes.UINT
    shell32.DragFinish.argtypes = [ctypes.c_void_p]

    widget.update_idletasks()  # ウィンドウを実際に作らせてから、その番号（HWND）を使う
    hwnd = widget.winfo_id()
    gwlp_wndproc = -4
    old_proc = user32.GetWindowLongPtrW(hwnd, gwlp_wndproc)
    if not old_proc:
        return False

    def read_paths(hdrop) -> list[str]:
        count = shell32.DragQueryFileW(hdrop, 0xFFFFFFFF, None, 0)
        paths = []
        for i in range(count):
            length = shell32.DragQueryFileW(hdrop, i, None, 0)
            buffer = ctypes.create_unicode_buffer(length + 1)
            shell32.DragQueryFileW(hdrop, i, buffer, length + 1)
            paths.append(buffer.value)
        return paths

    dropped: queue.SimpleQueue[list[str]] = queue.SimpleQueue()

    def wndproc(h, msg, wparam, lparam):
        if msg == WM_DROPFILES:
            # ここでは tkinter を呼ばない（モジュールの説明を参照）。キューに入れるだけ
            try:
                dropped.put(read_paths(wparam))
            except Exception:  # Windows からの呼び出しの中なので、例外は外に出さない
                pass
            finally:
                shell32.DragFinish(wparam)  # ドロップの情報を片付ける（必ず呼ぶ）
            return 0
        return user32.CallWindowProcW(old_proc, h, msg, wparam, lparam)

    def poll():
        """tkinter 側で定期的にキューを確認し、ドロップされたファイルを callback に渡す。"""
        try:
            if not widget.winfo_exists():
                return  # ウィンドウが閉じられたら確認をやめる
        except Exception:
            return
        while True:
            try:
                paths = dropped.get_nowait()
            except queue.Empty:
                break
            callback(paths)
        widget.after(POLL_MS, poll)

    proc = wndproc_type(wndproc)
    user32.SetWindowLongPtrW(hwnd, gwlp_wndproc, ctypes.cast(proc, ctypes.c_void_p))
    shell32.DragAcceptFiles(hwnd, True)
    # Python 側の関数が消されると Windows から呼べなくなり落ちるので、ウィジェットに持たせておく
    widget._file_drop_proc = proc
    widget.after(POLL_MS, poll)
    return True
