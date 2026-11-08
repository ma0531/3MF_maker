"""stl_maker を起動する。

使い方: .venv\\Scripts\\python main.py
"""

import sys
import tkinter as tk

from gui.main_window import MainWindow


def _enable_high_dpi() -> None:
    """Windows の高解像度画面で文字がぼやけないようにする。"""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass


def main() -> None:
    _enable_high_dpi()
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
