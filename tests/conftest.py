"""テスト全体で使う共通の設定。"""

from pathlib import Path

import pytest

from core.text_mesh import DEFAULT_FONT


@pytest.fixture
def jp_font() -> str:
    """日本語フォント（メイリオ）のパス。ない環境ではテストを飛ばす。"""
    if not Path(DEFAULT_FONT).is_file():
        pytest.skip(f"日本語フォントがありません: {DEFAULT_FONT}")
    return DEFAULT_FONT
