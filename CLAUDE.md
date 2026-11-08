# stl_maker 仕様書

## アプリ概要
テキストや画像から、3Dプリンター用のファイルを自動作成するPython GUIアプリ。

## 作れる形（この4つに限定）
1. 立体文字：テキストをそのまま押し出した文字
2. ネームプレート：台座（板）の上に立体文字を載せたもの
3. レリーフ：画像の明るさを高さに変換（明るい＝高い、反転オプションあり）
4. リトファン：光に透かして見る薄板（暗い＝厚い、明るい＝薄い）

## 出力形式
STL / OBJ / 3MF / AMF
- STL・OBJ・3MF は trimesh で書き出す
- AMF は trimesh が非対応のため、XML形式で自作の書き出し処理を作る

## ルール
- 単位はすべて mm
- 出力するメッシュは必ず watertight（穴のない閉じた形）にする
  - 保存前に mesh.is_watertight をチェックし、False ならエラーを出す
- 処理（core）と画面（gui）を分ける。core は GUI なしで単体で動くこと
- 日本語テキストに対応する（フォントファイルのパスを指定できるようにする）
  - Windows の既定フォント例：C:/Windows/Fonts/meiryo.ttc
- コメント・エラーメッセージは日本語

## フォルダ構成
stl_maker/
├─ CLAUDE.md
├─ main.py              … アプリ起動
├─ core/
│   ├─ text_mesh.py     … 立体文字・ネームプレート
│   ├─ image_mesh.py    … レリーフ・リトファン
│   └─ exporter.py      … 4形式の書き出し
├─ gui/
│   └─ main_window.py   … tkinter の画面
├─ output/              … 生成ファイルの保存先
└─ tests/               … pytest

## 使用ライブラリ
trimesh, numpy, Pillow, opencv-python, shapely, manifold3d, matplotlib, pytest

## パラメータの目安
- ネームプレート：幅・奥行き・台座の厚み・文字の高さ・余白
- レリーフ：幅（mm）・最小高さ・最大高さ・反転の有無
- リトファン：幅（mm）・最小厚み 0.8mm・最大厚み 3.0mm・解像度 0.2mm/画素
  - 画像は解像度に合わせて縮小してから処理する（大きすぎると重くなるため）

## 開発の進め方
1機能ずつ作り、作るたびに pytest と、スライサーでの目視確認を行う。
