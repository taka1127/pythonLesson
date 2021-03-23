# Pillow(画像操作モジュール)
from PIL import Image
# os=ファイル一覧取得 glob=画像データを扱う
import os, glob
# 画像データをnupmy配列に変換
import numpy as np
# データを分割するモジュール
from sklearn import model_selection

# パラメーターの初期化
# クラス分類
classes = ["car", "motorbike"]
# クラスの数
num_class = len(classes)
# imageサイズ224px
image_size = 224

# 画像の読み込みnumpy配列への変換
X = []  # 画像ファイルリスト
Y = []  # 正解ラベル（車？バイク？0or1の番号）リスト

# 連番を振る
for index, classlabel in enumerate(classes):
  # 画像の入っているディレクトリ
  photos_dir = "./" + classlabel
  # photos_dirに含まれる.jpgファイルだけを検出し代入
  files = glob.glob(photos_dir + "/*.jpg")
  for i, file in enumerate(files):
    # ファイルを開く
    image = Image.open(file)
    # 色の情報RGBの順にそろえる
    image = image.convert("RGB")
    # サイズをそろえる
    image = image.resize((image_size, image_size))
    # numpyの配列としてimageデータを変換 値が大きくなりすぎてスコアにばらつきが出るため255で割って値が0~1の間で推移するようにする
    data = np.asarray(image)
    # 最後尾に追加
    X.append(data)
    Y.append(index)
# numpyのarrayに変換
X = np.array(X)
Y = np.array(Y)

# XとYを分割しトレーニング用（X_trainとy_train）とテスト用（X_testとy_test）に分割する
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
# 一塊のデータとして格納
xy = (X_train, X_test, y_train, y_test)
# npyファイルに出力
np.save("./imagefiles224.npy", xy)