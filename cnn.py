# generate_data.pyで作ったnumpyデータを読み書き
import numpy as np
# tensorflowに含まれるkerasを使う
from tensorflow import keras
# シーケンシャルモデルを使用https://keras.io/guides/sequential_model/
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.utils import np_utils

# パラメーターの初期化
# クラス分類
classes = ["car", "motorbike"]
# クラスの数
num_class = len(classes)
# imageサイズ150px
image_size = 150

# X_train, X_test, y_train, y_testの順にデータをロード
X_train, X_test, y_train, y_test = np.load(
    "./imagefiles.npy", allow_pickle=True)
# 正解ラベルの位置だけ１がたつよう整数値のベクトルをone hot表現に変換https://qiita.com/JeJeNeNo/items/8a7c1781f6a6ad522adf
# to_categorical(データ, データの最大値（整数）)→例[0,1][1,0]...
y_train = np_utils.to_categorical(y_train, num_class)
y_test = np_utils.to_categorical(y_test, num_class)

# モデルの定義
# シーケンシャルモデルを宣言
model = Sequential()
# レイヤーの追加
""" 
・空間フィルタ – 畳み込み演算層
input_shape=縦・横ピクセルの3色を入力。 
"""
model.add(Conv2D(32, (3, 3), activation='relu',
          input_shape=(image_size, image_size, 3)))
""" 
・空間フィルタ – 畳み込み演算層。
「3×3」の大きさのフィルタを32枚使う（32種類の「3×3」のフィルタ）。
活性化関数「ReLU（Rectified Linear Unit）- ランプ関数」。フィルタ後の画像に実施。入力が0以下の時は出力0。入力が0より大きい場合はそのまま出力する。
"""
model.add(Conv2D(32, (3, 3), activation='relu'))
""" 
・「2×2」の大きさの最大プーリング層
MaxPooling2D＝データを圧縮しサイズを小さくし計算しやすくする
入力画像内の「2×2」の領域で最大の数値を出力する
"""
model.add(MaxPooling2D(pool_size=(2, 2)))
""" 
Dropout：過学習予防。今回は、全結合の層とのつながりを「25%」無効化しています。
"""
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu',))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

""" 
直列に並べる
1次元ベクトルに変換
"""
model.add(Flatten())
""" 
全結合層。出力256
"""
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
""" 
・全結合層
・Denseのところで、畳み込みニューラルネットワーク（CNN）の最終的な全結合層の出力次元数の決め方は、
判定するクラス数を指定（可変を考えnum_class）
・ノード出力。softmax関数：合計が1になるように
"""
model.add(Dense(num_class, activation='softmax'))

""" 
学習率を0.01でスタート
"""
# opt = SGD(lr=0.01) 最適化方法SGD Adam rmspropなどがある
opt = Adam()
"""
loss:損失関数
"""
model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])
# トレーニングを実行
model.fit(X_train, y_train, batch_size=32, epochs=20)
# 評価
score = model.evaluate(X_test, y_test, batch_size=32)
