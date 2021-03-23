# generate_data.pyで作ったnumpyデータを読み書き
import numpy as np
# tensorflowに含まれるkerasを使う
from tensorflow import keras
# シーケンシャルモデルを使用https://keras.io/guides/sequential_model/
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.utils import np_utils
# 転移学習
from tensorflow.python.keras.applications.vgg16 import VGG16

# パラメーターの初期化
# クラス分類
classes = ["car", "motorbike"]
# クラスの数
num_classes = len(classes)
# imageサイズ224px
image_size = 224

# X_train, X_test, y_train, y_testの順にデータをロード
X_train, X_test, y_train, y_test = np.load(
    "./imagefiles224.npy", allow_pickle=True)
# 正解ラベルの位置だけ１がたつよう整数値のベクトルをone hot表現に変換https://qiita.com/JeJeNeNo/items/8a7c1781f6a6ad522adf
# to_categorical(データ, データの最大値（整数）)→例[0,1][1,0]...
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
X_train = X_train.astype("float") / 255.0
X_test = X_test.astype("float") / 255.0

# モデルの定義
model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
# print('Model loaded')
# model.summary()

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))

model = Model(inputs=model.input, outputs=top_model(model.output))

# model.summary()
for layer in model.layers[:15]:
    layer.trainable = False
# opt = SGD(lr=0.01) 最適化方法SGD Adam rmspropなどがある
opt = Adam(lr=0.0001)
"""
loss:損失関数
"""
model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
# トレーニングを実行
model.fit(X_train, y_train, batch_size=32, epochs=3)
# 評価
score = model.evaluate(X_test, y_test, batch_size=32)
