import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from PIL import Image
import sys

# クラス分類
classes = ["car", "motorbike"]
# クラスの数
num_classes = len(classes)
# imageサイズ224px
image_size = 224

# 引数から画像ファイルを参照して読み込む
image = Image.open(sys.argv[1])
# 色の情報RGBの順にそろえる
image = image.convert("RGB")
# サイズをそろえる
image = image.resize((image_size, image_size))
# numpyの配列としてimageデータを変換 値が大きくなりすぎてスコアにばらつきが出るため255で割って値が0~1の間で推移するようにする
data = np.asarray(image) / 255.0
X = []
# 最後尾に追加
X.append(data)
# numpyのarrayに変換
X = np.array(X)

# モデルのロード
model = load_model('./vgg16_transfer.h5')

result = model.predict([X])[0]
predicted = result.argmax()
percentage = int(result[predicted] * 100)

print(classes[predicted],percentage)