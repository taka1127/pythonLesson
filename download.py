# flickrapiをimport
from flickrapi import FlickrAPI
# コマンドラインでアクセス
from urllib.request import urlretrieve
# os= OSに依存しているさまざまな機能を利用 time=時間制御 sys=システムのpathを取得
import os, time, sys
# flickrapiキー関係のファイル
import apiTest

# flickrapiキー
key = apiTest.flickrapiKey
# flickrapiパスワード
secret = apiTest.flickrapiSecret
# サーバー負荷がかからないよう1行間隔でリクエスト送信
wait_time = 1

# 検索キーワード sys.argv[1]=[download.py,検索キーワード]→ 1番目の引数を取得
keyword = sys.argv[1]
# ファイルをセーブするディレクトリ
savedir = "./" + keyword
# flickrにアクセスするためのクライエントオブジェクトを宣言
flickr = FlickrAPI(key, secret, format='parsed-json')
# 検索実行をresultに入れる(flickr.photos.search機能)以下引数
result = flickr.photos.search(

  text=keyword,
  # 写真枚数
  per_page=400,
  # 写真
  media='photos',
  # 最新のものから
  sort='relevanse',
  # 暴力的なものは避ける
  safe_search=1,
  # 余分に取得する情報 url_q=ダウンロード用URL
  extras = 'url_q, license'
)
# resultのphotosのついたキーをphotos変数に格納
photos = result['photos']

# photosを順番に取り出し、urlを確認しローカルのディレクトリに格納
# enumerate＝連番を振る
for i, photo in enumerate(photos['photo']):
  url_q = photo['url_q']
  # ファイルを保存するためのパスを作成
  filepath = savedir + '/' + photo['id'] + '.jpg'
  # もしデータが存在していれば飛ばす
  if os.path.exists(filepath): continue
  # ネット上からファイルをダウンロードし保存：urlretrieve(目的のURL, 保存先のファイル名)
  urlretrieve(url_q, filepath)
  # 1秒間停止し次のループ
  time.sleep(wait_time)