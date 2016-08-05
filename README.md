# フィードフォワードニューラルネットワークでニュース分類

## Requirements

* gensim
* mecab
* mecab-python
* chainer


## 使い方

1. 以下のURLよりlivedoorニュースコーパスをダウンロードする。

  http://www.rondhuit.com/download.html#ldcc

2. ダウンロードしたファイル解凍し、出来たフォルダを`news_docs`に改名する。


3. 以下のコマンドを叩く

```
$ python create_dic.py
# livedoorコーパスを元にgensimで辞書を作る
# => 辞書データlivedoordic.txtを作る

$ python estimation_chainer.py
# 学習を行う
```


## 参考

* ChainerのMNISTサンプル
  * https://github.com/pfnet/chainer/blob/master/examples/mnist/train_mnist.py

* Chainerでフィードフォワードニューラルネットワークを実装して文書分類する
  * http://qiita.com/ichiroex/items/9aa0bcada0b5bf6f9e1c

* scikit-learnとgensimでニュース記事を分類する
  * http://qiita.com/yasunori/items/31a23eb259482e4824e2
