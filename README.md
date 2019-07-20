# distribution_hypothesis

## どのようなプログラムを作ったか。

自然言語処理の機械学習。Wikipediaのも本文データを取得できるDBpediaからテキストデータをダウンロードし、word2vecを使用して学習。livedoorニュースコーパスを使用して**日本語ニュース記事を分類する**ものです。

---
### 開発環境
- MacOS 10.14.5
- python3.7.3
- pip 19.1.1
- pipenv
- Mecab
- Graphviz

- vscode
- JupyterLab
---
### 注意事項

※１　学習するためのデータなどが大きいため、必要に応じてダウンロードしてください。
今回使用したダウンロード先は下記の「各コードの説明」に随時書いていありますのでそちらの参照をお願いします。


※２　最初に文章から単語を空白区切りにしたり、学習させるためにかなりの時間を使います。（自分は６〜８時間かかりました）

※３　visualize.pyで意味ベクトルを視覚化するためにimport matplotlib.pyplot as pltと**matplotlib**を使用していますが、デフォルトでは日本語に対応していないため、そのまま実行すると日本語が描画される部分が◻︎◻︎◻︎...と**文字化け**を起こします。
下記のURLを参考に自分は解決したのでよろしくお願いします。
https://qiita.com/katuemon/items/5c4db01997ad9dc343e0

---
### 各コードの説明（上から順に、実行して欲しいコードになっています）

#### generate_corpus.py

###### DBpediaのnif_contextデータをtxtファイルに空白区切で分割するためのコード

DBpediaからwikipediaの本文テキストが含まれているNIF Contextデータをダウンロード

```

$ wget http://downloads.dbpedia.org/2016-10/core-i18n/ja/nif_context_ja_ttl.bz2

```
generate_corpus.pyを実行する

```

$ python generate_corpus.py nif_context_ja.ttl.bz2 > ja_wiki_corpus.txt

```

**自分のPC（Corei5）で３時間ほどかかりました。**
正しく分割されているかをチェック


```
$ head -n1 jawiki_corpus.txt
日常 生活 （ に ち じ ょうせいかつ ） は 、 毎日 繰り返さ れる 生活 の こと 。 具体 的 に は 、 日々 の 生活 の 中 で 繰り返さ れる 出来事 や 習慣 的 動作 、 そこ で 用い られる 物 の 考え方 や 知識 （ 常識 ） 、 接する 物品 （ その 一部 は 日 用品 と 呼ば れる ） など から 構成 さ れる 。 ...

```

出力結果が異なっても問題ありません。単語が空白区切りで分類されて格納されているかの確認をお願いします。


#### train.py

###### generate_corpus.pyで作ったコーパスからGensimを使って学習モデルを作成する。

```

$ python train.py ja_wiki_corpus.txt ja_wiki_word2vec.bin

```
**4~5時間かかりました。**

#### visualize.py

###### 学習したモデルであるbinファイルを使い、意味ベクトルを視覚化し、気になる単語に近い単語５０個を取得してプロットする

scikit-learnライブラリのt-SESアルゴリズムを用いて100次元ベクトルを２次元まで次元圧縮をする。


ファイルを実行（コード上で気になる単語は"料理"、取得する単語数は50に固定している）

```

$ python visualize.py ja_wiki_word2vec.bin word2vec_visualizetion.png

```

実行が終わるとword2vec_visualizetion.pngが生成される。

[![料理に近い単語５０](word2vec_visualizetion.png)]


---
### メインのニュースを分類する各コードの説明
---


#### 事前準備

※１　livedoorニュースコーパスのデータセットをダウンロード

```
$ wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
$ tar xzf ldcc-20140209.tar.gz
```

textというディレクトリが作成される。

| ディレクトリ名 | サービス名 |
|--------------------|--------------------|
| topic-news | トピックニュース |
| sports-watch | Sports Watch |
| it-life-hack | ITライフハック |
| kaden-channel | 家電チャンネル | 
| movie-enter | Movie Enter | 
| dokujo-tsushin | 独女通信 |
| smax | エスマックス |
| livedoor-homme | livedoor HOMME |
| peachy | Peachy |

表のように９種の記事のサービス名を表すディレクトリにテキストファイルで記事が保存されます。


※２　単語の意味ベクトルを読み込めるようにするため、学習したbinファイルをtxtファイル形式で保存し直す。
```
import gensim
model = gensim.models.KeyedVectors.load('ja_wiki_word2vec.bin')
model.wv.save_word2vec_format('ja_wiki_word2vec.txt')
```

#### dataset.py
###### データセットの読み込みと前処理を行う

#### nbow_model.py
###### Neural Bag-of-Wordsモデルを定義する。

#### nbow_train.py
###### モデルの訓練とテストを行う

ファイルを実行。

```
$ python nbow_train.py text ja_wiki_word2vec.txt 50 32
Epoch: 1 Accuracy: 13.94% (308/2210)
Epoch: 2 Accuracy: 19.91% (440/2210)
Epoch: 3 Accuracy: 19.77% (437/2210)
・
・・
・・・
Epoch: 50 Accuracy: 94.34% (2085/2210)
```

epoch 50 batch_size 32 で学習。自分の結果は94.34%でした。

以上が今回作成したニュース記事を分類する機械学習のプログラムになります。


---
### 今回のニュース分類とは関係がないコード
---

#### similar_words.py

###### 類語検索アルゴリズム

入力した単語から意味が近い単語を検索する。

```

$ python similar_words.py ja_wiki_word2vec.bin 飛行機
航空機 (類似度: 0.83)
セスナ機 (類似度: 0.80)
グライダー (類似度: 0.79)
```

---

#### word_analogy.py

###### アナロジー推論アルゴリズム

「パリ」＋「日本」ー「フランス」　＝　「東京」
上記のように「東京」を予測するアルゴリズム


```
$ python word_analogy.py ja_wiki_word2vec.bin ロンドン 日本 イギリス
予測結果: 東京
$ python word_analogy.py ja_wiki_word2vec.bin 父 女 男
予測結果: 母
```

