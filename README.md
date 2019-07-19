# distribution_hypothesis

## どのようなプログラムを作ったか。

自然言語処理の機械学習。word2vecを使い、分布仮説で入力したワードから、そのワードとよく一緒に使われる語を学習させるプログラム。
おまけで、入力したワードから類似性が高いワードを検索するプログラムと、アナロジー推論できるプログラム２種も作りました。


### 開発環境
- python3.7.3
- pip 19.1.1
- pipenv
- vscode
- JupyterLab
- Mecab
- Graphviz

- generate_corpus.py



- train.py

コーパスから学習させるプログラム

- visualize.py

学習モデルのbinファイルを使い、検索したいワードを入力して、そのワードと関連が高いワードの分布表を作って、画像ファイルとして保存する。


