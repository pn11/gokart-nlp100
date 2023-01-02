# gokart で言語処理100本ノックをやってみる

## gokart とは

[gokart](https://github.com/m3dev/gokart) はエムスリーが開発している機械学習パイプラインツール。 Spotify により開発されている [luigi](https://github.com/spotify/luigi) のラッパーになっていてより簡単に書くことができる。  
NLP の機械学習モデルを開発していると前処理、事前学習、ファインチューニング、可視化などなど工程が多く、管理が大変になる。パイプラインツールを使って楽になりたいということで、言語処理100本ノックの機械学習パートで試してみる (56, 57, 59は gokart 的に新しい操作がないため飛ばす)。

## 前準備

gokart がどんなものかまずは公式ドキュメントで動作を確かめてみると良い。

- [Intro To Gokart — gokart documentation](https://gokart.readthedocs.io/en/latest/intro_to_gokart.html)

上記と同じことを簡単にブログにも記録しておいた。

- [gokart 触ってみた - pn11's blog](https://pn11.github.io/blog/posts/2023/gokart-quickstart/)

## 参考リンク

- [機械学習プロジェクト向けPipelineライブラリgokartを用いた開発と運用 - エムスリーテックブログ](https://www.m3tech.blog/entry/2019/09/30/120229)  
  エムスリー公式なのでドキュメントとこれをまず読むのが良いと思われる
- [gokartを使ってみる - Re:ゼロから始めるML生活](https://www.nogawanogawa.com/entry/gokart)  
  題材が NLP (文書分類) なので参考にしやすい
- [【Techの道も一歩から】第42回「Luigiとgokartを試用して比べて特徴を掴む」 - Sansan Tech Blog](https://buildersbox.corp-sansan.com/entry/2021/10/06/110000)  
  luigi と gokart の比較が簡潔にまとまっていて分かりやすい
- [PythonのPipelineパッケージ比較：Airflow, Luigi, Gokart, Metaflow, Kedro, PipelineX - Qiita](https://qiita.com/Minyus86/items/70622a1502b92ac6b29c)  
  Gokart 以外のパイプラインツールもまとめた力作
- [gokart, redshellsによるMLOpsへの第一歩 - Qiita](https://qiita.com/yamasakih/items/11b14bb4712c9fcb7faf)  
  ドキュメントで扱われていない部分のコードの書き方がとても参考になる
- [【言語処理100本ノック 2020】第6章: 機械学習【Python】 - Amaru Note](https://amaru-ai.com/entry/2022/10/12/202559)  
  100本ノックのコードを書くにあたり大いに参考にさせて頂いた
