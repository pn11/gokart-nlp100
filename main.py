from io import BytesIO
import os
import shutil

import gokart
import luigi
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Step50DownloadDatasetTask(gokart.TaskOnKart):
    """ https://nlp100.github.io/ja/ch06.html#50-データの入手整形
    Zip ファイルのデータをダウンロードしてきて展開する。
    """
    # 最初にコード書いてる時やデバッグ時は `rerun = True` にして動かす
    # 公式ドキュメントでは適当なパラメータを作って変えることで rerun させろとのこと (https://gokart.readthedocs.io/en/latest/task_settings.html#rerun-task)
    # rerun = True
    def run(self):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip'
        filename = 'NewsAggregatorDataset.zip'
        if not os.path.exists(filename):
            data = requests.get(url).content
            with open(filename ,mode='wb') as f:
                f.write(data)
            outdirname = 'data/' + filename.replace('.zip', '')
            shutil.unpack_archive(filename, outdirname)

        colnames = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
        df = pd.read_csv('data/NewsAggregatorDataset/newsCorpora.csv', header=None, names=colnames, sep='\t', index_col='ID')
        self.dump(df)


class Step50SplitDatasetTask(gokart.TaskOnKart):
    """ https://nlp100.github.io/ja/ch06.html#50-データの入手整形
    データを train/valid/test に 8:1:1 で分割する。
    """
    # rerun = True
    def requires(self):
        return Step50DownloadDatasetTask()

    def run(self):
        df: pd.DataFrame = self.load()

        # 50-2: 情報源（publisher）が”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, “Daily Mail”の事例（記事）のみを抽出する．
        pub_list = ['Reuters','Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']
        df = df[df['PUBLISHER'].isin(pub_list)]

        print(df)

        seed = 12345

        # 50-3, 50-4: scikit-learn の train_test_split では2つにしか分割できないため2回に分けて3つに分割する (https://datascience.stackexchange.com/a/15136/126697)
        df_train, df_valid_test = train_test_split(df, test_size=0.2, random_state=seed)
        df_valid, df_test = train_test_split(df_valid_test, test_size=0.5, random_state=seed)
    
        print(len(df_train), len(df_valid), len(df_test))

        assert len(df_train) + len(df_valid) + len(df_test) == len(df)

        # 50-4: それぞれtrain.txt，valid.txt，test.txtというファイル名で保存する
        #       今回は使わないが一応保存しておく
        df_train[['CATEGORY', 'TITLE']].to_csv('output/train.txt', header=None, index=None, sep='\t')
        df_valid[['CATEGORY', 'TITLE']].to_csv('output/valid.txt', header=None, index=None, sep='\t')
        df_test[['CATEGORY', 'TITLE']].to_csv('output/test.txt', header=None, index=None, sep='\t')

        # 学習データと評価データを作成したら，各カテゴリの事例数を確認せよ．
        print('Train Data')
        print(df_train['CATEGORY'].value_counts())
        print('Validation Data')
        print(df_valid['CATEGORY'].value_counts())
        print('Test Data')
        print(df_test['CATEGORY'].value_counts())

        self.dump((df_train, df_valid, df_test))


class Step51ExtractFeatureTask(gokart.TaskOnKart):
    """ https://nlp100.github.io/ja/ch06.html#51-特徴量抽出
    特徴量を抽出する。今回は記事タイトルから TF-IDF を作成する。
    以下の記事では前処理などをちゃんとしていて参考になる。
    なお記事中で使われている `get_features_name` は今のバージョンではなくなっているので代わりに `get_features_name_out` を使う。
    https://amaru-ai.com/entry/2022/10/12/202559#51-特徴量抽出
    """

    def requires(self):
        return Step50SplitDatasetTask()

    def run(self):
        df_train, df_valid, df_test = self.load()

        # 10回以上出現する unigram, bi-gram について計算
        vec_tfidf = TfidfVectorizer(min_df=10, ngram_range=(1, 2))

        # valid は TF-IDF を計算するための train data に含めても良いが、今回はやらない
        tfidf_train = vec_tfidf.fit_transform(df_train['TITLE'])
        tfidf_valid = vec_tfidf.transform(df_valid['TITLE'])
        tfidf_test = vec_tfidf.transform(df_test['TITLE'])
        print(vec_tfidf.get_feature_names_out())

        print(tfidf_train)
        # DataFrame に変換
        df_train = pd.DataFrame(tfidf_train.toarray(), columns=vec_tfidf.get_feature_names_out())
        df_valid = pd.DataFrame(tfidf_valid.toarray(), columns=vec_tfidf.get_feature_names_out())
        df_test = pd.DataFrame(tfidf_test.toarray(), columns=vec_tfidf.get_feature_names_out())
        print(df_train)

        # 今回は使用しないが一応保存
        df_train.to_csv('output/train.feature.txt', index=None, sep='\t')
        df_valid.to_csv('output/valid.feature.txt', index=None, sep='\t')
        df_test.to_csv('output/test.feature.txt', index=None, sep='\t')

        self.dump((df_train, df_valid, df_test))


class Step52TrainTask(gokart.TaskOnKart):
    """ https://nlp100.github.io/ja/ch06.html#52-学習
    Scikit-learn のロジスティック回帰で学習する。
    データと特徴量を読み込むため requires を辞書型で複数指定する (https://gokart.readthedocs.io/en/latest/task_on_kart.html#taskonkart-load)
    """

    def requires(self):
        return {'data': Step50SplitDatasetTask(), 'feature': Step51ExtractFeatureTask()}

    def run(self):
        df_train, _, _ = self.load('data')
        X_train, _, _ = self.load('feature')
        y_train = df_train['CATEGORY']

        print(X_train)
        print(y_train)

        model = LogisticRegression(random_state=123, max_iter=10000)
        model.fit(X_train, y_train)

        self.dump(model)


class Step53PredictTask(gokart.TaskOnKart):
    """ https://nlp100.github.io/ja/ch06.html#53-予測
    学習したモデルで記事タイトルからカテゴリとその予測確率を計算する。
    52と同様に複数指定してロードする。
    また、保存 (dump) も辞書型でやってみる。 (https://gokart.readthedocs.io/en/latest/task_on_kart.html#taskonkart-dump)
    """
    def output(self):
        return {'pred': self.make_target('pred.pkl'), 'prob': self.make_target('prob.pkl')}

    def requires(self):
        return {
                   'data': Step50SplitDatasetTask(),
                   'feature': Step51ExtractFeatureTask(),
                   'model': Step52TrainTask()
               }

    def run(self):
        _, _, df_test = self.load('data')
        _, _, X_test = self.load('feature')
        model = self.load('model')
        y_test = df_test['CATEGORY']

        pred_test = model.predict(X_test)
        prob_test = model.predict_proba(X_test)

        print(pred_test)
        print(prob_test)

        self.dump(pred_test, 'pred')
        self.dump(prob_test, 'prob')


class Step54CalcAccuracyTask(gokart.TaskOnKart):
    """ https://nlp100.github.io/ja/ch06.html#54-正解率の計測
    正解率を計測する。正解率はテキスト形式で保存するようにしてみる。
    """
    # rerun = True

    def output(self):
        return self.make_target('test_accuracy.txt')

    def requires(self):
        return {'data': Step50SplitDatasetTask(), 'pred': Step53PredictTask()}

    def run(self):
        _, _, df_test = self.load('data')
        pred_test = self.load('pred')['pred']
        gt_test = df_test['CATEGORY']
        
        test_accuracy = accuracy_score(gt_test, pred_test)
        print(test_accuracy)
        self.dump(test_accuracy)


class Step55CalcConfusionMatrixTask(gokart.TaskOnKart):
    """ https://nlp100.github.io/ja/ch06.html#55-混同行列の作成
    混同行列を作成し画像として保存する。
    matplotlib のプロットの画像データを取得するために BytesIO を使っているが、
    もう少し簡単に書きたい。
    p. s. こちらの方が processor を作るやり方でやっておられた https://qiita.com/yamasakih/items/11b14bb4712c9fcb7faf
    """
    # rerun = True
    def output(self):
        return self.make_target('test_confusion_matrix.png')

    def requires(self):
        return {'data': Step50SplitDatasetTask(), 'pred': Step53PredictTask()}

    def run(self):
        _, _, df_test = self.load('data')
        pred_test = self.load('pred')['pred']
        gt_test = df_test['CATEGORY']
        test_cm = confusion_matrix(gt_test, pred_test)

        fig = plt.figure()
        sns.heatmap(test_cm, annot=True, cmap='Blues')
        figbin = BytesIO()
        fig.savefig(figbin, format='png')
        self.dump(figbin.getvalue())


class TrainLogisticTask(gokart.TaskOnKart):
    """学習用タスク。 パラメータを変えられるように Step52TrainTask を少し変更しただけだが、
    説明の都合上、別のタスクとして実装する。
    """
    C = luigi.FloatParameter(default=0.1)

    def output(self):
        return self.make_target(f"model-C{self.C}.pkl")

    def requires(self):
        return {'data': Step50SplitDatasetTask(), 'feature': Step51ExtractFeatureTask()}

    def run(self):
        df_train, _, _ = self.load('data')
        X_train, _, _ = self.load('feature')
        y_train = df_train['CATEGORY']

        print(f"Training C = {self.C}")
        model = LogisticRegression(random_state=123, max_iter=10000, C=self.C)
        model.fit(X_train, y_train)

        self.dump(model)


class Step58ChangeRegularizationParameterTask(gokart.TaskOnKart):
    """ https://nlp100.github.io/ja/ch06.html#58-正則化パラメータの変更
    ロジスティック回帰のパラメータ C を変化させて実行する。
    パラメータは luigi のものを使う。
    公式ドキュメント (https://gokart.readthedocs.io/en/latest/task_parameters.html) にはっきり記述がないが、
    おそらく Gokart のパラメータはタスクの管理用で、ハイパーパラメータなど用ではなさそう。
    並列で動かしたい。
    requires のところで動かすのはなんか気持悪い
    """
    rerun = True

    parameters = np.logspace(-5, 3, 9, base=10)

    def output(self):
        return self.make_target('param_C.png')

    def requires(self):
        tasks = {}
        for C in self.parameters:
            task_name = f"Logistic(C={C})"
            tasks[task_name] = TrainLogisticTask(C=C)
        tasks['data'] = Step50SplitDatasetTask()
        tasks['feature'] = Step51ExtractFeatureTask()
        return tasks

    def run(self):
        # test data をロード
        _, _, df_test = self.load('data')
        _, _, X_test = self.load('feature')
        gt_test = df_test['CATEGORY']

        # 各モデルをロードして予測
        accuracies = []
        for C in self.parameters:
            task_name = f"Logistic(C={C})"
            model = self.load(task_name)
            pred_test = model.predict(X_test)            
            accuracies.append(accuracy_score(gt_test, pred_test))
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xscale('log')
        ax.plot(self.parameters, accuracies)
        figbin = BytesIO()
        fig.savefig(figbin, format='png')
        self.dump(figbin.getvalue())


class RunAllTask(gokart.TaskOnKart):
    """ タスクが全部実行されるようにするためのタスク。
    """
    task_list = [
            Step54CalcAccuracyTask(),
            Step55CalcConfusionMatrixTask(),
            Step58ChangeRegularizationParameterTask()
        ]

    def requires(self):
        return {str(i): task for i, task in enumerate(RunAllTask.task_list)}


if __name__ == '__main__':
    gokart.run(['RunAllTask', '--local-scheduler', '--rerun'])
