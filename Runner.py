# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 19:06:50 2022

@author: kawano
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, ParameterGrid
from ThresholdBaseRejection import SingleThreshold, ClassWiseThreshold, SecondStageRejectOption
from CIlab_function import CIlab
from ThresholdOptimization import ThresholdEstimator, predict_proba_transformer
import file_output as output
import os
from GridSearchParameter import GridSearch


class runner():

    def __init__(self,
                 dataset,
                 algorithmID,
                 experimentID,
                 fname_train,
                 fname_test):
        """
        コンストラクタ

        Parameter
        ---------
        MoFGBMLライブラリの仕様に合わせています．
        dataset : dataset name : string, ex. "iris"

        algorithmID : string
                      "result"直下のディレクトリ名

        experimentID : string
                       出力ファイルのデイレクトリ名,
                       出力ファイルは "result\\algorithmID\\experimentID に出力されます．

        file_train : string
                     学習用データのファイル名

        file_test : string
                    評価用データのファイル名
        """

        self.dataset = dataset
        self.algorithmID = algorithmID
        self.experimentID = experimentID
        self.X_train, self.X_test, self.y_train, self.y_test = CIlab.load_train_test(fname_train, fname_test,
                                                                                     type_="numpy")
        self.output_dir = f"../results/threshold_base/{self.algorithmID}/{self.dataset}/{self.experimentID}/"

    def grid_search(self, model, param, cv=10):
        """
        grid_search function
        ハイパーパラメータをグリッドサーチにより決定し，best_modelを返す．

        Parameter
        ---------
        model : sklearn.classifier

        param : ハイパーパラメータの辞書のリスト

        cv : the number of CV, default is 10
        """

        gscv = GridSearchCV(model, param, cv=cv, verbose=0)

        gscv.fit(self.X_train, self.y_train)

        gs_result = pd.DataFrame.from_dict(gscv.cv_results_)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        gs_result.to_csv(self.output_dir + 'gs_result.csv')

        CIlab.output_dict(gscv.best_estimator_.get_params(), self.output_dir, "best_model_info.txt")

        return gscv.best_estimator_

    def run(self, pipe, params, train_file, test_file, core=4):
        """
        run function
        ARCs(Accuracy-Rejection Curves)で必要なデータを出力する関数.

        Parameter
        ---------
        pipe : Pipeline module
               ステップ：predict_proba_transfomer，ThresholdBaseRejectOption

        params : パラメータ辞書のリスト,
                 辞書のキーは，"kmax", "Rmax", "deltaT"にしてください．

        train_file : file name of result for trainning data, result is accuracy, reject rate, threshold

        test_file : file name of result for test data
        """


        def _run_one_param(param):
            return ThresholdEstimator(pipe, param, self.output_dir).fit(self.X_train, self.y_train)

        results = _run_one_param(params)
        result_list = [ResultItem(params, result_item['accuracy'], result_item['rejectrate'], result_item['threshold'])
                       for
                       result_item in results]
        # result_list = [_run_one_param(param) for param in params]

        train_result = [[result.accuracy, result.rejectrate, result.threshold] for result in result_list]

        output.to_csv(train_result, self.output_dir, train_file)

        test_result = [result.score(self.X_test, self.y_test) for result in result_list]

        output.to_csv(test_result, self.output_dir, test_file)

        return

    def run_second_stage(self, pipe, params, second_models, train_file, test_file, core=5):
        """
        run function
        2段階棄却オプションのARCs(Accuracy-Rejection Curves)で必要なデータを出力する関数.

        Parameter
        ---------
        pipe : Pipeline module
               ステップ：predict_proba_transfomer，ThresholdBaseRejectOption

        params : パラメータ辞書のリスト,
                 辞書のキーは，"kmax", "Rmax", "deltaT"にしてください．

        second_model : sklearn.ClassifierMixin
                       sklearnの識別器で使用される関数を実装したモデル
                       2段階目の判定で用いるモデル．

        train_file : file name of result for training data, result is accuracy, reject rate, threshold

        test_file : file name of result for test data
        """

        def _run_one_search_threshold(param):
            return ThresholdEstimator(pipe, param,self.output_dir).fit(self.X_train, self.y_train)

        result_list = _run_one_search_threshold(params)
        print("result_list:", result_list)
        # 学習用データの結果をまとめて出力
        train_result = [
            {'accuracy': result['accuracy'], 'rejectrate': result['rejectrate'], 'threshold': result['threshold']} for
            result in result_list]
        output.to_csv(train_result, self.output_dir, train_file)

        # 評価用データの結果をまとめて出力
        test_result = []
        for result in result_list:
            # 評価用データに対するスコア計算
            estimator = result['pipe'][-1]  # pipeの最後の要素(ThresholdBaseRejectOptionのインスタンス)を取得
            print("estimator:",estimator)
            proba_test = result['pipe'][0].transform(self.X_test)  # pipeの最初の要素で予測確率を計算
            print("proba_test:", proba_test)
            isReject = estimator.isReject(proba_test, result['threshold'])
            print("isReject:", isReject)
            test_accuracy, test_reject_rate = estimator.score(self.y_test, proba_test, isReject)
            test_result.append({'accuracy': test_accuracy, 'rejectrate': test_reject_rate})

        output.to_csv(test_result, self.output_dir, test_file)

        #ここからの処理内容を考える必要あり

        #proba_train = result_list[0].pipe[0].predict_proba
        #base_predict_train = result_list[0].pipe[-1].predict(proba_train)
        """
        # 2段階棄却オプション，やってることは上と同じ
        for key, model in second_models.items():

            model_predict = model.predict(self.X_train)

            RO_list = [SecondStageRejectOption(thresh_estimator, model) for thresh_estimator in result_list]

            isReject_list = [RO.isReject(proba_train, model_predict) for RO in RO_list]

            second_RO_train_result = [[RO.accuracy(self.y_train, base_predict_train, isReject),
                                       RO.rejectrate(isReject)] for RO, isReject in zip(RO_list, isReject_list)]

            output.to_csv(second_RO_train_result, f"{self.output_dir}/{key}/", "second-" + train_file)

            model_predict = model.predict(self.X_test)


            isReject_list = [RO.isReject(proba_test, model_predict) for RO in RO_list]

            second_RO_test_result = [[RO.accuracy(self.y_test, base_predict_test, isReject),
                                       RO.rejectrate(isReject)] for RO, isReject in zip(RO_list, isReject_list)]

            output.to_csv(second_RO_test_result, f"{self.output_dir}/{key}/", "second-" + test_file)
        """

    def output_const(self, dict_):
        CIlab.output_dict(dict_, self.output_dir, "Const.txt")


class ResultItem:
    def __init__(self, pipe, accuracy, rejectrate, threshold):
        self.param = pipe
        self.accuracy = accuracy
        self.rejectrate = rejectrate
        self.threshold = threshold


def main():
    dataset = "pima"

    param = {"max_depth": [5, 10, 20]}

    model = DecisionTreeClassifier()

    run = runner(dataset,
                 "RO-test",
                 "trial00-v2",
                 f"..\\dataset\\{dataset}\\a0_0_{dataset}-10tra.dat",
                 f"..\\dataset\\{dataset}\\a0_0_{dataset}-10tst.dat")

    best_model = run.grid_search(model, param)

    param = {"kmax": [700], "Rmax": np.arange(0, 0.51, 0.1), "deltaT": [0.001]}

    pipe = Pipeline(steps=[('predict_proba_transform', predict_proba_transformer(best_model)),
                           ('estimator', ClassWiseThreshold())])

    second_model = KNeighborsClassifier()

    run.run_second_stage(pipe, ParameterGrid(param), second_model)


if __name__ == "__main__":
    main()
