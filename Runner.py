
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


def extract_pareto_front(results):
    """
    非劣解（パレート最適）な accuracy / rejectrate のペアを抽出する。
    Parameters
    ----------
    results : list of tuples [(accuracy, rejectrate, model_name), ...]
    Returns
    -------
    pareto_front : list of tuples
        非劣解な点のみのリスト。
    """
    pareto_front = []
    for i, (acc_i, rej_i, model_i) in enumerate(results):
        dominated = False
        for j, (acc_j, rej_j, _) in enumerate(results):
            if i != j:
                if acc_j >= acc_i and rej_j <= rej_i:
                    if acc_j > acc_i or rej_j < rej_i:
                        dominated = True
                        break
        if not dominated:
            pareto_front.append((acc_i, rej_i, model_i))
    return pareto_front


class runner():
    def __init__(self, dataset, algorithmID, experimentID, fname_train, fname_test):
        self.dataset = dataset
        self.algorithmID = algorithmID
        self.experimentID = experimentID
        self.X_train, self.X_test, self.y_train, self.y_test = CIlab.load_train_test(fname_train, fname_test, type_="numpy")
        self.output_dir = f"./results/20250615/{self.algorithmID}/{self.dataset}/{self.experimentID}/"

    def grid_search(self, model, param, cv=10):
        gscv = GridSearchCV(model, param, cv=cv, verbose=0)
        gscv.fit(self.X_train, self.y_train)
        gs_result = pd.DataFrame.from_dict(gscv.cv_results_)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        gs_result.to_csv(self.output_dir + 'gs_result.csv')
        CIlab.output_dict(gscv.best_estimator_.get_params(), self.output_dir, "best_model_info.txt")
        return gscv.best_estimator_

    def load_model(self, model_name):
        if model_name == "Adaboost":
            from sklearn.ensemble import AdaBoostClassifier
            return AdaBoostClassifier()
        elif model_name == "DecisionTree":
            return DecisionTreeClassifier()
        elif model_name == "NaiveBayes":
            from sklearn.naive_bayes import GaussianNB
            return GaussianNB()
        elif model_name == "GaussianProcess":
            from sklearn.gaussian_process import GaussianProcessClassifier
            return GaussianProcessClassifier()
        elif model_name == "kNN":
            return KNeighborsClassifier()
        elif model_name == "MLP":
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier()
        elif model_name == "RF":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier()
        elif model_name == "LinearSVC":
            from sklearn.svm import LinearSVC
            from sklearn.calibration import CalibratedClassifierCV
            return CalibratedClassifierCV(LinearSVC())
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def run(self, pipe, params, train_file, test_file, core=4):
        def _run_one_param(param):
            return ThresholdEstimator(pipe, param, self.output_dir).fit(self.X_train, self.y_train)

        results = _run_one_param(params)
        result_list = []

        train_result = []
        test_result = []

        for r in results:
            threshold = r["threshold"]
            pipe = r["pipe"]
            pipe[-1].threshold = threshold  # 明示的に閾値を設定

            # --- trainデータに対する出力 ---
            proba_train = pipe[0].transform(self.X_train)
            print("proba_train shape:", proba_train.shape)
            print("proba_train sample:", proba_train[:3])
            isReject_train = pipe[-1].isReject(proba_train, threshold)

            if hasattr(pipe[-1], 'score'):
                acc_train, rej_train = pipe[-1].score(self.y_train, proba_train, isReject_train)
            else:
                # スコア関数が無い場合は手動計算
                accept_idx = ~isReject_train
                if np.sum(accept_idx) == 0:
                    acc_train = 0.0
                else:
                    y_pred = pipe[-1].predict(proba_train[accept_idx], reject_option=False)
                    acc_train = np.mean(y_pred == self.y_train[accept_idx])
                rej_train = np.mean(isReject_train)

            train_result.append([acc_train, rej_train, threshold])

            # --- testデータに対する出力 ---
            proba_test = pipe[0].transform(self.X_test)
            isReject_test = pipe[-1].isReject(proba_test, threshold)

            if hasattr(pipe[-1], 'score'):
                acc_test, rej_test = pipe[-1].score(self.y_test, proba_test, isReject_test)
            else:
                accept_idx = ~isReject_test
                if np.sum(accept_idx) == 0:
                    acc_test = 0.0
                else:
                    y_pred = pipe[-1].predict(proba_test[accept_idx], reject_option=False)
                    acc_test = np.mean(y_pred == self.y_test[accept_idx])
                rej_test = np.mean(isReject_test)

            test_result.append([acc_test, rej_test, threshold])
            result_list.append(ResultItem(pipe, acc_test, rej_test, threshold))

        output.to_csv(train_result, self.output_dir, train_file)
        output.to_csv(test_result, self.output_dir, test_file)
        return

    def run_second_stage(self, pipe, param_grid, second_models, train_file, test_file, thresh_type):
        result_list = []
        for param in param_grid:
            estimator = ThresholdEstimator(pipe, param, self.output_dir)
            result = estimator.fit(self.X_train, self.y_train)
            result_list.extend(result)

        proba_test_list = [res['pipe'][0].transform(self.X_test) for res in result_list]
        base_predict_test_list = [
            res['pipe'][-1].predict(proba, reject_option=False)
            for proba, res in zip(proba_test_list, result_list)
        ]

        # 1段階目の出力格納用
        first_train_result = []
        first_test_result = []

        # --- 1段階目の train/test 結果をまとめて取得 ---
        for res in result_list:
            estimator = res["pipe"][-1]
            estimator.threshold = res["threshold"]

            # --- train 評価 ---
            proba_train = res["pipe"][0].transform(self.X_train)
            base_predict_train = res["pipe"][-1].predict(proba_train, reject_option=False)
            first_isReject_train = estimator.isReject(proba_train, estimator.threshold)
            acc_train = estimator.accuracy(self.y_train, base_predict_train, first_isReject_train)
            rej_train = estimator.rejectrate(first_isReject_train)
            first_train_result.append([acc_train, rej_train, estimator.threshold.tolist()])

            # --- test 評価 ---
            proba_test = res["pipe"][0].transform(self.X_test)
            base_predict_test = res["pipe"][-1].predict(proba_test, reject_option=False)
            first_isReject_test = estimator.isReject(proba_test, estimator.threshold)
            acc_test = estimator.accuracy(self.y_test, base_predict_test, first_isReject_test)
            rej_test = estimator.rejectrate(first_isReject_test)
            first_test_result.append([acc_test, rej_test, estimator.threshold.tolist()])

        # --- 1段階目の出力 ---（Rejectrate昇順ソート + 非劣解抽出つき）

        # Rejectrate昇順ソート
        first_train_result.sort(key=lambda x: x[1])
        first_test_result.sort(key=lambda x: x[1])

        # 非劣解（パレート最適）のみ抽出
        pareto_train_result = extract_pareto_front([(acc, rej, "") for acc, rej, _ in first_train_result])
        pareto_test_result = extract_pareto_front([(acc, rej, "") for acc, rej, _ in first_test_result])

        # ソート後のパレート結果を出力（ファイル名に "pareto-" プレフィックスを追加）
        output.to_csv([[acc, rej, "-"] for acc, rej, _ in pareto_train_result],
                      self.output_dir, f"train-{thresh_type}.csv")

        output.to_csv([[acc, rej, "-"] for acc, rej, _ in pareto_test_result],
                      self.output_dir, f"test-{thresh_type}.csv")

        # 通常のすべての結果も残しておく（従来通り）
        output.to_csv(first_train_result, self.output_dir, f"non-sorted-train-{thresh_type}.csv")
        output.to_csv(first_test_result, self.output_dir, f"non-sorted-test-{thresh_type}.csv")

        # --- 2段階目の処理 ---
        all_results = []
        for model_name in second_models:
            model = self.load_model(model_name)
            model.fit(self.X_train, self.y_train)

            second_RO_test_result = []
            second_RO_train_result = []

            for i, res in enumerate(result_list):
                estimator = res['pipe'][-1]
                estimator.threshold = res["threshold"]

                # --- testデータ ---
                proba_test = proba_test_list[i]
                base_predict_test = base_predict_test_list[i]
                first_isReject_test = estimator.isReject(proba_test, estimator.threshold)

                model_predict_test = model.predict(self.X_test)
                RO_test = SecondStageRejectOption(estimator, model)
                second_isReject_test = RO_test.isReject(proba_test, model_predict_test)
                combined_isReject_test = np.logical_and(first_isReject_test, second_isReject_test)

                acc_test = RO_test.accuracy(self.y_test, base_predict_test, combined_isReject_test)
                rej_test = RO_test.rejectrate(combined_isReject_test)

                second_RO_test_result.append([acc_test, rej_test])
                all_results.append((acc_test, rej_test, model_name))

                # --- trainデータ ---
                proba_train = res['pipe'][0].transform(self.X_train)
                base_predict_train = res['pipe'][-1].predict(proba_train, reject_option=False)
                first_isReject_train = estimator.isReject(proba_train, estimator.threshold)

                model_predict_train = model.predict(self.X_train)
                RO_train = SecondStageRejectOption(estimator, model)
                second_isReject_train = RO_train.isReject(proba_train, model_predict_train)
                combined_isReject_train = np.logical_and(first_isReject_train, second_isReject_train)

                acc_train = RO_train.accuracy(self.y_train, base_predict_train, combined_isReject_train)
                rej_train = RO_train.rejectrate(combined_isReject_train)

                second_RO_train_result.append([acc_train, rej_train])

            # rejectrate 昇順にソート
            second_RO_test_result.sort(key=lambda x: x[1])
            second_RO_train_result.sort(key=lambda x: x[1])

            # 出力
            output.to_csv(second_RO_test_result, f"{self.output_dir}/{model_name}/", "second-" + test_file)
            output.to_csv(second_RO_train_result, f"{self.output_dir}/{model_name}/", "second-" + train_file)

        # 非劣解抽出して出力
        pareto_results = extract_pareto_front(all_results)
        df_pareto = pd.DataFrame(pareto_results, columns=["accuracy", "rejectrate", "model"])
        df_pareto = df_pareto.sort_values(by="rejectrate")

        # 出力ファイル名を方式に応じて変更
        if thresh_type == "single":
            filename = "pareto_summary_single.csv"
        elif thresh_type == "rwt":
            filename = "pareto_summary_rwt.csv"
        else:
            filename = "pareto_summary_cwt.csv"

        df_pareto.to_csv(f"{self.output_dir}/{filename}", index=False)

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
                 f"..\dataset\{dataset}\a0_0_{dataset}-10tra.dat",
                 f"..\dataset\{dataset}\a0_0_{dataset}-10tst.dat")
    best_model = run.grid_search(model, param)
    param = {"kmax": [700], "Rmax": np.arange(0, 0.51, 0.1), "deltaT": [0.001]}
    pipe = Pipeline(steps=[('predict_proba_transform', predict_proba_transformer(best_model)),
                           ('estimator', ClassWiseThreshold())])
    second_model = KNeighborsClassifier()
    run.run_second_stage(pipe, ParameterGrid(param), second_model)


if __name__ == "__main__":
    main()
