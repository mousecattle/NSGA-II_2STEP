# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 18:22:16 2022

@author: kawano
"""
import sys

print(sys.path)

import numpy as np

from CIlab_function import CIlab
from FuzzyClassifierFromCSV import FileInput
from Runner import runner
from ThresholdBaseRejection import SingleThreshold, ClassWiseThreshold,RuleWiseThreshold
from ThresholdOptimization import predict_proba_transformer
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
# モデル名とクラスの対応表
model_factory = {
    "Adaboost": AdaBoostClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "NaiveBayes": GaussianNB(),
    "GaussianProcess": GaussianProcessClassifier(),
    "kNN": KNeighborsClassifier(),
    "MLP": MLPClassifier(),
    "RF": RandomForestClassifier(),
    "LinearSVC": SVC(kernel="linear", probability=True),
}
def main():
    args = sys.argv

    dataset, algorithmID, experimentID, fname_train, fname_test, clf_name = args[1:]

    X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, type_="numpy")

    run = runner(dataset, algorithmID, experimentID, fname_train, fname_test)

    thresh_param = {"kmax": [500], "Rmax": np.ones(1), "deltaT": [0.001]}

    model_names = ["Adaboost", "DecisionTree", "NaiveBayes", "GaussianProcess", "kNN", "MLP", "RF", "LinearSVC"]
    second_models = {name: model_factory[name] for name in model_names}

    fuzzy_clf = FileInput.input_classify(clf_name)

    # --- Single Threshold ---
    pipe = Pipeline(steps=[('predict_proba_transform', predict_proba_transformer(fuzzy_clf)),('estimator', SingleThreshold())])
    #run.run(pipe, ParameterGrid(thresh_param), "train-single.csv", "test-single.csv")  # ★追加：1段階目出力
    run.run_second_stage(pipe, ParameterGrid(thresh_param), second_models, "train-single.csv", "test-single.csv","single")

    # --- Class-Wise Threshold ---
    pipe = Pipeline(steps=[('predict_proba_transform', predict_proba_transformer(fuzzy_clf)),('estimator', ClassWiseThreshold())])
    #run.run(pipe, ParameterGrid(thresh_param), "train-cwt.csv", "test-cwt.csv")  # ★追加：1段階目出力
    run.run_second_stage(pipe, ParameterGrid(thresh_param), second_models, "train-cwt.csv", "test-cwt.csv", "cwt")

    # --- Rule-Wise Threshold ---
    pipe = Pipeline(steps=[('predict_proba_transform', predict_proba_transformer(fuzzy_clf, base="rule")),
                           ('estimator', RuleWiseThreshold(fuzzy_clf.ruleset))])
    #run.run(pipe, ParameterGrid(thresh_param), "train-rwt.csv", "test-rwt.csv")  # ★追加：1段階目出力
    run.run_second_stage(pipe, ParameterGrid(thresh_param), second_models, "train-rwt.csv", "test-rwt.csv", "rwt")


if __name__ == "__main__":
    main()

    # dataset = "wisconsin"

    # algorithmID = "MoFGBML_Basic"

    # rr = 0
    # cc = 0

    # for rr in range(3):

    #     for cc in range(10):

    #         experimentID = f"trial{rr}{cc}"

    #         fname_train = f"../dataset/{dataset}/a{rr}_{cc}_{dataset}-10tra.dat"

    #         fname_test = f"../dataset/{dataset}/a{rr}_{cc}_{dataset}-10tst.dat"

    #         X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, type_ = "numpy")

    #         clf_name = f"../results/{algorithmID}/{dataset}/trial{rr}{cc}/VAR-0000600000.csv"

    #         fuzzy_clf = FileInput.best_classifier(clf_name, X_train, y_train)

    #         run = runner(dataset, algorithmID, experimentID, fname_train, fname_test)

    #         thresh_param = {"kmax" : [1000], "Rmax" : np.arange(0, 1.01, 0.01), "deltaT" : [0.001]}

    #         second_models = ["Adaboost", "DecisionTree", "NaiveBayes", "GaussianProcess", "kNN", "MLP", "RF", "SVM"]

    #         fuzzy_clf = FileInput.best_classifier(clf_name, X_train, y_train)

    #         pipe = Pipeline(steps = [('predict_proba_transform', predict_proba_transformer(fuzzy_clf)),
    #                                  ('estimator', SingleThreshold())])

    #         run.run_second_stage(pipe, ParameterGrid(thresh_param), second_models, "train-single.csv", "test-single.csv")

    #         pipe = Pipeline(steps = [('predict_proba_transform', predict_proba_transformer(fuzzy_clf)),
    #                                  ('estimator', ClassWiseThreshold())])

    #         run.run_second_stage(pipe, ParameterGrid(thresh_param), second_models, "train-cwt.csv", "test-cwt.csv")

    #         pipe = Pipeline(steps = [('predict_proba_transform', predict_proba_transformer(fuzzy_clf, base = "rule")),
    #                                  ('estimator', RuleWiseThreshold(fuzzy_clf.ruleset))])

    #         run.run_second_stage(pipe, ParameterGrid(thresh_param), second_models, "train-rwt.csv", "test-rwt.csv")
