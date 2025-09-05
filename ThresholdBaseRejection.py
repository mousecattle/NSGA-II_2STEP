# -*- coding: utf-8 -*-

"""
@author: kawano
"""
import numpy as np
import pandas as pd
import os


class ThresholdBaseRejectOption():
    """
    抽象クラス
    sklearnの推論器クラス
    """

    def isReject(self, predict_proba_, threshold):

        """
        予測の確信度と閾値を比較し，棄却するかを返す．

        Parameters
        ----------
        predict_proba_ : ndarray
                         入力に対する予測の確信度
        threshold : ndarray
                    閾値，単一の閾値の場合でもlistにしてください．

        Returns
        -------
        ndarray
        入力を棄却するかを表すarray
        棄却する場合 : True, 棄却しない場合 : False
        """

        pass

    def zeros_threshold(self):

        """
        閾値初期化用の関数

        Returns
        -------
        ndarray
        閾値の長さ分の0埋めされたリスト

        """

        pass

    def accuracy(self, y, predict_, isReject):

        """
        accuracyを求める．

        Parameters
        ----------
        y : ndarray
            教師ラベル.

        predict_ : ndarray
                   予測ラベル

        isReject : ndarray
                   predict_を棄却するかを表すarray
                   isReject()の返り値を渡すことを想定しています.

        Returns
        -------
        float : accuracy
        全ての予測ラベルを棄却する場合，accuracy は 1.0（誤識別率 0）として計算します．

        """
        #print("y",y)

        len_accept = np.count_nonzero(~isReject)

        # if all patterns are rejected, accuracy is 1.0
        if len_accept == 0:
            return 1.0

        return np.count_nonzero((predict_ == y) & (~isReject)) / len_accept

    def rejectrate(self, isReject):

        """
        reject rateを求める

        Parameters
        ----------
        isReject : ndarray
                   predict_を棄却するかを表すarray
                   isReject()の返り値を渡すことを想定しています.

        Returns
        -------
        float
        reject rateを返す．

        """

        return np.count_nonzero(isReject) / len(isReject)

    def fit(self, predict_proba_, y):

        return self

    def predict(self, predict_proba_, reject_option=False):

        """
        入力に対して，棄却する場合はNone，確信度が高い場合は予測ラベルを返す関数
        現状バグってます．

        Parameters
        ----------
        predict_proba_ : ndarray
                        入力に対する予測の確信度

        Returns
        -------
        list
        予測ラベル or None

        """
        if reject_option:
            return [np.argmax(proba) if ~self.isReject(proba, self.threshold) else None for proba in predict_proba_]

        return [np.argmax(proba) if not all(proba == 0) else None for proba in predict_proba_]

    def transform(self, X):

        return X


class SingleThreshold(ThresholdBaseRejectOption):
    """
    単一の閾値に基づく棄却オプション
    全てのパターンを１つの閾値で棄却の判定を行う．

    Reference
    --------------------------------------------------------------------------------
    C. K. Chow, “On optimum error and reject tradeoff,
    ” IEEE Trans. on Inform. Theory, vol. 16, pp.41-46, Jan. 1970.
    --------------------------------------------------------------------------------
    """

    def __init__(self):
        pass

    def zeros_threshold(self, y=None):
        return np.zeros(1)

    def isReject(self, predict_proba_, threshold):
        return np.max(predict_proba_, axis=1) < threshold

    def score(self, y_true, y_proba, isReject):
        """
        正解ラベル y_true、予測確率 y_proba、および isReject フラグから
        accuracy（正答率）と reject rate（棄却率）を計算する。

        Parameters
        ----------
        y_true : array-like
            正解ラベル

        y_proba : array-like
            各サンプルのクラス予測確率

        isReject : array-like (bool)
            棄却すべきサンプルのフラグ（True: 棄却）

        Returns
        -------
        accuracy : float
            棄却せずに予測した中での精度

        reject_rate : float
            全体の中で棄却された割合
        """
        y_pred = np.argmax(y_proba, axis=1)
        mask = ~isReject
        if np.sum(mask) == 0:
            return 0.0, 1.0  # すべて棄却された場合
        accuracy = np.mean(y_pred[mask] == y_true[mask])
        reject_rate = np.mean(isReject)
        return accuracy, reject_rate

class ClassWiseThreshold(ThresholdBaseRejectOption):
    """
    クラス毎の閾値に基づく棄却オプション

    Reference
    --------------------------------------------------------------------------------
    G. Fumera, F. Roli, and G. Giacinto, “Reject option with multiple thresholds,
    ” Pattern Recognition, vol. 33, no. 12, pp. 2099-2101, Dec. 2000.
    --------------------------------------------------------------------------------
    """

    def __init__(self):
        pass

    def zeros_threshold(self, y):
        return np.zeros(max(y) + 1)

    def isReject(self, predict_proba_, threshold):
        index_list = np.argmax(predict_proba_, axis=1)

        return np.array([proba[index] < threshold[index] for proba, index in zip(predict_proba_, index_list)])

    def score(self, y_true, predict_proba_, isReject):

        """
         評価データに対する正解率と棄却率を計算します。

        Parameters:
        - y_true: 実際のラベルの配列
        - predict_proba_: 予測確率の配列
        - isReject: 各データポイントが棄却されるかどうかを示すブール配列

        Returns:
        - accuracy: 正解率
        - reject_rate: 棄却率
        """

        #棄却されなかったデータポイントのインデックス
        accept_indices = ~isReject

        #棄却されなかったデータポイントに対する予測ラベル
        predictions = np.argmax(predict_proba_[accept_indices], axis=1)

        # 実際のラベルと予測ラベルを比較して正解数を計算
        y_true_accepted = y_true[accept_indices]
        num_correct = np.sum(predictions == y_true_accepted)

        # 正解率の計算
        accuracy = num_correct / len(y_true_accepted) if len(y_true_accepted) > 0 else 0

        # 棄却率の計算
        reject_rate = np.mean(isReject)

        return accuracy, reject_rate


class RuleWiseThreshold(ThresholdBaseRejectOption):
    """
    ルール毎の閾値に基づく棄却オプション
    現状では，CIlab_Classifierでしか使用できません．
    その他のルールベースの識別器で使用する際には，ルール毎の確信度を返す関数を実装してください．

    コンストラクタに使用するルールリストを渡してください．

    このクラスで使用するpredict_proba関数はクラス毎の確信度ではなく，ルール毎の確信度を返します．

    Reference
    --------------------------------------------------------------------------------
    川野弘陽，Eric Vernon，増山直輝，能島裕介，石渕久生，
    「複数の閾値を用いた棄却オプションの導入におけるファジィ識別器への影響調査」，
    インテリジェント・システム・シンポジウム 2021，オンライン，9 月，2021.
    --------------------------------------------------------------------------------
    """

    def __init__(self, ruleset):
        self.ruleset = np.array(ruleset)

    def zeros_threshold(self, y=None):
        return np.zeros(len(self.ruleset))

    def isReject(self, predict_proba_, threshold):
        proba_idx_list = np.argmax(predict_proba_, axis=1)
        old = np.array([proba[proba_idx] < threshold[proba_idx] for proba, proba_idx in
                        zip(predict_proba_, proba_idx_list)]).flatten()

        # if not all(new == old):
        # print(threshold)

        index_list = np.argmax(predict_proba_, axis=1)

        return np.array([proba[index] < threshold[index] for proba, index in zip(predict_proba_, index_list)])

    def predict(self, predict_proba_, reject_option=True):
        winner_rule_id = np.argmax(predict_proba_, axis=1)

        if reject_option:
            return [self.ruleset[np.argmax(proba)] \
                        if ~self.isReject(proba, self.threshold) else None for proba in predict_proba_]

        return np.array(
            [self.ruleset[np.argmax(proba)].class_label if not all(proba == 0) else None for proba in predict_proba_])

    def score(self, y_true, predict_proba_, isReject):
        """
         評価データに対する正解率と棄却率を計算します。

        Parameters:
        - y_true: 実際のラベルの配列
        - predict_proba_: 予測確率の配列
        - isReject: 各データポイントが棄却されるかどうかを示すブール配列

        Returns:
        - accuracy: 正解率
        - reject_rate: 棄却率
        """

        # 棄却されなかったデータポイントのインデックス
        accept_indices = ~isReject

        # 棄却されなかったデータポイントに対する予測ラベル
        predictions = np.argmax(predict_proba_[accept_indices], axis=1)

        # 実際のラベルと予測ラベルを比較して正解数を計算
        y_true_accepted = y_true[accept_indices]
        num_correct = np.sum(predictions == y_true_accepted)

        # 正解率の計算
        accuracy = num_correct / len(y_true_accepted) if len(y_true_accepted) > 0 else 0

        # 棄却率の計算
        reject_rate = np.mean(isReject)

        return accuracy, reject_rate


class SecondStageRejectOption(ThresholdBaseRejectOption):
    """
    ThresholdBaseRejectoptionでは，パターンの確信度と閾値から，棄却の判定を行うが，
    本クラスは，パターンと閾値から棄却の判定を行う．

    ２段階棄却オプションでは，探索した閾値に基づいて棄却と判定されたパターンに対して
    ファジィ識別器以外のモデルがファジィ識別器と同じ識別結果を出力した場合，棄却しない手法である．

    Reference
    -------------------------------------------------------------------------------------
    川野弘陽，Eric Vernon，増山直輝，能島裕介，石渕久生，
    「２段階棄却オプションを導入したファジィ識別器の精度と識別拒否のトレードオフ解析」，
    ファジィ・システム・シンポジウム 2022，オンライン，9 月，2022.
    -------------------------------------------------------------------------------------
    """

    def __init__(self, thresh_estimator, second_classifier):
        self.thresh_estimator = thresh_estimator

        self.second_classifier = second_classifier

    def isReject(self, predict_proba, second_predict, threshold=None):
        if threshold is None:
            threshold = self.thresh_estimator.threshold

        base_predict = self.thresh_estimator.predict(predict_proba, reject_option=False)
        return self.thresh_estimator.isReject(predict_proba, threshold) & (base_predict != second_predict)

    def score(self, y_true, predict_proba_, isReject):
        """
        評価データに対する正解率と棄却率を計算します。

        Parameters:
        - y_true: 実際のラベルの配列
        - predict_proba_: 予測確率の配列
        - isReject: 各データポイントが棄却されるかどうかを示すブール配列

        Returns:
        - accuracy: 正解率
        - reject_rate: 棄却率
        """
        # 棄却されなかったデータポイントのインデックス
        accept_indices = ~isReject

        # 棄却されたデータポイントの数
        num_rejected = np.sum(isReject)

        # 正解率の計算
        rule_ids = np.argmax(predict_proba_[accept_indices], axis=1)
        predictions = np.array([self.ruleset[rule_id].class_label for rule_id in rule_ids])
        y_true_accepted = y_true[accept_indices]
        num_correct = np.sum(predictions == y_true_accepted)
        accuracy = num_correct / len(y_true_accepted) if len(y_true_accepted) > 0 else 0

        # 棄却率の計算
        reject_rate = num_rejected / len(y_true)

        return accuracy, reject_rate

    # def accuracy(self, X, y, threshold = None):

    #     if threshold == None:

    #         threshold = self.thresh_estimator.threshold

    #     isReject = self.isReject(X, threshold)

    #     # if all patterns are rejected, accuracy is 1.0
    #     len_accept = np.count_nonzero(~isReject)

    #     if len_accept == 0:

    #         return 1.0

    #     predict_ = self.thresh_estimator.pipe[0].model.predict(X[~isReject])

    #     return np.count_nonzero(predict_ == y[~isReject]) / len_accept

    # def rejectrate(self, X, threshold = None):

    #     if threshold == None:

    #         threshold = self.thresh_estimator.threshold

    #     return np.count_nonzero(self.isReject(X, threshold)) / len(X)

