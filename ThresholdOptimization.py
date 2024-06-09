import numpy as np
import copy
from sklearn.pipeline import Pipeline
import os
import pandas as pd
import itertools
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.lhs import LHS


class ThresholdEstimator():
    """
    this class is threshold estimator of fumera's method

    Reference
    --------------------------------------------------------------------------------------------------------------------
    G. Fumera and F. Roli, “Multiple reject thresholds for improving classification reliability,”
    In Proceedings of the Joint IAPR International Workshopson Advances in Pattern Recognition, pp. 863–871, Aug. 2000.
    --------------------------------------------------------------------------------------------------------------------
    """

    def __init__(self,
                 pipe: Pipeline,
                 param,
                 output_dir
                 ):

        """
        コンストラクタ

        Parameter
        --------
        pipe  : Pipeline
                predict_proba_transformerの前処理＋ThresholdBaseRejectOptionの推論のパイプライン

        param : dictionary,
                key : "kmax", "Rmax", "deltaT"
        --------
        """

        self.param = param
        self.pipe = pipe
        self.output_dir = output_dir

    def fit(self, X, y):

        """
        パイプラインに適用させた後，self.paramに従い閾値の探索を行う．
        """
        self.pipe.fit(X, y)
        # _run_search から result_list を受け取る
        result = self._run_search(X, y)

        # ここでは単一の結果を含むリストを返します
        return result

    def _run_search(self, X, y):
        """
        def list_increment(list_, ite, value):

            list_copy = copy.deepcopy(list_)

            list_copy[ite] = list_copy[ite] + value

            return np.array(list_copy)

        def run_one_thresh(threshold):

            isReject = self.pipe[-1].isReject(self.predict_proba_, threshold)

            rejectrate = self.pipe[-1].rejectrate(isReject)

            accuracy = self.pipe[-1].accuracy(y, self.predict_, isReject)

            return {"threshold": threshold, "accuracy": accuracy, "rejectrate": rejectrate, "isReject": isReject}
        """
        self.threshold = self.pipe[-1].zeros_threshold(y)
        print(self.threshold)
        self.predict_proba_ = self.pipe.transform(X)

        self.predict_ = self.pipe[-1].predict(self.predict_proba_, reject_option=False)

        isReject = self.pipe[-1].isReject(self.predict_proba_, self.threshold)

        self.accuracy = self.pipe[-1].accuracy(y, self.predict_, isReject)

        self.rejectrate = 0.0

        self.isReject = np.ones(len(X), dtype=np.bool_) * False

        search_idx = np.arange(len(self.threshold))

        ignore_idx = np.array([])
        # 問題インスタンスの作成
        # NSGA2アルゴリズムの設定
        algorithm = NSGA2(
            pop_size=50,
            n_offsprings=50,
            sampling=LHS(),
            crossover=SBX(prob=0.9, prob_var=1.0, eta=15),
            mutation=PolynomialMutation(prob=1 / len(self.threshold), eta=20),
            eliminate_duplicates=True
        )
        problem = ThresholdOptimizationProblemNSGA2(self, X, y, threshold_length=len(self.threshold))

        # 最適化の実行
        result = minimize(problem,
                          algorithm,
                          termination=('n_gen', 101),
                          seed=1,
                          save_history=True,
                          verbose=True)

        # 最終世代の解集合を取得
        final_population = result.pop.get("X")

        # 解集合から必要な情報を抽出して辞書型のリストに保存
        result_list = []
        for solution in final_population:
            # ここで各解に対してaccuracy, rejectrate, thresholdを計算

            isReject = self.pipe[-1].isReject(self.predict_proba_, solution)
            accuracy = self.pipe[-1].accuracy(y, self.predict_, isReject)
            rejectrate = self.pipe[-1].rejectrate(isReject)
            result_list.append({"accuracy": accuracy, "rejectrate": rejectrate, "threshold": solution,
                                'pipe': self.pipe})  # ここで self.pipe を追加})

        return result_list

    def proba_score(self, y, predict, isReject):

        len_accept = np.count_nonzero(~isReject)

        # if all patterns are rejected, accuracy is 1.0
        accuracy = 1.0

        if len_accept != 0:
            accuracy = self.pipe[-1].accuracy(y, predict, isReject)

        return {"accuracy": accuracy,
                "rejectrate": self.pipe[-1].rejectrate(isReject)}

    def score(self, X, y, threshold=None):

        """
        探索した閾値に基づき，データセットのaccuracy，及びreject rateを求める．

        return : dict
                 key : "accuracy", "rejectrate"
        """
        if threshold is None:
            threshold = self.threshold

        isReject = self.func_isReject(X)

        predict = self.pipe[0].model.predict(X[~isReject])

        len_accept = np.count_nonzero(~isReject)

        # if all patterns are rejected, accuracy is 1.0
        accuracy = 1.0

        if len_accept != 0:
            accuracy = np.count_nonzero(predict == y[~isReject]) / len_accept

        return {"accuracy": accuracy,
                "rejectrate": self.pipe[-1].rejectrate(isReject)}

    def func_isReject(self, X):

        predict_proba_ = self.pipe.transform(X)

        return self.pipe[-1].isReject(predict_proba_, self.threshold)

    def proba_isReject(self, predict_proba):

        return self.pipe[-1].isReject(predict_proba, self.threshold)


class predict_proba_transformer():
    """
        Transfomerクラス（前処理を行うクラス）
        入力ベクトルを入力ベクトルに対するモデルの確信度に変換する．

        ここでは，sklearn.predict_proba()を使用する事を想定しています．
        """

    def __init__(self, _model, base=None):
        """
            コンストラクタ

            Parameter
            ---------
            model : sklearn.ClassifierMixin
                    sklearnの識別器で使用される関数を実装したモデル

            base : string
                   確信度のベースを指定する．
                   デフォルトは各クラスに対する確信度に変換するが，base = "rule"と指定することで各ルールに対する確信度に変換する．
                   "rule"を使用する場合は，現在はCIlab_Classifier.FuzzyClassifierをモデルとして使用して下さい．
            """

        self.model = _model

        self.base = base

    def fit(self, X, y):
        self.model.fit(X, y)

        self.transform(X)

        return self

    def transform(self, X):
        if self.base != None:
            return self.model.predict_proba(X, base=self.base)

        self.predict_proba = self.model.predict_proba(X)

        # for proba in predict_proba_:

        #     proba[np.argmin(proba)] = 0

        return self.predict_proba


class ThresholdOptimizationProblemNSGA2(ElementwiseProblem):
    def __init__(self, estimator, X, y, threshold_length):
        super().__init__(n_var=threshold_length,  # 閾値の数を変数の数として設定
                         n_obj=2,  # 目的関数の数（誤識別率と棄却率）
                         n_constr=0,  # 制約条件の数（棄却率がRmax以下）
                         xl=0,  # 変数の下限
                         xu=1)  # 変数の上限
        self.estimator = estimator
        self.X = X
        self.y = y

    def _evaluate(self, threshold, out, *args, **kwargs):
        print("threshold", threshold)
        isReject = self.estimator.pipe[-1].isReject(self.estimator.predict_proba_, threshold)
        accuracy = self.estimator.pipe[-1].accuracy(self.y, self.estimator.predict_, isReject)
        rejectrate = self.estimator.pipe[-1].rejectrate(isReject)

        # 目的関数の値を設定
        out["F"] = [1 - accuracy, rejectrate]

    def evaluate_individual(self, threshold):
        """ 個体の識別精度と棄却率を評価するメソッド """
        isReject = self.estimator.pipe[-1].isReject(self.estimator.predict_proba_, threshold)
        accuracy = self.estimator.pipe[-1].accuracy(self.y, self.estimator.predict_, isReject)
        rejectrate = self.estimator.pipe[-1].rejectrate(isReject)
        return accuracy, rejectrate


def save_generation_results(history, gen_numbers, estimator, X, y, output_dir, filename_prefix):
    for gen in gen_numbers:
        if gen < len(history):
            gen_data = history[gen].pop
            thresholds = gen_data.get("X")
            results = []
            for thresh in thresholds:
                # evaluate_individual を score に変更
                isReject = estimator.pipe[-1].isReject(estimator.predict_proba_, thresh)
                accuracy = estimator.pipe[-1].accuracy(y, estimator.predict_, isReject)
                rejectrate = estimator.pipe[-1].rejectrate(isReject)
                results.append([accuracy, rejectrate, thresh])
            print("results:", results)
            df = pd.DataFrame(results, columns=["Accuracy", "RejectRate", "Threshold"])
            output_dir_path = os.path.join(output_dir, f"{filename_prefix}_gen_{gen}.csv")
            # output_dir のディレクトリが存在するかどうかを確認
            if not os.path.exists(os.path.dirname(output_dir_path)):
                # 存在しない場合はディレクトリを作成
                os.makedirs(os.path.dirname(output_dir_path))

            # その後ファイルを保存
            df.to_csv(output_dir_path, index=False)
