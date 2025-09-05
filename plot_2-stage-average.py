# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 10:16:01 2023

@author: kawano
"""
from CIlab_function import CIlab
from FuzzyClassifierFromCSV import FileInput
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os

def make_setting_scale(dataset):

    setting = {"australian" : {"train" : {"x_min" : 0, "x_max" : 0.5, "y_min": 0.9, "y_max" : 0.94},
                               "test" : {"x_min" : 0, "x_max" : 0.5, "y_min": 0.85, "y_max" : 0.94},
                               "legend_size" : 14},

               "pima" : {"train" : {"x_min" : 0, "x_max" : 0.5, "y_min": 0.8, "y_max" : 0.86},
                         "test" : {"x_min" : 0, "x_max" : 0.5, "y_min": 0.74, "y_max" : 0.84},
                          "legend_size" : 15},

               "vehicle" : {"train" : {"x_min" : 0, "x_max" : 0.5, "y_min": 0.62, "y_max" : 0.9},
                            "test" : {"x_min" : 0, "x_max" : 0.5, "y_min": 0.62, "y_max" : 0.76},
                             "legend_size" : 12.2},

               "heart" : {"train" : {"x_min" : 0, "x_max" : 0.5, "y_min": 0.9, "y_max" : 0.95},
                           "test" : {"x_min" : 0, "x_max" : 0.5, "y_min": 0.8, "y_max" : 0.84},
                           "legend_size" : 15},

               "vowel" : {"train" : {"x_min" : 0, "x_max" : 0.5, "y_min": 0.64, "y_max" : 0.80},
                         "test" : {"x_min" : 0, "x_max" : 0.5, "y_min": 0.6, "y_max" : 0.70},
                         "legend_size" : 10.48},

               "satimage" : {"train" : {"x_min" : 0, "x_max" : 0.5, "y_min": 0.8, "y_max" : 0.95},
                            "test" : {"x_min" : 0, "x_max" : 0.5, "y_min": 0.8, "y_max" : 0.90},
                            "legend_size" : 14},

               "glass" : {"train" : {"x_min" : 0, "x_max" : 0.5, "y_min": 0.8, "y_max" : 0.9},
                           "test" : {"x_min" : 0, "x_max" : 0.5, "y_min": 0.64, "y_max" : 0.70},
                           "legend_size" : 14},

               "penbased" : {"train" : {"x_min" : 0, "x_max" : 0.5, "y_min": 0.8, "y_max" : 0.9},
                           "test" : {"x_min" : 0, "x_max" : 0.5, "y_min": 0.8, "y_max" : 0.9},
                             "legend_size" : 15},

               "texture" : {"train" : {"x_min" : 0, "x_max" : 0.5, "y_min": 0.85, "y_max" : 0.95},
                           "test" : {"x_min" : 0, "x_max" : 0.5, "y_min": 0.85, "y_max" : 0.95},
                             "legend_size" : 15},

               "phoneme" : {"train" : {"x_min" : 0, "x_max" : 0.5, "y_min": 0.8, "y_max" : 0.9},
                            "test" : {"x_min" : 0, "x_max" : 0.5, "y_min": 0.8, "y_max" : 0.9},
                            "legend_size" : 13}}


    return setting[dataset]


def make_setting(dataset):

    return {"dataset" : dataset, "marker":"o", "color":"tab:orange", "size":150, "algorithmID" : "FSS2022"}


def make_threshold_setting():

    setting = {
                "single" : {"linestyle" : "solid"},
                #"cwt" : {"linestyle" : "solid"}
                #rwt" : {"linestyle" : "solid"}
               }

    return setting


def make_model_setting():

    setting = {"Adaboost" : {"color" : "tab:blue", "label" : "AB"},
               "DecisionTree" : {"color" : "tab:orange", "label" : "DT"},
               "NaiveBayes" : {"color" : "tab:green", "label": "NB"},
               "GaussianProcess" : {"color" : "red", "label": "GP"},
               "kNN" : {"color" : "purple", "label" : "$\it{k}$NN"},
               "MLP" : {"color" : "brown", "label" :"MLP"},
               "RF" : {"color" : "hotpink", "label" : "RF"},
               "LinearSVC" : {"color" : "cyan", "label" : "SVM"}
               }

    return setting

class plotMLModelARC():

    def __init__(self):

        pass


    def figSetting(self, xMin, xMax, yMin, yMax):

        # figsizeで図のサイズを指定 横長にしたいなら左の数値を大きくする．
        fig, axes = plt.subplots(1, 1, figsize=(7, 7), dpi = 300)

        # X ticks (= Number of rule)
        xH = 0.1
        xticks = np.arange(0, xMax + 0.1, 0.1)

        xMax = xticks[len(xticks) - 1]

        xH = 0.02
        axes.set_xlim(xMin - xH, xMax + xH)
        axes.set_xticks(xticks)

        # Y ticks
        yH = 0.05
        yMin = (int)(yMin / yH)
        yMin = yMin * yH
        yMax = (int)((yMax + yH) / yH)
        yMax = yMax * yH

        yticks = np.arange(yMin, yMax + 0.02, 0.05)
        yMin = yticks[0]
        yMax = yticks[-1]

        yH = 0.01
        axes.set_ylim(yMin - yH, yMax + yH)
        axes.set_yticks(yticks)

        axes.set_xlabel("Reject rate", fontsize = 26)
        axes.set_ylabel("Accuracy", fontsize = 26)

        fig.subplots_adjust(left = 0.2)
        axes.grid(linewidth=0.4)
        axes.yaxis.set_minor_locator(AutoMinorLocator(3))
        axes.tick_params(which = 'major', length = 8, color = 'black', labelsize = 25)
        axes.tick_params(which = 'minor', length = 5, color = 'black', labelsize = 25)

        return fig, axes



    def _get_threshold_base(self, setting, model, num_stage, base, y_measure):

        def get_one_trial(files):

            dfs = [pd.read_csv(file) for file in files]

            columns = ["accuracy", "rejectrate"]

            dfs = [df[columns] for df in dfs]

            dfs = [df.drop_duplicates() for df in dfs]

            return step_mean(dfs)

        def step_mean(dfs, step = 0.02):

            def df_one_step(t_min, t_max):

                culc_df = [df.query(f"rejectrate > {t_min}").query(f"rejectrate <= {t_max}") for df in dfs]
                culc_df = [df for df in culc_df if not df.empty]

                criteria = 16

                if len(culc_df) >= criteria:

                    return pd.concat(culc_df)

                return None

            x_min = min([df["rejectrate"].min() for df in dfs]) - step
            x_max = max([df["rejectrate"].max() for df in dfs]) + step

            df_mean = [df_one_step(t, (t + step)) for t in np.arange(x_min, x_max, step)]

            return [df.mean() for df in df_mean if df is not None]

        algorithmID = "./results/20250615/FSS2022_test"

        dataset = setting["dataset"]

        RR = 3
        CC = 10

        files = [f"{algorithmID}/{dataset}/trial{rr}{cc}/{y_measure}-{base}.csv" for rr in range(RR) for cc in range(CC)]

        if num_stage == 2:

            files = [f"{algorithmID}/{dataset}/trial{rr}{cc}/{model}/second-{y_measure}-{base}.csv" for rr in range(RR) for cc in range(CC)]

        return pd.concat(get_one_trial(files))



    def plot_threshold_base(self, setting, y_measure, num_stage, fig, axes):

        setting_threshold = make_threshold_setting()

        bases = setting_threshold.keys()


        if num_stage == 1:
            color = "k"

            for base in bases:

                results = self._get_threshold_base(setting, None, 1, base, y_measure)

                plt.plot(results["rejectrate"],
                         results["accuracy"],
                         color = color,
                         label = "single",
                         linestyle = setting_threshold[base]["linestyle"])

            return fig, axes


        model_setting = make_model_setting()

        for base in bases:

            for model in model_setting.keys():

                results = self._get_threshold_base(setting, model, 2, base, y_measure)

                plt.plot(results["rejectrate"],
                         results["accuracy"],
                         color = model_setting[model]["color"],
                         label = model_setting[model]["label"],
                         alpha = 0.8,
                         linestyle = setting_threshold[base]["linestyle"])


        return fig, axes



    def plot_ARC(self, setting, x_measures = ["reject_train", "reject_test"], y_measures = ["train", "test"]):

        plt.rcParams["font.family"] = "Times New Roman"     #全体のフォントを設定
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["font.size"] = 14                      #フォントの大きさ
        plt.rcParams["xtick.minor.visible"] = False         #x軸補助目盛りの追加
        plt.rcParams["ytick.direction"] = "out"             #y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams["ytick.major.size"] = 10               #y軸主目盛り線の長さ
        plt.rcParams['figure.subplot.bottom'] = 0.15


        for x_measure, y_measure in zip(x_measures, y_measures):

            dataset = setting["dataset"]

            scale = make_setting_scale(dataset)

            xMin = scale[y_measure]["x_min"]
            xMax = scale[y_measure]["x_max"]
            yMin = scale[y_measure]["y_min"]
            yMax = scale[y_measure]["y_max"]

            fig, axes = self.figSetting(xMin, xMax, yMin, yMax)

            self.plot_threshold_base(setting, y_measure, 1, fig, axes)

            plt.rcParams['font.family'] = 'Times New Roman'


            self.plot_threshold_base(setting, y_measure, 2, fig, axes)

            ##凡例
            ##axes.legend(fontsize = 19.7, loc = "lower right")

            output_dir = f"./results/plots/all_models/{dataset}/"

            if not os.path.exists(output_dir):

                os.makedirs(output_dir)
            #CWT,RWT,singleを{y_measure}の後ろに追加
            fig.savefig(f"{output_dir}/{dataset}_{y_measure}_single.png")
            fig.savefig(f"{output_dir}/{dataset}_{y_measure}_single.png")

            # fig.savefig(f"{output_dir}/{dataset}_{y_measure}.png", dpi = 300)
            # fig.savefig(f"{output_dir}/{dataset}_{y_measure}.png", dpi = 300)


            # if not os.path.exists(output_dir + "png/"):

            #     os.makedirs(output_dir + "png/")

            # fig.savefig(f"{output_dir}/png/{dataset}_{y_measure}.png", dpi = 300)
            # fig.savefig(f"{output_dir}/png/{dataset}_{y_measure}.png", dpi = 300)

        return fig, axes


def run_dataset(dataset):

    setting = make_setting(dataset)

    plotMLModelARC().plot_ARC(setting)

    return f"finished_{dataset}"


if __name__ == "__main__":

    # datasets = ["pima", "australian", "vehicle", "heart", "phoneme", "glass", "satimage", "vowel"]
    # datasets = ["phoneme", "vowel", "texture"]
    datasets = ["pima", "australian", "vehicle", "heart", "phoneme", "glass", "vowel"]

    for dataset in datasets:

        print(run_dataset(dataset))