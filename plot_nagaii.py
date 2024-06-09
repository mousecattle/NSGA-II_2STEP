import matplotlib.pyplot as plt
import os
import pandas as pd


# グラフを保存する関数
def save_plot(data, gen_labels, output_path):
    plt.figure()
    for (acc, rej), gen_label in zip(data, gen_labels):
        plt.scatter(rej, acc, label=f'gen={gen_label}')
    plt.xlabel('Rejection Rate')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(output_path)
    plt.close()


# CSVからデータを読み込む関数
def read_data_from_csv(path):
    df = pd.read_csv(path)
    return df.iloc[:, 0], df.iloc[:, 1]  # 1列目と2列目のデータ


# メインの処理を行う関数
# メインの処理を行う関数
def plot_data_from_trials(trial_folders, generations, base_dir):
    gen_labels = ['0', '10', '100', '1000']  # 世代のラベル
    for trial in trial_folders:
        data = []
        for gen in generations:
            csv_path = os.path.join(base_dir, trial, f'threshold_results_gen_{gen}.csv')
            if os.path.exists(csv_path):
                acc, rej = read_data_from_csv(csv_path)
                data.append((acc, rej))
        output_path = os.path.join(base_dir, f'{trial}_plot.png')
        save_plot(data, gen_labels, output_path)


# トライアルフォルダのリスト
trial_folders = [f'trial{i:02d}' for i in range(30)]
generations = [0, 10, 100, 1000]
base_dir = '.'# 同一階層を指す

# データのプロットとグラフの保存
plot_data_from_trials(trial_folders, generations, base_dir)
