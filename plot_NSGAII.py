import pandas as pd
import matplotlib.pyplot as plt

# 仮のCSVファイルのパスを設定（実際のパスに置き換える）
csv_paths = ["threshold_results_gen_10.csv", "threshold_results_gen_100.csv", "threshold_results_gen_1000.csv"]

# CSVファイルからデータを読み込む
dataframes = [pd.read_csv(path) for path in csv_paths]

# 新しいグラフを作成
plt.figure(figsize=(10, 6))

# データフレームごとにループを行い、プロットする
for df in dataframes:
    plt.scatter(df['RejectRate'], df['Accuracy'], alpha=0.5)

# グラフのタイトルと軸ラベルを設定

plt.xlabel('Reject Rate')
plt.ylabel('Accuracy')

# 凡例を表示
plt.legend(['gen=10', 'gen=100', 'gen=1000'])

# グリッドを表示
plt.grid(True)

# グラフを表示
plt.show()