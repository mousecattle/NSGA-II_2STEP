# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 01:04:14 2022

@author: kawano
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor

import time
def detection_async(_request):
    result = subprocess.run(["python",
                             _request["pythonFile"],
                             _request["dataset"],
                             _request["algroithmID"],
                             _request["experimentID"],
                             _request["trainFile"],
                             _request["testFile"],
                             _request["fuzzy_clf"]])

    return result


def detection_async_parallel(_requests):
    results = []

    with ThreadPoolExecutor(max_workers=1) as executor:
        for result in executor.map(detection_async, _requests):
            results.append(result)

    return results


def run(dataset):
    import os



    requests = [{"dataset": dataset,
                 "pythonFile": "FSS2022_main.py",
                 "algroithmID": "FSS2022_test",
                 "experimentID": f"trial{i}{j}",
                 "trainFile": f"../dataset/{dataset}/a{i}_{j}_{dataset}-10tra.dat",
                 "testFile": f"../dataset/{dataset}/a{i}_{j}_{dataset}-10tst.dat",
                 "fuzzy_clf": f"../results/FSS2022/{dataset}/classify_{dataset}_{i}_{j}.csv"}
                for i in range(1) for j in range(1)]

    for result in detection_async_parallel(requests):
        print(result)


if __name__ == "__main__":

    start_time = time.time()
    #Datasets = ["australian","glass","heart","penbased","phoneme","pima","vehicle",]
    Datasets = ["australian"]

    for dataset in Datasets:
        run(dataset)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"プログラムの実行時間: {elapsed_time}秒")