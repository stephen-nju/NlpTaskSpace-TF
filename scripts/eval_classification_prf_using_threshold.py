# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:18073701@cnsuning.com
@license: Apache Licence
@software:PyCharm
@file: eval_classification_prf_using_threshold.py
@time: 2021/8/31 17:39
"""
import numpy as np
from functools import partial
import pandas as pd
import argparse
from sklearn.metrics import classification_report


# 数据格式，分为三列，分别是data,label,score
def threshold(x, score):
    if x < score:
        return 0
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    df = pd.read_csv(args.data_path, names=["data", "true_label", "score"], header=None)
    y_true = df["true_label"]
    y_pred = df["score"].apply(partial(threshold, args.threshold), axis=1)
    print(classification_report(y_true, y_pred, digits=4))
