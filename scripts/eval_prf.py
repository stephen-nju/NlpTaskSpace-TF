# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:18073701@cnsuning.com
@license: Apache Licence
@software:PyCharm
@file: eval_prf.py
@time: 2022/1/25 17:42
"""
import argparse
import json
import pickle

import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--label", default="../data/train_data/label.pickle", type=str)
    parse.add_argument("--eval_data", default="D:/Users/Desktop/out_student.txt", type=str)
    parse.add_argument("--ground_truth", default="D:/Users/Desktop/a.txt", type=str)
    args = parse.parse_args()

    with open(args.label, "rb") as g:
        label_list: list = pickle.load(g)

    label_encode = MultiLabelBinarizer()
    label_encode.fit([label_list])

    eval_data = {}
    with open(args.eval_data, "r", encoding="utf-8") as g:
        for line in g:
            raw = line.strip().split("\t")
            eval_data[raw[0]] = label_encode.transform(json.loads(raw[1]))

    truth_data = {}
    with open(args.ground_truth, "r", encoding="utf-8") as g:
        for line in g:
            raw = line.strip().split("\t")
            truth_data[raw[0]] = label_encode.transform([json.loads(raw[1])])

    y_true = []
    y_pred = []
    no_eval = 0
    for data in truth_data:
        if data in eval_data:
            y_true.append(truth_data[data])
            y_pred.append(eval_data[data])
        else:
            no_eval += 1
            print("true data not eval")

    print(no_eval)
    y_true = np.squeeze(np.asarray(y_true), axis=1)
    y_pred = np.squeeze(np.asarray(y_pred), axis=1)
    # y_true = np.asarray(list(itertools.chain.from_iterable(y_true)))
    # y_pred = np.asarray(list(itertools.chain.from_iterable(y_pred)))
    print(classification_report(y_true, y_pred))
