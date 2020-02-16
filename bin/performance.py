#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:25:26 2017, 20/11/2019

@author: Firoj Alam
"""
import numpy as np
from sklearn import metrics
import sys
import os
import sklearn.metrics as metrics
from sklearn import preprocessing
import pandas as pd
import re
import pandas as pd
from sklearn.metrics import roc_auc_score

def roc_auc_score_multiclass(actual_class, pred_class, average = "weighted"):

  #creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
    #creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc

    list_values = [v for v in roc_auc_dict.values()]
    average = np.average(list_values)
    return average


def performance_measure(y_true,y_pred,le):

    acc=P=R=F1=AUC=0.0
    report=""
    AUC = roc_auc_score_multiclass(y_true, y_pred)

    #print(roc_auc_multiclass)
    try:
       acc=metrics.accuracy_score(y_true,y_pred)
       P=metrics.precision_score(y_true,y_pred,average="weighted")
       R=metrics.recall_score(y_true,y_pred,average="weighted")
       F1=metrics.f1_score(y_true,y_pred,average="weighted")
       report=metrics.classification_report(y_true, y_pred)
    except Exception as e:
        print (e)
        pass
    return AUC,acc,P,R,F1,report


def performance_measure_cnn(y_true, y_prob, le):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_prob, axis=1)


    y_true = le.inverse_transform(y_true)
    y_pred = le.inverse_transform(y_pred)
    acc = P = R = F1 = AUC = 0.0
    report = ""
    AUC = roc_auc_score_multiclass(y_true, y_pred)
    try:
        acc = metrics.accuracy_score(y_true, y_pred)
        P = metrics.precision_score(y_true, y_pred, average="weighted")
        R = metrics.recall_score(y_true, y_pred, average="weighted")
        F1 = metrics.f1_score(y_true, y_pred, average="weighted")
        report = metrics.classification_report(y_true, y_pred)

    except Exception as e:
        print (e)
        pass

    return AUC,acc, P, R, F1, report


def format_conf_mat(y_true,y_pred,le):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)


    y_true = le.inverse_transform(y_true)
    y_pred = le.inverse_transform(y_pred)

    conf_mat = pd.crosstab(np.array(y_true), np.array(y_pred), rownames=['gold'], colnames=['pred'], margins=True)
    pred_columns = conf_mat.columns.tolist()
    gold_rows = conf_mat.index.tolist()
    conf_mat_str = ""
    header = "Pred\nGold"
    for h in pred_columns:
        header = header + "\t" + str(h)
    conf_mat_str = header + "\n"
    index = 0
    for r_index, row in conf_mat.iterrows():
        row_str = str(gold_rows[index])  # for class label (name)
        index += 1
        for col_item in row:
            row_str = row_str + "\t" + str(col_item)
        conf_mat_str = conf_mat_str + row_str + "\n"
    return conf_mat_str


def write_classified_label(y_true, y_prob, le, ids, out_file):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_prob, axis=1)

    probabilities = []
    for index, prob in zip(list(y_pred), list(y_prob)):
        probabilities.append(prob[index])

    y_true = le.inverse_transform(y_true)
    y_pred = le.inverse_transform(y_pred)

    out_file.write("id\tgold_label\tclassified_label\tconfidence\n")

    for id, ref, pred, prob in zip(ids, y_true, y_pred, probabilities):
        out_file.write(str(id) + "\t" + str(ref) + "\t" + str(pred) + "\t" + str(prob) + "\n")
    out_file.close

