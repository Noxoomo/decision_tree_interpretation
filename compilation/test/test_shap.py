from sklearn.metrics import mean_squared_error
from catboost import Pool, CatBoostRegressor
import json
import os

from random import random

import shap_calc
import shap_dependent
import shap_dependent_new
import unification
import compilation
import models_train


def test_binarization():
    y_train, y_test, X_train, X_test = models_train.load_data_features_txt()
    # model = models_train.train_xgboost_model(y_train, X_train)
    # y_pred = model.predict(X_test)
    print(X_train[:5])
    xs_binarized, medians = unification.binarize(X_train)
    # print(xs_binarized)
    # print(medians)
    print(X_train[0])
    print(xs_binarized[0])
    for x in xs_binarized[:100]:
        for x1 in xs_binarized[:100]:
            if x != x1:
                k = 0
                for i in range(len(x)):
                    if x[i] == x1[i]:
                        k += 1
                if k > 35:
                    print(k, x, x1)


def calc_diff(shap_my, shap, k):
    for i in range(k):
        print("shap catboost =", shap[i])
        print("my shap =", shap_my[i])
        diff = 0
        diff_rel = 0
        for j in range(len(shap_my[i])):
            diff += abs(shap[i][j] - shap_my[i][j])
            if max(abs(shap[i][j]), abs(shap[i][j])) > 0:
                diff_rel += abs(shap[i][j] - shap_my[i][j]) / max(abs(shap[i][j]), abs(shap_my[i][j]))
        print("diff =", diff / len(shap_my[0]))
        print("rel diff =", diff_rel / len(shap_my[0]))


def test_shap():
    y_train, y_test, X_train, X_test = models_train.load_data()
    train_pool = Pool(X_train, y_train)
    model = CatBoostRegressor(n_estimators=100, depth=4, learning_rate=1, loss_function='RMSE')
    model.fit(train_pool)
    test_pool = Pool(X_test)
    y_pred = model.predict(train_pool)
    rmse0 = mean_squared_error(y_train, y_pred) ** 0.5
    print('rmse of catboost model:', rmse0)
    xs_binarized, medians = unification.binarize(X_train)
    shap = model.get_feature_importance(data=train_pool, fstr_type='ShapValues')

    k = 2
    shap_my = shap_dependent.calc_shap_dependent_fast(xs_binarized, y_pred, xs_binarized[:k])
    calc_diff(shap_my, shap, k)
    shap_my1 = shap_dependent_new.calc_shap_dependent_fast(xs_binarized, y_pred, xs_binarized[:k])
    calc_diff(shap_my1, shap, k)
    m = len(X_train[0])
    shap_my2 = []
    for i in range(k):
        shap_my2.append([])
        c = (y_pred[i] - shap[i][m]) / sum(shap_my1[i])
        print("c =", c)
        if c < 0 or c > 100:
            c = 1
        for j in range(m):
            shap_my2[i].append(shap_my1[i][j] * c)

    calc_diff(shap_my2, shap, k)
