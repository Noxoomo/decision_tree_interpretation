import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from catboost import Pool, CatBoostRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import json
import numpy as np

from random import random

import shap_calc
import unification
import compilation


def save_xgb_model(xgb_model):
    json_model = xgb_model.get_dump(dump_format='json')
    with open('./data/models/model_xgb.json', 'w') as outfile:
        json.dump(json_model, outfile)


def test_features_txt_xgboost():
    df_train = pd.read_csv('./data/regression/features.txt', header=None, sep='\t')
    df_test = pd.read_csv('./data/regression/featuresTest.txt', header=None, sep='\t')

    y_train = df_train[1].values
    y_test = df_test[1].values
    X_train = df_train.drop([0, 1, 2, 3], axis=1).values
    X_test = df_test.drop([0, 1, 2, 3], axis=1).values

    model = XGBRegressor(n_estimators=2, learning_rate=0.02, max_depth=5, eta=1, subsample=0.8, reg_lambda=0,
                         reg_alpha=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse0 = mean_squared_error(y_test, y_pred) ** 0.5
    print("rmse features.txt =", rmse0)
    xgb_model = model.get_booster()
    save_xgb_model(xgb_model)
    json_model_xgb = xgb_model.get_dump(dump_format='json')
    xgb_model_unified = unification.unify_xgb_ensemble(json_model_xgb)
    with open('./data/models/model_xgb_unified.json', 'w') as outfile:
        json.dump(xgb_model_unified, outfile, indent=2)

    ys = compilation.trees_predict(xgb_model_unified, X_test, lambda x: x + 0.5)
    rmse1 = mean_squared_error(y_test, ys) ** 0.5
    print('rmse of xgb unified:', rmse1)
    assert (abs(rmse0 - rmse1) < 0.01)

    poly_xgb = compilation.tree_ensemble_to_polynomial(xgb_model_unified)
    ys = compilation.poly_predict(poly_xgb, X_test, lambda x: x + 0.5)
    rmse2 = mean_squared_error(y_test, ys) ** 0.5
    print('rmse of polynomial from xgb model:', rmse2)
    compilation.save_poly(poly_xgb, 'data/models/polynomial_xgb.json')
    assert (abs(rmse0 - rmse2) < 0.01)


def gen_data(n, m):
    xs = []
    ys = []
    for i in range(n):
        x = []
        y = 0.
        for j in range(m):
            f = random()
            x.append(f)
            y += f
        xs.append(x)
        ys.append((y + 0.1) / (m + 0.1))

    return xs[:n // 2], ys[:n // 2], xs[n // 2:], ys[n // 2:]


def test_shap():
    n = 500
    m = 4
    X_train, y_train, X_test, y_test = gen_data(n, m)
    # print(X_train, y_train, X_test, y_test)
    train_pool = Pool(X_train, y_train)
    model = CatBoostRegressor(n_estimators=20, depth=5, learning_rate=1, loss_function='RMSE')
    model.fit(train_pool)
    model.save_model("./data/models/model_catboost.json", format='json')

    test_pool = Pool(X_test)
    y_pred = model.predict(test_pool)
    rmse0 = mean_squared_error(y_test, y_pred) ** 0.5
    print('rmse of catboost model:', rmse0)

    # shap = model.get_feature_importance(data=train_pool, fstr_type='ShapValues')
    # print("x =", X_train)
    # print("shap =", shap)
    # print("shap average =", sum(shap) / len(shap))

    shap_my, diffs = shap_calc.calc_shap_loss(model, X_train, y_train, lambda y1, y2: (y1 - y2) ** 2)
    print('shap for loss =', shap_my)


    with open('./data/models/model_catboost.json') as f:
        json_model = json.load(f)
        model_unified = unification.unify_catboost_ensemble(json_model)
        poly = compilation.tree_ensemble_to_polynomial(model_unified)

        shap_rmse = shap_calc.calc_shap_rmse(poly, X_train, y_train)
        print('shap for rmse =', shap_rmse)

