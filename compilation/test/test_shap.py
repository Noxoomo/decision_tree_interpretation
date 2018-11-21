from sklearn.metrics import mean_squared_error
from catboost import Pool, CatBoostRegressor
import json
import os

from random import random

import shap_calc
import unification
import compilation


def save_xgb_model(xgb_model):
    json_model = xgb_model.get_dump(dump_format='json')
    with open('./data/models/model_xgb.json', 'w') as outfile:
        json.dump(json_model, outfile)


def gen_data_independent_features(n, m):
    return gen_data_dependent_features(n, m, 0)


# alpha=0 -- independent
def gen_data_dependent_features(n, m, alpha=0.5):
    xs = []
    ys = []
    for i in range(n):
        f = random()
        x = [int(f * 2)]
        y = f
        for j in range(m - 1):
            f = random()
            x.append(int((1 - alpha) * 2 * f + alpha * x[0]))
            y += f
        xs.append(x)
        ys.append(y / m)

    return xs[:n // 2], ys[:n // 2], xs[n // 2:], ys[n // 2:]


def test_shap_sum_zero():
    n = 200
    m = 3
    X_train, y_train, X_test, y_test = gen_data_independent_features(n, m)
    train_pool = Pool(X_train, y_train)
    model = CatBoostRegressor(n_estimators=20, depth=5, learning_rate=1, loss_function='RMSE')
    model.fit(train_pool)
    test_pool = Pool(X_test)
    y_pred = model.predict(test_pool)
    rmse0 = mean_squared_error(y_test, y_pred) ** 0.5
    print('rmse of catboost model:', rmse0)

    shap_loss_slow = shap_calc.calc_shap_loss(model, X_train, y_train, lambda y1, y2: y2)
    for shap_i in shap_loss_slow:
        assert (abs(shap_i) < 0.0001)


def test_shap_rmse_independent_features():
    n = 200
    m = 3
    X_train, y_train, X_test, y_test = gen_data_independent_features(n, m)
    train_pool = Pool(X_train, y_train)
    model = CatBoostRegressor(n_estimators=20, depth=5, learning_rate=1, loss_function='RMSE')
    model.fit(train_pool)
    os.makedirs("./data/models/", exist_ok=True)
    model.save_model("./data/models/model_catboost.json", format='json')
    test_pool = Pool(X_test)
    y_pred = model.predict(test_pool)
    rmse0 = mean_squared_error(y_test, y_pred) ** 0.5
    print('rmse of catboost model:', rmse0)

    shap_rmse_slow = shap_calc.calc_shap_rmse_slow(model, X_train, y_train)
    print('shap for rmse black box slow =', shap_rmse_slow)

    shap_rmse_slow2 = shap_calc.calc_shap_loss(model, X_train, y_train, lambda y1, y2: (y1 - y2) ** 2)

    with open('./data/models/model_catboost.json') as f:
        json_model = json.load(f)
        model_unified = unification.unify_catboost_ensemble(json_model)
        poly = compilation.tree_ensemble_to_polynomial(model_unified)

        shap_rmse = shap_calc.calc_shap_rmse_poly(poly, X_train, y_train)
        print('shap for rmse poly =', shap_rmse)

        for i in range(len(shap_rmse)):
            assert (abs(shap_rmse[i] - shap_rmse_slow[i]) < 0.0001)
            assert (abs(shap_rmse[i] - shap_rmse_slow2[i]) < 0.0001)


def test_shap_independent_features():
    n = 500
    m = 4
    X_train, y_train, X_test, y_test = gen_data_independent_features(n, m)
    train_pool = Pool(X_train, y_train)
    model = CatBoostRegressor(n_estimators=20, depth=5, learning_rate=1, loss_function='RMSE')
    model.fit(train_pool)
    os.makedirs("./data/models/", exist_ok=True)
    model.save_model("./data/models/model_catboost.json", format='json')

    test_pool = Pool(X_test)
    y_pred = model.predict(test_pool)
    rmse0 = mean_squared_error(y_test, y_pred) ** 0.5
    print('rmse of catboost model:', rmse0)

    shap = model.get_feature_importance(data=train_pool, fstr_type='ShapValues')
    print("shap catboost =", shap)

    y_pred_train = model.predict(train_pool)
    shap_my = shap_calc.calc_shap_dependent(X_train, y_pred_train)
    print("my shap =", shap_my)

    diff = 0
    for i in range(len(shap_my)):
        for j in range(len(shap_my[i])):
            assert (abs(shap_my[i][j] - shap[i][j]) < 0.05)
            diff += abs(shap_my[i][j] - shap[i][j])
    print("average_diff =", diff / len(shap_my) / len(shap_my[0]))


def test_shap_dependent_features():
    n = 500
    m = 4
    X_train, y_train, X_test, y_test = gen_data_dependent_features(n, m)
    train_pool = Pool(X_train, y_train)
    model = CatBoostRegressor(n_estimators=20, depth=5, learning_rate=1, loss_function='RMSE')
    model.fit(train_pool)
    os.makedirs("./data/models/", exist_ok=True)
    model.save_model("./data/models/model_catboost.json", format='json')

    test_pool = Pool(X_test)
    y_pred = model.predict(test_pool)
    rmse0 = mean_squared_error(y_test, y_pred) ** 0.5
    print('rmse of catboost model:', rmse0)

    shap = model.get_feature_importance(data=train_pool, fstr_type='ShapValues')
    print("shap catboost =", shap)

    y_pred_train = model.predict(train_pool)
    shap_my = shap_calc.calc_shap_dependent(X_train, y_pred_train)
    print("my shap =", shap_my)
    diff = 0
    for i in range(len(shap_my)):
        for j in range(len(shap_my[i])):
            assert (abs(shap_my[i][j] - shap[i][j]) < 0.5)
            diff += abs(shap_my[i][j] - shap[i][j])
    print("average_diff =", diff / len(shap_my) / len(shap_my[0]))
