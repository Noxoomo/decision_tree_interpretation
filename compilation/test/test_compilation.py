import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from catboost import Pool, CatBoostRegressor
import xgboost as xgb
import json
import unification
import compilation
import numpy as np


def train_lgb_model(y_train, y_test, X_train, X_test):
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'auc'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    model = lgb.train(params,
                      lgb_train,
                      num_boost_round=20,
                      valid_sets=lgb_eval,
                      early_stopping_rounds=5)

    return model


def train_xgb_model(y_train, y_test, X_train, X_test):
    xgdmat = xgb.DMatrix(X_train, y_train)
    params = {'eta': 0.1, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'objective': 'reg:logistic',
              'max_depth': 5, 'min_child_weight': 1}
    model = xgb.train(params, xgdmat)
    return model


def train_catboost_model(y_train, y_test, X_train, X_test):
    train_pool = Pool(X_train, y_train)

    model = CatBoostRegressor(iterations=10, depth=2, learning_rate=1, loss_function='RMSE')
    model.fit(train_pool)

    return model


def save_lgb_model(lgb_model):
    json_model = lgb_model.dump_model()
    with open('./data/models/model_lgb.json', 'w') as outfile:
        json.dump(json_model, outfile)


def save_xgb_model(xgb_model):
    json_model = xgb_model.get_dump(dump_format='json')
    with open('./data/models/model_xgb.json', 'w') as outfile:
        json.dump(json_model, outfile)


def save_catboost_model(catboost_model):
    catboost_model.save_model("./data/models/model_catboost.json", format='json')


def load_data():
    df_train = pd.read_csv('./data/regression/regression_train.csv', header=None, sep='\t')
    df_test = pd.read_csv('./data/regression/regression_test.csv', header=None, sep='\t')

    y_train = df_train[0].values
    y_test = df_test[0].values
    X_train = df_train.drop(0, axis=1).values
    X_test = df_test.drop(0, axis=1).values
    return y_train, y_test, X_train, X_test


def test_lgb_ensemble_compilation():
    y_train, y_test, X_train, X_test = load_data()

    lgb_model = train_lgb_model(y_train, y_test, X_train, X_test)
    y_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    print('rmse of lgb model:', mean_squared_error(y_test, y_pred) ** 0.5)
    rmse0 = mean_squared_error(y_test, y_pred) ** 0.5
    assert (rmse0 < 0.5)

    json_model_lgb = lgb_model.dump_model()
    lgb_model_unified = unification.unify_lgb_ensemble(json_model_lgb)
    ys = compilation.trees_predict(lgb_model_unified, X_test, lambda x: x)
    print('rmse of lgb unified:', mean_squared_error(y_test, ys) ** 0.5)
    rmse1 = mean_squared_error(y_test, ys) ** 0.5
    assert (abs(rmse0 - rmse1) < 0.01)

    poly_lgb = compilation.tree_ensemble_to_polynomial(lgb_model_unified)
    ys = compilation.poly_predict(poly_lgb, X_test, lambda x: x)
    print('rmse of polynomial from lgb model:', mean_squared_error(y_test, ys) ** 0.5)
    rmse2 = mean_squared_error(y_test, ys) ** 0.5
    assert (abs(rmse0 - rmse2) < 0.01)


def test_xgb_ensemble_compilation():
    y_train, y_test, X_train, X_test = load_data()

    xgb_model = train_xgb_model(y_train, y_test, X_train, X_test)
    test_dmat = xgb.DMatrix(X_test)
    y_pred = xgb_model.predict(test_dmat)
    print('rmse of xgb model:', mean_squared_error(y_test, y_pred) ** 0.5)
    rmse0 = mean_squared_error(y_test, y_pred) ** 0.5
    assert (rmse0 < 0.5)

    json_model_xgb = xgb_model.get_dump(dump_format='json')
    xgb_model_unified = unification.unify_xgb_ensemble(json_model_xgb)
    ys = compilation.trees_predict(xgb_model_unified, X_test, lambda x: 1 / (1 + np.exp(-x)))
    print('rmse of xgb unified:', mean_squared_error(y_test, ys) ** 0.5)
    rmse1 = mean_squared_error(y_test, ys) ** 0.5
    assert (abs(rmse0 - rmse1) < 0.01)

    poly_xgb = compilation.tree_ensemble_to_polynomial(xgb_model_unified)
    ys = compilation.poly_predict(poly_xgb, X_test, lambda x: 1 / (1 + np.exp(-x)))
    print('rmse of polynomial from xgb model:', mean_squared_error(y_test, ys) ** 0.5)
    rmse2 = mean_squared_error(y_test, ys) ** 0.5
    assert (abs(rmse0 - rmse2) < 0.01)


def test_catboost_ensemble_compilation():
    y_train, y_test, X_train, X_test = load_data()

    catboost_model = train_catboost_model(y_train, y_test, X_train, X_test)
    test_pool = Pool(X_test)
    y_pred = catboost_model.predict(test_pool)
    print('rmse of catboost model:', mean_squared_error(y_test, y_pred) ** 0.5)
    rmse0 = mean_squared_error(y_test, y_pred) ** 0.5
    assert (rmse0 < 0.5)
    save_catboost_model(catboost_model)
    with open('./data/models/model_catboost.json') as f:
        json_model_catboost = json.load(f)
        catboost_model_unified = unification.unify_catboost_ensemble(json_model_catboost)

        # print(catboost_model_unified)
        # print(catboost_model.get_dump())
        ys = compilation.trees_predict(catboost_model_unified, X_test, lambda x: x)
        # print(ys)
        # print(y_test)
        print('rmse of catboost unified:', mean_squared_error(y_test, ys) ** 0.5)
        rmse1 = mean_squared_error(y_test, ys) ** 0.5
        assert (abs(rmse0 - rmse1) < 0.01)
