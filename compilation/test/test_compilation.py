import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from catboost import Pool, CatBoostRegressor
from xgboost import XGBRegressor
import json

import decompilation
import unification
import compilation


import models_train


def test_lgb_ensemble_compilation():
    y_train, y_test, X_train, X_test = models_train.load_data()

    lgb_model = models_train.train_lgb_model(y_train, X_train)
    y_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    rmse0 = mean_squared_error(y_test, y_pred) ** 0.5
    print('rmse of lgb model:', rmse0)
    assert (rmse0 < 0.5)

    json_model_lgb = lgb_model.dump_model()
    lgb_model_unified = unification.unify_lgb_ensemble(json_model_lgb)
    ys = compilation.trees_predict(lgb_model_unified, X_test, lambda x: x)
    rmse1 = mean_squared_error(y_test, ys) ** 0.5
    print('rmse of lgb unified:', rmse1)
    assert (abs(rmse0 - rmse1) < 0.01)

    poly_lgb = compilation.tree_ensemble_to_polynomial(lgb_model_unified)
    ys = compilation.poly_predict(poly_lgb, X_test, lambda x: x)
    rmse2 = mean_squared_error(y_test, ys) ** 0.5
    print('rmse of polynomial from lgb model:', rmse2)
    assert (abs(rmse0 - rmse2) < 0.01)


def test_xgb_ensemble_compilation():
    y_train, y_test, X_train, X_test = models_train.load_data()

    model = XGBRegressor(n_estimators=10, learning_rate=0.02, max_depth=5, eta=1, subsample=0.8, reg_lambda=0,
                         reg_alpha=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse0 = mean_squared_error(y_test, y_pred) ** 0.5
    print("rmse of xgb =", rmse0)
    xgb_model = model.get_booster()
    models_train.save_xgb_model(xgb_model)
    json_model_xgb = xgb_model.get_dump(dump_format='json')
    xgb_model_unified = unification.unify_xgb_ensemble(json_model_xgb)
    with open('data/models/model_xgb_unified1.json', 'w') as outfile:
        json.dump(xgb_model_unified, outfile, indent=2)

    ys = compilation.trees_predict(xgb_model_unified, X_test, lambda x: x + 0.5)

    rmse1 = mean_squared_error(y_test, ys) ** 0.5
    print('rmse of xgb unified:', rmse1)
    assert (abs(rmse0 - rmse1) < 0.01)

    poly_xgb = compilation.tree_ensemble_to_polynomial(xgb_model_unified)
    ys = compilation.poly_predict(poly_xgb, X_test, lambda x: x + 0.5)
    rmse2 = mean_squared_error(y_test, ys) ** 0.5
    print('rmse of polynomial from xgb model:', rmse2)
    compilation.save_poly(poly_xgb, 'data/models/polynomial_xgb1.json')
    assert (abs(rmse0 - rmse2) < 0.01)


def test_catboost_ensemble_compilation():
    y_train, y_test, X_train, X_test = models_train.load_data()

    catboost_model = models_train.train_catboost_model(y_train, X_train)
    test_pool = Pool(X_test)
    y_pred = catboost_model.predict(test_pool)
    rmse0 = mean_squared_error(y_test, y_pred) ** 0.5
    print('rmse of catboost model:', rmse0)

    models_train.save_catboost_model(catboost_model)
    with open('./data/models/model_catboost.json') as f:
        json_model_catboost = json.load(f)
        catboost_model_unified = unification.unify_catboost_ensemble(json_model_catboost)
        with open('./data/models/model_catboost_unified.json', 'w') as outfile:
            json.dump(catboost_model_unified, outfile, indent=2)
        # ys = compilation.trees_predict(catboost_model_unified, X_test, lambda x: x)
        # rmse1 = mean_squared_error(y_test, ys) ** 0.5
        # print('rmse of catboost unified:', rmse1)
        # assert (abs(rmse0 - rmse1) < 0.01)

        poly_catboost = compilation.tree_ensemble_to_polynomial(catboost_model_unified)
        compilation.save_poly(poly_catboost, './data/models/model_catboost_poly.json')
        # ys = compilation.poly_predict(poly_catboost, X_test, lambda x: x)
        # rmse2 = mean_squared_error(y_test, ys) ** 0.5
        # print('rmse of polynomial from catboost model:', rmse2)
        # assert (abs(rmse0 - rmse2) < 0.01)

        model_decompiled = decompilation.polynomial_to_ensemble_greedy(poly_catboost)

        with open('data/models/model_catboost_decompiled.json', 'w') as outfile:
            json.dump(model_decompiled, outfile, indent=2)

        print("number of catboost trees =", len(catboost_model_unified))
        print("number of monomials =", len(poly_catboost))
        print("number of decompiled oblivious trees =", len(model_decompiled))


def test_compilation_simple_trees():
    tree1 = {"split_feature": 1,
             "threshold": 1,
             "left_child":
                 {"leaf_value": 1},
             "right_child":
                 {"leaf_value": 2}
             }
    tree2 = {"split_feature": 1,
             "threshold": 1.5,
             "left_child":
                 {"leaf_value": 0.5},
             "right_child":
                 {"leaf_value": 3}
             }
    tree = {"split_feature": 0,
            "threshold": 0.5,
            "left_child":
                tree1,
            "right_child":
                tree2}
    poly1 = compilation.tree_to_polynomial(tree1)
    assert (poly1 == {frozenset(): 2,
                      frozenset({(1, 1)}): -1})
    poly2 = compilation.tree_to_polynomial(tree2)
    assert (poly2 == {frozenset(): 3,
                      frozenset({(1, 1.5)}): -2.5})

    poly = compilation.tree_to_polynomial(tree)
    assert (poly == {frozenset(): 3,
                     frozenset({(0, 0.5)}): -1,
                     frozenset({(1, 1.5)}): -2.5,
                     frozenset({(0, 0.5), (1, 1)}): -1,
                     frozenset({(0, 0.5), (1, 1.5)}): 2.5})

    ensemble = [tree, tree, tree]
    poly_ensemble = compilation.tree_ensemble_to_polynomial(ensemble)
    assert (poly_ensemble == {frozenset(): 9,
                              frozenset({(0, 0.5)}): -3,
                              frozenset({(1, 1.5)}): -7.5,
                              frozenset({(0, 0.5), (1, 1)}): -3,
                              frozenset({(0, 0.5), (1, 1.5)}): 7.5})

    ensemble2 = [tree, tree1, tree2]
    poly_ensemble2 = compilation.tree_ensemble_to_polynomial(ensemble2)
    assert (poly_ensemble2 == {frozenset(): 8,
                               frozenset({(0, 0.5)}): -1,
                               frozenset({(1, 1)}): -1,
                               frozenset({(1, 1.5)}): -5.0,
                               frozenset({(0, 0.5), (1, 1)}): -1,
                               frozenset({(0, 0.5), (1, 1.5)}): 2.5})


def test_features_txt_xgboost():
    y_train, y_test, X_train, X_test = models_train.load_data_features_txt()

    model = XGBRegressor(n_estimators=100, learning_rate=0.02, max_depth=5, eta=1, subsample=0.8, reg_lambda=0,
                         reg_alpha=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse0 = mean_squared_error(y_test, y_pred) ** 0.5
    print("rmse features.txt =", rmse0)
    xgb_model = model.get_booster()
    models_train.save_xgb_model(xgb_model)
    json_model_xgb = xgb_model.get_dump(dump_format='json')
    xgb_model_unified = unification.unify_xgb_ensemble(json_model_xgb)
    with open('data/models/model_xgb_unified.json', 'w') as outfile:
        json.dump(xgb_model_unified, outfile, indent=2)

    # ys = compilation.trees_predict(xgb_model_unified, X_test, lambda x: x + 0.5)
    # rmse1 = mean_squared_error(y_test, ys) ** 0.5
    # print('rmse of xgb unified:', rmse1)
    # assert (abs(rmse0 - rmse1) < 0.01)

    poly_xgb = compilation.tree_ensemble_to_polynomial(xgb_model_unified)
    compilation.save_poly(poly_xgb, 'data/models/polynomial_xgb.json')
    # ys = compilation.poly_predict(poly_xgb, X_test, lambda x: x + 0.5)
    # rmse2 = mean_squared_error(y_test, ys) ** 0.5
    # print('rmse of polynomial from xgb model:', rmse2)
    # assert (abs(rmse0 - rmse2) < 0.01)

    model_decompiled = decompilation.polynomial_to_ensemble_greedy(poly_xgb)
    # ys = compilation.trees_predict(model_decompiled, X_test, lambda x: x + 0.5)
    # print("y_pred_decompiled =", ys)
    # diff = mean_squared_error(y_pred, ys)
    # print("diff =", diff)
    # assert (diff < 0.01)

    with open('data/models/model_xgb_features_txt_decompiled.json', 'w') as outfile:
        json.dump(model_decompiled, outfile, indent=2)

    print("number of xgboost trees =", len(xgb_model_unified))
    print("number of monomials =", len(poly_xgb))
    print("number of decompiled oblivious trees =", len(model_decompiled))


def test_decompilation_simple_trees():
    poly1 = {frozenset(): 2, frozenset({(1, 1)}): -1}
    tree1_expected = {"split_feature": 1,
                      "threshold": 1,
                      "left_child":
                          {"leaf_value": 1},
                      "right_child":
                          {"leaf_value": 2}
                      }
    ensemble1 = decompilation.polynomial_to_ensemble_greedy(poly1)
    assert (len(ensemble1) == 1)
    tree1 = ensemble1[0]
    # print(tree1)
    assert (tree1['split_feature'] == tree1_expected['split_feature'])
    assert (tree1['threshold'] == tree1_expected['threshold'])
    assert (tree1['threshold'] == tree1_expected['threshold'])
    assert (tree1['left_child']['leaf_value'] == tree1_expected['left_child']['leaf_value'])
    assert (tree1['right_child']['leaf_value'] == tree1_expected['right_child']['leaf_value'])


def test_decompilation():
    y_train, y_test, X_train, X_test = models_train.load_data()
    model = models_train.train_xgboost_model(y_train, X_train)

    y_pred = model.predict(X_test)
    # print("y_pred =", y_pred)
    xgb_model = model.get_booster()
    json_model_xgb = xgb_model.get_dump(dump_format='json')
    xgb_model_unified = unification.unify_xgb_ensemble(json_model_xgb)

    poly_xgb = compilation.tree_ensemble_to_polynomial(xgb_model_unified)

    model_decompiled = decompilation.polynomial_to_ensemble_greedy(poly_xgb)
    ys = compilation.trees_predict(model_decompiled, X_test, lambda x: x + 0.5)
    # print("y_pred_decompiled =", ys)
    diff = mean_squared_error(y_pred, ys)
    print("diff =", diff)
    assert (diff < 0.01)

    with open('data/models/model_xgb_before.json', 'w') as outfile:
        json.dump(xgb_model_unified, outfile, indent=2)

    with open('data/models/model_xgb_after.json', 'w') as outfile:
        json.dump(model_decompiled, outfile, indent=2)

    print("number of trees before decompilation =", len(xgb_model_unified))
    print("number of oblivious trees after decompilation =", len(model_decompiled))
