import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from catboost import Pool, CatBoostRegressor
from xgboost import XGBRegressor
import json


def train_lgb_model(y_train, X_train):
    lgb_train = lgb.Dataset(X_train, y_train)

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
                      num_boost_round=20)

    return model


def train_catboost_model(y_train, X_train):
    train_pool = Pool(X_train, y_train)

    model = CatBoostRegressor(n_estimators=100, depth=5, learning_rate=0.1, loss_function='RMSE')
    model.fit(train_pool)

    return model


def train_xgboost_model(y_train, X_train):
    model = XGBRegressor(n_estimators=50, learning_rate=0.02, max_depth=5, eta=1, subsample=0.8, reg_lambda=0,
                         reg_alpha=1)
    model.fit(X_train, y_train)

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


def load_data_amis():
    df = pd.read_csv('./data/regression/amis.csv', header=None, sep='\t')
    # df_test = pd.read_csv('./data/regression/featuresTest.txt', header=None, sep='\t')
    # print(df)
    l = len(df)
    y_train = df[0].values
    y_test = df[0].values
    X_train = df.drop(0, axis=1).values
    X_test = df.drop(0, axis=1).values
    print(X_train)
    print(y_train)
    return y_train, y_test, X_train, X_test


def load_data_features_txt():
    df_train = pd.read_csv('./data/regression/features.txt', header=None, sep='\t')
    df_test = pd.read_csv('./data/regression/featuresTest.txt', header=None, sep='\t')

    y_train = df_train[1].values
    y_test = df_test[1].values
    X_train = df_train.drop([0, 1, 2, 3], axis=1).values
    X_test = df_test.drop([0, 1, 2, 3], axis=1).values
    return y_train, y_test, X_train, X_test
