from sklearn.metrics import mean_squared_error
from catboost import Pool, CatBoostRegressor
from random import random, randint, seed
import matplotlib.pyplot as plt
import numpy as np
import json

import compilation
import shap_calc
import shap_dependent
import shap_dependent_new
import unification


# alpha=0 -- independent
def gen_data_dependent_features(n, m, alpha=0.5):
    xs = []
    ys = []
    for i1 in range(n):
        # f = random()
        f = np.random.normal()
        x = [f]
        y = f
        for j1 in range(m - 1):
            # f = random()
            f = np.random.normal()
            x.append((1 - alpha) * f + alpha * x[0])
            y += f
        xs.append(x)
        ys.append(y / m)

    xs_bin, medians = unification.binarize(xs)
    return xs_bin[:n // 2], ys[:n // 2], xs_bin[n // 2:], ys[n // 2:]


def sample(X, y, k):
    X_sampled = []
    y_sampled = []
    n = len(X)
    for it in range(n * k):
        i = randint(0, n - 1)
        X_sampled.append(X[i])
        y_sampled.append(y[i])

    return X_sampled, y_sampled


def calc_diff(shap_my, shap):
    diff = 0
    diff_rel = 0
    for i in range(len(shap_my)):
        for j in range(len(shap_my[i])):
            diff += abs(shap[i][j] - shap_my[i][j])
            if max(abs(shap[i][j]), abs(shap[i][j])) > 0:
                diff_rel += abs(shap[i][j] - shap_my[i][j]) / max(abs(shap[i][j]), abs(shap_my[i][j]))
    return diff / len(shap_my) / len(shap_my[0]), diff_rel / len(shap_my) / len(shap_my[0])


if __name__ == "__main__":
    n = 100000
    m = 10
    k = 1
    CNT_ITERATIONS = 2

    seed(1)

    average_diff = []
    average_rel_diff = []
    for it in range(CNT_ITERATIONS):
        alpha = it / CNT_ITERATIONS
        X_train, y_train, X_test, y_test = gen_data_dependent_features(n, m, alpha)
        train_pool = Pool(X_train, y_train)
        # print("X = ", X_train)
        # print("y = ", y_train)
        model = CatBoostRegressor(n_estimators=100, depth=4, learning_rate=1, loss_function='RMSE')
        model.fit(train_pool)

        test_pool = Pool(X_test)
        y_pred = model.predict(test_pool)
        rmse0 = mean_squared_error(y_test, y_pred) ** 0.5
        print("alpha =", alpha)
        print('rmse of catboost model:', rmse0)

        shap = model.get_feature_importance(data=train_pool, fstr_type='ShapValues')
        print("shap catboost =", shap[:4])

        X_train_sampled, y_train_sampled = sample(X_train, y_train, k)
        train_pool_sampled = Pool(X_train_sampled, y_train_sampled)

        y_pred_train = model.predict(train_pool)
        y_pred_train_sampled = model.predict(train_pool_sampled)

        shap_my = shap_dependent.calc_shap_dependent_fast(X_train_sampled, y_pred_train_sampled, X_train[:4])
        print("my shap =", shap_my)
        diff, diff_rel = calc_diff(shap_my, shap)
        print("diff =", diff)
        print("rel diff =", diff_rel)

        shap_my1 = shap_dependent_new.calc_shap_dependent_fast(X_train_sampled, y_pred_train_sampled, X_train[:4])
        print("my shap improved =", shap_my1)
        diff, diff_rel = calc_diff(shap_my1, shap)
        print("diff1 =", diff)
        print("rel diff1 =", diff_rel)

        shap_my2 = []
        for i in range(4):
            shap_my2.append([])
            c = (y_pred_train[i] - shap[i][m]) / sum(shap_my1[i])
            # print("c =", c)
            if c < 0 or c > 10:
                c = 1
            for j in range(m):
                shap_my2[i].append(shap_my1[i][j] * c)

        print("my shap improved normalized =", shap_my2)
        diff, diff_rel = calc_diff(shap_my2, shap)
        print("diff2 =", diff)
        print("rel diff2 =", diff_rel)
        print()

        average_diff.append(diff)
        average_rel_diff.append(diff_rel)

    print(average_diff)
    print(average_rel_diff)

    plt.plot(average_diff)
    plt.legend()
    plt.xlabel('features dependence')
    plt.ylabel('shap average diff')
    plt.savefig('data/graphs/shap_diff.png')
    plt.close()

    plt.plot(average_rel_diff)
    plt.legend()
    plt.xlabel('features dependence')
    plt.ylabel('shap average relative diff')
    plt.savefig('data/graphs/shap_rel_diff.png')
    plt.close()
