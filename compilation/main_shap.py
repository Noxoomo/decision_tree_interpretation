from sklearn.metrics import mean_squared_error
from catboost import Pool, CatBoostRegressor
from random import random
import matplotlib.pyplot as plt

import shap_calc


# alpha=0 -- independent
def gen_data_dependent_features(n, m, alpha=0.5):
    xs = []
    ys = []
    for i1 in range(n):
        f = random()
        x = [int(f * 2)]
        y = f
        for j1 in range(m - 1):
            f = random()
            x.append(int((1 - alpha) * 2 * f + alpha * x[0]))
            y += f
        xs.append(x)
        ys.append(y / m)

    return xs[:n // 2], ys[:n // 2], xs[n // 2:], ys[n // 2:]


if __name__ == "__main__":
    n = 2000
    m = 2
    CNT_ITERATIONS = 20

    average_diff = []
    average_rel_diff = []
    for it in range(CNT_ITERATIONS):
        alpha = it / CNT_ITERATIONS
        X_train, y_train, X_test, y_test = gen_data_dependent_features(n, m, alpha)
        train_pool = Pool(X_train, y_train)
        print(X_train)
        print(y_train)
        model = CatBoostRegressor(n_estimators=20, depth=4, learning_rate=1, loss_function='RMSE')
        model.fit(train_pool)

        test_pool = Pool(X_test)
        y_pred = model.predict(test_pool)
        rmse0 = mean_squared_error(y_test, y_pred) ** 0.5
        print("alpha =", alpha)
        print('rmse of catboost model:', rmse0)

        shap = model.get_feature_importance(data=train_pool, fstr_type='ShapValues')
        # print("shap catboost =", shap)

        y_pred_train = model.predict(train_pool)
        shap_my = shap_calc.calc_shap_dependent(X_train, y_pred_train)
        # print("my shap =", shap_my)
        diff = 0
        diff_rel = 0
        for i in range(len(shap_my)):
            for j in range(len(shap_my[i])):
                diff += abs(shap_my[i][j] - shap[i][j])
                if max(abs(shap_my[i][j]), abs(shap[i][j])) > 0:
                    diff_rel += abs(shap_my[i][j] - shap[i][j]) / max(abs(shap_my[i][j]), abs(shap[i][j]))

        average_diff.append(diff / len(shap_my) / len(shap_my[0]))
        average_rel_diff.append(diff_rel / len(shap_my) / len(shap_my[0]))
    print(average_diff)
    print(average_rel_diff)

    plt.plot(average_diff)
    plt.legend()
    plt.xlabel('features dependence')
    plt.ylabel('shap average diff')
    plt.savefig('data/graphs/shap_diff1.png')
    plt.close()

    plt.plot(average_rel_diff)
    plt.legend()
    plt.xlabel('features dependence')
    plt.ylabel('shap average relative diff')

    plt.savefig('data/graphs/shap_rel_diff1.png')
    plt.close()
