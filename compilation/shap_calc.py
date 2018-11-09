import numpy as np


def change_in_s(x1, x, s):
    m = len(x)
    x_new = x1.copy()
    for i in range(m):
        if (s & (1 << i)) > 0:
            x_new[i] = x[i]
    return x_new


def calc_loss_in_point_for_s(model, xs, ys, loss_f, x_id, s):
    res = 0.
    x = xs[x_id]
    y = ys[x_id]
    for x1 in xs:
        x1_changed = change_in_s(x1, x, s)
        y_pred = model.predict([x1_changed])[0]
        res += loss_f(y, y_pred)
    return res / len(xs)


def calc_shap_loss_in_point(model, f_id, xs, ys, loss_f, x_id):
    m = len(xs[0])

    fact = np.ones(m + 1)
    for i in range(1, m + 1):
        fact[i] = fact[i - 1] * i

    res = 0.
    diffs = np.zeros(1 << m)
    for s in range(1 << m):
        if (s & (1 << f_id)) == 0:
            s_size = bin(s).count('1')
            c = fact[s_size] * fact[m - s_size - 1] / fact[m]
            d = c * (calc_loss_in_point_for_s(model, xs, ys, loss_f, x_id, s | (1 << f_id))
                     - calc_loss_in_point_for_s(model, xs, ys, loss_f, x_id, s))
            res += d
            diffs[s] = d
    return res, diffs


def calc_shap_loss(model, xs, ys, loss_f):
    m = len(xs[0])
    res = np.zeros(m)
    diffs = np.zeros((m, 1 << m))
    for f_id in range(m):
        for x_id in range(len(xs)):
            p, diffs_x = calc_shap_loss_in_point(model, f_id, xs, ys, loss_f, x_id)
            # print(i, "+", p)
            res[f_id] += p
            diffs[f_id] += diffs_x

        res[f_id] /= len(xs)

    return res, diffs


def calc_shap_rmse_subsets_slow(model, f_id, x, x1):
    m = len(x)
    fact = np.ones(m + 1)
    for i in range(1, m + 1):
        fact[i] = fact[i - 1] * i

    res = 0.
    for s in range(1 << m):
        if (s & (1 << f_id)) == 0:
            s_size = bin(s).count('1')
            c = fact[s_size] * fact[m - s_size - 1] / fact[m]
            f1 = model.predict([change_in_s(x1, x, s | (1 << f_id))])[0]
            f2 = model.predict([change_in_s(x1, x, s)])[0]
            d = c * (f1 - f2)
            res += d

    return res


def calc_shap_rmse_subsets_poly(poly, f_id, x, x1):
    m = len(x)
    fact = np.ones(m + 1)
    for i in range(1, m + 1):
        fact[i] = fact[i - 1] * i

    res = 0.
    for mon, val in poly.items():
        monomial = dict(mon)
        cnt = [[0, 0], [0, 0]]
        for feature_id, thr in monomial.items():
            if feature_id != f_id:
                c1 = 1 if x[feature_id] < thr else 0
                c2 = 1 if x1[feature_id] < thr else 0
                cnt[c1][c2] += 1
        if cnt[0][0] > 0:
            continue
        for s_size in range(cnt[1][0], m - cnt[0][1]):
            c = fact[s_size] * fact[m - s_size - 1] / fact[m]
            if f_id in monomial and x[f_id] < monomial[f_id]:
                res += c * val * fact[m - cnt[0][1] - cnt[1][0] - 1] / fact[s_size - cnt[1][0]] \
                       / fact[m - cnt[0][1] - s_size - 1]
            if f_id in monomial and x1[f_id] < monomial[f_id]:
                res -= c * val * fact[m - cnt[0][1] - cnt[1][0] - 1] / fact[s_size - cnt[1][0]] \
                       / fact[m - cnt[0][1] - s_size - 1]
    return res


def calc_shap_rmse(poly, xs, ys):
    m = len(xs[0])
    res = np.zeros(m)
    for f_id in range(m):
        for x_id in range(len(xs)):
            for x1_id in range(x_id, len(xs)):
                p = -2 * (ys[x_id] - ys[x1_id]) * calc_shap_rmse_subsets_poly(poly, f_id, xs[x_id], xs[x1_id])
                res[f_id] += p
        res[f_id] /= len(xs) ** 2

    return res


def calc_shap_rmse_slow(poly, xs, ys):
    m = len(xs[0])
    res = np.zeros(m)
    for f_id in range(m):
        for x_id in range(len(xs)):
            for x1_id in range(x_id, len(xs)):
                p = -2 * (ys[x_id] - ys[x1_id]) * calc_shap_rmse_subsets_slow(poly, f_id, xs[x_id], xs[x1_id])
                res[f_id] += p
        res[f_id] /= len(xs) ** 2

    return res
