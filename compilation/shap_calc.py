import numpy as np

MAX_M = 25

fact = np.ones(MAX_M + 1)
for i in range(1, MAX_M + 1):
    fact[i] = fact[i - 1] * i


def __change_in_s(x1, x, s):
    m = len(x)
    x_new = [x[i] if s & (1 << i) else x1[i] for i in range(m)]
    return x_new


def __calc_loss_in_point_for_s(model, xs, loss_f, x, y, s):
    res = 0.
    for x1 in xs:
        x1_changed = __change_in_s(x1, x, s)
        y_pred = model.predict([x1_changed])[0]
        res += loss_f(y, y_pred)
    return res / len(xs)


def __calc_shap_loss_in_point(model, f_id, xs, ys, loss_f, x_id):
    m = len(xs[0])

    res = 0.
    for s in range(1 << m):
        if (s & (1 << f_id)) == 0:
            s_size = bin(s).count('1')
            c = fact[s_size] * fact[m - s_size - 1] / fact[m]
            sum_s_plus_f = __calc_loss_in_point_for_s(model, xs, loss_f, xs[x_id], ys[x_id], s | (1 << f_id))
            sum_s = __calc_loss_in_point_for_s(model, xs, loss_f, xs[x_id], ys[x_id], s)
            diff = sum_s_plus_f - sum_s
            d = c * diff
            res += d
    return res


def calc_shap_loss(model, xs, ys, loss_f):
    m = len(xs[0])
    res = np.zeros(m)
    for f_id in range(m):
        for x_id in range(len(xs)):
            p = __calc_shap_loss_in_point(model, f_id, xs, ys, loss_f, x_id)
            res[f_id] += p

        res[f_id] /= len(xs)

    return res


def __calc_shap_rmse_subsets_slow(model, f_id, x, x1):
    m = len(x)
    fact = np.ones(m + 1)
    for i in range(1, m + 1):
        fact[i] = fact[i - 1] * i

    res = 0.
    for s in range(1 << m):
        if (s & (1 << f_id)) == 0:
            s_size = bin(s).count('1')
            c = fact[s_size] * fact[m - s_size - 1] / fact[m]
            f1 = model.predict([__change_in_s(x1, x, s | (1 << f_id))])[0]
            f2 = model.predict([__change_in_s(x1, x, s)])[0]
            d = c * (f1 - f2)
            res += d

    return res


def __calc_shap_rmse_subsets_poly(poly, f_id, x, x1):
    m = len(x)

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


def __calc_shap_rmse(model, xs, ys, subsets_calc):
    m = len(xs[0])
    res = np.zeros(m)
    for f_id in range(m):
        for x_id in range(len(xs)):
            for x1_id in range(x_id, len(xs)):
                p = -2 * (ys[x_id] - ys[x1_id]) * subsets_calc(model, f_id, xs[x_id], xs[x1_id])
                res[f_id] += p
        res[f_id] /= len(xs) ** 2

    return res


def calc_shap_rmse_poly(poly, xs, ys):
    return __calc_shap_rmse(poly, xs, ys, __calc_shap_rmse_subsets_poly)


def calc_shap_rmse_slow(model, xs, ys):
    return __calc_shap_rmse(model, xs, ys, __calc_shap_rmse_subsets_slow)


def __calc_f_in_point_for_s(xs, ys, x, s):
    cnt = 0
    sum_f = 0
    for k in range(len(xs)):
        if __change_in_s(x, xs[k], s) == x:
            cnt += 1
            sum_f += ys[k]
    return sum_f / cnt


def __calc_shap_in_point(f_id, xs, ys, x):
    m = len(xs[0])
    res = 0.
    for s in range(1 << m):
        if (s & (1 << f_id)) == 0:
            s_size = bin(s).count('1')
            c = fact[s_size] * fact[m - s_size - 1] / fact[m]
            sum_s_plus_f = __calc_f_in_point_for_s(xs, ys, x, s | (1 << f_id))
            sum_s = __calc_f_in_point_for_s(xs, ys, x, s)
            diff = sum_s_plus_f - sum_s
            d = c * diff
            res += d
    return res


def calc_shap_dependent(xs, ys):
    m = len(xs[0])
    res = []
    for x in xs:
        res.append(np.zeros(m))
        for f_id in range(m):
            res[-1][f_id] = __calc_shap_in_point(f_id, xs, ys, x)

    return res
