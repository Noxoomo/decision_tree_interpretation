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
            res[f_id] += p
            diffs[f_id] += diffs_x

        res[f_id] /= len(xs)

    return res, diffs
