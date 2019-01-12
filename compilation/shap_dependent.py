import numpy as np

MAX_M = 60

fact = np.ones(MAX_M + 1)
for i1 in range(1, MAX_M + 1):
    fact[i1] = fact[i1 - 1] * i1


def __calc_intersection_features_ids(x, x1):
    res = []
    for i in range(len(x)):
        if x[i] == x1[i]:
            res.append(i)
    return res


def __do_for_all_subsets(s, f):
    for mask in range(1 << len(s)):
        s1 = set()
        for j in range(len(s)):
            if (mask & (1 << j)) > 0:
                s1.add(s[j])
        f(frozenset(s1))


def __add_subset(cnt, s):
    if s in cnt:
        cnt[s] += 1
    else:
        cnt[s] = 1


def __update_value(vals, cnt, y1, m, f_id, ids):
    s_size = len(ids)
    if f_id in ids:
        s_size -= 1
        c = fact[s_size] * fact[m - s_size - 1] / fact[m]
    else:
        ids_plus = set(ids)
        ids_plus.add(f_id)
        c = -fact[s_size] * fact[m - s_size - 1] / fact[m]
        if frozenset(ids_plus) not in cnt:
            c = 0

    vals[-1] += y1 * c / cnt[ids]


def __is_subset(s1, s):
    for x in s1:
        if x not in s:
            return False
    return True


def __calc_shap_dependent_in_point(xs, ys, x):
    cnt = {}
    # very_intersected_xs = []
    m = len(x)
    for x1 in xs:
        intersection = __calc_intersection_features_ids(x, x1)
        if len(intersection) > 20:
            continue
        __do_for_all_subsets(intersection, lambda ids: __add_subset(cnt, ids))

    res = []
    for f_id in range(m):
        res.append(0.)
        for i in range(len(xs)):
            x1 = xs[i]
            y1 = ys[i]
            intersection = __calc_intersection_features_ids(x, x1)
            if len(intersection) < 20:
                __do_for_all_subsets(intersection, lambda ids: __update_value(res, cnt, y1, m, f_id, ids))

    return res


def calc_shap_dependent_fast(xs, ys, xs_calc):
    return [__calc_shap_dependent_in_point(xs, ys, x) for x in xs_calc]
