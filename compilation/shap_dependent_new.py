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


def __update_value(vals, cnt_sets_by_size, cnt, y1, m, f_id, ids):
    s_size = len(ids)
    if f_id in ids:
        s_size -= 1
        c = fact[s_size] * fact[m - s_size - 1] / fact[m]
        cnt_sets_by_size[s_size].add(ids)
    else:
        ids_plus = set(ids)
        ids_plus.add(f_id)
        cnt_sets_by_size[s_size].add(frozenset(ids_plus))
        c = -fact[s_size] * fact[m - s_size - 1] / fact[m]
        if frozenset(ids_plus) not in cnt:
            c = 0

    vals[s_size] += y1 * c / cnt[ids]


def __is_subset(s1, s):
    for x in s1:
        if x not in s:
            return False
    return True


def __calc_shap_dependent_in_point(xs, ys, x):
    cnt = {}
    m = len(x)
    for x1 in xs:
        intersection = __calc_intersection_features_ids(x, x1)
        if len(intersection) < 20:
            __do_for_all_subsets(intersection, lambda ids: __add_subset(cnt, ids))

    res = []
    for f_id in range(m):
        res_s = [0. for x in range(m)]
        sets_by_size = [set() for x in range(m)]
        for i in range(len(xs)):
            x1 = xs[i]
            y1 = ys[i]
            intersection = __calc_intersection_features_ids(x, x1)
            if len(intersection) < 20:
                __do_for_all_subsets(intersection,
                                     lambda ids: __update_value(res_s, sets_by_size, cnt, y1, m, f_id, ids))
        res.append(0.)
        val_sum = 0.
        val_div = 0.
        for i in range(m):
            if len(sets_by_size[i]) > 0 and res_s[i] != 0:
                val_sum /= 2
                val_sum += res_s[i]
                val_div += len(sets_by_size[i])
                res[-1] += res_s[i] * fact[m - 1] / fact[i] / fact[m - i - 1] / len(sets_by_size[i])
            else:
                val_div *= 2
                res[-1] += val_sum / val_div * fact[m - 1] / fact[i] / fact[m - i - 1]
    return res


def calc_shap_dependent_fast(xs, ys, xs_calc):
    return [__calc_shap_dependent_in_point(xs, ys, x) for x in xs_calc]
