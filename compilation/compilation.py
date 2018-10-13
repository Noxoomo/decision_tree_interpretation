import json


def tree_to_polynomial(tree):
    poly = {}
    if 'leaf_value' in tree:
        poly[frozenset({}.items())] = tree['leaf_value']
    else:
        feature_id = tree['split_feature']
        thr = tree['threshold']
        left_poly = tree_to_polynomial(tree['left_child'])
        right_poly = tree_to_polynomial(tree['right_child'])
        poly = right_poly.copy()

        for monomial, val in left_poly.items():
            new_monomial = dict(monomial)
            new_monomial[feature_id] = min(thr, new_monomial[feature_id]) if feature_id in new_monomial else thr
            key = frozenset(new_monomial.items())
            if key in poly:
                poly[key] += val
            else:
                poly[key] = val

        for monomial, val in right_poly.items():
            new_monomial = dict(monomial)
            new_monomial[feature_id] = min(thr, new_monomial[feature_id]) if feature_id in new_monomial else thr
            key = frozenset(new_monomial.items())
            if key in poly:
                poly[key] -= val
            else:
                poly[key] = -val
    return poly


def tree_ensemble_to_polynomial(ensemble):
    poly_res = {}
    for tree in ensemble:
        poly = tree_to_polynomial(tree)
        for monomial, val in poly.items():
            if monomial in poly_res:
                poly_res[monomial] += val
            else:
                poly_res[monomial] = val
    return poly_res


def tree_predict_one(tree, x):
    if 'leaf_value' in tree:
        return tree['leaf_value']
    else:
        if x[tree['split_feature']] < tree['threshold']:
            return tree_predict_one(tree['left_child'], x)
        else:
            return tree_predict_one(tree['right_child'], x)


def trees_predict(trees, X_test, f):
    ys = []
    for x in X_test:
        ys.append(0.)
        for tree in trees:
            ys[-1] += tree_predict_one(tree, x)
        ys[-1] = f(ys[-1])
    return ys


def monomial_predict_one(monomial, x):
    for feature_id, thr in monomial.items():
        if x[feature_id] >= thr:
            return 0
    return 1


def poly_predict(poly, X_test, f):
    ys = []
    for x in X_test:
        ys.append(0.)
        for monomial, val in poly.items():
            ys[-1] += monomial_predict_one(dict(monomial), x) * val
        ys[-1] = f(ys[-1])
    return ys


def polynomial_to_json(poly):
    res = {}
    for monomial, val in poly.items():
        res[str(monomial)] = val
    return res


def save_poly(poly, file_name):
    with open(file_name, 'w') as outfile:
        json.dump(polynomial_to_json(poly), outfile, indent=2)
