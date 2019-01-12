
def __find_best_subset_greedy(poly):
    features = {}
    for monomial in poly.keys():
        items = dict(monomial)
        # print(items, len(items))
        if len(items) > len(features):
            features = items
    return features


def __tree_on_features(features, poly):
    tree = {}
    if len(features) == 0:
        tree['leaf_value'] = sum(poly.values())
    else:
        f_id, thr = features.popitem()
        tree['split_feature'] = f_id
        tree['threshold'] = thr
        poly_right = {}
        for monomial, val in poly.items():
            monomial_dict = dict(monomial)
            if f_id not in monomial_dict:
                poly_right[monomial] = val

        tree['left_child'] = __tree_on_features(features, poly)
        tree['right_child'] = __tree_on_features(features, poly_right)
        features[f_id] = thr
    return tree


def __is_subtree(monomial, features):
    for feature_id, thr in monomial.items():
        if feature_id not in features or features[feature_id] != thr:
            return False
    return True


def __divide_poly(features, poly):
    poly_subset = {}
    poly_not_subset = {}
    for monomial, val in poly.items():
        if __is_subtree(dict(monomial), features):
            poly_subset[monomial] = val
        else:
            poly_not_subset[monomial] = val

    return poly_subset, poly_not_subset


def polynomial_to_ensemble_greedy(poly):
    ensemble = []
    while len(poly.items()) > 0:
        features = __find_best_subset_greedy(poly)
        poly_subset, poly = __divide_poly(features, poly)
        tree = __tree_on_features(features, poly_subset)
        ensemble.append(tree)

    return ensemble
