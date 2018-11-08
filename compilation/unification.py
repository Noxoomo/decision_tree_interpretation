import json


def unify_lgb_tree(tree):
    out_tree = {}
    if 'leaf_value' in tree is not None:
        out_tree['leaf_value'] = tree['leaf_value']
        out_tree['count'] = tree['leaf_count']
    else:
        out_tree['split_feature'] = tree['split_feature']
        out_tree['threshold'] = tree['threshold']
        out_tree['left_child'] = unify_lgb_tree(tree['left_child'])
        out_tree['right_child'] = unify_lgb_tree(tree['right_child'])
        out_tree['count'] = out_tree['left_child']['count'] + out_tree['right_child']['count']
    return out_tree


def unify_lgb_ensemble(lgb_json):
    return [unify_lgb_tree(tree['tree_structure']) for tree in lgb_json['tree_info']]


def unify_xgb_tree(tree):
    out_tree = {}
    if 'leaf' in tree is not None:
        out_tree['leaf_value'] = tree['leaf']
        out_tree['count'] = 1
    else:
        out_tree['split_feature'] = int(tree['split'][1:])
        out_tree['threshold'] = tree['split_condition']
        out_tree['left_child'] = unify_xgb_tree(tree['children'][0])
        out_tree['right_child'] = unify_xgb_tree(tree['children'][1])
        out_tree['count'] = out_tree['left_child']['count'] + out_tree['right_child']['count']
    return out_tree


def unify_xgb_ensemble(xgb_json):
    return [unify_xgb_tree(json.loads(tree)) for tree in xgb_json]


def unify_catboost_tree(tree):
    out_tree = {}
    if len(tree['leaf_values']) == 1:
        out_tree['leaf_value'] = tree['leaf_values'][0]
        out_tree['count'] = tree['leaf_weights'][0]
    else:
        out_tree['split_feature'] = tree['splits'][-1]['float_feature_index']
        out_tree['threshold'] = tree['splits'][-1]['border_id']

        sz = len(tree['leaf_weights'])
        left_tree = {'leaf_weights': tree['leaf_weights'][:sz // 2], 'leaf_values': tree['leaf_values'][:sz // 2],
                     'splits': tree['splits'][:-1]}
        right_tree = {'leaf_weights': tree['leaf_weights'][sz // 2:], 'leaf_values': tree['leaf_values'][sz // 2:],
                      'splits': tree['splits'][:-1]}

        out_tree['left_child'] = unify_catboost_tree(left_tree)
        out_tree['right_child'] = unify_catboost_tree(right_tree)
        out_tree['count'] = out_tree['left_child']['count'] + out_tree['right_child']['count']
    return out_tree


def unify_catboost_ensemble(catboost_json):
    return [unify_catboost_tree(tree) for tree in catboost_json['oblivious_trees']]


def set_nodes_weights_tree(tree, xs):
    if 'leaf_value' not in tree:
        xs_l = list(filter(lambda x: x[tree['split_feature']] < tree['threshold'], xs))
        xs_r = list(filter(lambda x: x[tree['split_feature']] >= tree['threshold'], xs))
        set_nodes_weights_tree(tree['left_child'], xs_l)
        set_nodes_weights_tree(tree['right_child'], xs_r)
        tree['count'] = tree['left_child']['count'] + tree['right_child']['count']
    else:
        tree['count'] = len(xs)


def set_nodes_weights_ensemble(ensemble, xs):
    for tree in ensemble:
        set_nodes_weights_tree(tree, xs)
