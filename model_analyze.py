import numpy as np
from scipy.stats import ks_2samp
from itertools import combinations

from analyzer import extract_feats_from_pairs
from utils import Unimplementation


stages_1d = ['bidirectional_GRU_stage',
             'transformer_encoder_stage',
             'simple_dense_stage',
             'conformer_encoder_stage']


def is_1d(block):
    return block in stages_1d


def get_block_keys(config):
    return sorted([key for key in config.keys()
                   if key.startswith('BLOCK') and not key.endswith('ARGS')])


def count_blocks(config, criteria=is_1d):
    keys = get_block_keys(config)
    return sum([criteria(config[key]) for key in keys])


def get_ks_test_values(values, perfs, min_samples=1, a=0.05, verbose=False):
    n_values = len(values)
    comb = list(combinations(range(n_values), 2))
    pvalues = [[] for _ in range(n_values)]

    for j, k in comb:
        if len(perfs[j]) >= min_samples and len(perfs[k]) >= min_samples:
            pvalue = ks_2samp(perfs[j], perfs[k]).pvalue
            pvalues[j].append(pvalue)
            pvalues[k].append(pvalue)

            if verbose:
                print(f'{values[j]}({len(perfs[j])})    vs    '
                      f'{values[k]}({len(perfs[k])}): {pvalue:.5f}')

    if verbose:
        print()
    return pvalues


def extract_feats_from_pairs(pairs):
    feats = {}
    for pair in pairs:
        c = pair['config']
        for key in c.keys():
            if isinstance(c[key], dict):
                if key in feats:
                    feats[key] = [
                        feats[key][0].intersection(set(c[key].keys()))]
                else:
                    feats[key] = [set(c[key].keys())]
            else:
                if key in feats:
                    feats[key] = feats[key].union([c[key]])
                else:
                    feats[key] = set([c[key]])

    # features from *_ARGS
    keys = tuple(feats.keys())
    for key in keys:
        if not isinstance(feats[key], set):
            if len(feats[key][0]) > 0:
                for name in feats[key][0]:
                    new_name = f'{key}.{name}'
                    for pair in pairs:
                        value = pair['config'][key][name]
                        if isinstance(value, list):
                            value = str(value)
                        value = set([value])
                        if new_name in feats:
                            feats[new_name] = feats[new_name].union(value)
                        else:
                            feats[new_name] = value
            del feats[key]
    return feats


def analyzer(search_space, results, train_config):
    min_samples = train_config.min_samples
    keyword = 'objective_score'
    common_features = extract_feats_from_pairs(results)

    table = {feat: [] for feat in common_features}
    table[keyword] = []

    for result in results:
        perf = result['perf']

        for feat in common_features.keys():
            # find value
            if '.' in feat:
                front, end = feat.split('.')
                value = result['config'][front][end]
            else:
                value = result['config'][feat]
            if not isinstance(value, (int, float, str)):
                value = str(value)

            table[feat].append(value)

        score = perf[keyword]
        if isinstance(score, list):
            score = score[-1]
        table[keyword].append(score)

    table['count1d'] = [count_blocks(p['config']) for p in results]
    
    total = [v for k, v in common_features.items() 
                if k.startswith('BLOCK') or k in ['SED', 'DOA']]
    stages = total[0]
    for s in total[1:]:
        stages = stages.union(s)

    for stage in stages:
        table[stage] = [count_blocks(p['config'], lambda p: p == stage)
                        for p in results]
    table = {k: np.array(v) for k, v in table.items()}
    scores = sorted(table[keyword]) # minimize score 이므로 key값 사용하지 않고 밑에 loop에서도 수정함
    frontier = [[]]
    criteria = np.inf
    for s0 in scores:
        if s0 < criteria:
            criteria = s0
            frontier[0].append(s0)

    result_table = {}
    for rv in table.keys():
        if rv == keyword:
            continue

        unique_values = sorted(np.unique(table[rv]))
        if len(unique_values) == 1:
            continue

        print(f'{rv}')
        perfs = [table[keyword][table[rv] == value]
                    for value in unique_values]
        pvalues = get_ks_test_values(
            unique_values, perfs, min_samples=train_config.min_samples, 
            a=0.05, verbose=train_config.verbose)
        n_samples = [len(p) for p in perfs]
        import pdb; pdb.set_trace()
        for i, pv in enumerate(pvalues):
            if len(pv) > 0:
                print(f'{unique_values[i]}: '
                        f'[{min(pv):.5f}, {max(pv):.5f}] '
                        f'({np.mean(pv):.5f}) '
                        f'n_samples={len(perfs[i])}, '
                        f'{keyword}(min={np.min(perfs[i]):.5f}, '
                        f'mean={np.mean(perfs[i]):.5f}, '
                        f'median={np.median(perfs[i]):.5f}, '
                        f'max={np.max(perfs[i]):.5f})')
        print()
