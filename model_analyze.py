import numpy as np
from scipy.stats import ks_2samp
from itertools import combinations

from analyzer import extract_feats_from_pairs
from utils import Unimplementation


stages_1d = ['bidirectional_GRU_stage',
             'transformer_encoder_stage',
             'simple_dense_stage',
             'conformer_encoder_stage']


def update_search_space(search_space, name, unit, mode='del'):
    name = name.split('.')
    if len(name) == 1:
        del search_space[name[0]][unit]
    elif len(name) == 2:
        del search_space[name[0]][name[1]][unit]
    else:
        raise ValueError(f'something wrong to name, {name}')

    # 제거되는 것들을 모두 기록할 json 만들기
    Unimplementation()


def narrow_search_space(search_space, table):
    '''
        table: [pvalue, min, mean, median, max], name, unit
    '''
    threshold = 1
    
    if len(table) == 0:
        return False, search_space

    table = sorted(table, key=lambda x: x[0][0]) # pvalue
    best = table[0]
    
    comparison = []
    for result in table:
        if best[-2] == result[-2] and best[-1] != result[-1]:
            comparison.append(result)
    
    import pdb; pdb.set_trace()
    low, high = 0, 0 # best의 score가 제일 높은 지 낮은 지 판단
    for case in comparison:
        if case[0][1] <= best[0][1] and case[0][3] <= best[0][3]: # min과 median 비교
            high += 1
        elif case[0][1] > best[0][1] and case[0][3] > best[0][3]:
            low += 1

    if low == high == 0: # 결정할만한 게 없는 경우
        return False, search_space

    # best보다 낮은 것이 있으면 best까지 제거 best보다 높은 것이 있으면 그것들 모두 제거
    Unimplementation()
            
    return True, search_space


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
                        value = pair['config'].get(key, None)
                        if value == None:
                            continue
                        value = value[name]
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
    interests = ['SED', 'DOA'] + [f'BLOCK{i}' for i in range(search_space['num1d'][-1] + search_space['num2d'][-1])]

    common_features = {}
    for k,v in extract_feats_from_pairs(results).items():
        for interest in interests:
            if interest in k:
                common_features[k] = v
                break
    # common_features = extract_feats_from_pairs(results)

    table = {feat: [] for feat in common_features}
    table[keyword] = []

    for result in results:
        perf = result['perf']

        for feat in common_features.keys():
            # find value
            if '.' in feat:
                front, end = feat.split('.')
                value = result['config'].get(front, {})
                value = value.get(end, None)
                if value == None:
                    continue
            else:
                value = result['config'].get(feat, None)
                if value == None:
                    continue
            if not isinstance(value, (int, float, str)):
                value = str(value)

            table[feat].append(value)

        score = perf[keyword]
        if isinstance(score, list):
            score = score[-1]
        table[keyword].append(score)

    # table['count1d'] = [count_blocks(p['config']) for p in results]
    
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

    result_table = []
    for rv in table.keys():
        if rv == keyword:
            continue

        unique_values = sorted(np.unique(table[rv]))
        if len(unique_values) == 1:
            continue

        perfs = [table[keyword][table[rv] == value]
                    for value in unique_values]
        pvalues = get_ks_test_values(
            unique_values, perfs, min_samples=train_config.min_samples, 
            a=train_config.threshold, verbose=train_config.verbose)
        n_samples = [len(p) for p in perfs]
        for i in range(len(pvalues)):
            result_table.append([
                [pvalues[i][0], np.min(perfs[i]), np.mean(perfs[i]), np.median(perfs[i]), np.max(perfs[i])], # [pvalue, min, mean, median, max]
                rv,
                unique_values[i]
            ])
    return result_table
