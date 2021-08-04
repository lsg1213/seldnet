import json

import numpy as np
from scipy.stats import ks_2samp
from itertools import combinations

from analyzer import extract_feats_from_pairs
from utils import Unimplementation, ValueErrorjson
from writer_manager import Writer
from modules import stages_1d, stages_2d


def delete_unit(search_space, name, unit, writer):
    check = False

    if isinstance(unit, str):
        unit = json.loads(unit)

    for dimension in search_space[name[0]].keys():
        for layer in search_space[name[0]][dimension].keys():
            if layer != 'num':
                if name[1] in search_space[name[0]][dimension][layer]:
                    check = True
                    if not (layer == 'mother_stage' and 'kernel_size' in name[1] and unit == 0):
                        search_space[name[0]][dimension][layer][name[1]].remove(unit)

    if check:
        return search_space
    raise ValueErrorjson(search_space, name, unit, msg='nothing was deleted!!', writer=writer)


def delete_stage(search_space, name, unit, writer):
    check = False
    
    for dimension in search_space[name[0]].keys():
        try:
            for layer in search_space[name[0]][dimension].keys():
                if unit == layer:
                    check = True
                    del search_space[name[0]][dimension][unit]
                    break
        except:
            import pdb; pdb.set_trace()

    if check:
        return search_space
    raise ValueErrorjson('nothing was deleted!!', writer, search_space, name, unit)


def update_search_space(search_space, name, unit, writer):
    name = name.replace('_ARGS', '') if '_ARGS' in name else name
    name = name.split('.')

    if len(name) == 1:
        search_space = delete_stage(search_space, name, unit, writer)
    elif len(name) == 2:
        search_space = delete_unit(search_space, name, unit, writer)
    else:
        raise ValueError(f'something wrong to name, {name}')


def stage_filter(name, unit):
    def _stage_filter(result):
        return result['config'][name[0]] != unit
    return _stage_filter
    

def unit_filter(name, unit):
    def _stage_filter(result):
        return result['config'][name[0]][name[1]] != unit
    return _stage_filter


def result_filtering(results, name, unit):
    name = name.split('.')
    if len(name) == 1:
        # stage filtering
        results = list(filter(stage_filter(name, unit), results))
    elif len(name) == 2:
        # unit filtering
        results = list(filter(unit_filter(name, unit), results))
    return results


def narrow_search_space(search_space, table, results, writer):
    '''
        table: [pvalue, min, mean, median, max], name, unit
    '''
    threshold = 1
    check = False
    
    if len(table) == 0:
        return False, search_space

    table = sorted(table, key=lambda x: x[0][0]) # pvalue
    best = table[0]
    
    comparison = []
    for result in table:
        if best[-2] == result[-2] and best[-1] != result[-1]:
            comparison.append(result)
    
    low, high = 0, 0 # best의 score가 제일 높은 지 낮은 지 판단, high는 score보다 best가 높은 것 개수, low는 score보다 best가 낮은 것 개수
    removed_case = []
    for case in comparison:
        if case[0][1] <= best[0][1] and case[0][3] <= best[0][3]: # min과 median 비교
            removed_case.append({
                'versus': f'{best[-2]}: {best[-1]} vs {case[-1]}',
                'min': f'{best[0][1]} vs {case[0][1]}',
                'median': f'{best[0][3]} vs {case[0][3]}',
                'result': f'{best[-1]} was deleted'
            })
            update_search_space(search_space, best[-2], best[-1], writer)
            results = result_filtering(results, best[-2], best[-1])
            check = True
        elif case[0][1] > best[0][1] and case[0][3] > best[0][3]:
            removed_case.append({
                'versus': f'{best[-2]}: {best[-1]} vs {case[-1]}',
                'min': f'{best[0][1]} vs {case[0][1]}',
                'median': f'{best[0][3]} vs {case[0][3]}',
                'result': f'{case[-1]} was deleted'
            })
            update_search_space(search_space, case[-2], case[-1], writer)
            results = result_filtering(results, case[-2], case[-1])
            check = True
    writer.dump(removed_case, f'removed_space_{writer.index}.json')
    
    return check, search_space, results


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
