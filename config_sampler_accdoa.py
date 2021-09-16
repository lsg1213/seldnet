import copy
import random
from random import choice
from collections import OrderedDict
from itertools import product
from copy import deepcopy

from utils import *
from search_utils import search_space_sanity_check
from modules import stages_1d, stages_2d


def get_classifier_config(search_space, model_config, stage_name):
    sp = search_space[stage_name]['search_space_1d']
    model_config[stage_name] = choice([i for i in sp.keys() if i != 'num'])

    model_arg_config = {}
    for k, v in sp[model_config[stage_name]].items():
        v = choice(v)
        model_arg_config[k] = v

    model_config[stage_name + '_ARGS'] = model_arg_config


def mother_stage_constraint(search_space, name, model_arg_config):
    def mother_stage_checker(arg, idx, unit, mode='same'):
        arg_search_space = search_space[name]['search_space_2d']['mother_stage'][arg]
        check = True
        for i in arg_search_space:
            if mode == 'same':
                if i[idx] == unit:# 범위 안에 unit이 있는 케이스가 있으면 통과 unit인 케이스가 없으면 check True로 재뽑기
                    check = False
            elif mode == 'dif':
                if i[idx] != unit:
                    check = False
        return check
    
    # kernel size가 0인 경우는 나중에 analyze에서 예외처리 해주기
    if model_arg_config['filters0'] == 0:
        if mother_stage_checker('connect0', 0, 1):
            return True
        if mother_stage_checker('connect1', 1, 0):
            return True
        if mother_stage_checker('connect2', 1, 0):
            return True

    if model_arg_config['filters2'] == 0:
        if mother_stage_checker('connect2', 2, 1):
            return True
        
        if model_arg_config['filters0'] != 0:
            if mother_stage_checker('connect2', 1, 1):
                return True

        if model_arg_config['connect2'][2] == 0:
            if model_arg_config['filters1'] != 0:
                return True

            if model_arg_config['connect2'][1] == 0:
                if model_arg_config['filters0'] != 0:
                    return True
    return False


def get_block_config(search_space, model_config, stage_name='BLOCK'):
    num2d = choice(search_space['num2d'])
    num1d = choice(search_space['num1d'])
    max_block_num = max(search_space['num1d']) + max(search_space['num2d'])
    
    i = 0
    idx_1d, idx_2d = 0, 0
    while idx_1d + idx_2d < num1d + num2d:
        name = stage_name + str(i)
        sp = copy.deepcopy(search_space[name])
        
        if idx_2d < num2d:
            if len(sp['search_space_2d']) == 0:
                num2d -= 1
                continue
            model_config[name] = choice([i for i in sp['search_space_2d'].keys() if i != 'num'])

            model_arg_config = {}
            if model_config[name] == 'mother_stage':
                check = True # 조건을 다 만족할 때까지 새로 돌리기
                while check:
                    model_arg_config = {}
                    for k, v in sp['search_space_2d'][model_config[name]].items():
                        if k == 'filters1':
                            v = [i for i in v if i > 0]
                        v = choice(v)
                        model_arg_config[k] = v
                    check = mother_stage_constraint(search_space, name, model_arg_config)
            else:
                for k, v in sp['search_space_2d'][model_config[name]].items():
                    v = choice(v)
                    model_arg_config[k] = v
            idx_2d += 1
        elif idx_1d < num1d:
            model_config[name] = choice([i for i in sp['search_space_1d'].keys() if i != 'num'])

            model_arg_config = {}
            for k, v in sp['search_space_1d'][model_config[name]].items():
                v = choice(v)
                model_arg_config[k] = v
            idx_1d += 1

        model_config[name + '_ARGS'] = model_arg_config
        i += 1
    
    identities = range(num1d + num2d, max_block_num)
    for i in identities:
        model_config[f'BLOCK{i}'] = 'identity_block'
        model_config[f'BLOCK{i}_ARGS'] = {}


def get_config(train_config, search_space, input_shape, postprocess_fn=None):
    search_space_sanity_check(search_space)

    model_config = {
        'n_classes': train_config.n_classes
    }

    get_block_config(search_space, model_config)

    get_classifier_config(search_space, model_config, stage_name='DOA')

    if postprocess_fn:
        model_config = postprocess_fn(model_config)
    return model_config


def get_max_configs(train_config, search_space, input_shape, postprocess_fn):
    model_configs = []
    num2d = search_space['num2d']
    num1d = search_space['num1d']
    blocks = [i for i in search_space.keys() if not 'num' in i]

    queue = [(0, None)]
    # blocks
    while len(queue) > 0:
        current_block_num, model_config = queue[0]
        del queue[0]
        if model_config == None:
            model_config = {
                'n_classes': train_config.n_classes
            }
        elif len([i for i in model_config.keys() if i.startswith('BLOCK') and not '_ARGS' in i]) == num1d + num2d:
            model_configs.append(model_config)
            continue

        if current_block_num < num2d:
            for stage in stages_2d:
                tmp_config = deepcopy(model_config)
                
                tmp_config[f'BLOCK{current_block_num}'] = stage
                tmp_config[f'BLOCK{current_block_num}_ARGS'] = {}
                for key, value in search_space[f'BLOCK{current_block_num}']['search_space_2d'][stage].items():
                    if isinstance(value[0], (int, float)):
                        tmp_config[f'BLOCK{current_block_num}_ARGS'][key] = max(value)
                    elif isinstance(value[0], (list, tuple)):
                        if 'strides' in key:
                            tmp_config[f'BLOCK{current_block_num}_ARGS'][key] = sorted(value, key=lambda x: sum(x))[0]
                        else:
                            tmp_config[f'BLOCK{current_block_num}_ARGS'][key] = sorted(value, key=lambda x: sum(x))[-1]
                    elif isinstance(value[0], str):
                        tmp_config[f'BLOCK{current_block_num}_ARGS'][key] = value[-1]
                queue.append((current_block_num + 1, tmp_config))
        elif current_block_num < num1d + num2d:
            for stage in stages_1d:
                tmp_config = deepcopy(model_config)
                tmp_config[f'BLOCK{current_block_num}'] = stage
                tmp_config[f'BLOCK{current_block_num}_ARGS'] = {}
                for key, value in search_space[f'BLOCK{current_block_num}']['search_space_1d'][stage].items():
                    if isinstance(value[0], (int, float)):
                        tmp_config[f'BLOCK{current_block_num}_ARGS'][key] = max(value)
                    elif isinstance(value[0], str):
                        tmp_config[f'BLOCK{current_block_num}_ARGS'][key] = value[-1]
                queue.append((current_block_num + 1, tmp_config))

    # DOA part
    for model_config in model_configs:
        for DOA in search_space[f'DOA']['search_space_1d'].keys():
            model_config['DOA'] = DOA
            model_config['DOA_ARGS'] = {}

            for key, value in search_space[f'DOA']['search_space_1d'][DOA].items():
                if isinstance(value[0], int):
                    model_config[f'DOA_ARGS'][key] = max(value)
                elif isinstance(value[0], str):
                    model_config[f'DOA_ARGS'][key] = value[-1]
        if postprocess_fn:
            model_config = postprocess_fn(model_config)
    return model_configs

def config_sampling(search_space: OrderedDict):
    sample = copy.deepcopy(search_space)

    # key must be sorted first
    # block type must be sampled first and its arguments later
    for key in sample.keys():
        if not key.endswith('_ARGS'):
            sample[key] = random.sample(sample[key], 1)[0]
        else:
            block_type = key.replace('_ARGS', '')
            sample[key] = config_sampling(sample[key][sample[block_type]])

    return sample
