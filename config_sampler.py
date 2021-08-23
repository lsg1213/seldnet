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
            import pdb; pdb.set_trace()
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

    get_classifier_config(search_space, model_config, stage_name='SED')
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

    # SED, DOA part
    for model_config in model_configs:
        for SED, DOA in product(search_space[f'SED']['search_space_1d'].keys(), search_space[f'DOA']['search_space_1d'].keys()):
            model_config['SED'] = SED
            model_config['SED_ARGS'] = {}
            model_config['DOA'] = DOA
            model_config['DOA_ARGS'] = {}

            for key, value in search_space[f'SED']['search_space_1d'][SED].items():
                if isinstance(value[0], int):
                    model_config[f'SED_ARGS'][key] = max(value)
                elif isinstance(value[0], str):
                    model_config[f'SED_ARGS'][key] = value[-1]
                    
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


def conv_temporal_sampler(search_space_2d: dict, 
                          search_space_1d: dict,
                          n_blocks: int,
                          input_shape,
                          default_config=None,
                          config_postprocess_fn=None,
                          constraint=None):
    '''
    search_space_2d: modules with 2D outputs
    search_space_1d: modules with 1D outputs
    input_shape: (without batch dimension)
    default_config: the process will sample model config
                    starting from default_config
                    if not given, it will start from an
                    empty dict
    constraint: func(model_config) -> bool

    assume body parts can take 2D or 1D modules
    + sed, doa parts only take 1D modules
    '''
    search_space_sanity_check(search_space_2d)
    search_space_sanity_check(search_space_1d)

    search_space_total = copy.deepcopy(search_space_2d)
    search_space_total.update(search_space_1d)
    
    modules_2d = search_space_2d.keys()
    modules_1d = search_space_1d.keys()

    if default_config is None:
        default_config = {}

    count = 0
    while True:
        if (count % 10000) == 0:
            if len(modules_1d) == 0:
                n_2d = n_blocks
            else:
                n_2d = random.randint(0, n_blocks)

            if count != 0:
                print(f'{count}th iters. check constraint')
        count += 1

        # body parts
        model_config = copy.deepcopy(default_config)

        for i in range(n_blocks):
            pool = modules_2d if i < n_2d else modules_1d
            module = random.sample(pool, 1)[0]
            model_config[f'BLOCK{i}'] = module
            model_config[f'BLOCK{i}_ARGS'] = {
                k: random.sample(v, 1)[0]
                for k, v in search_space_total[module].items()}

        for head in ['SED', 'DOA']:
            module = random.sample(modules_1d, 1)[0]
            model_config[f'{head}'] = module
            model_config[f'{head}_ARGS'] = {
                k: random.sample(v, 1)[0]
                for k, v in search_space_total[module].items()}

        if config_postprocess_fn is not None:
            model_config = config_postprocess_fn(model_config)

        if constraint is None or constraint(model_config, input_shape):
            return model_config


def vad_architecture_sampler(search_space_2d: dict, 
                             search_space_1d: dict,
                             n_blocks: int,
                             input_shape,
                             default_config=None,
                             config_postprocess_fn=None,
                             constraint=None):
    search_space_sanity_check(search_space_2d)
    search_space_sanity_check(search_space_1d)

    search_space_total = copy.deepcopy(search_space_2d)
    search_space_total.update(search_space_1d)
    
    modules_2d = search_space_2d.keys()
    modules_1d = search_space_1d.keys()

    if default_config is None:
        default_config = {}

    count = 0
    while True:
        if (count % 10000) == 0:
            if len(modules_1d) == 0:
                n_2d = n_blocks
            else:
                n_2d = random.randint(0, n_blocks)

            if count != 0:
                print(f'{count}th iters. check constraint')
        count += 1

        model_config = copy.deepcopy(default_config)

        for i in range(n_blocks):
            pool = modules_2d if i < n_2d else modules_1d
            module = random.sample(pool, 1)[0]
            model_config[f'BLOCK{i}'] = module
            model_config[f'BLOCK{i}_ARGS'] = {
                k: random.sample(v, 1)[0]
                for k, v in search_space_total[module].items()}

        if config_postprocess_fn is not None:
            model_config = config_postprocess_fn(model_config)

        if constraint is None or constraint(model_config, input_shape):
            return model_config


# def search_space_sanity_check(search_space: dict):
#     for name in search_space:
#         # check whether each value is valid
#         for v in search_space[name].values():
#             if not isinstance(v, (list, tuple)):
#                 raise ValueError(f'values of {name} must be tuple or list')
#             if len(v) == 0:
#                 raise ValueError(f'len of value in {name} must be > 0')


def complexity(model_config: OrderedDict, 
               input_shape,
               mapping_dict: dict):
    block = None
    total_complexity = {} 

    for key in model_config.keys():
        if block is None:
            block = model_config[key]
        else:
            complexity, output_shape = mapping_dict[block](model_config[key], 
                                                           input_shape)
            total_complexity = dict_add(total_complexity, complexity)
            input_shape = output_shape
            block = None

    return total_complexity


if __name__ == '__main__':
    import complexity

    search_space_2d = {
        'simple_conv_block': 
            {'filters': [[16], [24], [32], [48], [64], [96], [128], [192], [256]], 
             'pool_size': [[[1, 1]], [[1, 2]], [[1, 4]]]},
        'another_conv_block': 
            {'filters': [16, 24, 32, 48, 64, 96, 128, 192, 256],
             'depth': [1, 2, 3, 4, 5, 6, 7, 8],
             'pool_size': [1, (1, 2), (1, 4)]},
        'res_basic_stage': 
            {'filters': [16, 24, 32, 48, 64, 96, 128, 192, 256],
             'depth': [1, 2, 3, 4, 5, 6, 7, 8],
             'strides': [1, (1, 2), (1, 4)],
             'groups': [1, 2, 4, 8, 16, 32, 64]},
        'res_bottleneck_stage': 
            {'filters': [16, 24, 32, 48, 64, 96, 128, 192, 256],
             'depth': [1, 2, 3, 4, 5, 6, 7, 8],
             'strides': [1, (1, 2), (1, 4)],
             'groups': [1, 2, 4, 8, 16, 32, 64],
             'bottleneck_ratio': [0.25, 0.5, 1, 2, 4, 8]},
        'dense_net_block': 
            {'growth_rate': [4, 6, 8, 12, 16, 24, 32, 48],
             'depth': [1, 2, 3, 4, 5, 6, 7, 8],
             'strides': [1, (1, 2), (1, 4)],
             'bottleneck_ratio': [0.25, 0.5, 1, 2, 4, 8],
             'reduction_ratio': [0.5, 1, 2]},
        'sepformer_block': 
            {'pos_encoding': [None, 'basic', 'rff'],
             'n_head': [1, 2, 4, 8],
             'ff_multiplier': [0.25, 0.5, 1, 2, 4, 8],
             'kernel_size': [1, 3]},
        'xception_basic_block':
            {'filters': [16, 24, 32, 48, 64, 96, 128, 192, 256],
             'strides': [(1, 2)],
             'mid_ratio': [1]},
        'identity_block': 
            {},
    }
    search_space_1d = {
        'bidirectional_GRU_block':
            {'units': [[16], [24], [32], [48], [64], [96], [128], [192], [256]]}, 
        'transformer_encoder_block':
            {'n_head': [1, 2, 4, 8],
             'ff_multiplier': [0.25, 0.5, 1, 2, 4, 8],
             'kernel_size': [1, 3]},
        'simple_dense_block':
            {'units': [[16], [24], [32], [48], [64], [96], [128], [192], [256]], 
             'dense_activation': [None, 'relu']},
    }

    def sample_constraint(min_flops=None, max_flops=None, 
                          min_params=None, max_params=None):
        # this contraint was designed for conv_temporal
        def _contraint(model_config, input_shape):
            def get_complexity(block_type):
                return getattr(complexity, f'{block_type}_complexity')

            shape = input_shape[-3:]
            total_cx = {}

            total_cx, shape = complexity.conv2d_complexity(
                shape, model_config['filters'], model_config['first_kernel_size'],
                padding='same', prev_cx=total_cx)
            total_cx, shape = complexity.norm_complexity(shape, prev_cx=total_cx)
            total_cx, shape = complexity.pool2d_complexity(
                shape, model_config['first_pool_size'], padding='same', 
                prev_cx=total_cx)

            # main body parts
            blocks = [b for b in model_config.keys()
                      if b.startswith('BLOCK') and not b.endswith('_ARGS')]
            blocks.sort()

            for block in blocks:
                # input shape check
                if model_config[block] not in search_space_1d and len(shape) != 3:
                    return False

                try:
                    cx, shape = get_complexity(model_config[block])(
                        model_config[f'{block}_ARGS'], shape)
                    total_cx = dict_add(total_cx, cx)
                except ValueError as e:
                    return False

            # sed + doa
            try:
                cx, sed_shape = get_complexity(model_config['SED'])(
                    model_config['SED_ARGS'], shape)
                cx, sed_shape = complexity.linear_complexity(
                    sed_shape, model_config['n_classes'], prev_cx=cx)
                total_cx = dict_add(total_cx, cx)

                cx, doa_shape = get_complexity(model_config['DOA'])(
                    model_config['DOA_ARGS'], shape)
                cx, doa_shape = complexity.linear_complexity(
                    doa_shape, 3*model_config['n_classes'], prev_cx=cx)
                total_cx = dict_add(total_cx, cx)
            except ValueError as e:
                return False

            # total complexity contraint
            if min_flops and total_cx['flops'] < min_flops:
                return False
            if max_flops and total_cx['flops'] > max_flops:
                return False
            if min_params and total_cx['params'] < min_params:
                return False
            if max_params and total_cx['params'] > max_params:
                return False
            return True
        return _contraint

    default_config = {
        'filters': 16,
        'first_kernel_size': 5,
        'first_pool_size': [5, 1],
        'n_classes': 14}

    input_shape = [300, 64, 4]
    min_flops, max_flops = 750_000_000, 1_333_333_333

    import models # for test
    import tensorflow.keras.backend as K

    for i in range(100):
        model_config = conv_temporal_sampler(
            search_space_2d,
            search_space_1d,
            n_blocks=4,
            input_shape=input_shape,
            default_config=default_config,
            constraint=sample_constraint(min_flops, max_flops))
        print(complexity.conv_temporal_complexity(model_config, input_shape))

        # for test
        model = models.conv_temporal(input_shape, model_config)
        print(model.output_shape, 
              sum([K.count_params(p) for p in model.trainable_weights]))

