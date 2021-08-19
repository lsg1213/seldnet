import argparse
import json
import os
import time
from copy import deepcopy
from itertools import product

import tensorflow as tf

from data_loader import *
from metrics import *
from transforms import *
from config_sampler import get_config
from search_utils import postprocess_fn
from model_flop import get_flops
from model_size import get_model_size
from model_analyze import analyzer, narrow_search_space
from writer_manager import Writer
from modules import stages_1d, stages_2d
import models


args = argparse.ArgumentParser()

# args.add_argument('--name', type=str, required=True,
#                   help='name must be {name}_{divided index} ex) 2021_1')
args.add_argument('--dataset_path', type=str, 
                  default='/root/datasets/DCASE2021/feat_label')
args.add_argument('--n_samples', type=int, default=500)
args.add_argument('--min_samples', type=int, default=32)
args.add_argument('--verbose', action='store_true')
args.add_argument('--threshold', type=float, default=0.05)

args.add_argument('--batch_size', type=int, default=256)
args.add_argument('--n_repeat', type=int, default=5)
args.add_argument('--epoch', type=int, default=12)
args.add_argument('--lr', type=float, default=1e-3)
args.add_argument('--n_classes', type=int, default=12)
args.add_argument('--gpus', type=str, default='-1')
args.add_argument('--config', action='store_true', help='if true, reuse config')
args.add_argument('--new', action='store_true')
args.add_argument('--multi', action='store_true')
args.add_argument('--score', action='store_true')

train_config = args.parse_args()


def get_search_space(target):
    search_space = {}
    blocks = [i for i in target.keys() if i.startswith('BLOCK') and not '_ARGS' in i]
    for block in blocks:
        search_space[block] = {}
        if target[block] in stages_1d:
            search_space[block]['search_space_1d'] = {}
            args = {}
            for k, v in target[block+'_ARGS'].items():
                if k == 'multiplier' or k == 'ff_multiplier':
                    v = float(v)
                if isinstance(v, float):
                    v = sorted(list(set([v * i / 20 for i in range(1, 11)])))
                elif isinstance(v, int):
                    v = sorted(list(set([int(v * i / 20) for i in range(1, 11)])))
                elif k == 'pos_encoding':
                    v = [None, 'basic', 'rff']
                args[k] = v
            search_space[block]['search_space_1d'][target[block]] = args
        elif target[block] in stages_2d:
            search_space[block]['search_space_2d'] = {}
            args = {}
            for k,v in target[block+'_ARGS'].items():
                if k == 'connect0':
                    v = [0,1]
                elif k == 'connect1':
                    v = list(map(list, product(range(2), range(2))))
                elif k == 'connect2':
                    v = list(map(list, product(range(2), range(2), range(2))))
                elif k == 'strides':
                    v = [(1, 1), (1, 2), (1, 3)]
                args[k] = v
            search_space[block]['search_space_2d'][target[block]] = args
        else:
            raise ValueError()
    return search_space


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = train_config.gpus
    writer = Writer(train_config)
    mirrored_strategy = tf.distribute.MirroredStrategy()
    if train_config.config:
        train_config = vars(train_config)
        train_config.update(writer.load(os.path.join(os.path.join('result', train_config['name']), 'train_config.json')))
        train_config = argparse.Namespace(**train_config)
    # if train_config.gpus != '-1':
    #     gpus = tf.config.experimental.list_physical_devices('GPU')
    #     print(gpus)
    #     if gpus:
    #         try:
    #             tf.config.experimental.set_virtual_device_configuration(
    #                 gpus[0],
    #                 [tf.config.experimental.VirtualDeviceConfiguration(
    #                     memory_limit=10240)])
    #         except RuntimeError as e:
    #             print(e)
    del train_config.gpus
    del train_config.config
    del train_config.new
    
    name = train_config.name
    if name.endswith('.json'):
        name = os.path.splitext(name)[0]

    input_shape = [300, 64, 7]
    with open('model_config/SS5.json', 'r') as f:
        target = json.load(f)

    if 'first_pool_size' in target.keys():
        del target['first_pool_size']

    
    
    
    # datasets
    if train_config.multi:
        with mirrored_strategy.scope():
            trainset = get_dataset(train_config, mode='train')
            valset = get_dataset(train_config, mode='val')
    else:
        trainset = get_dataset(train_config, mode='train')
        valset = get_dataset(train_config, mode='val')
    
    # Evaluator
    evaluator = SELDMetrics(doa_threshold=20, n_classes=train_config.n_classes)

    if not os.path.exists(writer.train_config_path):
        writer.train_config_dump()
    else:
        loaded_train_config = writer.train_config_load()
        tmp = deepcopy(vars(train_config))
        if 'multi' in tmp:
            del tmp['multi']
            
        if loaded_train_config != tmp:
            for k, v in tmp.items():
                print(k, ':', v)
            raise ValueError('train config doesn\'t match')

    while True:
        writer.index += 1 # 차수

        current_result_path = os.path.join(writer.result_path, f'result_{writer.index}.json')
        results = []

        # search space
        search_space_path = os.path.join(writer.result_path, f'search_space_{writer.index}.json')
        if os.path.exists(search_space_path):
            search_space = writer.load(search_space_path)
        elif writer.index == 1:
            search_space = get_search_space(target)
            writer.dump(search_space, search_space_path)
        else:
            writer.dump(search_space, search_space_path)

        while len(results) < train_config.n_samples:
            # resume
            if os.path.exists(current_result_path):
                results = writer.load(current_result_path)
            current_number = len(results)
            while True:
                model_config = get_config(train_config, search_space, input_shape=input_shape, postprocess_fn=postprocess_fn)
                # 학습
                start = time.time()
                outputs = train_and_eval(
                    train_config, model_config, 
                    input_shape, 
                    trainset, valset, evaluator, mirrored_strategy)
                if isinstance(outputs, bool) and outputs == True:
                    print('Model config error! RETRY')
                    continue
                break

            outputs['time'] = time.time() - start

            # eval
            if train_config.score:
                outputs['objective_score'] = np.array(outputs['val_seld_score'])[-1]
            else:
                outputs['objective_score'] = get_objective_score(outputs)

            # 결과 저장
            if os.path.exists(current_result_path):
                results = writer.load(current_result_path)
                if len(results) >= train_config.n_samples:
                    break
            results.append({'config': model_config, 'perf': outputs})
            writer.dump(results, current_result_path)
        
        
        # 분석
        check = True
        table = analyzer(search_space, results, train_config)
        tmp_table = list(filter(lambda x: x[0][0] <= train_config.threshold and x[-2] != 'identity_block' and x[-1] != 'identity_block', table))
        if len(tmp_table) == 0:
            print('MODEL SEARCH COMPLETE!!')
            return

        while check:
            table = analyzer(search_space, results, train_config)
            table = list(filter(lambda x: x[-2] != 'identity_block' and x[-1] != 'identity_block', table))
            # 단순히 좁힐 게 있는 지 탐지
            tmp_table = list(filter(lambda x: x[0][0] <= train_config.threshold, table))
            # search space 줄이기
            check, search_space, results = narrow_search_space(search_space, table, tmp_table, results, train_config, writer)


if __name__ == '__main__':
    main()

