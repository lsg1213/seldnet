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
from search import get_dataset


def get_search_space(target):
    blocks = [i for i in target.keys() if i.startswith('BLOCK') and not '_ARGS' in i]
    num1d, num2d = 0, 0
    for block in blocks:
        if target[block] in stages_1d:
            num1d += 1
        elif target[block] in stages_2d:
            num2d += 1
    search_space = {'num1d': [num1d],'num2d': [num2d]}
    for block in blocks + ['SED', 'DOA']:
        search_space[block] = {}
        search_space[block]['search_space_1d'] = {}
        search_space[block]['search_space_2d'] = {}

        if target[block] in stages_1d:
            args = {}
            for k, v in target[block+'_ARGS'].items():
                if k == 'multiplier' or k == 'ff_multiplier':
                    v = float(v)
                if isinstance(v, float):
                    if 'dropout_rate' in k:
                        v = sorted(list(set([v * i / 20 for i in range(10, 21)])))
                    else:
                        v = sorted(list(set([v * i / 20 for i in range(10, 21) if v * i / 20 != 0])))
                elif isinstance(v, int):
                    v = sorted(list(set([int(v * i / 20) for i in range(10, 21) if int(v * i / 20) != 0])))
                elif k == 'pos_encoding':
                    v = [None, 'basic', 'rff']
                elif isinstance(v, (str, list)):
                    v = [v]
                args[k] = v
            search_space[block]['search_space_1d'][target[block]] = args
        elif target[block] in stages_2d:
            args = {}
            for k,v in target[block+'_ARGS'].items():
                if k == 'connect0':
                    v = [[0], [1]]
                elif k == 'connect1':
                    v = list(map(list, product(range(2), range(2))))
                elif k == 'connect2':
                    v = list(map(list, product(range(2), range(2), range(2))))
                elif k == 'strides':
                    v = [(1, 1), (1, 2), (1, 3)]
                if isinstance(v, float):
                    if 'dropout_rate' in k:
                        v = sorted(list(set([v * i / 20 for i in range(10, 21)])))
                    else:
                        v = sorted(list(set([v * i / 20 for i in range(10, 21) if v * i / 20 != 0])))
                elif isinstance(v, int):
                    if k == 'filter1':
                        v = sorted(list(set([int(v * i / 20) for i in range(10, 21) if int(v * i / 20) != 0])))
                    else:
                        v = sorted(list(set([int(v * i / 20) for i in range(10, 21)])))
                args[k] = v
            search_space[block]['search_space_2d'][target[block]] = args
        elif target[block] == 'identity_block':
            continue
        else:
            raise ValueError()

    return search_space


def train_and_eval(train_config,
                   model_config: dict,
                   input_shape,
                   trainset: tf.data.Dataset,
                   valset: tf.data.Dataset,
                   evaluator,
                   mirrored_strategy):
    performances = {}
    try:
        optimizer = tf.keras.optimizers.Adam(train_config.lr)
        if train_config.multi:
            with mirrored_strategy.scope():
                model = models.conv_temporal(input_shape, model_config)
                model.compile(optimizer=optimizer,
                            loss={'sed_out': tf.keras.losses.BinaryCrossentropy(),
                                    'doa_out': tf.keras.losses.MSE},
                            loss_weights=[1, 1000])
        else:
            model = models.conv_temporal(input_shape, model_config)
            model.compile(optimizer=optimizer,
                        loss={'sed_out': tf.keras.losses.BinaryCrossentropy(),
                                'doa_out': tf.keras.losses.MSE},
                        loss_weights=[1, 1000])

        model.summary()
    except tf.errors.ResourceExhaustedError:
        print('!!!!!!!!!!!!!!!model error occurs!!!!!!!!!!!!!!!')
        if not os.path.exists('error_models'):
            os.makedirs('error_models')
        configs = []
        if os.path.exists(os.path.join('error_models', 'error_model.json')):
            with open(os.path.join('error_models', 'error_model.json'), 'r') as f:
                configs = json.load(f)
        else:
            configs = [model_config]
        with open(os.path.join('error_models', 'error_model.json'), 'w') as f:
            json.dump(model_config, f, indent=4)
        return True
    history = model.fit(trainset, validation_data=valset, epochs=train_config.epoch).history

    if len(performances) == 0:
        for k, v in history.items():
            performances[k] = v
    else:
        for k, v in history.items():
            performances[k] += v

    evaluator.reset_states()
    for x, y in valset:
        y_p = model(x, training=False)
        evaluator.update_states(y, y_p)
    scores = evaluator.result()
    scores = {
        'val_error_rate': scores[0].numpy().tolist(),
        'val_f1score': scores[1].numpy().tolist(),
        'val_der': scores[2].numpy().tolist(),
        'val_derf': scores[3].numpy().tolist(),
        'val_seld_score': calculate_seld_score(scores).numpy().tolist(),
    }
    if 'val_error_rate' in performances.keys():
        for k, v in scores.items():
            performances[k].append(v)
    else:
        for k, v in scores.items():
            performances[k] = [v]

    performances.update({
        'flops': get_flops(model),
        'size': get_model_size(model)
    })
    del model, optimizer, history
    return performances


def main(train_config):
    os.environ['CUDA_VISIBLE_DEVICES'] = train_config.gpus
    writer = Writer(train_config, result_folder='small_result')
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
    target = writer.load(os.path.join('model_config', train_config.target if os.path.splitext(train_config.target)[-1] == '.json' else train_config.target + '.json'))

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
            count = 0
            while True:
                try:
                    model_config = get_config(train_config, search_space, input_shape=input_shape, postprocess_fn=postprocess_fn)
                    # 학습
                    start = time.time()
                    outputs = train_and_eval(
                        train_config, model_config, 
                        input_shape, 
                        trainset, valset, evaluator, mirrored_strategy)
                except ValueError:
                    count += 1
                    if count % 100000 == 0:
                        print(f'config count: {count}')
                    continue
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
    args = argparse.ArgumentParser()

    args.add_argument('--name', type=str, required=True,
                    help='name must be {name}_{divided index} ex) 2021_1')
    args.add_argument('--target', type=str, default='SS5')
    args.add_argument('--dataset_path', type=str, 
                    default='/root/datasets/DCASE2021/feat_label')
    args.add_argument('--n_samples', type=int, default=200)
    args.add_argument('--min_samples', type=int, default=16)
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
    main(train_config)

