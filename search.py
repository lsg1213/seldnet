import argparse
import json
import os
import time
from copy import deepcopy

import tensorflow as tf

from data_loader import *
from metrics import *
from transforms import *
from config_sampler import get_config
from search_utils import postprocess_fn
from model_flop import get_flops
from model_size import get_model_size
from model_analyze import analyzer, narrow_search_space, table_filter
from writer_manager import Writer
import models


args = argparse.ArgumentParser()

args.add_argument('--name', type=str, required=True,
                  help='name must be {name}_{divided index} ex) 2021_1')
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
args.add_argument('--loss', action='store_true')


input_shape = [300, 64, 7]


'''            SEARCH SPACES           '''
block_2d_num = [1, 2, 3]
block_1d_num = [0, 1, 2]
search_space_2d = {
    'mother_stage':
        {'mother_depth': [1, 2, 3],
        'filters0': [0, 3, 4, 6, 8, 12, 16, 24, 32],
        'filters1': [3, 4, 6, 8, 12, 16, 24, 32],
        'filters2': [0, 3, 4, 6, 8, 12, 16, 24, 32],
        'kernel_size0': [1, 3, 5],
        'kernel_size1': [1, 3, 5],
        'kernel_size2': [1, 3, 5],
        'connect0': [[0], [1]],
        'connect1': [[0, 0], [0, 1], [1, 0], [1, 1]],
        'connect2': [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
        'strides': [(1, 1), (1, 2), (1, 3)]},
}
search_space_1d = {
    'bidirectional_GRU_stage':
        {'GRU_depth': [1, 2],
        'GRU_units': [16, 24, 32, 48, 64, 96, 128, 192, 256]}, 
    'transformer_encoder_stage':
        {'transformer_depth': [1, 2],
        'transformer_n_head': [1, 2, 4],
        'transformer_key_dim': [2, 3, 4, 6, 8, 12, 16, 24, 32],
        'ff_multiplier': [0.25, 0.5, 1, 2],
        'transformer_kernel_size': [1, 3, 5]},
    'simple_dense_stage':
        {'dense_depth': [1, 2],
            'dense_units': [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256],
            'dense_activation': ['relu'],
            'dropout_rate': [0., 0.2, 0.5]},
    'conformer_encoder_stage':
        {'conformer_depth': [1],
        'conformer_key_dim': [2, 3, 4, 6, 8, 12, 16],
        'conformer_n_head': [1, 2, 4],
        'conformer_kernel_size': [4, 6, 8, 12, 16, 24, 32],
        'multiplier': [1, 2, 3],
        'pos_encoding': [None, 'basic', 'rff']},
}


def get_learning_rate(train_config, input_shape, model_config, mirrored_strategy, trainset, valset):
    various_lr = [train_config.lr / i ** 2 for i in range(1, 5)]
    model = models.conv_temporal(input_shape, model_config)

    weights = model.get_weights()
    val_losses = []
    for lr in various_lr:
        optimizer = tf.keras.optimizers.Adam(lr)
        model.set_weights(weights)
        if train_config.multi:
            with mirrored_strategy.scope():
                model = models.conv_temporal(input_shape, model_config)
                model.compile(optimizer=optimizer,
                            loss={'sed_out': tf.keras.losses.BinaryCrossentropy(),
                                    'doa_out': tf.keras.losses.MSE},
                            loss_weights=[1, 1000])
        else:
            model.compile(optimizer=optimizer,
                        loss={'sed_out': tf.keras.losses.BinaryCrossentropy(),
                                'doa_out': tf.keras.losses.MSE},
                        loss_weights=[1, 1000])
        history = model.fit(trainset, validation_data=valset).history
        val_losses.append(history['val_loss'])
        if len(val_losses) == 2:
            if val_losses[0] < val_losses[1]:
                break
    return various_lr[tf.concat(val_losses, 0).numpy().argmin()], weights


def train_and_eval(train_config,
                   model_config: dict,
                   input_shape,
                   trainset: tf.data.Dataset,
                   valset: tf.data.Dataset,
                   evaluator,
                   mirrored_strategy):
    # while True:
    #     count = 0
    #     try:
    #         selected_lr, weights = get_learning_rate(train_config, input_shape, model_config, mirrored_strategy, trainset, valset)
    #         break
    #     except tf.errors.ResourceExhaustedError:
    #         print('!!!!!!!!!!!!!!!model error occurs!!!!!!!!!!!!!!!')
    #         if not os.path.exists('error_models'):
    #             os.makedirs('error_models')
    #         configs = []
    #         if os.path.exists(os.path.join('error_models', 'error_model.json')):
    #             with open(os.path.join('error_models', 'error_model.json'), 'r') as f:
    #                 configs = json.load(f)
    #         else:
    #             configs = [model_config]
    #         with open(os.path.join('error_models', 'error_model.json'), 'w') as f:
    #             json.dump(model_config, f, indent=4)
    #         return True
    #     except ValueError:
    #         count += 1
    #         if count % 10000 == 0:
    #             print("iter:", count)
    #         continue
    selected_lr = train_config.lr

    performances = {}
    try:
        optimizer = tf.keras.optimizers.Adam(selected_lr)
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
    # model.set_weights(weights)
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
        'selected_lr': selected_lr
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


# reference: https://github.com/IRIS-AUDIO/SELD.git
def random_ups_and_downs(x, y):
    stddev = 0.25
    offsets = tf.linspace(tf.random.normal([], stddev=stddev),
                          tf.random.normal([], stddev=stddev),
                          x.shape[-3])
    offsets_shape = [1] * len(x.shape)
    offsets_shape[-3] = offsets.shape[0]
    offsets = tf.reshape(offsets, offsets_shape)
    x = tf.concat([x[..., :4] + offsets, x[..., 4:]], axis=-1)
    return x, y


def get_dataset(config, mode: str = 'train'):
    path = config.dataset_path
    x, y = load_seldnet_data(os.path.join(path, 'foa_dev_norm'),
                             os.path.join(path, 'foa_dev_label'),
                             mode=mode, n_freq_bins=64)
    if mode == 'train':
        sample_transforms = [
            # random_ups_and_downs,
            # lambda x, y: (mask(x, axis=-2, max_mask_size=16), y),
        ]
        batch_transforms = [foa_intensity_vec_aug]
    else:
        sample_transforms = []
        batch_transforms = []
    batch_transforms.append(split_total_labels_to_sed_doa)

    dataset = seldnet_data_to_dataloader(
        x, y,
        train= mode == 'train',
        batch_transforms=batch_transforms,
        label_window_size=60,
        batch_size=config.batch_size,
        sample_transforms=sample_transforms,
        loop_time=config.n_repeat
    )

    # return dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def search_space_filter(target, unit):
    target = target.split('.')
    try:
        unit = json.loads(unit)
    except:
        pass

    def _search_space_filter(results):
        aa = deepcopy(target)
        if len(target) == 1:
            v = results['config'].get(target[0])
            if v is None:
                raise ValueError()
        elif len(target) == 2:
            v = results['config'].get(target[0])
            if v is None:
                raise ValueError()
            if target[1] == 'depth' and v.get(target[1]) is None:
                if not target[1] in v.keys():
                    target[1] = [i for i in v.keys() if 'depth' in i][0]
            v = v.get(target[1])
            if v is None:
                try:
                    target[1] = '_'.join(target[1].split('_')[1:])
                    v = v.get(target[1])
                except:
                    import pdb; pdb.set_trace()
                if v is None:
                    raise ValueError()
        if type(v) != type(unit):
            raise TypeError('value and unit must be same type')
        return v != unit
    return _search_space_filter


def main():
    train_config = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = train_config.gpus
    writer = Writer(train_config)
    mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
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
        if not os.path.exists(current_result_path):
            writer.dump([], current_result_path)
        results = []

        # search space
        search_space_path = os.path.join(writer.result_path, f'search_space_{writer.index}.json')
        if os.path.exists(search_space_path):
            search_space = writer.load(search_space_path)
        elif writer.index == 1:
            # search space initializing
            specific_search_space = {'num2d': block_2d_num, 'num1d': block_1d_num}
            for i in range(specific_search_space['num2d'][-1] + specific_search_space['num1d'][-1]):
                specific_search_space[f'BLOCK{i}'] = {
                    'search_space_2d': search_space_2d,
                    'search_space_1d': search_space_1d,
                }

            specific_search_space['SED'] = {'search_space_1d': search_space_1d}
            specific_search_space['DOA'] = {'search_space_1d': search_space_1d}
            search_space = specific_search_space
            writer.dump(search_space, search_space_path)
        else:
            writer.dump(search_space, search_space_path)

        while len(results) < train_config.n_samples:
            # resume
            if os.path.exists(current_result_path):
                results = writer.load(current_result_path)
                if len(results) >= train_config.n_samples:
                    break
            current_number = len(results)
            while True:
                model_config = get_config(train_config, search_space, input_shape=input_shape, postprocess_fn=postprocess_fn)
                # 학습
                start = time.time()
                try:
                    outputs = train_and_eval(
                        train_config, model_config, 
                        input_shape, 
                        trainset, valset, evaluator, mirrored_strategy)
                    if isinstance(outputs, bool) and outputs == True:
                        print('Model config error! RETRY')
                        continue
                except ValueError:
                    pass
                break

            outputs['time'] = time.time() - start

            # eval
            if train_config.score:
                outputs['objective_score'] = np.array(outputs['val_seld_score'])[-1]
            elif train_config.loss:
                outputs['objective_score'] = np.array(outputs['val_loss'])[-1]
            else:
                outputs['objective_score'] = get_objective_score(outputs)

            # 결과 저장
            if os.path.exists(current_result_path):
                results = writer.load(current_result_path)
                if len(results) >= train_config.n_samples:
                    break
            results.append({'config': model_config, 'perf': outputs})
            writer.dump(results, current_result_path)
        
        # 그동안 모든 결과 부르기
        results = []
        results = [writer.load(res_path) for res_path in sorted(glob(os.path.join(writer.result_path, 'result_*')))]
        results = [x for y in results for x in y]
        # 그동안 삭제했던 부분 부르기
        removed_space = [writer.load(removed_path) for removed_path in sorted(glob(os.path.join(writer.result_path, 'removed_space_*')))]
        removed_space = [x for y in removed_space for x in y]

        for remove in removed_space:
            target = remove['versus'].split(':')[0]
            unit = ' '.join(remove['result'].split(' ')[:-2])
            results = list(filter(search_space_filter(target, unit), results))

        # 분석
        check = True
        table = analyzer(search_space, results, train_config)
        tmp_table = table_filter(table, search_space, train_config.threshold)
        if len(tmp_table) == 0:
            print('MODEL SEARCH COMPLETE!!')
            return

        while check:
            table = analyzer(search_space, results, train_config)
            table = list(filter(lambda x: x[-2] != 'identity_block' and x[-1] != 'identity_block', table))

            tmp_table = table_filter(table, search_space, train_config.threshold)
            # search space 줄이기
            check, search_space, results, end = narrow_search_space(search_space, table, tmp_table, results, train_config, writer)
            if end:
                print('MODEL SEARCH END!!')
                return


if __name__=='__main__':
    main()

