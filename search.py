import argparse
import json
import os

import tensorflow as tf

from data_loader import *
from metrics import *
from transforms import *
from config_sampler import get_config
from search_utils import postprocess_fn


args = argparse.ArgumentParser()

args.add_argument('--name', type=str, required=True,
                  help='name must be {name}_{divided index} ex) 2021_1')
args.add_argument('--dataset_path', type=str, 
                  default='/root/datasets/DCASE2021/feat_label')
args.add_argument('--n_samples', type=int, default=250)
args.add_argument('--n_blocks', type=int, default=3)
# args.add_argument('--min_flops', type=int, default=200_000_000)
# args.add_argument('--max_flops', type=int, default=240_000_000)

args.add_argument('--batch_size', type=int, default=256)
args.add_argument('--n_repeat', type=int, default=5)
args.add_argument('--epoch', type=int, default=10)
args.add_argument('--lr', type=int, default=1e-3)
args.add_argument('--n_classes', type=int, default=12)
args.add_argument('--gpus', type=str, default='-1')

input_shape = [300, 64, 7]


'''            SEARCH SPACES           '''
search_space = {
    'search_space_2d': {
        'num': [0, 1, 2, 3, 4, 5],
        'mother_stage':
            {'depth': [1, 2, 3],
            'filters0': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                        3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
            'filters1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                        3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
            'filters2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                        3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
            'kernel_size0': [1, 3, 5],
            'kernel_size1': [1, 3, 5],
            'kernel_size2': [1, 3, 5],
            'connect0': [[0], [1]],
            'connect1': [[0, 0], [0, 1], [1, 0], [1, 1]],
            'connect2': [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
            'strides': [(1, 1), (1, 2), (1, 3)]},
    },
    'search_space_1d': {
        'num': [0, 1, 2, 3, 4, 5],
        'bidirectional_GRU_stage':
            {'depth': [1, 2, 3],
            'units': [16, 24, 32, 48, 64, 96, 128, 192, 256]}, 
        'transformer_encoder_stage':
            {'depth': [1, 2, 3],
            'n_head': [1, 2, 4, 8, 16],
            'key_dim': [2, 3, 4, 6, 8, 12, 16, 24, 32, 48],
            'ff_multiplier': [0.25, 0.5, 1, 2, 4, 8],
            'kernel_size': [1, 3, 5]},
        'simple_dense_stage':
            {'depth': [1, 2, 3],
             'units': [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
             'dense_activation': ['relu'],
             'dropout_rate': [0., 0.2, 0.5]},
        'conformer_encoder_stage':
            {'depth': [1, 2],
            'key_dim': [2, 3, 4, 6, 8, 12, 16, 24, 32, 48],
            'n_head': [1, 2, 4, 8, 16],
            'kernel_size': [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
            'multiplier': [1, 2, 4],
            'pos_encoding': [None, 'basic', 'rff']},
    }
}


def train_and_eval(train_config,
                   model_config: dict,
                   input_shape,
                   trainset: tf.data.Dataset,
                   valset: tf.data.Dataset,
                   evaluator):
    model = models.conv_temporal(input_shape, model_config)
    optimizer = tf.keras.optimizers.Adam(train_config.lr)

    model.compile(optimizer=optimizer,
                  loss={'sed_out': tf.keras.losses.BinaryCrossentropy(),
                        'doa_out': tf.keras.losses.MSE},
                  loss_weights=[1, 1000])

    history = model.fit(trainset,
                        validation_data=valset, epoch=train_config.epoch)

    evaluator.reset_states()
    for x, y in valset:
        evaluator.update_states(y, model(x, training=False))
    scores = evaluator.result()
    scores = {
        'val_error_rate': scores[0].numpy().tolist(),
        'val_f1score': scores[1].numpy().tolist(),
        'val_der': scores[2].numpy().tolist(),
        'val_derf': scores[3].numpy().tolist(),
        'val_seld_score': calculate_seld_score(scores).numpy().tolist(),
    }

    performances = {
        **history.history,
        **scores,
        **(model_complexity.conv_temporal_complexity(model_config, 
                                                     input_shape)[0])
    }
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
            random_ups_and_downs,
            lambda x, y: (mask(x, axis=-2, max_mask_size=16), y),
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

    return dataset


if __name__=='__main__':
    train_config = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = train_config.gpus
    del train_config.gpus
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=10240)])
        except RuntimeError as e:
            print(e)

    name = train_config.name
    if name.endswith('.json'):
        name = os.path.splitext(name)[0]

    input_shape = [300, 64, 7]

    # datasets
    # trainset = get_dataset(train_config, mode='train')
    # valset = get_dataset(train_config, mode='val')
    
    # Evaluator
    evaluator = SELDMetrics(doa_threshold=20, n_classes=train_config.n_classes)

    default_config = {
        'n_classes': train_config.n_classes
    }
    results = {'train_config': vars(train_config)}
    start_idx = 0

    # result folder
    result_path = os.path.join('result', name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    # train config
    train_config_path = os.path.join(result_path, 'train_config.json')
    if not os.path.exists(train_config_path):
        with open(train_config_path, 'w') as f:
            json.dump(vars(train_config), f, indent=4)
    else:
        with open(train_config_path, 'r') as f:
            loaded_train_config = json.load(f)
        if loaded_train_config != vars(train_config):
            raise ValueError('train config doesn\'t match')

    index = 0
    while True:
        index += 1 # 차수
        current_result_path = os.path.join(result_path, f'result_{str(index)}.json')
        results = []

        # resume
        if os.path.exists(current_result_path):
            with open(current_result_path, f'result_{str(index)}.json', 'r') as f:
                results = json.load(f)


        # search space
        search_space_path = os.path.join(result_path, f'search_space_{index}.json')
        if os.path.exists(search_space_path):
            with open(search_space_path, 'r') as f:
                search_space = json.load(f)
        else:
            with open(search_space_path, 'w') as f:
                json.dump(search_space, f)

        start_epoch = len(results)
        for i in range(start_epoch, train_config.n_samples):
            model_configs = get_config(train_config, search_space, input_shape=input_shape, postprocess_fn=postprocess_fn)

            # 학습
            start = time.time()
            outputs = train_and_eval(
                train_config, model_config, 
                input_shape, 
                trainset, valset, evaluator)
            outputs['time'] = time.time() - start

            # eval

            # 결과 저장
            with open(current_result_path, f'result_{str(index)}.json', 'w') as f:
                json.dump(results, f, indent=4)

        # search space 줄이기

        # search space 기록 남기기

        # search space 저장



