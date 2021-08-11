import os

import tensorflow as tf

from search import search_space_1d, search_space_2d, block_2d_num, block_1d_num, args
from config_sampler import get_max_configs
from search_utils import postprocess_fn
from search import get_dataset

train_config = args.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = train_config.gpus

block_2d_num = max(block_2d_num)
block_1d_num = max(block_1d_num)

search_space = {'num2d': block_2d_num, 'num1d': block_1d_num}

for i in range(search_space['num2d'] + search_space['num1d']):
    search_space[f'BLOCK{i}'] = {
        'search_space_2d': search_space_2d,
        'search_space_1d': search_space_1d,
    }

search_space['SED'] = {'search_space_1d': search_space_1d}
search_space['DOA'] = {'search_space_1d': search_space_1d}

configs = get_max_configs(train_config, search_space, [300,64,7], postprocess_fn)
trainset = get_dataset(train_config, mode='train')

for config in configs:
    optimizer = tf.keras.optimizers.Adam(train_config.lr)

    model = models.conv_temporal([300,64,7], model_config)
    model.summary()

    model.compile(optimizer=optimizer,
                  loss={'sed_out': tf.keras.losses.BinaryCrossentropy(),
                        'doa_out': tf.keras.losses.MSE},
                  loss_weights=[1, 1000])
    model.fit(trainset)

