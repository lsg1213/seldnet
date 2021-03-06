import os
import json

import tensorflow as tf

from search import search_space_1d, search_space_2d, block_2d_num, block_1d_num, args
from config_sampler import get_max_configs
from search_utils import postprocess_fn
from search import get_dataset
import models


train_config = args.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = train_config.gpus
mirrored_strategy = tf.distribute.MirroredStrategy()
train_config.n_repeat = 1

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

model_configs = get_max_configs(train_config, search_space, [300,64,7], postprocess_fn)
if train_config.multi:
    with mirrored_strategy.scope():
        valset = get_dataset(train_config, mode='train')
else:
    valset = get_dataset(train_config, mode='train')

for model_config in model_configs:
    optimizer = tf.keras.optimizers.Adam(train_config.lr)
    if train_config.multi:
        with mirrored_strategy.scope():
            model = models.conv_temporal([300,64,7], model_config)
            model.summary()

            model.compile(optimizer=optimizer,
                        loss={'sed_out': tf.keras.losses.BinaryCrossentropy(),
                                'doa_out': tf.keras.losses.MSE},
                        loss_weights=[1, 1000])
    else:
        model = models.conv_temporal([300,64,7], model_config)
        model.summary()

        model.compile(optimizer=optimizer,
                    loss={'sed_out': tf.keras.losses.BinaryCrossentropy(),
                            'doa_out': tf.keras.losses.MSE},
                    loss_weights=[1, 1000])
    for k,v in model_config.items():
        print(k)
        print(v)
        print('------------------------------------')
    with open('too_big_search.json', 'w') as f:
        json.dump(model_config, f, indent=4)
    model.fit(valset)
    os.system('rm -rf too_big_search.json')
print('All search space is available in this GPU')

