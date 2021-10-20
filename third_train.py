import argparse
import os
import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow.python.keras.api.keras as keras
import tensorflow_addons as tfa
from glob import glob
from numpy import inf
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data_loader import *
from metrics import SELDMetrics, calculate_seld_score
from transforms import *
from utils import adaptive_clip_grad, apply_kernel_regularizer

args = argparse.ArgumentParser()
    
args.add_argument('--name', type=str, required=True)

args.add_argument('--gpus', type=str, default='-1')
args.add_argument('--resume', action='store_true')    
args.add_argument('--abspath', type=str, default='/root/datasets')
args.add_argument('--output_path', type=str, default='./output')
args.add_argument('--ans_path', type=str, default='/root/datasets/DCASE2021/metadata_dev/')


# training
args.add_argument('--decay', type=float, default=0.9)
args.add_argument('--sed_th', type=float, default=0.3)
args.add_argument('--lr', type=float, default=0.003)
args.add_argument('--final_lr', type=float, default=0.0001)
args.add_argument('--batch', type=int, default=512)
args.add_argument('--agc', type=bool, default=False)
args.add_argument('--epoch', type=int, default=60)
args.add_argument('--lr_patience', type=int, default=5, 
                    help='learning rate decay patience for plateau')
args.add_argument('--patience', type=int, default=100, 
                    help='early stop patience')
args.add_argument('--use_acs', type=bool, default=True)
args.add_argument('--use_tfm', type=bool, default=True)
args.add_argument('--use_tdm', action='store_true')
args.add_argument('--schedule', action='store_true')
args.add_argument('--loop_time', type=int, default=5, 
                    help='times of train dataset iter for an epoch')
args.add_argument('--lad_doa_thresh', type=int, default=20)


def resnet_block(inp, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
    if stride == 2:
        x = keras.layers.AveragePooling2D((2, 2))(inp)
    else:
        x = inp
    x = keras.layers.Conv2D(planes, (3,3), padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Conv2D(planes, (3,3), padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    
    if downsample is not None:
        inp = downsample(inp)
    x = keras.layers.ReLU()(x + inp)
    return x


def resnet_layer(inp, planes, blocks, strides=1, dilate=False):
    inplanes = inp.shape[-1]
    expansion = 1
    downsample = None

    if strides != 1 or inplanes != planes * expansion:
        layers = []
        if strides == 2:
            layers.append(keras.layers.AveragePooling2D((2, 2)))
        layers.append(keras.layers.Conv2D(planes * expansion, kernel_size=1, strides=1, use_bias=False))
        layers.append(keras.layers.BatchNormalization())
        downsample = keras.Sequential(layers)
    
    x = resnet_block(inp, planes, strides, downsample)
    for _ in range(1, blocks):
        x = resnet_block(x, planes)
    return x


def resnet(inp, layers, replace_stride_with_dilation=None):
    if replace_stride_with_dilation is None:
        replace_stride_with_dilation = [False, False, False]
    if len(replace_stride_with_dilation) != 3:
        raise ValueError(f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}")
    x = resnet_layer(inp, 64, layers[0], strides=1)
    x = resnet_layer(x, 128, layers[1], strides=2, dilate=replace_stride_with_dilation[0])
    x = resnet_layer(x, 256, layers[2], strides=2, dilate=replace_stride_with_dilation[1])
    x = resnet_layer(x, 512, layers[3], strides=2, dilate=replace_stride_with_dilation[2])
    return x


def conv_block(inp, out_channel, pool_type='avg', pool_size=(2,2)):
    x = keras.layers.Conv2D(out_channel, kernel_size=(3,3), padding='same', use_bias=False)(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(out_channel, kernel_size=(3,3), padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    if pool_type == 'avg':
        x = keras.layers.AveragePooling2D(pool_size)(x)
    elif pool_type == 'max':
        x = keras.layers.MaxPooling2D(pool_size)(x)
    elif pool_type == 'avg+max':
        x1 = keras.layers.AveragePooling2D(pool_size)(x)
        x2 = keras.layers.MaxPooling2D(pool_size)(x)
        x = x1 + x2
    else:
        raise Exception('Wrong pool_type')
    return x

# https://github.com/qiuqiangkong/audioset_tagging_cnn/tree/master/pytorch
def get_model(input_shape):
    inp = keras.layers.Input(shape = input_shape)
    x = keras.layers.BatchNormalization(-2)(inp) # mel 기준으로 bn
    x = conv_block(x, 64)
    x = keras.layers.Dropout(0.2)(x)
    x = resnet(x, layers=[2, 2, 2, 2])
    x = keras.layers.AveragePooling2D((1,2))(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.TimeDistributed(keras.layers.Flatten())(x)

    x = keras.layers.Bidirectional(keras.layers.GRU(256, return_sequences=True), merge_mode='sum')(x)
    x = keras.layers.Bidirectional(keras.layers.GRU(256, return_sequences=True), merge_mode='sum')(x)
    x = keras.layers.Conv1DTranspose(256, 2, strides=2)(x)

    x = keras.layers.Dense(36)(x)
    x = keras.layers.Activation('tanh')(x)
    return keras.Model(inputs=inp, outputs=x)

@tf.function
def get_accdoa_labels(accdoa_in, nb_classes, sed_th=0.3):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:]
    sed = tf.cast(tf.sqrt(x**2 + y**2 + z**2) > sed_th, tf.float32)
    return sed, accdoa_in


def generate_trainstep(criterion, config):
    # These are statistics from the train dataset
    # train_samples = tf.convert_to_tensor(
    #     [[58193, 32794, 29801, 21478, 14822, 
    #     9174, 66527,  6740,  9342,  6498, 
    #     22218, 49758]],
    #     dtype=tf.float32)
    # cls_weights = tf.reduce_mean(train_samples) / train_samples
    @tf.function
    def trainstep(model, x, y, optimizer):
        with tf.GradientTape() as tape:
            y_p = model(x, training=True)
            loss = criterion(y[1], y_p)

            # regularizer
            # loss += tf.add_n([l.losses[0] for l in model.layers
            #                   if len(l.losses) > 0])

        grad = tape.gradient(loss, model.trainable_variables)
        # apply AGC
        if config.agc:
            grad = adaptive_clip_grad(model.trainable_variables, grad)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

        return y_p, loss
    return trainstep


def generate_teststep(criterion):
    @tf.function
    def teststep(model, x, y, optimizer=None):
        y_p = model(x, training=False)
        loss = criterion(y[1], y_p)
        return y_p, loss
    return teststep


def generate_iterloop(criterion, evaluator, writer, 
                      mode, config=None):
    if mode == 'train':
        step = generate_trainstep(criterion, config)
    else:
        step = generate_teststep(criterion)

    def iterloop(model, dataset, epoch, optimizer=None):
        evaluator.reset_states()
        losses = keras.metrics.Mean()

        with tqdm(dataset) as pbar:
            for x, y in pbar:
                preds, loss = step(model, x, y, optimizer)
                y, preds = y, get_accdoa_labels(preds, preds.shape[-1] // 3, config.sed_th)
                
                evaluator.update_states(y, preds)
                metric_values = evaluator.result()
                seld_score = calculate_seld_score(metric_values)

                losses(loss)
                if mode == 'train':
                    status = OrderedDict({
                        'mode': mode,
                        'epoch': epoch,
                        'lr': optimizer.learning_rate.numpy(),
                        'loss': losses.result().numpy(),
                        'ER': metric_values[0].numpy(),
                        'F': metric_values[1].numpy(),
                        'DER': metric_values[2].numpy(),
                        'DERF': metric_values[3].numpy(),
                        'seldscore': seld_score.numpy()
                    })
                else:
                    status = OrderedDict({
                    'mode': mode,
                    'epoch': epoch,
                    'loss': losses.result().numpy(),
                    'ER': metric_values[0].numpy(),
                    'F': metric_values[1].numpy(),
                    'DER': metric_values[2].numpy(),
                    'DERF': metric_values[3].numpy(),
                    'seldscore': seld_score.numpy()
                    })
                pbar.set_postfix(status)

        writer.add_scalar(f'{mode}/{mode}_ErrorRate', metric_values[0].numpy(),
                          epoch)
        writer.add_scalar(f'{mode}/{mode}_F', metric_values[1].numpy(), epoch)
        writer.add_scalar(f'{mode}/{mode}_DoaErrorRate', 
                          metric_values[2].numpy(), epoch)
        writer.add_scalar(f'{mode}/{mode}_DoaErrorRateF', 
                          metric_values[3].numpy(), epoch)
        writer.add_scalar(f'{mode}/{mode}_Loss', 
                          losses.result().numpy(), epoch)
        writer.add_scalar(f'{mode}/{mode}_seldScore', 
                          seld_score.numpy(), epoch)

        return seld_score.numpy()
    return iterloop


def random_ups_and_downs(x, y):
    stddev = 0.25
    offsets = tf.linspace(tf.random.normal([], stddev=stddev),
                          tf.random.normal([], stddev=stddev),
                          x.shape[-3])
    offsets_shape = [1] * len(x.shape)
    offsets_shape[-3] = offsets.shape[0]
    offsets = tf.reshape(offsets, offsets_shape)
    x = tf.concat([x[..., :4] + offsets, x[..., 4:]], -1)
    return x, y


def get_dataset(config, mode: str = 'train'):
    path = os.path.join(config.abspath, 'DCASE2021/')
    x = joblib.load(os.path.join(path, f'foa_dev_{mode}_mel_512.joblib'))
    y = joblib.load(os.path.join(path, f'foa_dev_{mode}_label.joblib'))
    if config.use_tfm and mode == 'train':
        sample_transforms = [
            # random_ups_and_downs,
            # lambda x, y: (mask(x, axis=-2, max_mask_size=8, n_mask=6), y),
            # lambda x, y: (mask(x, axis=-2, max_mask_size=16, period=80), y),
            make_spec_augment(16, 32, 2, 2)
        ]
    else:
        sample_transforms = []
    batch_transforms = [split_total_labels_to_sed_doa]
    if config.use_acs and mode == 'train':
        batch_transforms.insert(0, foa_intensity_vec_aug)
    dataset = seldnet_data_to_dataloader(
        x, y,
        train= mode == 'train',
        batch_transforms=batch_transforms,
        label_window_size=10,
        batch_size=config.batch,
        sample_transforms=sample_transforms,
        loop_time=config.loop_time
    )
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    n_classes = 12
    name = '_'.join(['2', str(config.lr), str(config.final_lr)])
    if config.schedule:
        name += '_schedule'
    config.name = name + '_' + config.name

    # data load
    trainset = get_dataset(config, 'train')
    valset = get_dataset(config, 'val')
    testset = get_dataset(config, 'test')

    tensorboard_path = os.path.join('./tensorboard_log', config.name)
    if not os.path.exists(tensorboard_path):
        print(f'tensorboard log directory: {tensorboard_path}')
        os.makedirs(tensorboard_path)
    writer = SummaryWriter(logdir=tensorboard_path)

    model_path = os.path.join('./saved_model', config.name)
    if not os.path.exists(model_path):
        print(f'saved model directory: {model_path}')
        os.makedirs(model_path)

    x, _ = [(x, y) for x, y in valset.take(1)][0]
    input_shape = x.shape[1:]
    model = get_model(input_shape)
    model.summary()

    optimizer = keras.optimizers.Adam(config.lr)
    criterion = keras.losses.MSE


    if config.resume:
        _model_path = sorted(glob(model_path + '/*.hdf5'))
        if len(_model_path) == 0:
            raise ValueError('the model does not exist, cannot be resumed')
        model = keras.models.load_model(_model_path[0])

    best_score = inf
    evaluator = SELDMetrics(
        doa_threshold=config.lad_doa_thresh, n_classes=n_classes, sed_th=config.sed_th)

    train_iterloop = generate_iterloop(
        criterion, evaluator, writer, 'train', config=config)
    val_iterloop = generate_iterloop(
        criterion, evaluator, writer, 'val', config=config)
    test_iterloop = generate_iterloop(
        criterion, evaluator, writer, 'test', config=config)

    lr_decay_patience = 0
    for epoch in range(config.epoch):

        # train loop
        train_iterloop(model, trainset, epoch, optimizer)
        score = val_iterloop(model, valset, epoch)
        test_iterloop(model, testset, epoch)

        if best_score > score:
            os.system(f'rm -rf {model_path}/bestscore_{best_score}.hdf5')
            best_score = score
            keras.models.save_model(
                model, 
                os.path.join(model_path, f'bestscore_{best_score}.hdf5'), 
                include_optimizer=False)
            lr_decay_patience = 0
        else:
            if not config.schedule:
                lr_decay_patience += 1
                print(f'lr_decay_patience: {lr_decay_patience}')
            if lr_decay_patience >= config.lr_patience and config.decay != 1:
                print(f'lr: {optimizer.learning_rate.numpy():.3} -> {(optimizer.learning_rate * config.decay).numpy():.3}')
                optimizer.learning_rate = optimizer.learning_rate * config.decay
                lr_decay_patience = 0
        if config.schedule:
            decay_coefficient = (config.final_lr / config.lr) ** (1 / config.epoch)
            print(f'lr: {optimizer.learning_rate.numpy():.3} -> {(optimizer.learning_rate * decay_coefficient).numpy():.3}')
            optimizer.learning_rate = optimizer.learning_rate * decay_coefficient

    # end of training
    print(f'epoch: {epoch}')


if __name__=='__main__':
    main(args.parse_args())
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # input_shape = [80, 128, 7]
    # model = get_model(input_shape)
    # model.summary()
    # from model_flop import get_flops
    # print(get_flops(model))
    