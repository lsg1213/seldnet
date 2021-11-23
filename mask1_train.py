'''
이 파일은 mask를 이용한 source seperation 모델을 제작하기 위해 만든 파일임
'''

import argparse
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import MSE
import tensorflow_addons as tfa
from glob import glob
from numpy import inf
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data_loader import *
from metrics import SELDMetrics, calculate_seld_score
from transforms import *
from utils import adaptive_clip_grad, apply_kernel_regularizer


    
class ARGS:
    def __init__(self):
        self.args = argparse.ArgumentParser()
        self.set('--name', type=str, default='stft')
        self.set('--gpus', type=str, default='1')
        self.set('--resume', action='store_true')    
        self.set('--abspath', type=str, default='/root/datasets')
        self.set('--output_path', type=str, default='./output')
        self.set('--ans_path', type=str, default='/root/datasets/DCASE2021/metadata_dev/')
        self.set('--norm', type=bool, default=True)
        self.set('--decay', type=float, default=0.9)
        self.set('--sed_th', type=float, default=0.3)
        self.set('--lr', type=float, default=1e-3)
        self.set('--final_lr', type=float, default=0.00001)
        self.set('--batch', type=int, default=8)
        self.set('--agc', type=bool, default=False)
        self.set('--epoch', type=int, default=200)
        self.set('--lr_patience', type=int, default=5, help='learning rate decay patience for plateau')
        self.set('--patience', type=int, default=10, help='early stop patience')
        self.set('--use_acs', type=bool, default=True)
        self.set('--use_tfm', type=bool, default=True)
        self.set('--use_tdm', action='store_true')
        self.set('--schedule', type=bool, default=True)
        self.set('--loop_time', type=int, default=5, help='times of train dataset iter for an epoch')
        self.set('--lad_doa_thresh', type=int, default=20)
        self.set('--nfft', type=int, default=1024)
        self.set('--hop', type=int, default=480)
        self.set('--len', type=int, default=4)
        self.set('--steps_per_epoch', type=int, default=400)
        
    def set(self, name, type=str, default=None, action=None, help=''):
        # if action == 'store_true':
        #     type = bool
        #     default = False
        # name = name.split('--')[-1]
        # setattr(self, name, type(default))
        if action is not None:
            self.args.add_argument(name, action=action, help=help)
        else:
            self.args.add_argument(name, type=type, default=default, help=help)

    def get(self):
        return self.args.parse_args()
        
args = ARGS().get()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus


""" COMPLEX-SPECTROGRAMS """
def complex_to_magphase(complex_tensor, y=None):
    if tf.rank(complex_tensor) == 4:
        n_chan = complex_tensor.shape[-1] // 2
        real = complex_tensor[..., :n_chan]
        img = complex_tensor[..., n_chan:]
    elif tf.rank(complex_tensor) == 5:
        n_chan = complex_tensor.shape[-2] // 2
        real = complex_tensor[..., :n_chan, :]
        img = complex_tensor[..., n_chan:, :]

    mag = tf.math.sqrt(real**2 + img**2)
    phase = tf.math.atan2(img, real)

    magphase = tf.concat([mag, phase], axis=-1)

    if y is None:
        return magphase
    return magphase, y


def magphase_to_complex(magphase):
    if tf.rank(magphase) == 4:
        n_chan = magphase.shape[-1] // 2
        mag = magphase[..., :n_chan]
        phase = magphase[..., n_chan:]
    elif tf.rank(magphase) == 5:
        n_chan = magphase.shape[-2] // 2
        mag = magphase[..., :n_chan, :]
        phase = magphase[..., n_chan:, :]

    real = mag * tf.cos(phase)
    img = mag * tf.sin(phase)

    return tf.concat([real, img], axis=-1)


def resnet_block(inp, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
    if stride == 2:
        x = tf.keras.layers.AveragePooling2D((2, 2))(inp)
    else:
        x = inp
    x = tf.keras.layers.Conv2D(planes, (3,3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(planes, (3,3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    if downsample is not None:
        inp = downsample(inp)
    x = tf.keras.layers.ReLU()(x + inp)
    return x


def resnet_layer(inp, planes, blocks, strides=1, dilate=False):
    inplanes = inp.shape[-1]
    expansion = 1
    downsample = None

    if strides != 1 or inplanes != planes * expansion:
        layers = []
        if strides == 2:
            layers.append(tf.keras.layers.AveragePooling2D((2, 2)))
        layers.append(tf.keras.layers.Conv2D(planes * expansion, kernel_size=1, strides=1, use_bias=False))
        layers.append(tf.keras.layers.BatchNormalization())
        downsample = tf.keras.Sequential(layers)
    
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
    x = tf.keras.layers.Conv2D(out_channel, kernel_size=(3,3), padding='same', use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(out_channel, kernel_size=(3,3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    if pool_type == 'avg':
        x = tf.keras.layers.AveragePooling2D(pool_size)(x)
    elif pool_type == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size)(x)
    elif pool_type == 'avg+max':
        x1 = tf.keras.layers.AveragePooling2D(pool_size)(x)
        x2 = tf.keras.layers.MaxPooling2D(pool_size)(x)
        x = x1 + x2
    else:
        raise Exception('Wrong pool_type')
    return x


# @tf.function
def get_accdoa_labels(accdoa_in, nb_classes, sed_th=0.3):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:]
    sed = tf.cast(tf.sqrt(x**2 + y**2 + z**2) > sed_th, tf.float32)
    return sed, accdoa_in


def mask_train(criterion, config):
    @tf.function
    def maskstep(model, x, y, optimizer):
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
    return maskstep


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
        losses = tf.keras.metrics.Mean()

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


class CustomModel(tf.keras.Model):
    def __init__(self, **kwargs) -> None:
        super(CustomModel, self).__init__(**kwargs)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y, splited_x, splited_y = data
        if 'mag' in self.name:
            x = complex_to_magphase(x)
        # splited_y = (batch, time, one_hot class, 3)

        resolution = x.shape[1] // y[0].shape[1]
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            
            masked_x = x[...,tf.newaxis] * y_pred # (batch, frame_num, freq, chan=8, class=12)
            results = tf.zeros_like(masked_x)
            for i in range(splited_x.shape[-1]):
                sy, sx = splited_y[0][...,i,:], splited_x[...,i]
                results += sx[..., tf.newaxis] * tf.repeat(sy, resolution, axis=1)[..., tf.newaxis, tf.newaxis, :]
            if 'mag' in self.name:
                masked_x = magphase_to_complex(masked_x)
            loss = self.compiled_loss(results, masked_x)

        # Compute gradients
        trainable_vars = self.trainable_variables
        
        gradients = tape.gradient(loss, trainable_vars)
        # gradients = adaptive_clip_grad(self.trainable_variables, gradients)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        
        self.compiled_metrics.update_state(results, masked_x)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


# https://github.com/JusperLee/Looking-to-Listen-at-the-Cocktail-Party/blob/master/model/AO_model/AO_model.py
def get_model(input_shape):
    inp = tf.keras.layers.Input(shape = input_shape)
    class_num = 12
    
    x = inp
    conv1 = Convolution2D(96, kernel_size=(1, 7), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='conv1')(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = Convolution2D(96, kernel_size=(7, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='conv2')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    conv3 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='conv3')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    conv4 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(2, 1), name='conv4')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)

    conv5 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(4, 1), name='conv5')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    conv6 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(8, 1), name='conv6')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    conv7 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(16, 1), name='conv7')(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    conv8 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(32, 1), name='conv8')(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    conv9 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='conv9')(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    conv10 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(2, 2), name='conv10')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)

    conv11 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(4, 4), name='conv11')(conv10)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)

    conv12 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(8, 8), name='conv12')(conv11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)

    conv13 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(16, 16), name='conv13')(conv12)
    conv13 = BatchNormalization()(conv13)
    conv13 = Activation('relu')(conv13)

    conv14 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(32, 32), name='conv14')(conv13)
    conv14 = BatchNormalization()(conv14)
    conv14 = Activation('relu')(conv14)

    conv15 = Convolution2D(8, kernel_size=(1, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='conv15')(conv14)
    conv15 = BatchNormalization()(conv15)
    conv15 = Activation('relu')(conv15)

    AVfusion = TimeDistributed(Flatten())(conv15)

    # lstm = Bidirectional(LSTM(200, return_sequences=True),merge_mode='sum')(AVfusion)
    lstm = Bidirectional(LSTM(400, return_sequences=True),merge_mode='sum')(AVfusion)

    fc1 = Dense(600, name="fc1", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed=27))(lstm)
    fc2 = Dense(600, name="fc2", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed=42))(fc1)
    fc3 = Dense(600, name="fc3", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed=65))(fc2)

    complex_mask = Dense(inp.shape[-2] * inp.shape[-1] * class_num, name="complex_mask", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=87))(fc3)

    complex_mask_out = Reshape((inp.shape[-3], inp.shape[-2], -1, class_num))(complex_mask)
    return CustomModel(inputs=inp, outputs=complex_mask_out)


def seperate_single_class(x, y):
    resolution = x.shape[1] // y.shape[1]
    class_num = y.shape[-1] // 4

    classy = y[..., :class_num]
    classy = np.repeat(classy, resolution, axis=1)

    x = x[classy.sum(-1) == 1]
    y = y[y[..., :class_num].sum(-1) == 1]
    return x, y


class Pipeline_Dataset:
    def __init__(self, config, x, y, snr=-10) -> None:
        '''
            x: (time, freq, chan)
            y: (time, class)
        '''
        self.config = config
        self.class_num = y.shape[-1] // 4
        self.resolution = x.shape[1] // y.shape[1]
        self.x, self.y = seperate_single_class(x, y) # seperated x
        self.freq_num = self.x.shape[1]
        self.chan = self.x.shape[2]
        self.seperate_with_class()
        self.snr = snr
        self.label_len = config.len * 10
        self.frame_num = self.label_len * self.resolution

    def seperate_with_class(self):
        for i in range(self.class_num):
            setattr(self, f'x_{i}', self.x[np.repeat(self.y[...,i], self.resolution, axis=0) == 1])
            setattr(self, f'y_{i}', self.y[self.y[...,i] == 1])
        del self.x, self.y

    def gen(self):
        while True:
            class_list = []
            res_x = np.zeros([self.frame_num] + [*self.x_0.shape[1:]] + [3], dtype=self.x_0.dtype)
            res_y = np.zeros([self.label_len] + [*self.y_0.shape[1:]] + [3], dtype=self.y_0.dtype)
            sample_num = int(tf.random.uniform((), minval=1, maxval=3 + 1))
            for i in range(sample_num):
                x_class = np.random.randint(self.class_num)
                while x_class in class_list:
                    x_class = np.random.randint(self.class_num)
                x = getattr(self, f'x_{x_class}')
                y = getattr(self, f'y_{x_class}')
                x_offset = np.random.randint(x.shape[0] - self.frame_num)
                x = np.copy(x[x_offset:x_offset+self.frame_num])
                y = np.copy(y[x_offset // self.resolution: x_offset // self.resolution + self.label_len])

                mask_offset = np.random.randint(y.shape[0] // 2)
                mask_len = np.random.randint(y.shape[0] // 2, y.shape[0] - mask_offset)
                mask = np.concatenate([
                    np.zeros([mask_offset, 1], dtype=y.dtype),
                    np.ones([mask_len, 1], dtype=y.dtype),
                    np.zeros([y.shape[0] - mask_offset - mask_len, 1], dtype=y.dtype)
                ], 0)
                y *= mask
                x *= np.repeat(mask[..., np.newaxis], self.resolution, axis=0)[:x.shape[0]]
                snr = np.random.rand() * self.snr
                x = 10 ** (snr / 20) * x
                res_x[..., i] = x
                res_y[..., i] = y
            yield res_x.sum(-1), res_y.sum(-1), res_x, np.swapaxes(res_y, -2, -1)

    def get(self):
        return tf.data.Dataset.from_generator(self.gen,
            (tf.float32, tf.float32, tf.float32, tf.float32), 
            (tf.TensorShape([self.frame_num, self.freq_num, self.chan]), 
             tf.TensorShape([self.label_len, self.class_num * 4]), 
             tf.TensorShape([self.frame_num, self.freq_num, self.chan, 3]),
             tf.TensorShape([self.label_len, 3, self.class_num * 4]))
        ).prefetch(AUTOTUNE)


def get_maskdata(config, mode: str = 'train'):
    path = os.path.join(config.abspath, 'DCASE2021')
    name = os.path.join(path, f'foa_dev_{mode}_stft_{config.nfft}_{config.hop}')
    with ThreadPoolExecutor() as pool:
        x = list(pool.map(lambda x: joblib.load(x), sorted(glob(name + '/*.joblib'))))

    y = joblib.load(os.path.join(path, f'foa_dev_{mode}_label.joblib'))
    
    x = np.stack(x, 0).transpose(0,2,3,1)
    dataset = Pipeline_Dataset(config, x, y).get()
    batch_transforms = [split_total_labels_to_sed_doa]

    dataset = dataset.batch(config.batch)
    for transforms in batch_transforms:
        dataset = apply_ops(dataset, transforms)
        
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)


def get_dataset(config, mode: str = 'train'):
    path = os.path.join(config.abspath, 'DCASE2021')
    name = os.path.join(path, f'foa_dev_{mode}_stft_{config.nfft}_{config.hop}')
    seconds = config.len
    with ThreadPoolExecutor() as pool:
        x = list(pool.map(lambda x: joblib.load(x), sorted(glob(name + '/*.joblib'))))
    x = np.stack(x, 0).transpose(0,2,3,1)
    y = joblib.load(os.path.join(path, f'foa_dev_{mode}_label.joblib'))
    resolution = x.shape[1] // y.shape[1]

    x = x.reshape([-1, seconds * 10 * resolution] + [*x.shape[2:]])
    x = np.concatenate([x.real, x.imag], -1)
    y = y.reshape([-1, seconds * 10, y.shape[2]])
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    batch_transforms = [split_total_labels_to_sed_doa]

    frame_num = 30
    dataset = dataset.batch(config.batch, drop_remainder=False)

    if mode == 'train':
        dataset.shuffle(x.shape[0])
    for transforms in batch_transforms:
        dataset = apply_ops(dataset, transforms)
        
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)


class_name = {
    0 : 'alarm',
    1 : 'crying baby',
    2 : 'crash',
    3 : 'barking dog',
    4 : 'female scream',
    5 : 'female speech',
    6 : 'footsteps',
    7 : 'knocking on door',
    8 : 'male scream',
    9 : 'male speech',
    10 : 'ringing phone',
    11 : 'piano'}


class sample(tf.keras.callbacks.Callback):
    def __init__(self, config, dataset, path='sample'):
        super(sample, self).__init__()
        if not os.path.exists(os.path.join('sample', config.name)):
            os.makedirs(os.path.join('sample', config.name))
        self.config = config
        self.dataset = dataset
        self.path = path
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch >= 10 - 1:
            for x, _, splited_x, splited_y in self.dataset.take(1):
                results = self.model(x, training=False)
                masked_results_all = x[..., tf.newaxis] * results
                y = tf.argmax(splited_y[0], -1)
                y = tf.reduce_max(y, -2)
                masked_results = tf.gather(masked_results_all, y, axis=-1, batch_dims=1)
                real_imag_idx = masked_results.shape[-2] // 2
                masked_results = tf.complex(masked_results[...,:real_imag_idx,:], masked_results[...,real_imag_idx:,:])

                splited = tf.complex(splited_x[...,:real_imag_idx,:], splited_x[...,real_imag_idx:,:])
                for i in range(2):
                    wave_results = tf.signal.inverse_stft(tf.transpose(masked_results[i], [2,3,0,1]), 1024, 480, 1024)
                    wave_results = tf.transpose(wave_results, [2, 0, 1])
                    raw_results = tf.signal.inverse_stft(tf.transpose(splited[i], [2,3,0,1]), 1024, 480, 1024)
                    raw_results = tf.transpose(raw_results, [2, 0, 1])
                    for num in range(masked_results.shape[-1]):
                        if tf.reduce_sum(splited_y[0], (-3,-1))[i,num] == 0:
                            continue
                        wave = wave_results[..., num]
                        wave = tf.audio.encode_wav(wave, 24000)
                        name = class_name[int(y[i][num])].replace(' ', '_')
                        tf.io.write_file(os.path.join(self.path, self.config.name, f'{epoch + 1}_{i}_{name}.wav'), wave)

                        wave = tf.audio.encode_wav(raw_results[..., num], 24000)
                        tf.io.write_file(os.path.join(self.path, self.config.name, f'{epoch + 1}_{i}_{name}_raw.wav'), wave)
                    raw_results = tf.reduce_sum(raw_results, -1)
                    wave = tf.audio.encode_wav(raw_results, 24000)
                    tf.io.write_file(os.path.join(self.path, self.config.name, f'{epoch + 1}_{i}_all.wav'), wave)


def _mse_100000times(y_true, y_pred):
    return MSE(y_true, y_pred) * 100000


def mse_100000times(y_true, y_pred):
    score = tf.py_function(func=_mse_100000times, inp=[y_true, y_pred], Tout=tf.float32,  name='mse_100000times') # tf 2.x
    return score


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    n_classes = 12
    name = '_'.join(['maskmodel', str(config.lr), str(config.final_lr)])
    if config.schedule:
        name += '_schedule'
    # if config.norm:
    #     name += '_norm'
    config.name = name + '_' + config.name

    # data load
    maskset = get_maskdata(config)
    # trainset = get_dataset(config, 'train')
    # valset = get_dataset(config, 'val')
    # testset = get_dataset(config, 'test')

    tensorboard_path = os.path.join('./tensorboard_log', config.name)
    if not os.path.exists(tensorboard_path):
        print(f'tensorboard log directory: {tensorboard_path}')
        os.makedirs(tensorboard_path)
    writer = SummaryWriter(logdir=tensorboard_path)

    model_path = os.path.join('./saved_model', config.name)
    if not os.path.exists(model_path):
        print(f'saved model directory: {model_path}')
        os.makedirs(model_path)

    x = [x for x, _, _, _ in maskset.take(1)][0]
    input_shape = x.shape[1:]
    model = get_model(input_shape)
    setattr(model, 'train_config', config)
    kernel_regularizer = tf.keras.regularizers.l1_l2(l1=0, l2=0.0001)
    # model = apply_kernel_regularizer(model, kernel_regularizer)

    model.summary()
    if not os.path.exists(os.path.join('saved_model', config.name)):
        os.makedirs(os.path.join('saved_model', config.name))
    optimizer = tf.keras.optimizers.Adam(config.lr)
    criterion = tf.keras.losses.MSE
    callbacks = [ReduceLROnPlateau(monitor='mse_100000times', factor=1 / 2**0.5, patience=3, verbose=1, mode='min'),
                 ModelCheckpoint(os.path.join('saved_model', config.name, "maskmodel.h5"), monitor='mse_100000times', save_best_only=True, verbose=1),
                 EarlyStopping(patience=config.patience, monitor='mse_100000times', verbose=1, mode='min', restore_best_weights=True),
                 sample(config, maskset,path='sample')]
    model.compile(optimizer=optimizer, loss=criterion, metrics=[mse_100000times])
    model.fit(maskset, epochs=config.epoch, batch_size=config.batch, steps_per_epoch=config.steps_per_epoch, callbacks=callbacks,
              use_multiprocessing=True)


if __name__=='__main__':
    main(args)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # input_shape = [320, 128, 7]
    # model = get_model(input_shape)
    # model.summary()
    # from model_flop import get_flops
    # print(get_flops(model))
    
