'''
이 파일은 custom loss를 실험하기 위해 제작된 파일로 third_train.py 기반으로 진행
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
        self.set('--name', type=str, default='')
        self.set('--gpus', type=str, default='3')
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpus
        self.set('--resume', action='store_true')    
        self.set('--abspath', type=str, default='/root/datasets')
        self.set('--output_path', type=str, default='./output')
        self.set('--ans_path', type=str, default='/root/datasets/DCASE2021/metadata_dev/')
        self.set('--norm', type=bool, default=True)
        self.set('--decay', type=float, default=0.9)
        self.set('--sed_th', type=float, default=0.3)
        self.set('--lr', type=float, default=0.001)
        self.set('--final_lr', type=float, default=0.0001)
        self.set('--batch', type=int, default=8)
        self.set('--agc', type=bool, default=False)
        self.set('--epoch', type=int, default=100)
        self.set('--lr_patience', type=int, default=5, help='learning rate decay patience for plateau')
        self.set('--patience', type=int, default=100, help='early stop patience')
        self.set('--use_acs', type=bool, default=True)
        self.set('--use_tfm', type=bool, default=True)
        self.set('--use_tdm', action='store_true')
        self.set('--schedule', type=bool, default=True)
        self.set('--loop_time', type=int, default=5, help='times of train dataset iter for an epoch')
        self.set('--lad_doa_thresh', type=int, default=20)
        self.set('--nfft', type=int, default=1024)
        self.set('--hop', type=int, default=480)
        self.set('--len', type=int, default=4)
        
    def set(self, name, type=str, default=None, action=None, help=''):
        if action == 'store_true':
            type = bool
            default = False
        name = name.split('--')[-1]
        setattr(self, name, type(default))
        
args = ARGS()

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
        # splited_y = (batch, time, one_hot class, 3)

        resolution = x.shape[1] // y[0].shape[1]
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            
            masked_x = x[...,tf.newaxis] * y_pred # (batch, frame_num, freq, chan=8, class=12)
            results = tf.zeros_like(masked_x)
            for i in range(splited_x.shape[-1]):
                sy, sx = splited_y[0][...,i,:], splited_x[...,i]
                results += sx[..., tf.newaxis] * tf.repeat(sy, resolution, axis=1)[..., tf.newaxis, tf.newaxis, :]
            loss = self.compiled_loss(masked_x, results)

        # Compute gradients
        trainable_vars = self.trainable_variables
        
        gradients = tape.gradient(loss, trainable_vars)
        # gradients = adaptive_clip_grad(self.trainable_variables, gradients)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        
        self.compiled_metrics.update_state(x, y_pred)

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
            x1_class = tf.random.uniform((), minval=0, maxval=self.class_num, dtype=tf.int32)
            x2_class = tf.random.uniform((), minval=0, maxval=self.class_num, dtype=tf.int32)
            x3_class = tf.random.uniform((), minval=0, maxval=self.class_num, dtype=tf.int32)
            while x1_class == x2_class:
                x2_class = tf.random.uniform((), minval=0, maxval=self.class_num, dtype=tf.int32)
            while x1_class == x3_class or x2_class == x3_class:
                x3_class = tf.random.uniform((), minval=0, maxval=self.class_num, dtype=tf.int32)

            x1 = getattr(self, f'x_{x1_class}')
            x2 = getattr(self, f'x_{x2_class}')
            x3 = getattr(self, f'x_{x3_class}')
            y1 = getattr(self, f'y_{x1_class}')
            y2 = getattr(self, f'y_{x2_class}')
            y3 = getattr(self, f'y_{x3_class}')
            x1_offset = tf.random.uniform((), minval=0, maxval=x1.shape[0] - self.frame_num, dtype=tf.int32)
            x2_offset = tf.random.uniform((), minval=0, maxval=x2.shape[0] - self.frame_num, dtype=tf.int32)
            x3_offset = tf.random.uniform((), minval=0, maxval=x3.shape[0] - self.frame_num, dtype=tf.int32)
            x1 = x1[x1_offset:x1_offset+self.frame_num]
            x2 = x2[x2_offset:x2_offset+self.frame_num]
            x3 = x3[x3_offset:x3_offset+self.frame_num]
            y1 = y1[x1_offset // self.resolution: x1_offset // self.resolution + self.label_len]
            y2 = y2[x2_offset // self.resolution: x2_offset // self.resolution + self.label_len]
            y3 = y3[x3_offset // self.resolution: x3_offset // self.resolution + self.label_len]

            mask1_offset = tf.random.uniform((), minval=0, maxval=y1.shape[0] // 2, dtype=tf.int32)
            mask1_len = tf.random.uniform((), minval=y1.shape[0] // 2, maxval=y1.shape[0] - mask1_offset, dtype=tf.int32)
            mask1 = tf.concat([
                tf.zeros([mask1_offset, 1], dtype=y1.dtype),
                tf.ones([mask1_len, 1], dtype=y1.dtype),
                tf.zeros([y1.shape[0] - mask1_offset - mask1_len, 1], dtype=y1.dtype)
            ], 0)
            y1 *= mask1
            x1 *= tf.repeat(mask1[..., tf.newaxis], self.resolution, axis=0)[:x1.shape[0]]

            mask2_offset = tf.random.uniform((), minval=0, maxval=y2.shape[0] // 2, dtype=tf.int32)
            mask2_len = tf.random.uniform((), minval=y2.shape[0] // 2, maxval=y2.shape[0] - mask2_offset, dtype=tf.int32)
            mask2 = tf.concat([
                tf.zeros([mask2_offset, 1], dtype=y2.dtype),
                tf.ones([mask2_len, 1], dtype=y2.dtype),
                tf.zeros([y2.shape[0] - mask2_offset - mask2_len, 1], dtype=y2.dtype)
            ], 0)
            y2 *= mask2
            x2 *= tf.repeat(mask2[..., tf.newaxis], self.resolution, axis=0)[:x2.shape[0]]

            mask3_offset = tf.random.uniform((), minval=0, maxval=y3.shape[0] // 2, dtype=tf.int32)
            mask3_len = tf.random.uniform((), minval=y3.shape[0] // 2, maxval=y3.shape[0] - mask3_offset, dtype=tf.int32)
            mask3 = tf.concat([
                tf.zeros([mask3_offset, 1], dtype=y3.dtype),
                tf.ones([mask3_len, 1], dtype=y3.dtype),
                tf.zeros([y3.shape[0] - mask3_offset - mask3_len, 1], dtype=y3.dtype)
            ], 0)
            y3 *= mask3
            x3 *= tf.repeat(mask3[..., tf.newaxis], self.resolution, axis=0)[:x3.shape[0]]
            
            snr = tf.random.uniform((), minval=self.snr, maxval=0, dtype=x1.dtype)
            x = x1 + 10 ** (snr / 20) * x2
            snr = tf.random.uniform((), minval=self.snr, maxval=0, dtype=x1.dtype)
            x += 10 ** (snr / 20) * x3
            y = y1 + y2 + y3
            yield x, y, tf.stack([x1, x2, x3], -1), tf.stack([y1, y2, y3], -2)

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


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    n_classes = 12
    name = '_'.join(['2', str(config.lr), str(config.final_lr)])
    if config.schedule:
        name += '_schedule'
    if config.norm:
        name += '_norm'
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

    optimizer = tf.keras.optimizers.Adam(config.lr)
    criterion = tf.keras.losses.MSE
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=1 / 2**0.5, patience=5, verbose=1, mode='min'),
                 ModelCheckpoint("maskmodel.h5", monitor='val_loss', save_best_only=True, verbose=1),
                 EarlyStopping(patience=config.patience, verbose=1, mode='min')]
    model.compile(optimizer=optimizer, loss=criterion)
    model.fit(maskset, epochs=config.epoch, batch_size=config.batch, steps_per_epoch=200, callbacks=callbacks,
              validation_batch_size=config.batch * 4, validation_steps=20, validation_data=maskset, use_multiprocessing=True)
    exit()
    if config.resume:
        _model_path = sorted(glob(model_path + '/*.hdf5'))
        if len(_model_path) == 0:
            raise ValueError('the model does not exist, cannot be resumed')
        model = tf.keras.models.load_model(_model_path[0])

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
            tf.keras.models.save_model(
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
            # decay_coefficient = (config.final_lr / config.lr) ** (1 / config.epoch)
            # print(f'lr: {optimizer.learning_rate.numpy():.3} -> {(optimizer.learning_rate * decay_coefficient).numpy():.3}')
            # optimizer.learning_rate = optimizer.learning_rate * decay_coefficient
            decay_coefficient = (config.final_lr - config.lr) / config.epoch
            print(f'lr: {optimizer.learning_rate.numpy():.3} -> {(optimizer.learning_rate + decay_coefficient).numpy():.3}')
            optimizer.learning_rate = optimizer.learning_rate + decay_coefficient

            

    # end of training
    print(f'epoch: {epoch}')


if __name__=='__main__':
    main(args)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # input_shape = [320, 128, 7]
    # model = get_model(input_shape)
    # model.summary()
    # from model_flop import get_flops
    # print(get_flops(model))
    
