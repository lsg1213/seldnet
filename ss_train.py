import argparse
import os
import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.tensor_shape import TensorShape
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


distributed_strategy = None
EPSILON = 1e-6

args = argparse.ArgumentParser()
    
args.add_argument('--name', type=str, required=True)
args.add_argument('--mask', type=int, required=True)

args.add_argument('--gpus', type=str, default='-1')
args.add_argument('--resume', action='store_true')    
args.add_argument('--abspath', type=str, default='/root/datasets')
args.add_argument('--output_path', type=str, default='./output')
args.add_argument('--ans_path', type=str, default='/root/datasets/DCASE2021/metadata_dev/')
args.add_argument('--norm', type=bool, default=True)
args.add_argument('--nfft', type=int, default=1024)
args.add_argument('--hop', type=int, default=480)
args.add_argument('--len', type=int, default=4)
args.add_argument('--mel', type=int, default=128)


# training
args.add_argument('--masktrain', action='store_true')
args.add_argument('--pretrain', action='store_true')
args.add_argument('--pt', type=str, default='')
args.add_argument('--decay', type=float, default=0.9)
args.add_argument('--sed_th', type=float, default=0.3)
args.add_argument('--lr', type=float, default=0.0002)
args.add_argument('--final_lr', type=float, default=0.00001)
args.add_argument('--batch', type=int, default=8)
args.add_argument('--agc', type=bool, default=False)
args.add_argument('--epoch', type=int, default=60)
args.add_argument('--lr_patience', type=int, default=5, 
                    help='learning rate decay patience for plateau')
args.add_argument('--patience', type=int, default=100, 
                    help='early stop patience')
args.add_argument('--use_acs', type=bool, default=True)
args.add_argument('--use_tfm', type=bool, default=True)
args.add_argument('--use_tdm', action='store_true')
args.add_argument('--schedule', type=bool, default=True)
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
    x = keras.layers.Convolution2D(512, (3, 1), (2, 1), use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.AveragePooling2D((1,2))(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.TimeDistributed(keras.layers.Flatten())(x)

    x = keras.layers.Bidirectional(keras.layers.GRU(256, return_sequences=True), merge_mode='sum')(x)
    x = keras.layers.Bidirectional(keras.layers.GRU(256, return_sequences=True), merge_mode='sum')(x)
    x = keras.layers.Conv1DTranspose(256, 2, strides=2)(x)
    x = keras.layers.Conv1DTranspose(256, 2, strides=2)(x)
    x = keras.layers.Conv1DTranspose(256, 2, strides=2)(x)

    x = keras.layers.Dense(36)(x)
    x = keras.layers.Activation('tanh')(x)
    return keras.Model(inputs=inp, outputs=x)

@tf.function
def get_accdoa_labels(accdoa_in, nb_classes, sed_th=0.3):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:]
    sed = tf.cast(tf.sqrt(x**2 + y**2 + z**2) > sed_th, tf.float32)
    return sed, accdoa_in


def foa_intensity_vectors_tf(spectrogram, eps=1e-8):
    # complex_specs: [chan, time, freq]
    conj_zero = tf.math.conj(spectrogram[0])
    IVx = tf.math.real(conj_zero * spectrogram[3])
    IVy = tf.math.real(conj_zero * spectrogram[1])
    IVz = tf.math.real(conj_zero * spectrogram[2])

    norm = tf.math.sqrt(IVx**2 + IVy**2 + IVz**2)
    norm = tf.math.maximum(norm, eps)
    IVx = IVx / norm
    IVy = IVy / norm
    IVz = IVz / norm

    # apply mel matrix without db ...
    return tf.stack([IVx, IVy, IVz], 0)


def stft_to_mel_intensity_vector(config):
    sr = 24000
    mel_mat = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=config.mel,
                                                    num_spectrogram_bins=config.nfft//2+1,
                                                    sample_rate=sr,
                                                    lower_edge_hertz=0,
                                                    upper_edge_hertz=sr//2)
    
    def _stft_to_mel_intensity_vector(stft):
        # (batch, time, freq, (4+4 real, imag), 13)
        chan = stft.shape[-2] // 2
        stft = tf.transpose(stft, [3,4,0,1,2])
        stft_real = stft[:chan,...]
        stft = tf.complex(stft_real, stft[chan:,...])
        inten_vec = foa_intensity_vectors_tf(stft)

        mel = tf.matmul(stft_real, mel_mat)
        mel = tf.math.log(mel + EPSILON)
        inten_vec = tf.matmul(inten_vec, mel_mat)
        stft = tf.concat([stft_real, inten_vec], 0)
        return tf.transpose(out, [2,3,4,0,1])
    return _stft_to_mel_intensity_vector


def generate_trainstep(criterion, config):
    global distributed_strategy
    # These are statistics from the train dataset
    # train_samples = tf.convert_to_tensor(
    #     [[58193, 32794, 29801, 21478, 14822, 
    #     9174, 66527,  6740,  9342,  6498, 
    #     22218, 49758]],
    #     dtype=tf.float32)
    # cls_weights = tf.reduce_mean(train_samples) / train_samples
    transform_function = stft_to_mel_intensity_vector(config)
    def trainstep(maskmodel, model, x, y, optimizer):
        with tf.GradientTape() as tape:
            masked_x = maskmodel(x, training=config.masktrain)
            x = tf.concat([masked_x, x[..., tf.newaxis]], -1)
            x = transform_function(x)
            x = tf.reshape(x, [*x.shape[:-2]] + [-1])
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
    @tf.function
    def distributed_train_step(dist_inputs):
        per_replica_losses = distributed_strategy.run(trainstep, args=(dist_inputs,))
        return distributed_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    if distributed_strategy is None:
        return tf.function(trainstep)
    else: 
        return distributed_train_step


def generate_teststep(criterion, config):
    transform_function = stft_to_mel_intensity_vector(config)
    @tf.function
    def teststep(maskmodel, model, x, y, optimizer=None):
        masked_x = maskmodel(x, training=False)
        x = tf.concat([masked_x, x[..., tf.newaxis]], -1)
        x = transform_function(x)
        x = tf.reshape(x, [*x.shape[:-2]] + [-1])
        y_p = model(x, training=False)
        loss = criterion(y[1], y_p)
        return y_p, loss
    return teststep


def generate_iterloop(criterion, evaluator, writer, 
                      mode, config=None):
    if mode == 'train':
        step = generate_trainstep(criterion, config)
    else:
        step = generate_teststep(criterion, config)

    def iterloop(maskmodel, model, dataset, epoch, optimizer=None):
        evaluator.reset_states()
        losses = keras.metrics.Mean()

        with tqdm(dataset) as pbar:
            for x, y in pbar:
                preds, loss = step(maskmodel, model, x, y, optimizer)
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


def get_stftdata(config, mode: str = 'train'):
    path = os.path.join(config.abspath, 'DCASE2021')
    name = os.path.join(path, f'foa_dev_{mode}_stft_{config.nfft}_{config.hop}')
    with ThreadPoolExecutor() as pool:
        x = list(pool.map(lambda x: joblib.load(x), sorted(glob(name + '/*.joblib'))))

    y = joblib.load(os.path.join(path, f'foa_dev_{mode}_label.joblib'))
    
    x = np.stack(x, 0).transpose(0,2,3,1)
    resolution = x.shape[1] // y.shape[1]
    x = x.reshape([-1, config.len * 10 * resolution] + [*x.shape[2:]])
    y = y.reshape([-1, config.len * 10, y.shape[-1]])
    
    def gen():
        for X, Y in zip(x, y):
            yield X, Y
    dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.float32), 
              output_shapes=(tf.TensorShape([config.len * 10 * resolution] + [*x.shape[2:]]),
                             tf.TensorShape([config.len * 10, y.shape[-1]])))

    batch_transforms = [split_total_labels_to_sed_doa]

    dataset = dataset.batch(config.batch)
    if mode == 'train':
        dataset = dataset.shuffle(x.shape[0] // config.batch)
    for transforms in batch_transforms:
        dataset = apply_ops(dataset, transforms)
        
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    if len(config.gpus.split(',')) > 2:
        global distributed_strategy
        distributed_strategy = tf.distribute.MirroredStrategy()
    n_classes = 12
    name = '_'.join(['ss2', str(config.lr), str(config.final_lr)]) # np: none pretrained mask, pm: pretrained mask
    if config.masktrain:
        name += '_masktrain'
    if config.pretrain:
        name += '_pretrain'
    if config.schedule:
        name += '_schedule'
    if config.norm:
        name += '_norm'
    config.name = name + '_' + config.name


    # data load
    trainset = get_stftdata(config, 'train')
    valset = get_stftdata(config, 'val')
    testset = get_stftdata(config, 'test')
    if distributed_strategy is not None:
        trainset = distributed_strategy.experimental_distribute_dataset(trainset)
        valset = distributed_strategy.experimental_distribute_dataset(valset)
        testset = distributed_strategy.experimental_distribute_dataset(testset)

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
    stft_shape = [config.len * 10 * 5] + [*input_shape[1:]]
    input_shape = [config.len * 10 * 5, config.mel, (input_shape[-1] // 2 + 3) * 13]
    if distributed_strategy is not None:
        with distributed_strategy.scope():
            model = get_model(input_shape)
    else:
        model = get_model(input_shape)

    if config.mask == 1:
        import mask1_train
        if distributed_strategy is not None:
            with distributed_strategy.scope():
                maskmodel = mask1_train.get_model(stft_shape)
                if not config.pretrain:
                    maskmodel.load_weights(os.path.join('saved_model', config.pt, 'maskmodel.h5'))
        else:
            maskmodel = mask1_train.get_model(stft_shape)
            if not config.pretrain:
                maskmodel.load_weights(os.path.join('saved_model', config.pt, 'maskmodel.h5'))
    kernel_regularizer = tf.keras.regularizers.l1_l2(l1=0, l2=0.0001)
    # model = apply_kernel_regularizer(model, kernel_regularizer)

    model.summary()

    if distributed_strategy is not None:
        with distributed_strategy.scope():
            optimizer = keras.optimizers.Adam(config.lr)
    else:
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
        train_iterloop(maskmodel, model, trainset, epoch, optimizer)
        score = val_iterloop(maskmodel, model, valset, epoch)
        test_iterloop(maskmodel, model, testset, epoch)

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
            # decay_coefficient = (config.final_lr / config.lr) ** (1 / config.epoch)
            # print(f'lr: {optimizer.learning_rate.numpy():.3} -> {(optimizer.learning_rate * decay_coefficient).numpy():.3}')
            # optimizer.learning_rate = optimizer.learning_rate * decay_coefficient
            decay_coefficient = (config.final_lr - config.lr) / config.epoch
            print(f'lr: {optimizer.learning_rate.numpy():.3} -> {(optimizer.learning_rate + decay_coefficient).numpy():.3}')
            optimizer.learning_rate = optimizer.learning_rate + decay_coefficient

            

    # end of training
    print(f'epoch: {epoch}')


if __name__=='__main__':
    main(args.parse_args())
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # input_shape = [320, 128, 7]
    # model = get_model(input_shape)
    # model.summary()
    # from model_flop import get_flops
    # print(get_flops(model))
    