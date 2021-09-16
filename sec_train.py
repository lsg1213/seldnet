import argparse
import os
import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from glob import glob
from numpy import inf
from tensorboardX import SummaryWriter
from tqdm import tqdm

import layers
import losses
import models
from data_loader import *
from metrics import * 
from swa import SWA
from transforms import *
from utils import adaptive_clip_grad, apply_kernel_regularizer
from params import get_param

args = argparse.ArgumentParser()
    
args.add_argument('--name', type=str, required=True)

args.add_argument('--gpus', type=str, default='-1')
args.add_argument('--resume', action='store_true')    
args.add_argument('--abspath', type=str, default='/root/datasets')
args.add_argument('--output_path', type=str, default='./output')
args.add_argument('--ans_path', type=str, default='/root/datasets/DCASE2021/metadata_dev/')


# training
args.add_argument('--lr', type=float, default=0.001)
args.add_argument('--decay', type=float, default=0.5)
args.add_argument('--batch', type=int, default=256)
args.add_argument('--agc', type=bool, default=False)
args.add_argument('--epoch', type=int, default=100)
args.add_argument('--lr_patience', type=int, default=80, 
                    help='learning rate decay patience for plateau')
args.add_argument('--patience', type=int, default=100, 
                    help='early stop patience')
args.add_argument('--use_acs', type=bool, default=True)
args.add_argument('--use_tfm', type=bool, default=True)
args.add_argument('--use_tdm', action='store_true')
args.add_argument('--loop_time', type=int, default=5, 
                    help='times of train dataset iter for an epoch')
args.add_argument('--lad_doa_thresh', type=int, default=20)


def bn_conv_block(inp, chn, kernel_size=3, dilation=1, pad=1, stride=1, test=False):
    h = tf.keras.layers.BatchNormalization()(inp)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.ZeroPadding2D(padding=pad)(h)
    out = tf.keras.layers.Conv2D(chn, kernel_size, strides=stride, dilation_rate=dilation)(h)
    return out


def dilated_dense_block(inp, growth_rate, num_layers, kernel_size=3, pad=1, dilation=True):
    '''
    Define Dilated Dense Block
    '''
    h = bn_conv_block(inp, growth_rate*num_layers, dilation=1, kernel_size=kernel_size, pad=pad)

    if num_layers > 1:
        lst = []
        for i in range(num_layers):
            # Split Variable(h) and append them in lst.
            lst.append(h[..., i*growth_rate:(i+1)*growth_rate])

        def update(inp_, n):
            for j in range(num_layers-n-1):
                lst[j+1+n] += inp_[..., j*growth_rate:(j+1)*growth_rate]

        for i in range(num_layers - 1):
            d = int(2**(i+1)) if dilation else 1
            tmp = bn_conv_block(lst[i], growth_rate*(num_layers-i), # (num_layers - 1 - i)에서 수정
                                dilation=d, kernel_size=kernel_size, pad=pad*d)
            update(tmp, i)
        # concatenate the splitted and updated Variables from the lst
        h = tf.keras.layers.Concatenate()(lst)
    return h[..., -growth_rate:]


def d3_block(inp, num_layers, growth_rate, n_blocks, kernel_size=3, pad=1, dilation=True, test=False):
    '''
    Define D3Block
    '''
    h = dilated_dense_block(inp, growth_rate*n_blocks, num_layers, kernel_size=kernel_size, pad=pad, dilation=dilation)
    if n_blocks > 1:
        lst = []
        for i in range(n_blocks):
            lst.append(h[..., i*growth_rate:(i+1)*growth_rate])

        def update(inp_, n):
            for j in range(n_blocks-n-1):
                lst[j+1+n] += inp_[..., j*growth_rate:(j+1)*growth_rate]

        for i in range(n_blocks-1):
            tmp = dilated_dense_block(lst[i], growth_rate*(n_blocks-i-1), num_layers, kernel_size=kernel_size, pad=pad, dilation=dilation)
            update(tmp, i)
        h = tf.keras.layers.Concatenate()(lst)
    return h[..., -growth_rate:]


def single_lstm(inp, hidden_size, dropout=0., bidirectional=False):
    if bidirectional:
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size, dropout=dropout, return_sequences=True))(inp)
    else:
        x = tf.keras.layers.LSTM(hidden_size, dropout=dropout, return_sequences=True)(inp)
    x = tf.keras.layers.Dense(inp.shape[-1])(x)
    return x


def DPRNN(inp, hidden_size, output_size, dropout=0, num_layers=1, bidirectional=True):
    x = inp # B, dim1, dim2, N
    output = inp
    for i in range(num_layers):
        if i == num_layers - 1:
            hidden_size = output_size
        dim1, dim2, N = x.shape[1:]
        row_input = tf.keras.layers.Permute([2, 1, 3])(x) # B, dim2, dim1, N
        row_input = tf.reshape(row_input, [-1, dim1, N])
        row_output = single_lstm(row_input, hidden_size, dropout=dropout, bidirectional=bidirectional)
        row_output = tf.reshape(row_output, [-1,dim2,dim1,N])
        row_output = tf.keras.layers.Permute([2, 1, 3])(row_output)
        row_output = tfa.layers.GroupNormalization(1, epsilon=1e-8)(row_output)
        
        col_input = tf.reshape(x, [-1, dim2, N])
        col_output = single_lstm(col_input, hidden_size, dropout=dropout, bidirectional=bidirectional)
        col_output = tf.reshape(col_output, [-1,dim1,dim2,N])
        col_output = tfa.layers.GroupNormalization(1, epsilon=1e-8)(col_output)

        output += row_output + col_output
    return output # (B, dim1, dim2, output_size)


# https://github.com/sony/ai-research-code/blob/596d4ba79737de3bcf4f0f8bd934195c90c957c7/d3net/music-source-separation/model.py#L60
# https://github.com/ShiZiqiang/dual-path-RNNs-DPRNNs-based-speech-separation/blob/master/models.py
def get_model(input_shape):
    inp = tf.keras.layers.Input(shape = input_shape)
    x = d3_block(inp, 4, 16, 2)
    x = tf.keras.layers.AveragePooling2D(strides=(2,2), padding='same')(x)
    x = d3_block(x, 4, 24, 2)
    x = tf.keras.layers.AveragePooling2D(strides=(2,2), padding='same')(x)
    x = d3_block(x, 4, 32, 2)
    x = tf.keras.layers.AveragePooling2D(strides=(2,2), padding='same')(x)
    x = d3_block(x, 4, 40, 2)

    x = DPRNN(x, 100, 160, num_layers=4, bidirectional=True)
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 0))(x)
    x = tf.keras.layers.AveragePooling2D(strides=(2,2), padding='same')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    x = tf.keras.layers.Dense(36)(x)
    x = tf.keras.layers.UpSampling1D(3)(x)
    x = tf.keras.layers.Activation('tanh')(x)

    return tf.keras.Model(inputs=inp, outputs=x)


def get_accdoa_labels(accdoa_in, nb_classes):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:]
    sed = tf.cast(tf.sqrt(x**2 + y**2 + z**2) > 0.5, tf.float32)
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
        losses = tf.keras.metrics.Mean()

        with tqdm(dataset) as pbar:
            for x, y in pbar:
                preds, loss = step(model, x, y, optimizer)
                y, preds = y, get_accdoa_labels(preds, preds.shape[-1] // 3)
                
                evaluator.update_states(y, preds)
                metric_values = evaluator.result()
                seld_score = calculate_seld_score(metric_values)

                losses(loss)
                pbar.set_postfix(
                    OrderedDict({
                        'mode': mode,
                        'epoch': epoch, 
                        'ER': metric_values[0].numpy(),
                        'F': metric_values[1].numpy(),
                        'DER': metric_values[2].numpy(),
                        'DERF': metric_values[3].numpy(),
                        'seldscore': seld_score.numpy()
                    }))

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
    path = os.path.join(config.abspath, 'DCASE2021/feat_label/')

    x, y = load_seldnet_data(os.path.join(path, 'foa_dev_norm'),
                             os.path.join(path, 'foa_dev_label'), 
                             mode=mode, n_freq_bins=64)
    if config.use_tfm and mode == 'train':
        sample_transforms = [
            random_ups_and_downs,
            # lambda x, y: (mask(x, axis=-2, max_mask_size=8, n_mask=6), y),
            lambda x, y: (mask(x, axis=-2, max_mask_size=16), y),
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
        label_window_size=60,
        batch_size=config.batch,
        sample_transforms=sample_transforms,
        loop_time=config.loop_time
    )
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)


def ensemble_outputs(model, xs: list, 
                     win_size=300, step_size=5, batch_size=256):
    @tf.function
    def predict(model, x, batch_size):
        windows = tf.signal.frame(x, win_size, step_size, axis=0)

        sed, doa = [], []
        for i in range(int(np.ceil(windows.shape[0]/batch_size))):
            y_p = model(windows[i*batch_size:(i+1)*batch_size], training=False)
            s, d = get_accdoa_labels(y_p, y_p.shape[-1] // 3)
            sed.append(s)
            doa.append(d)
        sed = tf.concat(sed, 0)
        doa = tf.concat(doa, 0)

        # windows to seq
        total_counts = tf.signal.overlap_and_add(
            tf.ones((sed.shape[0], win_size//step_size), dtype=sed.dtype),
            1)[..., tf.newaxis]
        sed = tf.signal.overlap_and_add(tf.transpose(sed, (2, 0, 1)), 1)
        sed = tf.transpose(sed, (1, 0)) / total_counts
        doa = tf.signal.overlap_and_add(tf.transpose(doa, (2, 0, 1)), 1)
        doa = tf.transpose(doa, (1, 0)) / total_counts

        return sed, doa

    # assume 0th dim of each sample is time dim
    seds = []
    doas = []
    
    for x in xs:
        sed, doa = predict(model, x, batch_size)
        seds.append(sed)
        doas.append(doa)

    return list(zip(seds, doas))


def generate_evaluate_fn(test_xs, test_ys, evaluator, batch_size=256,
                         writer=None):
    def evaluate_fn(model, epoch):
        start = time.time()
        evaluator.reset_states()
        e_outs = ensemble_outputs(model, test_xs, batch_size=batch_size)

        for y, pred in zip(test_ys, e_outs):
            evaluator.update_states(y, pred)

        metric_values = evaluator.result()
        seld_score = calculate_seld_score(metric_values).numpy()
        er, f, der, derf = list(map(lambda x: x.numpy(), metric_values))

        if writer is not None:
            writer.add_scalar('ENS_T/ER', er, epoch)
            writer.add_scalar('ENS_T/F', f, epoch)
            writer.add_scalar('ENS_T/DER', der, epoch)
            writer.add_scalar('ENS_T/DERF', derf, epoch)
            writer.add_scalar('ENS_T/seldScore', seld_score, epoch)
        print('ensemble outputs')
        print(f'ER: {er:4f}, F: {f:4f}, DER: {der:4f}, DERF: {derf:4f}, '
              f'SELD: {seld_score:4f} '
              f'({time.time()-start:.4f} secs)')
        return seld_score, metric_values
    return evaluate_fn


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    swa_start_epoch = 80
    swa_freq = 2
    n_classes = 12

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

    path = os.path.join(config.abspath, 'DCASE2021/feat_label/')
    test_xs, test_ys = load_seldnet_data(
        os.path.join(path, 'foa_dev_norm'),
        os.path.join(path, 'foa_dev_label'), 
        mode='test', n_freq_bins=64)
    test_ys = list(map(
        lambda x: split_total_labels_to_sed_doa(None, x)[-1], test_ys))

    x, _ = [(x, y) for x, y in valset.take(1)][0]
    input_shape = x.shape[1:]
    model = get_model(input_shape)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(config.lr)
    criterion = tf.keras.losses.MSE

    # stochastic weight averaging
    swa = SWA(model, swa_start_epoch, swa_freq)

    if config.resume:
        _model_path = sorted(glob(model_path + '/*.hdf5'))
        if len(_model_path) == 0:
            raise ValueError('the model does not exist, cannot be resumed')
        model = tf.keras.models.load_model(_model_path[0])

    best_score = inf
    early_stop_patience = 0
    lr_decay_patience = 0
    evaluator = SELDMetrics(
        doa_threshold=config.lad_doa_thresh, n_classes=n_classes)

    train_iterloop = generate_iterloop(
        criterion, evaluator, writer, 'train', 
        config=config)
    val_iterloop = generate_iterloop(
        criterion, evaluator, writer, 'val')
    test_iterloop = generate_iterloop(
        criterion, evaluator, writer, 'test')
    evaluate_fn = generate_evaluate_fn(
        test_xs, test_ys, evaluator, config.batch*4, writer=writer)

    for epoch in range(config.epoch):
        if epoch == swa_start_epoch:
            tf.keras.backend.set_value(optimizer.lr, config.lr * 0.5)

        if epoch % 10 == 0:
            evaluate_fn(model, epoch)

        # train loop
        train_iterloop(model, trainset, epoch, optimizer)
        score = val_iterloop(model, valset, epoch)
        test_iterloop(model, testset, epoch)

        swa.on_epoch_end(epoch)

        if best_score > score:
            os.system(f'rm -rf {model_path}/bestscore_{best_score}.hdf5')
            best_score = score
            early_stop_patience = 0
            lr_decay_patience = 0
            tf.keras.models.save_model(
                model, 
                os.path.join(model_path, f'bestscore_{best_score}.hdf5'), 
                include_optimizer=False)
        else:
            '''
            if lr_decay_patience == config.lr_patience and config.decay != 1:
                optimizer.learning_rate = optimizer.learning_rate * config.decay
                print(f'lr: {optimizer.learning_rate.numpy()}')
                lr_decay_patience = 0
            '''
            if early_stop_patience == config.patience:
                print(f'Early Stopping at {epoch}, score is {score}')
                break
            early_stop_patience += 1
            lr_decay_patience += 1

    # end of training
    print(f'epoch: {epoch}')
    swa.on_train_end()

    seld_score, *_ = evaluate_fn(model, epoch)

    tf.keras.models.save_model(
        model, 
        os.path.join(model_path, f'SWA_best_{seld_score:.5f}.hdf5'),
        include_optimizer=False)





if __name__=='__main__':
    main(args.parse_args())
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # model = get_model([300, 64, 7])
    # from model_flop import get_flops
    # model.summary()
    # print(get_flops(model))