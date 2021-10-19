import argparse
import os
import time
from collections import OrderedDict
from math import ceil, log

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
from transforms import *
from utils import adaptive_clip_grad, apply_kernel_regularizer
from params import get_param

args = argparse.ArgumentParser()
    
args.add_argument('--name', type=str, required=True)
args.add_argument('--model', type=str, default='DPRNN')

args.add_argument('--gpus', type=str, default='-1')
args.add_argument('--resume', action='store_true')    
args.add_argument('--abspath', type=str, default='/root/datasets')
args.add_argument('--output_path', type=str, default='./output')
args.add_argument('--ans_path', type=str, default='/root/datasets/DCASE2021/metadata_dev/')
args.add_argument('--norm', action='store_true')
args.add_argument('--data', type=str, default='mel', choices=['mel','stft'])


# training
args.add_argument('--lr', type=float, default=0.001)
args.add_argument('--resolution', type=int, default=10)
args.add_argument('--iters', type=int, default=10000)
args.add_argument('--decay', type=float, default=0.9)
args.add_argument('--batch', type=int, default=32)
args.add_argument('--agc', type=bool, default=False)
args.add_argument('--epoch', type=int, default=400000)
args.add_argument('--lr_patience', type=int, default=40000, 
                    help='learning rate decay patience for plateau')
args.add_argument('--patience', type=int, default=100000, 
                    help='early stop patience')
args.add_argument('--use_acs', type=bool, default=True)
args.add_argument('--use_tfm', type=bool, default=True)
args.add_argument('--use_tdm', action='store_true')
args.add_argument('--loop_time', type=int, default=5, 
                    help='times of train dataset iter for an epoch')
args.add_argument('--lad_doa_thresh', type=int, default=20)


def bn_conv_block(inp, chn, kernel_size=3, dilation=1, pad=1, stride=1):
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
            tmp = bn_conv_block(lst[i], growth_rate*(num_layers - i - 1),
                                dilation=d, kernel_size=kernel_size, pad=pad*d)
            update(tmp, i)
        # concatenate the splitted and updated Variables from the lst
        h = tf.keras.layers.Concatenate()(lst)
    return h[..., -growth_rate:]


def d3_block(inp, num_layers, growth_rate, n_blocks, kernel_size=3, pad=1, dilation=True):
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
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size, dropout=dropout, return_sequences=True)))(inp)
    else:
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.LSTM(hidden_size, dropout=dropout, return_sequences=True))(inp)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(inp.shape[-1]))(x)
    return x


def DPRNN(inp, hidden_size, output_size, dropout=0, num_layers=1, bidirectional=True):
    output = inp # B, dim1, dim2, N
    for i in range(num_layers):
        if i == num_layers - 1:
            hidden_size = output_size
        row_input = tf.keras.layers.Permute([2, 1, 3])(output) # B, dim2, dim1, N
        row_output = single_lstm(row_input, hidden_size, dropout=dropout, bidirectional=bidirectional)
        row_output = tf.keras.layers.Permute([2, 1, 3])(row_output)
        row_output = tfa.layers.GroupNormalization(1, epsilon=1e-8)(row_output)
        output += row_output
        
        col_output = single_lstm(output, hidden_size, dropout=dropout, bidirectional=bidirectional)
        col_output = tfa.layers.GroupNormalization(1, epsilon=1e-8)(col_output)

        output += col_output
    return output # (B, dim1, dim2, output_size)


# https://github.com/sony/ai-research-code/blob/596d4ba79737de3bcf4f0f8bd934195c90c957c7/d3net/music-source-separation/model.py#L60
# https://github.com/ShiZiqiang/dual-path-RNNs-DPRNNs-based-speech-separation/blob/master/models.py
def get_model(input_shape, config):
    inp = tf.keras.layers.Input(shape = input_shape)
    x = d3_block(inp, 4, 16, 2)
    x = tf.keras.layers.AveragePooling2D(strides=(2,2), padding='valid')(x)
    x = d3_block(x, 4, 24, 2)
    x = tf.keras.layers.AveragePooling2D(strides=(2,2), padding='valid')(x)
    x = d3_block(x, 4, 32, 2)
    x = tf.keras.layers.AveragePooling2D(strides=(2,2), padding='valid')(x)
    x = d3_block(x, 4, 40, 2)
    
    if config.model == 'DPRNN':
        x = DPRNN(x, 160, 160, num_layers=4, bidirectional=True)
    x = tf.keras.layers.AveragePooling2D((1,2), padding='valid')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    if config.model == 'GRU':
        x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(160, return_sequences=True))(x)
    x = tf.keras.layers.Dense(36)(x)
    if config.data == 'stft':
        x = tf.keras.layers.Conv1D(52, 1, use_bias=False, data_format='channels_first')(x)
    elif config.data == 'mel':
        x = tf.keras.layers.Conv1D(60, 1, use_bias=False, data_format='channels_first')(x)

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


def generate_valstep(criterion, config):
    @tf.function
    def valstep(model, x, y, optimizer=None):
        y_p = model(x, training=False)
        loss = criterion(y[1], y_p)
        return y_p, loss
    return valstep
    

def generate_teststep(criterion, config):
    @tf.function
    def teststep(model, x):
        y_p = model(x, training=False)
        return y_p
    return teststep


def generate_iterloop(criterion, evaluator, writer, 
                      mode, config=None):
    if mode == 'train':
        step = generate_trainstep(criterion, config)
    elif mode == 'val':
        step = generate_valstep(criterion, config)
    else:
        step = generate_teststep(criterion, config)

    @tf.function
    def overlap_and_criterion(preds, y, criterion):
        preds = tf.transpose(preds, [2, 0, 1])
        total_counts = tf.signal.overlap_and_add(tf.ones_like(preds), config.frame_step // config.resolution)[..., :y[1].shape[0]]
        preds = tf.signal.overlap_and_add(preds, config.frame_step // config.resolution)[..., :y[1].shape[0]]
        preds /= total_counts
        preds = tf.transpose(preds, [1, 0])[tf.newaxis,...]
        loss = criterion(y[1][tf.newaxis,...], preds)
        return preds, loss

    def iterloop(model, dataset, epoch, optimizer=None):
        evaluator.reset_states()
        losses = tf.keras.metrics.Mean()

        with tqdm(dataset) as pbar:
            for x, y in pbar:
                if mode in ('test'):
                    smalldata = tf.data.Dataset.from_tensor_slices(x)
                    smalldata = smalldata.batch(config.batch).prefetch(AUTOTUNE)
                    preds = tf.concat([step(model, x_) for x_ in smalldata], 0)
                    preds, loss = overlap_and_criterion(preds, y, criterion)
                else:
                    preds, loss = step(model, x, y, optimizer)
                preds = get_accdoa_labels(preds, preds.shape[-1] // 3)
                
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

                # if mode == 'train' and epoch * config.iters < 50000:
                #     final_lr = 0.001
                #     lr_coefficient = (final_lr / config.lr) ** (1 / 50000) # 50000 root (0.001 / 0.0001)
                #     next_lr = min(optimizer.learning_rate * (lr_coefficient ** config.batch), final_lr / (32 / config.batch))
                #     optimizer.learning_rate = next_lr
                #     tf.keras.backend.set_value(optimizer.lr, next_lr)

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


class Custom_dataset:
    def __init__(self, config, mode='val') -> None:
        self.path = os.path.join(config.abspath, 'DCASE2021')
        self.config = config
        self.mode = mode
        self.load_raw_data()

    def load_raw_data(self):
        self.x = joblib.load(os.path.join(self.path, f'foa_dev_{self.mode}_stft_480.joblib'))
        self.y = joblib.load(os.path.join(self.path, f'foa_dev_{self.mode}_label.joblib'))

    @staticmethod
    def generator(x, y):
        def _generator():
            for x_, y_ in zip(x, y):
                yield x_, y_
        return _generator

    def get(self):
        frame_num = 512
        x, y = self.x, self.y

        if self.mode == 'val':
            x = x.reshape([-1, x.shape[-2], x.shape[-1]])
            y = y.reshape([-1, y.shape[-1]])
            label_num = ceil(frame_num / (x.shape[0] / y.shape[0]))
            frame_len = label_num * (x.shape[0] // y.shape[0])
            x = np.pad(x, ((0, frame_len - (x.shape[0] % frame_len)), (0,0), (0,0)))
            # frame_len 520, frame_num 512
            x = x.reshape((-1, frame_len, x.shape[-2], x.shape[-1]))[:,:frame_num]
            y = np.pad(y, ((0, label_num - (y.shape[0] % label_num)),(0,0)))
            y = y.reshape((-1, label_num, y.shape[-1]))

            dataset = tf.data.Dataset.from_generator(self.generator(x, y), output_signature=(
                tf.TensorSpec((frame_num, x.shape[-2], None), dtype=x.dtype),
                tf.TensorSpec((label_num, y.shape[-1]), dtype=y.dtype)
            ))
        elif self.mode == 'test':
            hop_len =  20 * (x.shape[1] // y.shape[1])
            config = vars(self.config)
            config['frame_step'] = hop_len
            config['resolution'] = x.shape[1] // y.shape[1]
            self.config = argparse.Namespace(**config)

            def frame(data):
                return tf.signal.frame(data, frame_num, hop_len, axis=0, pad_end=True).numpy()
            
            x = np.stack(list(map(frame, x)), 0)
            dataset = tf.data.Dataset.from_generator(self.generator(x, y), output_signature=(
                tf.TensorSpec((x.shape[1], frame_num, x.shape[-2], None), dtype=x.dtype),
                tf.TensorSpec((y.shape[1], y.shape[-1]), dtype=y.dtype)
            ))
        dataset = apply_ops(dataset, get_intensity_vector)
        if self.mode == 'val':
            dataset = dataset.batch(self.config.batch, drop_remainder=False)
        dataset = apply_ops(dataset, split_total_labels_to_sed_doa)

        return dataset.prefetch(AUTOTUNE)


# only for mel-spectrogram dataset
def get_mel_dataset(config, mode: str = 'train'):
    path = os.path.join(config.abspath, 'DCASE2021/feat_label/')

    x, y = load_seldnet_data(os.path.join(path, 'foa_dev_norm'),
                             os.path.join(path, 'foa_dev_label'), 
                             mode=mode, n_freq_bins=64)
    
    if mode == 'train':
        if config.use_tfm:
            sample_transforms = [
                random_ups_and_downs,
                # lambda x, y: (mask(x, axis=-2, max_mask_size=8, n_mask=6), y),
                lambda x, y: (mask(x, axis=-2, max_mask_size=16), y),
            ]
        else:
            sample_transforms = []
        batch_transforms = [split_total_labels_to_sed_doa]
        if config.use_acs:
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
    else:
        def frame(data):
            return tf.signal.frame(data, config.frame_num, config.frame_step, axis=0, pad_end=True).numpy()
        
        x = np.stack(list(map(frame, x)), 0)
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = apply_ops(dataset, split_total_labels_to_sed_doa)
    return dataset


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    config.epoch = config.epoch // config.iters
    config.model = config.model.upper()
    if config.model not in ('DPRNN', 'GRU'):
        raise argparse.ArgumentError(None, 'model must be DPRNN OR GRU')
    name = '_'.join(['1', config.data, config.model, str(config.lr)])
    if config.norm:
        name += '_norm'
    config.name = name + '_' + config.name
    n_classes = 12

    # data load
    if config.data == 'stft':
        trainsetloader = Pipline_Trainset_Dataloader(os.path.join(config.abspath, 'DCASE2021'), batch=config.batch, iters=config.iters, 
                            batch_preprocessing=[
                                split_total_labels_to_sed_doa
                            ],
                            sample_preprocessing=[
                                # swap_channel,
                                get_intensity_vector,
                                # make_spec_augment(100, 27, 2, 2),
                            ])
        valset = Custom_dataset(config, 'val')
        testset = Custom_dataset(config, 'test')

    # --------------------------- mel dataset ------------------------------------
    elif config.data == 'mel':
        trainset = get_mel_dataset(config, 'train')
        config = vars(config)
        config['frame_step'] = 100
        config['frame_num'] = 300
        config['resolution'] = 5
        config = argparse.Namespace(**config)
        valset = get_mel_dataset(config, 'val')
        testset = get_mel_dataset(config, 'test')
    # -----------------------------------------------------------------------------

    tensorboard_path = os.path.join('./tensorboard_log', config.name)
    if not os.path.exists(tensorboard_path):
        print(f'tensorboard log directory: {tensorboard_path}')
        os.makedirs(tensorboard_path)
    writer = SummaryWriter(logdir=tensorboard_path)

    model_path = os.path.join('./saved_model', config.name)
    if not os.path.exists(model_path):
        print(f'saved model directory: {model_path}')
        os.makedirs(model_path)

    # path = os.path.join(config.abspath, 'DCASE2021/feat_label/')
    # test_xs, test_ys = load_seldnet_data(
    #     os.path.join(path, 'foa_dev_norm'),
    #     os.path.join(path, 'foa_dev_label'), 
    #     mode='test', n_freq_bins=64)
    # test_ys = list(map(
    #     lambda x: split_total_labels_to_sed_doa(None, x)[-1], test_ys))

    x, _ = [(x, y) for x, y in valset.get().take(1)][0]
    input_shape = x.shape[1:]
    model = get_model(input_shape, config)
    model.summary()

    # optimizer = tfa.optimizers.AdamW(1e-6, config.lr)
    optimizer = tf.optimizers.Adam(config.lr)
    criterion = tf.keras.losses.MSE

    # stochastic weight averaging

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
        criterion, evaluator, writer, 'val', config=config)
    test_iterloop = generate_iterloop(
        criterion, evaluator, writer, 'test', config=config)
    # evaluate_fn = generate_evaluate_fn(
    #     test_xs, test_ys, evaluator, config.batch, writer=writer)

    for epoch in range(config.epoch):
        if config.data == 'stft':
            trainset = next(trainsetloader)

        # normalize

        # if epoch % 10 == 0:
        #     evaluate_fn(model, epoch)
            
        # train loop
        # train_iterloop(model, trainset, epoch, optimizer)
        score = val_iterloop(model, valset.get(), epoch)
        test_iterloop(model, testset.get(), epoch)

        

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
            if 50000 <= config.iters * epoch:
                lr_decay_patience += 1
                early_stop_patience += 1
                print(f'lr_decay_patience: {lr_decay_patience}')
                if lr_decay_patience * config.iters >= config.lr_patience and config.decay != 1:
                    print(f'iters: {epoch * config.iters}, lr: {optimizer.learning_rate.numpy():.5} -> {(optimizer.learning_rate * config.decay).numpy():.5}')
                    optimizer.learning_rate = optimizer.learning_rate * config.decay
                    lr_decay_patience = 0

                if early_stop_patience * config.iters == config.patience:
                    print(f'Early Stopping at {epoch * config.iters}, score is {score}')
                    break

    # end of training
    print(f'iters: {epoch * config.iters}')

    # seld_score, *_ = evaluate_fn(model, epoch * config.iters)





if __name__=='__main__':
    main(args.parse_args())
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # model = get_model([512, 241, 7])
    # from model_flop import get_flops
    # model.summary()
    # print(get_flops(model))
