import json
import os
import time
from collections import OrderedDict
import argparse

import numpy as np
import tensorflow as tf
from glob import glob
from numpy import inf
from tensorboardX import SummaryWriter
from tqdm import tqdm

from search_utils import postprocess_fn
import layers
import losses
import models
from data_loader import *
from metrics import * 
from params import get_param
from swa import SWA
from transforms import *
from utils import adaptive_clip_grad, AdaBelief, apply_kernel_regularizer

from accdoa_search import search_space_1d, search_space_2d, block_1d_num, block_2d_num


args = argparse.ArgumentParser()
    
args.add_argument('--name', type=str, required=True)

args.add_argument('--gpus', type=str, default='-1')
args.add_argument('--resume', action='store_true')    
args.add_argument('--abspath', type=str, default='/root/datasets')
args.add_argument('--config_mode', type=str, default='')
args.add_argument('--doa_loss', type=str, default='MSE', 
                    choices=['MAE', 'MSE', 'MSLE', 'MMSE'])
args.add_argument('--model', type=str, default='accdoa')
args.add_argument('--model_config', type=str, default='SS5')
args.add_argument('--output_path', type=str, default='./output')
args.add_argument('--ans_path', type=str, default='/root/datasets/DCASE2021/metadata_dev/')
args.add_argument('--multi', action='store_true')


# training
args.add_argument('--lr', type=float, default=0.001)
args.add_argument('--decay', type=float, default=0.5)
args.add_argument('--batch', type=int, default=256)
args.add_argument('--agc', type=bool, default=False)
args.add_argument('--epoch', type=int, default=100)
args.add_argument('--loss_weight', type=str, default='1,1000')
args.add_argument('--lr_patience', type=int, default=80, 
                    help='learning rate decay patience for plateau')
args.add_argument('--patience', type=int, default=100, 
                    help='early stop patience')
args.add_argument('--freq_mask_size', type=int, default=16)
args.add_argument('--time_mask_size', type=int, default=24)
args.add_argument('--tfm_period', type=int, default=100)
args.add_argument('--use_acs', type=bool, default=True)
args.add_argument('--use_tfm', type=bool, default=True)
args.add_argument('--use_tdm', action='store_true')
args.add_argument('--loop_time', type=int, default=5, 
                    help='times of train dataset iter for an epoch')
args.add_argument('--tdm_epoch', type=int, default=2,
                    help='epochs of applying tdm augmentation. If 0, don\'t use it.')
args.add_argument('--accdoa', type=bool, default=True)

# metric
args.add_argument('--lad_doa_thresh', type=int, default=20)
args.add_argument('--sed_loss', type=str, default='BCE',
                    choices=['BCE','FOCAL'])
args.add_argument('--focal_g', type=float, default=2)
args.add_argument('--focal_a', type=float, default=0.25)


def get_accdoa_labels(accdoa_in, nb_classes):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:]
    sed = tf.cast(tf.sqrt(x**2 + y**2 + z**2) > 0.5, tf.float32)
    return sed, accdoa_in


def generate_trainstep(sed_loss, doa_loss, loss_weights, config, label_smoothing=0, accdoa=True):
    # These are statistics from the train dataset
    train_samples = tf.convert_to_tensor(
        [[58193, 32794, 29801, 21478, 14822, 
        9174, 66527,  6740,  9342,  6498, 
        22218, 49758]],
        dtype=tf.float32)
    cls_weights = tf.reduce_mean(train_samples) / train_samples
    @tf.function
    def trainstep(model, x, y, optimizer):
        with tf.GradientTape() as tape:
            y_p = model(x, training=True)
            sed, doa = y
            if accdoa:
                loss = doa_loss(doa, y_p)
            else:
                sed_pred, doa_pred = y_p

                if label_smoothing > 0:
                    sed = sed * (1-label_smoothing) + 0.5 * label_smoothing

                # sloss = tf.reduce_mean(sed_loss(sed, sed_pred) * cls_weights)
                # dloss = doa_loss(doa, doa_pred, cls_weights)
                sloss = tf.reduce_mean(sed_loss(sed, sed_pred))
                dloss = doa_loss(doa, doa_pred)

                loss = sloss * loss_weights[0] + dloss * loss_weights[1]

                # regularizer
                # loss += tf.add_n([l.losses[0] for l in model.layers
                #                   if len(l.losses) > 0])

        grad = tape.gradient(loss, model.trainable_variables)
        # apply AGC
        if config.agc:
            grad = adaptive_clip_grad(model.trainable_variables, grad)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

        if accdoa:
            return y_p, loss
        else:
            return y_p, sloss, dloss
    return trainstep


def generate_teststep(sed_loss, doa_loss, accdoa=True):
    @tf.function
    def teststep(model, x, y, optimizer=None):
        y_p = model(x, training=False)
        if accdoa:
            loss = doa_loss(y[1], y_p)
            return y_p, loss
        else:
            sloss = sed_loss(y[0], y_p[0])
            dloss = doa_loss(y[1], y_p[1])
            return y_p, sloss, dloss
    return teststep


def generate_iterloop(sed_loss, doa_loss, evaluator, mode,  
                      writer=None, loss_weights=None, config=None, accdoa=True):
    if mode == 'train':
        step = generate_trainstep(sed_loss, doa_loss, loss_weights, config, accdoa)
    else:
        step = generate_teststep(sed_loss, doa_loss, accdoa)

    def iterloop(model, dataset, epoch, optimizer=None):
        evaluator.reset_states()
        if accdoa:
            losses = tf.keras.metrics.Mean()
        else:
            ssloss = tf.keras.metrics.Mean()
            ddloss = tf.keras.metrics.Mean()

        with tqdm(dataset) as pbar:
            for x, y in pbar:
                if accdoa:
                    preds, loss = step(model, x, y, optimizer)
                    preds = get_accdoa_labels(preds, y[1].shape[-1] // 3)
                else:
                    preds, sloss, dloss = step(model, x, y, optimizer)

                evaluator.update_states(y, preds)
                metric_values = evaluator.result()
                seld_score = calculate_seld_score(metric_values)

                if accdoa:
                    losses(loss)
                else:
                    ssloss(sloss)
                    ddloss(dloss)
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

        if writer is not None:
            writer.add_scalar(f'{mode}/{mode}_ErrorRate', metric_values[0].numpy(),
                            epoch)
            writer.add_scalar(f'{mode}/{mode}_F', metric_values[1].numpy(), epoch)
            writer.add_scalar(f'{mode}/{mode}_DoaErrorRate', 
                            metric_values[2].numpy(), epoch)
            writer.add_scalar(f'{mode}/{mode}_DoaErrorRateF', 
                            metric_values[3].numpy(), epoch)
            
            if accdoa:
                writer.add_scalar(f'{mode}/{mode}_Loss', 
                                losses.result().numpy(), epoch)
            else:
                writer.add_scalar(f'{mode}/{mode}_sedLoss', 
                                ssloss.result().numpy(), epoch)
                writer.add_scalar(f'{mode}/{mode}_doaLoss', 
                                ddloss.result().numpy(), epoch)
            writer.add_scalar(f'{mode}/{mode}_seldScore', 
                            seld_score.numpy(), epoch)

        if accdoa:
            return seld_score.numpy(), losses.result().numpy()
        else:
            return seld_score.numpy(), ssloss.result().numpy(), ddloss.result().numpy()
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
    return dataset


def ensemble_outputs(model, xs: list, 
                     win_size=300, step_size=5, batch_size=256, accdoa=True):
    @tf.function
    def predict(model, x, batch_size):
        windows = tf.signal.frame(x, win_size, step_size, axis=0)

        sed, doa = [], []
        for i in range(int(np.ceil(windows.shape[0]/batch_size))):
            outs = model(windows[i*batch_size:(i+1)*batch_size], training=False)
            if accdoa:
                s, d = get_accdoa_labels(outs, outs.shape[-1] // 3)
            else:
                s, d = outs
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
    # HyperParameters
    n_classes = 12
    swa_start_epoch = 80
    swa_freq = 2
    # data load
    trainset = get_dataset(config, 'train')
    valset = get_dataset(config, 'val')
    testset = get_dataset(config, 'test')

    path = os.path.join(config.abspath, 'DCASE2021/feat_label/')
    # test_xs, test_ys = load_seldnet_data(
    #     os.path.join(path, 'foa_dev_norm'),
    #     os.path.join(path, 'foa_dev_label'), 
    #     mode='test', n_freq_bins=64)
    # test_ys = list(map(lambda x: split_total_labels_to_sed_doa(None, x)[-1], test_ys))

    # extract data size
    x, y = [(x, y) for x, y in trainset.take(1)][0]
    input_shape = x.shape
    sed_shape, doa_shape = tf.shape(y[0]), tf.shape(y[1])
    print('-----------data shape------------')
    print()
    print(f'data shape: {input_shape}')
    print(f'label shape(sed, doa): {sed_shape}, {doa_shape}')
    print()
    print('---------------------------------')
    
    specific_search_space = {'num2d': block_2d_num, 'num1d': block_1d_num}
    for i in range(max(specific_search_space['num2d']) + max(specific_search_space['num1d'])):
        specific_search_space[f'BLOCK{i}'] = {
            'search_space_2d': search_space_2d,
            'search_space_1d': search_space_1d,
        }

    specific_search_space['SED'] = {'search_space_1d': search_space_1d}
    specific_search_space['DOA'] = {'search_space_1d': search_space_1d}
    search_space = specific_search_space
    from config_sampler import get_config

    max_num = 200
    if not os.path.exists('sampling_result'):
        os.makedirs('sampling_result')
    num = len(glob('sampling_result/*.json'))
    if num == max_num:
        print('Already done')
        return
    # model load
    while num < max_num:
        while True:
            model_config = get_config(argparse.Namespace(n_classes=12), search_space, input_shape=input_shape, postprocess_fn=postprocess_fn)
            model_config['n_classes'] = n_classes
            try:
                model = getattr(models, config.model)(input_shape, model_config)
            except ValueError:
                continue
            break
        model.summary()

        # model = apply_kernel_regularizer(model, kernel_regularizer)

        optimizer = tf.keras.optimizers.Adam(config.lr)
        # optimizer = AdaBelief(config.lr)
        if config.sed_loss == 'BCE':
            sed_loss = tf.keras.backend.binary_crossentropy
        else:
            sed_loss = losses.focal_loss
        # fix doa_loss to MMSE_with_cls_weights (because of class weights)
        # doa_loss = losses.MMSE_with_cls_weights
        try:
            doa_loss = getattr(tf.keras.losses, config.doa_loss)
        except:
            doa_loss = getattr(losses, config.doa_loss)

        # stochastic weight averaging
        swa = SWA(model, swa_start_epoch, swa_freq)

        best_score = inf
        early_stop_patience = 0
        lr_decay_patience = 0
        evaluator = SELDMetrics(
            doa_threshold=config.lad_doa_thresh, n_classes=n_classes)

        train_iterloop = generate_iterloop(
            sed_loss, doa_loss, evaluator, 'train', 
            loss_weights=list(map(int, config.loss_weight.split(','))), config=config)
        val_iterloop = generate_iterloop(
            sed_loss, doa_loss, evaluator, 'val')
        test_iterloop = generate_iterloop(
            sed_loss, doa_loss, evaluator, 'test')
        # evaluate_fn = generate_evaluate_fn(
        #     test_xs, test_ys, evaluator, config.batch*4, writer=writer)

        train_slosses, train_dlosses, val_slosses, val_dlosses, train_scores, test_score, val_score = [], [], [], [], [], [], []
        if config.accdoa:
            train_losses = []
            val_losses = []
        try:
            for epoch in range(config.epoch):
                if epoch == swa_start_epoch:
                    tf.keras.backend.set_value(optimizer.lr, config.lr * 0.5)

                # train loop
                if config.accdoa:
                    train_score, train_loss = train_iterloop(model, trainset, epoch, optimizer)
                    score, val_loss = val_iterloop(model, valset, epoch)
                else:
                    train_score, train_sloss, train_dloss = train_iterloop(model, trainset, epoch, optimizer)
                    score, val_sloss, val_dloss = val_iterloop(model, valset, epoch)
                testscore, _, = test_iterloop(model, testset, epoch)
                test_score.append(float(testscore))
                val_score.append(float(score))
                if config.accdoa:
                    train_losses.append(float(train_loss))
                    val_losses.append(float(val_loss))
                else:
                    train_slosses.append(float(train_sloss))
                    train_dlosses.append(float(train_dloss))
                    val_slosses.append(float(val_sloss))
                    val_dlosses.append(float(val_dloss))
                train_scores.append(float(train_score))

                swa.on_epoch_end(epoch)

                if best_score > score:
                    best_score = score
                    early_stop_patience = 0
                    lr_decay_patience = 0
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
        except tf.errors.ResourceExhaustedError:
            if tf.__version__ >= '2.6.0':
                if tf.config.list_physical_devices('GPU'):
                    trainset = get_dataset(config, 'train')
                    valset = get_dataset(config, 'val')
                    testset = get_dataset(config, 'test')
                    tf.config.experimental.reset_memory_stats('GPU:0')
            print('resource exhuasted, get another model')
            continue

        # end of training
        if not os.path.exists('sampling_result'):
            os.makedirs('sampling_result')

        num = len(glob('sampling_result/*.json'))
        if num == max_num:
            print('done')
            return
        
        with open(os.path.join('sampling_result', f'{num}.json'), 'w') as f:
            if config.accdoa:
                json.dump([model_config, train_losses, val_losses, train_scores, test_score, val_score], f, indent=4)
            else:
                json.dump([model_config, train_slosses, train_dlosses, val_slosses, val_dlosses, train_scores, test_score, val_score], f, indent=4)


if __name__=='__main__':
    main(args.parse_args())

