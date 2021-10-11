import random as rnd
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tensorflow_io as tfio
import joblib

from feature_extractor import *
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from data_utils import spec_augment
AUTOTUNE = tf.data.experimental.AUTOTUNE


def data_loader(dataset, 
                preprocessing=None,
                sample_transforms=None, 
                batch_transforms=None,
                deterministic=False,
                loop_time=None,
                batch_size=32) -> tf.data.Dataset:
    '''
    INPUT
        preprocessing: a list of preprocessing ops
                       output of preprocessing ops will be cached
        sample_transforms: a list of samplewise augmentations
        batch_transforms: a list of batchwise augmentations
        deterministic: set to False for efficiency,
                       if the order of the data is critical, set to True
        inf_loop: whether to loop infinitely (will run .repeat() after .cache())
                  this can also increase efficiency
        batch_size: batch size
    '''
    if not isinstance(dataset, tf.data.Dataset):
        dataset = tf.data.Dataset.from_tensor_slices(dataset)

    def apply_ops(dataset, operations):
        if operations is None:
            return dataset

        if not isinstance(operations, (list, tuple)):
            operations = [operations]

        for op in operations:
            dataset = dataset.map(
                op, num_parallel_calls=AUTOTUNE, deterministic=deterministic)

        return dataset
    
    dataset = apply_ops(dataset, preprocessing)
    dataset = dataset.cache()
    dataset = dataset.repeat(loop_time)
    dataset = apply_ops(dataset, sample_transforms)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = apply_ops(dataset, batch_transforms)

    return dataset


def load_seldnet_data(feat_path, label_path, mode='train', n_freq_bins=64):
    from glob import glob
    import os

    assert mode in ['train', 'val', 'test']
    splits = {
        'train': [1, 2, 3, 4],
        'val': [5],
        'test': [6]
    }

    # load splits according to the mode
    if not os.path.exists(feat_path):
        raise ValueError(f'no such feat_path ({feat_path}) exists')
    features = sorted(glob(os.path.join(feat_path, '*.npy')))
    features = [np.load(f).astype('float32') for f in features 
                if int(f[f.rfind(os.path.sep)+5]) in splits[mode]]

    if not os.path.exists(label_path):
        raise ValueError(f'no such label_path ({label_path}) exists')
    labels = sorted(glob(os.path.join(label_path, '*.npy')))
    labels = [np.load(f).astype('float32') for f in labels
              if int(f[f.rfind(os.path.sep)+5]) in splits[mode]]

    if len(features[0].shape) == 2:
        def extract(x):
            x = np.reshape(x, (x.shape[0], -1, n_freq_bins))
            return x.transpose(0, 2, 1)

        features = list(map(extract, features))
    else:
        # already in shape of [time, freq, chan]
        pass
    
    return features, labels


def load_wav_and_label(feat_path, label_path, mode='train', class_num=12):
    '''
        output
        x: wave form -> (data_num, channel(4), time)
        y: label(padded) -> (data_num, time, 56)
    '''
    
    f_paths = sorted(glob(os.path.join(feat_path, f'dev-{mode}', '*.wav')))
    l_paths = sorted(glob(os.path.join(label_path, f'dev-{mode}', '*.csv')))

    splits = {
        'train': [1, 2, 3, 4],
        'val': [5],
        'test': [6]
    }

    f_paths = [f for f in f_paths 
            if int(f[f.rfind(os.path.sep)+5]) in splits[mode]]
    l_paths = [f for f in l_paths 
            if int(f[f.rfind(os.path.sep)+5]) in splits[mode]]

    if len(f_paths) != len(l_paths):
        raise ValueError('# of features and labels are not matched')
    
    def preprocess_label(labels, max_label_length=600):
        cur_len = labels.shape[0]
        max_len = max_label_length

        if cur_len < max_len: 
            labels = tf.pad(labels, ((0, max_len-cur_len), (0,0)))
        else:
            labels = labels[:max_len]
        return labels
    sr = tf.audio.decode_wav(tf.io.read_file(f_paths[0]))[1]
    with ThreadPoolExecutor() as pool:
        x = list(pool.map(lambda x: tf.audio.decode_wav(tf.io.read_file(x))[0], f_paths))
    with ThreadPoolExecutor() as pool:
        y = list(pool.map(lambda x: preprocess_label(extract_labels(x, class_num)), l_paths))
    return x, y, int(sr)


def seldnet_data_to_dataloader(features: [list, tuple], 
                               labels: [list, tuple], 
                               train=True, 
                               label_window_size=60,
                               drop_remainder=True,
                               shuffle_size=None,
                               batch_size=32,
                               loop_time=1,
                               **kwargs):
    total_length = labels[0].shape[0]
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    # shapes of seldnet features and labels 
    # features: [time_features, freq, chan]
    # labels:   [time_labels, 4*classes]
    # for each 5 input time slices, a single label time slices was designated
    # features' shape [time_f, freq, chan] -> [time_l, resolution, freq, chan]
    features = np.reshape(features, (labels.shape[0], -1, *features.shape[1:]))

    # windowing
    n_samples = features.shape[0] // label_window_size
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(label_window_size, drop_remainder=drop_remainder)
    dataset = dataset.map(lambda x,y: (tf.reshape(x, (-1, *x.shape[2:])), y),
                          num_parallel_calls=AUTOTUNE)
    del features, labels
    if train == False:
        batch_size = total_length // label_window_size
    dataset = data_loader(dataset, batch_size=batch_size, 
            loop_time=loop_time if train else 1, **kwargs)
    
    if train:
        if shuffle_size is None:
            shuffle_size = n_samples // batch_size
        dataset = dataset.shuffle(shuffle_size)

    return dataset.prefetch(AUTOTUNE)


def get_TDMset(TDM_PATH):
    from glob import glob
    tdm_path = os.path.join(TDM_PATH, 'foa_dev_tdm')
    class_num = len(glob(tdm_path + '/*label_*.joblib'))

    def load_data(cls):
        return tf.convert_to_tensor(joblib.load(os.path.join(tdm_path, f'tdm_noise_{cls}.joblib')), dtype=tf.float32)

    def load_label(cls):
        return tf.convert_to_tensor(joblib.load(os.path.join(tdm_path, f'tdm_label_{cls}.joblib')))
    
    with ThreadPoolExecutor() as pool:
        tdm_x = list(pool.map(load_data, range(class_num)))
        tdm_y = list(pool.map(load_label, range(class_num)))
    return tdm_x, tdm_y


def TDM_aug(x: list, y: list, tdm_x, tdm_y, sr=24000, label_resolution=0.1, max_overlap_num=5, max_overlap_per_frame=2, min_overlap_sec=1, max_overlap_sec=5):
    '''
        x: list(tf.Tensor): shape(sample number, channel(4), frame(1440000))
        y: list(tf.Tensor): shape(sample number, time(600), class+cartesian(14+42))
        tdm_x: list(tf.Tensor): shape(class_num, channel(4), frame)
        tdm_y: list(tf.Tensor): shape(class_num, time, class+cartesian(14+42))
    '''
    class_num = y[0].shape[-1] // 4
    min_overlap_sec = int(min_overlap_sec / label_resolution) 
    max_overlap_sec = int(max_overlap_sec / label_resolution)
    sr = int(sr * label_resolution)

    def add_sample(i):
        weight = 1 / tf.convert_to_tensor([k.shape[0] for k in tdm_y])
        weight /= tf.reduce_sum(weight)
        selected_cls = tf.random.categorical(tf.math.log(weight[tf.newaxis,...]), max_overlap_num)[0] # (max_overlap_num,)

        def _add_sample(cls):
            frame_y_num = y[i].shape[0]
            sample_time = tf.random.uniform((), min_overlap_sec, max_overlap_sec,dtype=tf.int64) # to milli second
            offset = tf.random.uniform((), 0, frame_y_num - sample_time, dtype=tf.int64) # offset as label
            td_offset = tf.random.uniform((),0, tdm_y[cls].shape[0] - sample_time, dtype=sample_time.dtype) # 뽑을 노이즈에서의 랜덤 offset
            
            frame_y = y[i][offset:offset+sample_time] # (sample_time, 56)
            nondup_class = 1 - frame_y[..., cls]

            valid_index = tf.cast(tf.reduce_sum(frame_y[...,:class_num], -1) < max_overlap_per_frame, nondup_class.dtype) * nondup_class # 1프레임당 최대 클래스 개수보다 작으면서 겹치지 않는 노이즈를 넣을 수 있는 공간 찾기

            if tf.reduce_sum(valid_index) == 0: # 만약 넣을 수 없다면 이번에는 노이즈 안 넣음
                return tf.zeros((), dtype=tf.int64)

            tdm_frame_y = tdm_y[cls][td_offset:td_offset+sample_time] * valid_index[...,tf.newaxis] # valid한 프레임만 남기기
            y[i] += tf.pad(tdm_frame_y, ((offset, frame_y_num - offset - sample_time),(0,0))) # 레이블 부분 완료
            tdm_frame_x = tdm_x[cls][..., td_offset * sr: (td_offset + sample_time) * sr] * tf.repeat(tf.cast(valid_index, dtype=x[i].dtype), sr, axis=0)[tf.newaxis, ...]
            x[i] += tf.pad(tdm_frame_x, ((0,0), (offset * sr, x[i].shape[-1] - (offset + sample_time) * sr)))
            return tf.zeros((), dtype=tf.int64)
        
        j = tf.constant(0)
        cond = lambda i, j: j < len(selected_cls)
        def body(i, j):
            _add_sample(selected_cls[j])
            return i, j + 1
        tf.while_loop(cond, body, (i, j))
        return tf.zeros((), dtype=tf.int32)

    tf.map_fn(add_sample, tf.range(len(x)))
    return x, y


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
    return tf.stack([IVx, IVy, IVz], axis=0)


def gcc_features_tf(complex_specs, n_mels):
    n_chan = complex_specs.shape[0]
    gcc_feat = []
    for m in range(n_chan):
        for n in range(m+1, n_chan):
            R = tf.math.conj(complex_specs[m]) * complex_specs[n]
            print(R.shape)
            cc = tf.signal.irfft(tf.math.exp(1.j*tf.complex(tf.math.angle(R),0.0)))
            cc = tf.concat([cc[-n_mels//2:], cc[:(n_mels+1)//2]], axis=0)
            gcc_feat.append(cc)

    return tf.stack(gcc_feat, axis=0)


def get_preprocessed_x_tf(wav, sr, mode='foa', n_mels=64,
                          multiplier=5, max_label_length=600, win_length=1024,
                          hop_length=480, n_fft=1024):
    mel_mat = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=n_mels,
                                                    num_spectrogram_bins=n_fft//2+1,
                                                    sample_rate=sr,
                                                    lower_edge_hertz=0,
                                                    upper_edge_hertz=sr//2)


    spectrogram = tf.signal.stft(wav, win_length, hop_length, n_fft, pad_end=True)
    norm_spec = tf.math.abs(spectrogram)
    mel_spec = tf.matmul(norm_spec, mel_mat)
    mel_spec = tfio.experimental.audio.dbscale(mel_spec, top_db=80)
    features = [mel_spec]
        
    if mode == 'foa':
        foa = foa_intensity_vectors_tf(spectrogram)
        foa = tf.matmul(foa, mel_mat)
        features.append(foa)
        
    elif mode == 'mic':
        gcc = gcc_features_tf(spectrogram, n_mels=n_mels)
        features.append(gcc)
    
    else:
        raise ValueError('invalid mode')
    
    features = tf.concat(features, axis=0)
    features = tf.transpose(features, perm=[1, 2, 0])
    
    cur_len = features.shape[0]
    max_len = max_label_length * multiplier
    
    if cur_len < max_len: 
        pad = tf.constant([[0, max_len-cur_len], [0,0], [0,0]])
        features = tf.pad(features, pad, 'constant')
    else:
        features = features[:max_len]
    return features


class Pipline_Trainset_Dataloader:
    def __init__(self, path, batch, frame_num=512, frame_len=0.02, iters=10000, accdoa=True, sample_preprocessing=[], batch_preprocessing=[]):
        self.x = joblib.load(os.path.join(path, 'foa_dev_train_stft_480.joblib')) # (sample_num, frame_num, freqs, chan)
        self.y = joblib.load(os.path.join(path, 'foa_dev_train_label.joblib')) # (sample_num, label_frame_num, SED+DOA)
        self.sr = 24000
        if self.x.shape[0] % self.y.shape[0] != 0:
            raise ValueError('data resolution is wrong')
        self.frame_num = frame_num
        self.label_num = tf.cast(tf.math.ceil(self.frame_num / 10), dtype=tf.int64) # batch 1개에 들어갈(transform 이후 512개의 frame) 레이블 길이 미리 계산
        self.iters = iters
        self.sample_preprocessing = sample_preprocessing
        self.batch_preprocessing = batch_preprocessing
        self.class_num = self.y.shape[-1] // 4
        self.batch = batch
        self.mel_bin = 64
        self.resolution = self.x.shape[-3] // self.y.shape[-2]
        self.equilizer = biquad_equilizer(self.sr)
        self.mono_y_idx = self.get_mono_y_idx()
        self.mono_x_idx = self.get_x_from_y(self.mono_y_idx, self.resolution)
        # self.sample_preprocessing.append(self.EMDA)
        self.sample_preprocessing.insert(0, self.seperate_real_imag)

    @tf.function
    def get_x_from_y(self, y, resolution): # 97769~97779 이상
        pad = tf.pad(tf.range(0, self.resolution, dtype=self.mono_y_idx.dtype)[tf.newaxis,...], ((1,0),(0,0)))[tf.newaxis, ...]
        out = tf.reshape(tf.transpose(y[...,tf.newaxis] * [[[1], [resolution]]] + pad, (0,2,1)), [-1, y.shape[-1]])
        return out

    @tf.function
    def seperate_real_imag(self, x, y):
        if x.dtype in [tf.complex64, tf.complex128]:
            x = tf.concat([tf.math.real(x), tf.math.imag(x)], -1)
        return x, y

    def get_sliced_data(self):
        sample_num = tf.random.uniform((), minval=0, maxval=self.x.shape[0], dtype=tf.int64)
        label_offset = tf.random.uniform((), minval=0, maxval=self.y.shape[1] - self.label_num, dtype=tf.int64)
        feature_offset = label_offset * self.resolution

        x = self.x[sample_num][feature_offset: feature_offset + self.frame_num] # (frame, freq, chan)
        y = self.y[sample_num][label_offset:label_offset + self.label_num] # (frame_label, SED+DOA)
        # if label_offset >= self.y.shape[1] - self.label_num:
        #     x = tf.concat([x, self.x[(sample_num + 1) % self.x.shape[0]][:self.frame_num - x.shape[0]]], 0)
        #     y = tf.concat([y, self.y[(sample_num + 1) % self.y.shape[0]][:self.label_num - y.shape[0]]], 0)
        return x, y

    def data_generator(self):
        for i in range(self.iters):
            x, y = self.get_sliced_data()
            for sample_preprocess in self.sample_preprocessing:
                x, y = sample_preprocess(x, y)
            yield x, y

    def __next__(self):
        trainset = tf.data.Dataset.from_generator(self.data_generator, output_types=(tf.float32, tf.float32), 
                    output_shapes=([self.frame_num] + [*self.x.shape[2:-1]] + [self.x.shape[-1] * 2], [self.label_num] + [*self.y.shape[2:]]))

        trainset = trainset.batch(self.batch)
        print('batch_preprocessing')
        for pre in self.batch_preprocessing:
            trainset = trainset.map(pre)

        return trainset.prefetch(tf.data.AUTOTUNE)

    def get_mono_y_idx(self):
        idx = tf.cast(tf.where(tf.reduce_sum(self.y[...,:self.class_num], -1) == 1), tf.int64)
        return idx

    @tf.function
    def get_mono_frame(self, x, y):
        y_frame_size = tf.random.uniform((), minval=1, maxval=y.shape[0], dtype=tf.int32) # y에 넣을 frame 크기 구하기
        y_offset = tf.random.uniform((), maxval=y.shape[0] - y_frame_size, dtype=tf.int32) # 프레임 크기를 고려하여 offset 설정
        mono_y_frame_size = y_frame_size
        mono_y_offset = tf.random.uniform((), maxval=self.mono_y_idx.shape[0] - mono_y_frame_size, dtype=tf.int32)
        x_frame_size = y_frame_size * self.resolution
        x_offset = y_offset * self.resolution
        mono_x_frame_size = x_frame_size
        mono_x_offset = mono_y_offset

        yy = self.mono_y_idx[mono_y_offset:mono_y_offset+mono_y_frame_size]
        xx = self.mono_x_idx[mono_x_offset:mono_x_offset+mono_x_frame_size]
        mono_y_frame = tf.gather_nd(self.y, yy)
        mono_x_frame = tf.gather_nd(self.x, xx)

        y_frame_offset = tf.random.uniform((), maxval=y.shape[0] - tf.shape(mono_y_frame)[0], dtype=tf.int32)
        mono_y_frame = tf.pad(mono_y_frame, ((y_frame_offset, y.shape[0] - y_frame_offset - tf.shape(mono_y_frame)[0]),(0,0)))
        x_frame_offset = y_frame_offset * self.resolution
        mono_x_frame = tf.pad(mono_x_frame, ((x_frame_offset, x.shape[0] - x_frame_offset - tf.shape(mono_x_frame)[0]),(0,0),(0,0)))
        return mono_x_frame, mono_y_frame

    @tf.function
    def filtering(self, x, y, mono_x_frame, mono_y_frame):
        cond = tf.reduce_sum(tf.cast((y[...,:self.class_num] + mono_y_frame[...,:self.class_num]) < 2, tf.float32), -1) == self.class_num # 합성 후 같은 class가 같은 frame에 존재하는 지 여부 탐색
        cond = tf.logical_and(cond, tf.reduce_sum(y[..., :self.class_num], -1) < 2) # 원래 class가 2개 미만인 프레임만 합성
        y = tf.where(cond[..., tf.newaxis], x=y + mono_y_frame, y=y)
        cond = tf.repeat(cond, self.resolution)[:x.shape[0], tf.newaxis, tf.newaxis]
        
        ratio = tf.random.uniform((), minval=1e-3, maxval=1 - 1e-3)
        x = tf.complex(tf.math.real(x) * ratio, tf.math.imag(x))
        mono_x_frame = tf.complex(tf.math.real(mono_x_frame) * (1 - ratio), tf.math.imag(mono_x_frame))

        x = tf.where(cond, x=mono_x_frame + x, y=x)
        return x, y

    @tf.function
    def EMDA(self, x, y):
        # x: (frame, freq, chan)
        # y: (frame_label, SED+DOA)

        mono_x_frame, mono_y_frame = self.get_mono_frame(x, y)

        # equilizer가 완성되면 사용
        # mono_x_frame = self.equilizer(mono_x_frame)
        # x = self.equilizer(x)

        # class number in same frame constraint
        x, y = self.filtering(x, y, mono_x_frame, mono_y_frame)
        return x, y

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=20000)])
        except RuntimeError as e:
            print(e)
    path = '/root/datasets/DCASE2021'
    sample_preprocessing = []
    batch_preprocessing = [spec_augment]
    trainsetloader = Pipline_Trainset_Dataloader(path, batch=32, sample_preprocessing=sample_preprocessing, batch_preprocessing=batch_preprocessing)
    trainset = next(trainsetloader)
    import pdb; pdb.set_trace()
    
 
