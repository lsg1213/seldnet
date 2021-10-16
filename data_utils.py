import os

import numpy as np
import torchaudio
import tensorflow as tf
import scipy.signal as ss


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)


def degree_to_radian(degree):
    return degree * np.pi / 180


def radian_to_degree(radian):
    return radian / np.pi * 180


# https://github.com/DemisEom/SpecAugment/blob/master/SpecAugment/spec_augment_tensorflow.py
@tf.function
def frequency_masking(mel_spectrogram, frequency_masking_para=27, frequency_mask_num=2):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    # Step 2 : Frequency masking
    fbank_size = tf.shape(mel_spectrogram)
    n, v = fbank_size[0], fbank_size[1]
    for i in range(frequency_mask_num):
        f = tf.random.uniform([], minval=0, maxval=frequency_masking_para, dtype=tf.int32)
        v = tf.cast(v, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=v-f, dtype=tf.int32)

        # warped_mel_spectrogram[f0:f0 + f, :] = 0
        mask = tf.concat((tf.ones(shape=(n, v - f0 - f, 1)),
                          tf.zeros(shape=(n, f, 1)),
                          tf.ones(shape=(n, f0, 1)),
                          ), 1)
        mel_spectrogram = mel_spectrogram * mask
    return tf.cast(mel_spectrogram, dtype=tf.float32)


@tf.function
def time_masking(x, y, tau, time_masking_para=100, time_mask_num=2):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    fbank_size = tf.shape(x)
    n, v = fbank_size[0], fbank_size[1]
    axis = 0
    resolution = int(tf.math.ceil(fbank_size[0] / tf.shape(y)[0]))

    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = tf.random.uniform([], minval=0, maxval=time_masking_para // resolution, dtype=tf.int32)
        t0 = tf.random.uniform([], minval=0, maxval=int(tf.math.ceil(tau / resolution)) - t, dtype=tf.int32)

        # x[:, t0:t0 + t] = 0
        mask = tf.concat((tf.ones(shape=(tf.shape(y)[axis]-t0-t, v, 1)),
                          tf.zeros(shape=(t, v, 1)),
                          tf.ones(shape=(t0, v, 1)),
                          ), axis)
        y *= mask[:,:1,0]
        x = x * tf.repeat(mask, repeats=resolution, axis=axis)[:x.shape[axis]]
    return tf.cast(x, dtype=tf.float32), y


@tf.function
def swap_channel(x, y):
    # x = (window, freq, chan)
    flip = tf.cast(tf.random.uniform((3,), 0, 2, dtype=tf.int32), tf.float32)
    class_num = y.shape[-1] // 4
    y = tf.reshape(y, [-1] + [*y.shape[1:-1]] + [4, class_num])
    cartesian = y[..., -3:, :]

    cartesian = (1 - 2*tf.reshape(flip, (-1, 3, 1))) * cartesian


    perm = 2 * tf.random.uniform((1,), maxval=2, dtype=tf.int32) # -1 결정
    perm = tf.concat([perm, tf.ones_like(perm), 2-perm], axis=-1)
    correct_shape = tf.constant([0,1,2], dtype=perm.dtype)
    
    check = tf.reduce_sum(tf.cast(perm != correct_shape, tf.int32), -1, keepdims=True)
    feat_perm = (perm + check) % 3
    
    cartesian = tf.gather(cartesian, feat_perm, axis=-2, batch_dims=1)
    y = tf.concat([y[..., :-3, :], cartesian], axis=-2)
    y = tf.reshape(y, [-1] + [*y.shape[1:-2]] + [4*y.shape[-1]])

    reshaped_x = x[..., 1:4]
    reshaped_x = tf.complex(tf.math.real(reshaped_x), tf.math.imag(reshaped_x) * flip)
    reshaped_x = tf.gather(x[..., 1:4], perm, axis=-1)
    x = tf.concat([x[..., :1], reshaped_x], axis=-1)
    return x, y


@tf.function
def spec_augment(x, y):
    # x = (window, freq, chan)
    tau = tf.shape(x)[0]

    x = frequency_masking(x)
    x, y = time_masking(x, y, tau=tau)

    return x, y
    

def biquad_equalizer(sampling_rate, central_freq=[100., 6000.], g=[-8.,8.], Q=[1.,9.]):
    '''
    central_freq: central frequency
    g: gain
    Q: Q-factor
    '''
    def _band_biquad_equalizer(feat):
        gain = tf.random.uniform((), minval=g[0], maxval=g[1])
        central_frequency = tf.random.uniform((), minval=central_freq[0], maxval=central_freq[1])
        Qfactor = tf.random.uniform((), minval=Q[0], maxval=Q[1])

        w0 = 2 * np.math.pi * central_frequency / sampling_rate
        A = tf.exp(gain / 40.0 * tf.math.log(10.))
        alpha = tf.sin(w0) / 2 / Qfactor

        b0 = 1 + alpha * A
        b1 = -2 * tf.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * tf.cos(w0)
        a2 = 1 - alpha / A

        frf = tf.cast(ss.freqz(tf.stack([b0, b1, b2], 0), tf.stack([a0, a1, a2], 0), worN=feat.shape[0])[1][..., np.newaxis, np.newaxis], feat.dtype)
        return biquad(feat, frf)

    @tf.function
    def biquad(feat, frf):
        return feat * frf
    
    return _band_biquad_equalizer

@tf.function
def stft(wav):
    wav = tf.transpose(wav, [1,0])
    out = tf.signal.stft(wav, 480, 240, 480, pad_end=True)
    return out


if __name__ == '__main__':
    from data_loader import load_wav_and_label
    import joblib
    import tensorflow as tf
    import numpy as np
    from tqdm import tqdm
    
    modes = ('train', 'val', 'test')
    path = '/root/datasets/DCASE2021'
    x_ = []
    for mode in modes:
        print(mode)
        x, y, sr = load_wav_and_label(os.path.join(path, 'foa_dev'),
                                os.path.join(path, 'metadata_dev'),
                                mode=mode)
        x = np.stack([stft(i).numpy() for i in tqdm(x)], 0).transpose(0,2,3,1)
        y = np.stack(y, 0)
        joblib.dump(x, os.path.join(path, f'foa_dev_{mode}_stft_480.joblib'))
        joblib.dump(y, os.path.join(path, f'foa_dev_{mode}_label.joblib'))
        x_.append(x)
    x_ = np.concatenate(x_, 0)
    mean = x_.mean(0)
    std = x_.std(0)
    for mode in modes:
        print(mode)
        x = joblib.load(os.path.join(path, f'foa_dev_{mode}_stft_480.joblib'))
        joblib.dump((x - mean) / std, os.path.join(path, f'foa_dev_{mode}_stft_480_norm.joblib'))
