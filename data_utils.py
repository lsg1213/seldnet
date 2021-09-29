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
    n, v = fbank_size[1], fbank_size[2]
    for i in range(frequency_mask_num):
        f = tf.random.uniform([], minval=0, maxval=frequency_masking_para, dtype=tf.int32)
        v = tf.cast(v, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=v-f, dtype=tf.int32)

        # warped_mel_spectrogram[f0:f0 + f, :] = 0
        mask = tf.concat((tf.ones(shape=(1, n, v - f0 - f, 1)),
                          tf.zeros(shape=(1, n, f, 1)),
                          tf.ones(shape=(1, n, f0, 1)),
                          ), 2)
        mel_spectrogram = mel_spectrogram * mask
    return tf.cast(mel_spectrogram, dtype=tf.float32)


def time_masking(mel_spectrogram, y, tau, time_masking_para=100, time_mask_num=2):
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
    fbank_size = tf.shape(mel_spectrogram)
    n, v = fbank_size[1], fbank_size[2]
    axis = 1
    resolution = fbank_size[1] // tf.shape(y)[1]

    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = tf.random.uniform([], minval=0, maxval=time_masking_para // resolution, dtype=tf.int32)
        t0 = tf.random.uniform([], minval=0, maxval=tau // 5 - t, dtype=tf.int32)

        # mel_spectrogram[:, t0:t0 + t] = 0
        mask = tf.concat((tf.ones(shape=(1, n-t0-t, v, 1)),
                          tf.zeros(shape=(1, t, v, 1)),
                          tf.ones(shape=(1, t0, v, 1)),
                          ), axis)
        y *= mask
        mel_spectrogram = mel_spectrogram * tf.repeat(mask, repeats=resolution, axis=axis)
    return tf.cast(mel_spectrogram, dtype=tf.float32), y


def spec_augment(x, y):
    # x = (batch, time, freq, chan)
    mel_spectrogram = x
    v = mel_spectrogram.shape[0]
    tau = mel_spectrogram.shape[1]

    warped_frequency_spectrogram = frequency_masking(mel_spectrogram)
    warped_frequency_time_sepctrogram, y = time_masking(warped_frequency_spectrogram, y, tau=tau)

    return warped_frequency_time_sepctrogram, y


def EMDA(raw_x, raw_y, sr=24000):
    # raw_x, raw_y : mono class sound
    equilizer = biquad_equilizer(sr)
    def _EMDA(x, y):
        # x = (time, freq, chan)
        if x.shape[0] % y.shape[0] != 0:
            raise ValueError('resolution is not matched')
        resolution = x.shape[0] // y.shape[0]
        class_num = y.shape[-1] // 4
        
        import pdb; pdb.set_trace()
        y_frame_size = tf.random.uniform([1], maxval=y.shape[0], dtype=tf.int32)[0] # y에 넣을 frame 크기 구하기
        y_offset = tf.random.uniform([1], maxval=y.shape[0] - y_frame_size, dtype=tf.int32)[0] # 프레임 크기를 고려하여 offset 설정
        mono_y_frame_size = y_frame_size
        mono_y_offset = tf.random.uniform([1], maxval=raw_y.shape[0] - mono_y_frame_size, dtype=tf.int32)[0]

        x_frame_size = y_frame_size * resolution
        x_offset = y_offset * resolution
        mono_x_frame_size = x_frame_size
        mono_x_offset = mono_y_offset * resolution

        mono_y_frame = raw_y[mono_y_offset:mono_y_offset + mono_y_frame_size]
        y_frame_offset = tf.random.uniform([1], maxval=y.shape[0] - tf.shape(mono_y_frame)[0], dtype=tf.int32)[0]
        mono_y_frame = tf.pad(mono_y_frame, ((y_frame_offset, y.shape[0] - y_frame_offset - tf.shape(mono_y_frame)[0]),(0,0)))
        mono_y_frame = tf.reshape(mono_y_frame, [*y.shape])
        
        mono_x_frame = raw_x[mono_x_offset:mono_x_offset + mono_x_frame_size * resolution]
        x_frame_offset = y_frame_offset * resolution
        mono_x_frame = tf.pad(mono_x_frame, ((x_frame_offset, x.shape[0] - x_frame_offset - tf.shape(mono_x_frame)[0]),(0,0),(0,0)))
        mono_x_frame = tf.reshape(mono_x_frame, [*x.shape])
        mono_x_frame = equilizer(mono_x_frame)
        
        ratio = tf.random.uniform([1], minval=1e-3, maxval=1 - 1e-3)[0]
        sampled_x_frame = equilizer(x)
        # class number in same frame constraint
        cond = tf.reduce_sum(tf.cast((y[...,:class_num] + mono_y_frame[...,:class_num]) < 2, tf.float32), -1) == 1 # 합성 후 같은 class가 같은 frame에 존재하는 지 여부 탐색
        cond = tf.logical_and(cond, tf.reduce_sum(y[..., :class_num], -1) < 2) # 원래 class가 2개 미만인 프레임만 합성
        y = tf.where(cond, x=y + mono_y_frame, y=y)
        x = tf.where(tf.repeat(cond, resolution), x=ratio * sampled_x_frame + (1 - ratio) * mono_x_frame, y=x)
        import pdb; pdb.set_trace()
        # mono_x_frame = tf.gather(raw_x, tf.range())

        return x, y
    return _EMDA
    

def biquad_equilizer(sampling_rate, central_freq=[100, 6000], g=[-8,8], Q=[1,9]):
    '''
    central_freq: central frequency
    g: gain
    Q: Q-factor
    '''
    def _band_biquad_equilizer(wav):
        gain = tf.random.uniform([1], minval=g[0], maxval=g[1])[0]
        central_frequency = tf.random.uniform([1], minval=central_freq[0], maxval=central_freq[1])[0]
        Qfactor = tf.random.uniform([1], minval=Q[0], maxval=Q[1])[0]

        # wav *= gain

        w0 = 2 * np.math.pi * central_frequency / sampling_rate
        bw_Hz = central_frequency / Qfactor

        a0 = tf.constant(1.0, dtype=wav.dtype)
        a2 = tf.exp(-2 * np.math.pi * bw_Hz / sampling_rate)
        a1 = -4 * a2 / (1 + a2) * tf.cos(w0)

        b0 = tf.sqrt(1 - a1 * a1 / (4 * a2)) * (1 - a2)

        b1 = tf.constant(0., dtype=wav.dtype)
        b2 = tf.constant(0., dtype=wav.dtype)
        return biquad(wav, b0, b1, b2, a0, a1, a2)

    def biquad(wav, b0, b1, b2, a0, a1, a2):
        return lfilter(wav, tf.concat([a0, a1, a2], 0), tf.concat([b0, b1, b2], 0))

    def lfilter(wav, a_coeffs, b_coeffs):
        # (time, chan)
        wav = ss.filtfilt(b_coeffs, a_coeffs, wav, axis=1)
        return tf.clip_by_value(wav, -1., 1.)
    return _band_biquad_equilizer


if __name__ == '__main__':
    import tensorflow as tf
    aa = tf.random.uniform([300,3])
    bb = biquad_equilizer(24000)(aa)
    import pdb; pdb.set_trace()
    print()