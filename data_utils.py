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


@tf.function
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
    

def biquad_equilizer(sampling_rate, central_freq=[100, 6000], g=[-8,8], Q=[1,9]):
    '''
    central_freq: central frequency
    g: gain
    Q: Q-factor
    '''
    def _band_biquad_equilizer(wav):
        gain = tf.random.uniform((), minval=g[0], maxval=g[1])
        central_frequency = tf.random.uniform((), minval=central_freq[0], maxval=central_freq[1])
        Qfactor = tf.random.uniform((), minval=Q[0], maxval=Q[1])

        # wav *= gain

        w0 = 2 * np.math.pi * central_frequency / sampling_rate
        A = tf.exp(gain / 40.0 * tf.math.log(10))
        alpha = tf.sin(w0) / 2 / Q

        b0 = 1 + alpha * A
        b1 = -2 * tf.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * tf.cos(w0)
        a2 = 1 - alpha / A
        return biquad(wav, b0, b1, b2, a0, a1, a2)

    def biquad(wav, b0, b1, b2, a0, a1, a2):
        return lfilter(wav, tf.stack([a0, a1, a2], 0), tf.stack([b0, b1, b2], 0))

    def lfilter(wav, a_coeffs, b_coeffs):
        dtype = wav.dtype
        # (time, chan)
        wav = ss.filtfilt(b_coeffs, a_coeffs, wav, axis=0)
        return tf.cast(tf.clip_by_value(wav, -1., 1.), dtype=dtype)
    return _band_biquad_equilizer


def stft(wav):
    wav = tf.transpose(wav, [1,0])
    out = tf.signal.stft(wav, 480, 240, 480, pad_end=True)
    return out


if __name__ == '__main__':
    from data_loader import load_wav_and_label
    import tensorflow as tf
    import numpy as np
    from tqdm import tqdm
    path = '/root/datasets/DCASE2021'
    x, y, sr = load_wav_and_label(os.path.join(path, 'foa_dev'),
                             os.path.join(path, 'metadata_dev'),
                             mode='train')
    x = np.stack([stft(i).numpy() for i in tqdm(x)], 0)
    y = np.stack(y, 0)
    joblib.dump(x, os.path.join(path, 'foa_dev_stft_480.joblib'))
    joblib.dump(y, os.path.join(path, 'foa_dev_label.joblib'))

