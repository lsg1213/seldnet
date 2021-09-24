import numpy as np
import os
import torchaudio


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


def EMDA(raw_x, raw_y):
    # raw_x, raw_y : mono class sound
    def _EMDA(x, y):
        # x = (time, freq, chan)
        if x.shape[0] % y.shape[0] != 0:
            raise ValueError('resolution is not matched')
        resolution = x.shape[0] // y.shape[0]
        import pdb; pdb.set_trace()
        return x, y
    return _EMDA