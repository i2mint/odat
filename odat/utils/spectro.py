import numpy as np
from numpy import array, hanning, fft  # TODO: Get rid of this once we use C-based spectr
from numpy.core.multiarray import zeros
from numpy.core.umath import ceil, log

# from omodel import DFLT_TILE_SIZE, DFLT_SR, DFLT_MEL_KWARGS, DFLT_MEL_MIN_AMP

DFLT_WIN_FUNC = hanning

DFLT_WF_NUM_TYPE = 'int16'
DFLT_SR = 44100
DFLT_TILE_SIZE = 2048
DFLT_HOP_SIZE = DFLT_TILE_SIZE
DFLT_TILE_STEP = DFLT_TILE_SIZE

DFLT_MEL_MIN_AMP = 1e-10
DFLT_MEL_KWARGS = dict(n_fft=2048, hop_length=512, n_mels=128)


class ParamValAssertionError(AssertionError):
    pass


class SampleRateAssertionError(ParamValAssertionError):
    pass


def tile_fft(tile, window=DFLT_WIN_FUNC, amp_function=np.abs):
    """Compute the power fft for a single tile
    """
    if callable(window):
        window = window(len(tile))
    fft_amplitudes = amp_function(np.fft.rfft(tile * window))
    return fft_amplitudes


def stft(y, n_fft=2048, hop_length=None, win_func=DFLT_WIN_FUNC):
    '''
    :param y: Mx0 audio
    :param n_fft: window size
    :param hop_length: hop size
    :return: S - DxN stft matrix
    '''

    if hop_length is None:
        hop_length = n_fft

    if win_func is not None:
        win = win_func(n_fft)
    else:
        win = 1

    # calculate STFT
    M = len(y)
    N = int(ceil(1.0 * (M - n_fft) / hop_length) + 1)  # no. windows
    S = zeros((n_fft, N), dtype='complex')
    for f in range(N - 1):
        S[:, f] = y[f * hop_length:n_fft + f * hop_length] * win
    x_end = y[(N - 1) * hop_length:]
    S[:len(x_end), N - 1] = x_end
    S[:, N - 1] *= win
    S = fft.fft(S, axis=0)
    S = S[:n_fft // 2 + 1, :]

    return S


def get_wf_to_spectr_func(tile_size=DFLT_TILE_SIZE, tile_step=None, win_func=DFLT_WIN_FUNC):
    """
    Get a spectr_of_wf function that computes spectrograms (lists of spectrums really) for given tile_size and tile_step

    :param tile_size: n_fft, also called "window size" or "buffer size"
    :param tile_step: By default, tile_size (i.e. non-overlapping and gapless sfft window support
    :return: a spectr_of_wf function that computes spectrograms
    >>> from random import randint
    >>>
    >>> tile_size = 2048
    >>> n_tiles = 21
    >>> spectr_of_wf = get_wf_to_spectr_func(tile_size)
    >>>
    >>> wf = [randint(-30000, 30000) for x in range(n_tiles * tile_size)]
    >>> spectr_of_wf(wf).shape
    (1025, 21)
    >>>
    >>> wf = [randint(-30000, 30000) for x in range(n_tiles * tile_size - 1)]
    >>> spectr_of_wf(wf).shape
    (1025, 20)
    >>>
    >>> wf = [randint(-30000, 30000) for x in range(n_tiles * tile_size + 1)]
    >>> spectr_of_wf(wf).shape
    (1025, 21)
    >>>
    >>> tile_step = 1024
    >>> spectr_of_wf = get_wf_to_spectr_func(tile_size, tile_step=tile_step)
    >>>
    >>> wf = [randint(-30000, 30000) for x in range(n_tiles * tile_size)]
    >>> spectr_of_wf(wf).shape
    (1025, 41)
    >>> spectr_of_wf = get_wf_to_spectr_func()
    >>> spectra = spectr_of_wf([0.] * 2048 * 3)
    >>> list(sum(spectra))
    [0.0, 0.0, 0.0]
    >>> all([x == 0 for x in sum(spectra)])
    True
    """

    if tile_step is None:
        tile_step = tile_size

    def spectr_of_wf(wf):
        n_tiles = max(1 + (len(wf) - tile_size) // tile_step, 0)
        return spectr(array(wf), sr=None, tile_size=tile_size, tile_step=tile_step)[:n_tiles, :].T

    return spectr_of_wf


def wf_to_spectr_func(tile_size=DFLT_TILE_SIZE, tile_step=None, win_func=None):
    """Like get_wf_to_spectr_func, but with frequencies in columns"""
    _wf_to_spectr = get_wf_to_spectr_func(tile_size, tile_step, win_func=win_func)

    def wf_to_spectr(wf):
        return _wf_to_spectr(wf).T

    return wf_to_spectr


dflt_wf_to_spectr = wf_to_spectr_func()


class Spectr(object):
    def __init__(self, tile_size=DFLT_TILE_SIZE, tile_step=None, win_func=DFLT_WIN_FUNC, **lin_trans_kwargs):
        self.tile_size = tile_size
        if tile_step is None:
            tile_step = tile_size
        self.tile_step = tile_step

    def __call__(self, wf):
        n_tiles = max(1 + (len(wf) - self.tile_size) // self.tile_step, 0)
        return spectr(array(wf), sr=None, tile_size=self.tile_size, tile_step=self.tile_step)[:n_tiles, :].T


def abs_stft(wf, tile_size=DFLT_TILE_SIZE, tile_step=None, win_func=DFLT_WIN_FUNC):
    if len(wf) > 0:
        return abs(stft(wf, n_fft=tile_size, hop_length=tile_step, win_func=win_func)).T
    else:
        return np.array([])


def spectr(wf, sr=None, tile_size=DFLT_TILE_SIZE, tile_step=None, win_func=DFLT_WIN_FUNC):
    tile_step = tile_step or tile_size
    return abs_stft(wf, tile_size=tile_size, tile_step=tile_step, win_func=win_func)


# Note: Removing until librosa gets it's act together
# try:
#     import librosa
#
#
#     def log_mel_of_wf_sr(wf, sr, common_sr=DFLT_SR):
#         if common_sr is None:
#             feat_mat = librosa.feature.melspectrogram(wf, sr=sr, **DFLT_MEL_KWARGS)
#         else:
#             _assert_sample_rate_if_not_none(common_sr, sr)
#             feat_mat = librosa.feature.melspectrogram(wf, sr=common_sr, **DFLT_MEL_KWARGS)
#         return librosa.db_to_amplitude(feat_mat, ref=DFLT_MEL_MIN_AMP).T  # TODO: Look into
#
#
#     def onset_strength_and_spectral_contrast(wf, sr):
#         onset_strength = librosa.onset.onset_strength(wf, sr)
#         spectral_contrast = librosa.feature.spectral_contrast(wf, sr)
#         return np.vstack((onset_strength, spectral_contrast)).T  # TODO: Look into
#
#
#     # TODO: Look into this!
#     def intensity_and_spectral_features(wf, sr, tile_size=DFLT_TILE_SIZE, tile_step=None):
#         tile_step = tile_step or tile_size
#         S = abs_stft(wf, tile_size=tile_size, tile_step=tile_step)
#         log_intensity = log(S.sum(axis=0) + 1)
#         onset_strength = librosa.onset.onset_strength(sr=sr, S=S)
#         spectral_centroid = librosa.feature.spectral_centroid(sr=sr, S=S, n_fft=tile_size, hop_length=tile_step)
#         spectral_contrast = librosa.feature.spectral_contrast(sr=sr, S=S, n_fft=tile_size, hop_length=tile_step)
#         return np.vstack((log_intensity, onset_strength, spectral_centroid, spectral_contrast)).T  # TODO: Look into
# except ImportError as e:
#     from warnings import warn
#
#     warn(str(e))


def intensity_and_normalized_spectr(wf, sr=None, tile_size=DFLT_TILE_SIZE,
                                    tile_step=None, intensity_normalization_factor=1):
    tile_step = tile_step or tile_size
    S = abs_stft(wf, tile_size=tile_size, tile_step=tile_step)
    intensity = S.sum(axis=0)
    S /= intensity
    return np.vstack((log(intensity + 1) / intensity_normalization_factor, S))


def intensity_normalized_spectr(wf, sr=None, tile_size=DFLT_TILE_SIZE, tile_step=None):
    tile_step = tile_step or tile_size
    S = abs_stft(wf, tile_size=tile_size, tile_step=tile_step)
    intensity = S.sum(axis=0)
    S /= intensity
    return S


def log_spectr(wf, sr=None, tile_size=DFLT_TILE_SIZE, tile_step=None):
    return log(1 + spectr(wf, sr=sr, tile_size=tile_size, tile_step=tile_step))


def _assert_sample_rate_if_not_none(assert_sr, sr=None):
    if sr is not None and sr != assert_sr:
        raise SampleRateAssertionError("Sample rate was {}: Should be {}".format(sr, assert_sr))
