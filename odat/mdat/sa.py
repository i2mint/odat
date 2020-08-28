# @title Dacc
from functools import partial
import time
from inspect import signature
from io import BytesIO
import os

# from oplot.plot_stats import *
import numpy as np
import pandas as pd
import soundfile as sf

from py2store import FilesOfZip, KvReader, wrap_kvs, filtered_iter, cached_keys
from odat.utils.chunkers import fixed_step_chunker

ddir = lambda o: [a for a in dir(o) if not a.startswith('_')]

DFLT_CHK_SIZE = 2048
DFLT_CHK_STEP = None
DFLT_CHUNKER = partial(fixed_step_chunker, chk_size=DFLT_CHK_SIZE, chk_step=DFLT_CHK_STEP)


def make_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def print_args_for(obj):
    name = getattr(obj, '__name__', getattr(obj.__class__, '__name__'))
    print(f"{name} construction arguments:\n\t" + ', '.join(signature(obj).parameters.keys()))


# DFLT_CHUNKER = lambda x: fixed_step_chunker(x, chk_size=2048)
DFLT_SPECTRA_TRANS = lambda x: 20 * np.log10(np.abs(x + 0.001))

normalize_path = lambda p: os.path.abspath(os.path.expanduser(p))


def print_if_verbose(verbose=True, *args, **kwargs):
    if verbose:
        if len(args) > 0 and len(args[0]) > 0:
            return print(*args, **kwargs)


class TimerAndFeedback:
    """Context manager that will serve as a timer, with custom feedback prints (or logging, or any callback)
    >>> with TimerAndFeedback():
    ...     time.sleep(0.5)
    Took 0.5 seconds
    >>> with TimerAndFeedback("doing something...", "... finished doing that thing"):
    ...     time.sleep(0.5)
    doing something...
    ... finished doing that thing
    Took 0.5 seconds
    >>> with TimerAndFeedback(verbose=False) as feedback:
    ...     time.sleep(1)
    >>> # but you still have access to some stats
    >>> _ = feedback  # For example: feedback.elapsed=1.0025296140000002 (start=1.159414532, end=2.161944146)
    """

    def __init__(self, start_msg="", end_msg="", verbose=True, print_func=print):
        self.start_msg = start_msg
        if end_msg:
            end_msg += '\n'
        self.end_msg = end_msg
        self.verbose = verbose
        self.print_func = print_func  # change print_func if you want to log, etc. instead

    def print_if_verbose(self, *args, **kwargs):
        if self.verbose:
            if len(args) > 0 and len(args[0]) > 0:
                return self.print_func(*args, **kwargs)

    def __enter__(self):
        self.print_if_verbose(self.start_msg)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        self.print_if_verbose(self.end_msg + f"Took {self.elapsed:0.1f} seconds")

    def __repr__(self):
        return f"elapsed={self.elapsed} (start={self.start}, end={self.end})"


dflt_path_templates = (
    "{name}",  # dataname IS the path
    "gdrive/My Drive/{name}",  # can be found in gdrive
    "~/Downloads/{name}",  # can be found in Downloads
    "~/odata/{name}",  # can be found in ~/odata
    "~/{name}",  # can be found in home folder
)


def find_datapath(name, path_templates=dflt_path_templates):
    path_options = map(lambda x: normalize_path(x.format(name=name)), path_templates)
    return next(filter(os.path.exists, path_options), None)


def mk_wf_store(zip_file_path, wf_folder, verbose=True):
    wf_folder_size = len(wf_folder)

    @wrap_kvs(
        key_of_id=lambda x: x[wf_folder_size + 1:],
        id_of_key=lambda x: os.path.join(wf_folder, x),
        obj_of_data=lambda b: sf.read(BytesIO(b), dtype='int16')[0])
    @filtered_iter(lambda x: x.endswith('.wav'))
    class WfStore(FilesOfZip):
        """Waveform access. Keys are .wav filenames and values are numpy arrays of int16 waveform."""
        pass

    # making the two stores on which the dacc will be build
    wfs = WfStore(zip_file_path)  # access the wf
    wfs_list = list(wfs)
    t = ", ".join(list(wfs)[:3])
    print_if_verbose(verbose, f"{len(wfs_list)} wfs: {t}...")

    return wfs


def mk_context_store(zip_file_path, verbose=True):
    @filtered_iter(lambda x: x.endswith('.csv'))
    @wrap_kvs(obj_of_data=lambda b: pd.read_csv(BytesIO(b)))
    class CSVStore(FilesOfZip):
        pass

    context_store = CSVStore(zip_file_path)
    print_if_verbose(verbose, f"context_store keys: {list(context_store)}")

    return context_store


class AnnotStore(KvReader):
    """Annotations access. Keys are .wav filenames and values are tags."""

    def __init__(self, annots_df, key_cols=None):
        self.annots_df = annots_df.set_index(key_cols)
        self.annots_df['filename'] = self.annots_df.index

    def __iter__(self):
        yield from self.annots_df.index.values

    def __getitem__(self, k):
        return self.annots_df.loc[k]

    def __len__(self):
        return len(self.annots_df)

    def __contains__(self, k):
        return k in self.annots_df.index


def is_iterable(x):
    from typing import Iterable
    return isinstance(x, Iterable)


class Dacc:
    """The Dacc is an object that will allow access to both the wf store and the annotation store
    and let us get chunks, spectra and fvs. Again, don't spend time trying to understand everything
    """

    def __init__(self,
                 wfs,
                 annots,
                 annot_to_tag=lambda x: x.name,
                 extra_info=None,
                 chunker=DFLT_CHUNKER,
                 spectra_trans=DFLT_SPECTRA_TRANS):
        self.wfs = wfs
        self.annots = annots
        self.annot_to_tag = annot_to_tag
        if extra_info is None:
            self.extra_info = lambda x: None
        else:
            self.extra_info = extra_info
        self.chunker = chunker
        self.spectra_trans = spectra_trans

    # this allows you to create an instance of the dacc class where the data is filtered out
    # you could also filter out externaly, but this can be convenient
    @classmethod
    def with_key_filt(cls, key_filt, wfs, annots, annot_to_tag, extra_info, chunker, spectra_trans):
        filtered_annots = cached_keys(annots, keys_cache=key_filt)
        filtered_wfs = cached_keys(wfs, keys_cache=key_filt)
        return cls(filtered_wfs, filtered_annots, annot_to_tag, extra_info, chunker, spectra_trans)

    def _get_tile_fft(self, tile):
        fft_amplitudes = self.spectra_trans(np.fft.rfft(tile))
        return fft_amplitudes

    def wf_tag_gen(self):
        """Yields the (wf, tag) pairs"""
        for k in self.annots:
            try:
                wf = self.wfs[k]
                annot = self.annots[k]
                yield wf, self.annot_to_tag(annot), self.extra_info(annot)
            except KeyError:
                pass

    def chk_tag_gen(self):
        """Yields the (chunk, tag) pairs"""
        for wf, tag, extra in self.wf_tag_gen():
            for chk in self.chunker(wf):
                yield chk, tag, extra

    def fft_tag_gen(self):
        """Yields the (fft, tag) pairs"""
        for chk, tag, extra in self.chk_tag_gen():
            yield self._get_tile_fft(chk), tag, extra


def make_train_test(context_store,
                    context_csv,
                    context_file_name_col='filename',
                    context_file_partition_name='dataset',
                    wfs=None,
                    train_str='Train',
                    test_str='Test',
                    annot_to_tag=lambda x: x.name,
                    chunker=DFLT_CHUNKER):
    """Make the necessary stores for the train test split given in the context_csv"""
    # accessing the annotations
    assert wfs is not None, "You need to specify a wf store (wfs)"
    annots = AnnotStore(context_store[context_csv], key_cols=context_file_name_col)
    df = annots.annots_df
    train_keys = list(df[df[context_file_partition_name] == train_str].index)
    test_keys = list(df[df[context_file_partition_name] == test_str].index)

    dacc_train = Dacc.with_key_filt(key_filt=train_keys,
                                    wfs=wfs,
                                    annots=annots,
                                    annot_to_tag=annot_to_tag,
                                    chunker=chunker)

    dacc_test = Dacc.with_key_filt(key_filt=test_keys,
                                   wfs=wfs,
                                   annots=annots,
                                   annot_to_tag=annot_to_tag,
                                   chunker=chunker)

    return dacc_train, dacc_test


dataname = 'SAS.zip'
wf_folder = 'wav'
zip_file_path = find_datapath(dataname)
wfs = mk_wf_store(zip_file_path, wf_folder)
context_store = mk_context_store(zip_file_path)
context_name = 'Context.csv'

# get the split
# dacc_train, dacc_test = make_train_test(context_store=context_store,
#                                         context_csv=context_name,
#                                         context_file_name_col='filename',
#                                         context_file_partition_name='dataset',
#                                         wfs=wfs,
#                                         train_str='Train',
#                                         test_str='Test',
#                                         annot_to_tag=lambda x: x.name,
#                                         #                                         extra_info=lambda x: x.truth,
#                                         chunker=DFLT_CHUNKER)
