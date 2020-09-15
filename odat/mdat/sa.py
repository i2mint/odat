from functools import partial
from collections import deque
from itertools import dropwhile, islice, chain
from io import BytesIO
import os
import numpy as np
import pandas as pd
import soundfile as sf
from py2store import FilesOfZip, KvReader, wrap_kvs, filt_iter, cached_keys
from odat.utils.chunkers import clever_chunker

DFLT_CHK_SIZE = 2048
DFLT_CHK_STEP = None
DFLT_CHUNKER = partial(clever_chunker, tile_size=DFLT_CHK_SIZE, tile_step=DFLT_CHK_STEP)
# A function to be applied to the raw fft output
DFLT_SPECTRA_TRANS = lambda x: np.abs(x)


######################################### STORES ###########################################
def mk_wf_store(zip_file_path, wf_folder):
    """
    Make a store accessing the wav files in the zip
    :param zip_file_path: str, path to the zip file
    :param wf_folder: str, the name of the folder containing the wav files within the zip file
    :return: a store with key being the name of the wav files and values the array of samples
    """
    wf_folder_size = len(wf_folder)

    @wrap_kvs(
        key_of_id=lambda x: x[wf_folder_size + 1:],
        id_of_key=lambda x: os.path.join(wf_folder, x),
        obj_of_data=lambda b: sf.read(BytesIO(b), dtype='int16')[0])
    @filt_iter(filt=lambda x: x.endswith('.wav'))
    class WfStore(FilesOfZip):
        """Waveform access. Keys are .wav filenames and values are numpy arrays of int16 waveform."""
        pass

    # making the two stores on which the dacc will be build
    wfs = WfStore(zip_file_path)  # access the wf
    # wfs_list = list(wfs)
    # t = ", ".join(list(wfs)[:3])
    return wfs


def mk_context_store(zip_file_path):
    """
    Make a store accessing the csv files in the zip
    :param zip_file_path: str, path to the zip file
    :return: a store with key being the name of the csv files and values the panda dataframe of the
             files
    """

    @filt_iter(filt=lambda x: x.endswith('.csv'))
    @wrap_kvs(obj_of_data=lambda b: pd.read_csv(BytesIO(b)))
    class CSVStore(FilesOfZip):
        pass

    context_store = CSVStore(zip_file_path)
    return context_store


class AnnotStore(KvReader):
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


######################### FILE PATHS AND STORES INITIALIZATION #########################
dflt_path_templates = (
    "{name}",  # dataname IS the path
    "~/Downloads/{name}",  # can be found in Downloads
    "~/odata/{name}",  # can be found in ~/odata
    "~/{name}",  # can be found in home folder
)
normalize_path = lambda p: os.path.abspath(os.path.expanduser(p))


def find_datapath(name, path_templates=dflt_path_templates):
    path_options = map(lambda x: normalize_path(x.format(name=name)), path_templates)
    return next(filter(os.path.exists, path_options), None)


dataname = 'SAS.zip'
wf_folder = 'wav'
zip_file_path = find_datapath(dataname)
wfs = mk_wf_store(zip_file_path, wf_folder)
context_store = mk_context_store(zip_file_path)
context_name = 'context.csv'
annots = AnnotStore(context_store[context_name], key_cols='filename')


####################################### DATA ACCESS #######################################
class Dacc:
    """The Dacc is an object that will allow access to both the wf store and the annotation store
    and let us get chunks, spectra and fvs. Again, don't spend time trying to understand everything
    """

    def __init__(self,
                 wfs=wfs,
                 annots=annots,
                 annot_to_tag=lambda x: x.name,
                 extra_info=lambda x: x.truth,
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
