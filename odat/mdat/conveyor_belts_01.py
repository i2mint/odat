from py2store import FilesOfZip, filt_iter, wrap_kvs, lazyprop
import soundfile as sf
from io import BytesIO
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from odat.utils.spectro import wf_to_spectr_func, DFLT_WIN_FUNC
from odat.util import find_datapath

dataname = 'conveyor_states.zip'
ASSERT_SR = 44100
DFLT_ZIP_FILE_PATH = find_datapath(dataname)

def get_wf(wav_bytes, assert_sr=ASSERT_SR):
    wf, sr = sf.read(BytesIO(wav_bytes))
    if assert_sr is not None:
        assert sr == assert_sr, f"Sample rate was {sr}. I expected {assert_sr}"
    return wf


key_to_tag_pattern = re.compile('(\w+).wav$')


def key_to_tag(key):
    tag, *_ = key_to_tag_pattern.search(key).groups()
    return tag


@wrap_kvs(obj_of_data=get_wf)
@filt_iter(filt=lambda x: not x.startswith('__MACOSX') and not x.endswith('DS_Store'))
class Wavs(FilesOfZip):
    pass


dflt_wf_to_spectr = wf_to_spectr_func(
    tile_size=2048, tile_step=None, win_func=DFLT_WIN_FUNC)


@dataclass
class Dacc:
    zip_filepath: str = DFLT_ZIP_FILE_PATH
    key_to_tag: Callable = key_to_tag
    wf_to_spectr: Callable = dflt_wf_to_spectr

    @lazyprop
    def wfs(self):
        return Wavs(self.zip_filepath)

    def wf_tag_gen(self):
        for k, wf in self.wfs.items():
            tag = self.key_to_tag(k)
            yield wf, tag

    def tag_spectr_gen(self):
        for k, wf in self.wfs.items():
            tag = self.key_to_tag(k)
            for spectra in self.wf_to_spectr(wf):
                yield tag, spectra

    @lazyprop
    def _spectra_tags(self):
        tags, spectras = list(map(np.array, zip(*self.tag_spectr_gen())))
        return spectras, tags

    @property
    def spectras(self):
        return self._spectra_tags[0]

    @property
    def tags(self):
        return self._spectra_tags[1]

    def get_xy_sample(self, n=400):
        idx = np.random.choice(range(len(self.tags)), size=n, replace=False)
        return self.spectras[idx], self.tags[idx]


_meta = dict(
    name='conveyor_belts_01.zip',
    description='Conveyor belts',
    mk_dacc=Dacc,
)
