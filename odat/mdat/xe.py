import os
from functools import partial
from py2store import myconfigs
from py2store.ext.audio import WavLocalFileStore
from slang.core import KvDataSource
import re
import numpy as np

from odat.utils.chunkers import fixed_step_chunker

DFLT_CHUNKER = partial(fixed_step_chunker, chk_size=2048)

config_filename = 'xe.json'
DFLT_LOCAL_SOURCE_DIR = myconfigs.get_config_value(config_filename, 'local_source_dir')

p_first_path_component = re.compile('[^/]+')


def first_path_component(x):
    m = p_first_path_component.match(x)
    if m:
        return m.group(0)


def mk_kv_data_source(local_source_dir=DFLT_LOCAL_SOURCE_DIR,
                      key_to_tag=first_path_component, key_filt=None):
    assert local_source_dir is not None, "You need to specify the directory where your data is"
    s = WavLocalFileStore(os.path.join(local_source_dir, '{tag}/{filename}.wav'))
    return KvDataSource(s, key_to_tag=key_to_tag, key_filt=key_filt)


def mk_dacc(local_source_dir=DFLT_LOCAL_SOURCE_DIR,
            key_to_tag=first_path_component,
            wf_to_chk=DFLT_CHUNKER,
            key_filt=None):
    assert local_source_dir is not None, "You need to specify the directory where your data is"
    s = WavLocalFileStore(os.path.join(local_source_dir, '{tag}/{filename}.wav'))

    class Dacc(KvDataSource):
        def chk_tag_gen(self):
            for k, tag, chk in self.key_tag_chks_gen(wf_to_chk=DFLT_CHUNKER):
                yield np.array(chk), tag

        def wf_tag_gen(self):
            for k, tag, wf in self.key_tag_wf_gen():
                yield wf, tag

    return Dacc(s, key_to_tag=key_to_tag, key_filt=key_filt)


_meta = dict(
    name='xe',
    description='Fridge compressor data',
    mk_dacc=mk_dacc
)
