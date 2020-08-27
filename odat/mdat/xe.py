import os

from py2store import myconfigs
from py2store.ext.audio import WavLocalFileStore
from slang.core import KvDataSource
import re

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
