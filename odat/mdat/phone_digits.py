from py2store.slib.s_zipfile import ZipReader
from py2store import wrap_kvs, filt_iter
import soundfile as sf
from io import BytesIO
from slang import KvDataSource
import re
import numpy as np
from sklearn.preprocessing import normalize


def mk_dacc(zip_dir):
    return Dacc(zip_dir=zip_dir)


def mk_ds(zip_dir):
    s = ZipReader(zip_dir)
    _wf_store = filt_iter(s, filt=lambda x: x.startswith("train/"))
    _wf_store = filt_iter(_wf_store, filt=lambda x: x.endswith(".wav"))
    wf_store = wrap_kvs(_wf_store, obj_of_data=lambda v: sf.read(BytesIO(v))[0])
    path_component = re.compile("[^_]+")

    def key(x):
        m = path_component.match(x)
        if m:
            return int(m.group(0)[-1])

    ds = KvDataSource(wf_store, key_to_tag=key)
    return ds


class Dacc:
    def __init__(self, zip_dir):
        self.ds = mk_ds(zip_dir)

    def wf_tag_gen(self):
        for _, tag, wf in self.ds.key_tag_wf_gen():
            normal_wf = normalize(np.float32(wf).reshape(1, -1))[0]
            yield normal_wf, tag

    def chk_tag_gen(self, chunker):
        for wf, tag in self.wf_tag_gen():
            for chk in chunker(wf):
                yield chk, tag

    def snips_tag_gen(self, snipper):
        for wf, tag in self.wf_tag_gen():
            yield snipper.wf_to_snips(wf), tag


_meta = dict(
    name="phone_digits_new", description="Phone digits new data", mk_dacc=mk_dacc
)
