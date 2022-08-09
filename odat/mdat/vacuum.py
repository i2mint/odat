from py2store import myconfigs
import numpy as np
from functools import partial
from sklearn.preprocessing import normalize
from hear import WavLocalFileStore
from dol import wrap_kvs
import soundfile as sf
from io import BytesIO
from odat.utils.chunkers import fixed_step_chunker

DFLT_CHUNKER = partial(fixed_step_chunker, chk_size=2048)

config_filename = "vacuum.json"
DFLT_LOCAL_SOURCE_DIR = myconfigs.get_config_value(config_filename, "local_source_dir")


def mk_dacc(root_dir=DFLT_LOCAL_SOURCE_DIR):
    return Dacc(root_dir=root_dir)


def wf_from_bytes(bytes):
    return sf.read(BytesIO(bytes), dtype="float32")[0]


def WfStore(root_store):
    obj_of_data = wf_from_bytes
    return wrap_kvs(root_store, obj_of_data=obj_of_data)


class Dacc:
    def __init__(self, root_dir=DFLT_LOCAL_SOURCE_DIR):
        self.wfs = WavLocalFileStore(root_dir)

    def wf_tag_gen(self):
        for key in self.wfs:
            signal = self.wfs[key]
            normal_wf = normalize(np.float32(signal).reshape(1, -1))[0]

            yield normal_wf, key

    def chk_tag_gen(self, chunker=DFLT_CHUNKER):
        for wf, tag in self.wf_tag_gen():
            for chk in chunker(wf):
                yield chk, tag


if __name__ == "__main__":
    dacc = mk_dacc()
    print(next(dacc.chk_tag_gen()))
