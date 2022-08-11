from py2store import myconfigs
import numpy as np
from functools import partial
from sklearn.preprocessing import normalize
from hear import WavLocalFileStore
from dol import wrap_kvs
import soundfile as sf
from io import BytesIO
from odat.utils.chunkers import fixed_step_chunker
from slang.featurizers import tile_fft
import pandas as pd


DFLT_CHUNKER = partial(fixed_step_chunker, chk_size=2048)
DFLT_FEATURIZER = tile_fft

config_filename = "vacuum.json"


def create_source_dir(fname):
    try:
        result = myconfigs.get_config_value(config_filename, "local_source_dir")
        return result
    except TypeError:
        print(f"A config_filename called {config_filename} needs to be present")


DFLT_LOCAL_SOURCE_DIR = create_source_dir(config_filename)
DFLT_ANNOTS_COLS = ["srefs", "tag", "train_info", "full_tag"]


def extract_annot_info(sref):
    train_info, classification = sref.split("/")
    annot, full_annot, *rest = classification.split(".")
    return sref, annot, train_info, full_annot


def annot_columns(srefs):
    return list(map(extract_annot_info, srefs))


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

    def mk_annots(self):
        srefs = self.wfs.keys()
        annots = annot_columns(srefs)
        return annots

    def mk_annots_df(self):
        annots = self.mk_annots()
        columns = DFLT_ANNOTS_COLS
        df = pd.DataFrame(annots, columns=columns)
        return df

    def wf_tag_train_gen(self):
        for key in self.wfs:
            signal = self.wfs[key]
            train = key.split("/")[0]
            tag = key.split("/")[1].split(".")[0]
            normal_wf = normalize(np.float32(signal).reshape(1, -1))[0]

            yield normal_wf, tag, train

    def chk_tag_train_gen(self, chunker=DFLT_CHUNKER):
        for wf, tag, train in self.wf_tag_train_gen():
            for chk in chunker(wf):
                yield chk, tag, train

    def fvs_tag_train_gen(self, featurizer=DFLT_FEATURIZER):
        for chk, tag, train in self.chk_tag_train_gen():
            yield featurizer(chk), tag, train

    def mk_Xy(self):  # TODO use a groupby here
        X_train, y_train, X_test, y_test = [], [], [], []
        for fv, tag, train in self.fvs_tag_train_gen():
            if train == "train":
                X_train.append(fv)
                y_train.append(tag)
            elif train == "test":
                X_test.append(fv)
                y_test.append(tag)
            else:
                continue
        return np.array(X_train), y_train, np.array(X_test), y_test


if __name__ == "__main__":
    dacc = mk_dacc()
    print(next(dacc.chk_tag_train_gen()))
