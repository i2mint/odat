import os
from collections import Counter
import pandas as pd
import soundfile as sf
from io import BytesIO

from slang.chunkers import mk_chunker
from py2store import filt_iter, wrap_kvs
from haggle import KaggleDatasets


DFLT_CHUNKER = mk_chunker()


def mk_dacc(
    kaggle_path: str,
    annots_path: str,
    wrangle_func: lambda df: df,
    extension='.wav',
    key_path='',
):
    assert (
        kaggle_path is not None
    ), 'You need to specify the kaggle directory where the data is.'
    assert (
        annots_path is not None
    ), 'You need to specify the file storing your annotations.'
    return Dacc(kaggle_path, annots_path, wrangle_func, extension, key_path)


class Dacc:
    def __init__(self, kaggle_path, annots_path, wrangle_func, extension, key_path):
        self.kaggle_path = kaggle_path
        self.h = KaggleDatasets()
        self.s = self.h[self.kaggle_path]
        self.annots_df = None
        self.wfs = None
        self.set_annots_df(annots_path, wrangle_func)
        self.set_wfs(extension=extension, key_path=key_path)

    def get_files_df(self, path_sep=os.path.sep):
        """Get a dataframe containing all of the files in the kaggle dataset"""
        return pd.DataFrame([x.split(path_sep) for x in self.s])

    def file_extension_counts(self):
        """Get a counter of the different filetypes in the kaggle dataset"""
        return Counter(map(lambda k: os.path.splitext(k)[1], self.s))

    def get_files_of_type(self, extension='.csv'):
        """Returns the contents of the kaggle dataset filtered to one extension type"""
        return filt_iter(self.s, filt=lambda k: k.endswith(extension))

    def preview_file(self, path):
        """Converts and returns a csv at path to a dataframe"""
        return pd.read_csv(BytesIO(self.s[path]))

    def set_annots_df(self, path, func):
        """Sets annots_df with the path to a csv and a function to wrangle the dataframe"""
        annots_df = pd.read_csv(BytesIO(self.s[path]))
        self.annots_df = func(annots_df)

    def set_wfs(self, extension, key_path):
        """Sets wfs with the extension of the sounds files and the file_path to match annots_df"""
        _wfs = self.get_files_of_type(extension=extension)
        wfs = wrap_kvs(
            _wfs,
            obj_of_data=lambda v: sf.read(BytesIO(v))[0],
            key_of_id=lambda _id: _id[len(key_path) : -len(extension)],
            id_of_key=lambda k: key_path + k + extension,
        )
        self.wfs = wfs

    def key_tag_gen(self, tag_column, file_key_col='file_path'):
        """Get a (key, tag) pairs iterator for given tags"""
        for row_key, row in self.annots_df.iterrows():
            yield row[file_key_col], row[tag_column]

    def wf_tag_gen(self, tag_column, file_key_col='file_path'):
        """Get a (wf, tag) pairs iterator for given tags"""
        for key, tag in self.key_tag_gen(tag_column, file_key_col=file_key_col):
            yield self.wfs[str(key)], tag

    def chk_tag_gen(self, tag_column, chunker=DFLT_CHUNKER, file_key_col='file_path'):
        """Get a (chk, tag) pairs iterator for given tags"""
        for wf, tag in self.wf_tag_gen(tag_column, file_key_col=file_key_col):
            for chk in chunker(wf):
                yield chk, tag


_meta = dict(name='kgl', description='Kaggle data set', mk_dacc=mk_dacc)
