import os
from itertools import chain
from collections import Counter
from warnings import warn
import re
from io import BytesIO

import pandas as pd
import numpy as np
import soundfile as sf

from py2store.stores.s3_store import S3BinaryStore

from py2store import (
    LocalJsonStore,
    add_ipython_key_completions,
    mk_read_only,
    lazyprop,
)
from odat import ODAT_DIR, ojoin, pjoin

odat_subdir = 'iatis'
dflt_data_rootdir = ojoin(odat_subdir)

from py2store import myconfigs

try:
    ro_client_kwargs = myconfigs['oto.ini']['s3_all_ro']
except Exception:
    warn("You don't have the myconfigs['oto.ini']['s3_all_ro'], which I need to be able to access s3. "
         "This means you won't be able to access the audio data.")


def delete_empty_records(s):
    """To delete all empty (i.e. length 0) items of a store"""
    for k, v in s.items():
        if len(v) == 0:
            del s[k]


def mk_subdict_extractor(fields):
    """Makes a function that generates field-inclusion defined subdicts from a dict iterator."""

    def extract(dict_list):
        for dd in dict_list:
            yield {k: dd[k] for k in fields}

    return extract


from py2store import KvReader, kv_wrap


# Pattern: Explicit store pattern (dict wrapping store). Todo: Use relevant tools
class DfGroupReader(KvReader):
    def __init__(self, df, by=None, **groupby_kwargs):
        self.src = df.groupby(by, **groupby_kwargs)

    def __iter__(self):
        yield from self.src

    def __contains__(self, k):
        return k in self.src

    def __len__(self):
        return len(self.src)

    def __getitem__(self, k):
        return self.src.get_group(k)


@kv_wrap.outcoming_vals(lambda x: x["_id"].values.tolist())
class TagSrefs(DfGroupReader):
    pass


s3_path_p = re.compile('^(?P<protocol>[^:]+)://(?P<bucket>[^/]+)/(?P<path>.*)')


def parse_sref(k):
    m = s3_path_p.match(k)
    if m:
        protocol, bucket, path = m.groups()
        return protocol, bucket, path
    else:
        raise KeyError(f"Your key couldn't be parsed: {k}")


def mk_sref_wav_reader(resource_kwargs):
    def get_wfsr(k):
        protocol, bucket, path = parse_sref(k)
        assert protocol == 's3', f"That wasn't an s3 path: {k}"
        b = S3BinaryStore(bucket, '', resource_kwargs)[path]
        return sf.read(BytesIO(b))

    return get_wfsr


get_wfsr = mk_sref_wav_reader(ro_client_kwargs)


class Dacc:
    _fv_mgc_subdir = 'iatis_fv_mgc'
    _fv_mgc_subpath = '{group}/{user}'
    _sref_tag_df_pkl = 'sref_tag_df.pkl'
    cache_sref_tag_df = True

    def __init__(self, data_rootdir=dflt_data_rootdir):
        self.data_rootdir = data_rootdir

    def djoin(self, *paths):
        return os.path.join(self.data_rootdir, *paths)

    def random_wfsrs_tagged(self, tag, n=1):
        srefs = self.sref_tag_store[tag]
        n_srefs = len(srefs)
        srefs_selected = np.random.choice(srefs, n, replace=n_srefs < n)
        for sref in srefs_selected:
            yield sref, get_wfsr(sref)

    def tag_wf_gen(self, tags, assert_sr):
        for sref, tag, (wf, sr) in self.sref_tag_wfsr_gen(tags):
            if assert_sr is not None:
                assert sr == assert_sr, f"You are asserting sr={assert_sr}, but I got {sr}"
            yield tag, wf

    def sref_tag_wfsr_gen(self, tags):
        for tag in tags:
            for sref in self.sref_tag_store[tag]:
                yield sref, tag, get_wfsr(sref)

    @lazyprop
    def sref_tag_store(self):
        return TagSrefs(self.sref_tag_df, 'tag')

    # Pattern: Caching. TODO: Use local file py2store caching for this
    @lazyprop
    def sref_tag_df(self):
        """df containing tag and _id (s3 key) columns.
        (Took 1mn to make on my computer)
        """
        if os.path.isfile(self._path_sref_tag_df_pkl):
            df = pd.read_pickle(self._path_sref_tag_df_pkl)
        else:
            extract_tag_and_id = mk_subdict_extractor(('_id', 'tag'))
            df = pd.DataFrame(list(chain.from_iterable(map(extract_tag_and_id,
                                                           self.fv_mgc_store.values()))))
            if self.cache_sref_tag_df:
                df.to_pickle(self._path_sref_tag_df_pkl)

        return df

    @lazyprop
    def _path_sref_tag_df_pkl(self):
        return self.djoin(self._sref_tag_df_pkl)

    @lazyprop
    def _path_fv_mgc_path_format(self):
        return self.djoin(self._fv_mgc_subdir,
                          self._fv_mgc_subpath)

    @lazyprop
    def fv_mgc_store(self):
        return mk_read_only(LocalJsonStore(self._path_fv_mgc_path_format))

    @lazyprop
    def tag_counts(self):
        return pd.Series(Counter(self.sref_tag_df['tag'])).sort_values(ascending=False)
