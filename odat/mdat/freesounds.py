"""
Data accessor for the freesounds dataset:
https://zenodo.org/records/2552860#.XFD05fwo-V4

"""

from collections import Counter
from dol import wrap_kvs, add_prefix_filtering, add_ipython_key_completions
from dol import FilesOfZip
from config2py import config_getter


def freesounds_dataset_postget(path, bytes_):
    """A decoder that will decode the bytes of the freesounds dataset into
    the appropriate data type:
    - .wav files are decoded into numpy arrays
    - .csv files are decoded into pandas dataframes
    """
    import pandas as pd
    import numpy as np
    import io
    import recode

    if path.endswith('.wav'):
        wf, sr = recode.decode_wav_bytes(bytes_)  # sr is the sample rate
        # ... which we'll just ignore (but we might want to verify that all the same)
        return np.array(wf)
    elif path.endswith('.csv'):
        return pd.read_csv(io.BytesIO(bytes_))
    else:
        return bytes_


@add_ipython_key_completions  # adds tab-completion of keys in ipython
@add_prefix_filtering(
    relativize_prefix=True
)  # adds the ability to filter by key prefix
@wrap_kvs(postget=freesounds_dataset_postget)  # adds bytes-to-data decoding
class FreesoundsDataset(FilesOfZip):
    """A base store for the freesounds audio dataset."""


from functools import partial
from slang import KvDataSource


def _mk_dacc(
    raw_store=None,
    *,
    audio_key='audio_train/',
    annots_key='train_post_competition.csv',
    annots_table_key_col='fname',
    annots_table_tag_col='label',
):
    if raw_store is None:
        zip_filepath = config_getter('freesounds_audio_dataset_local_zip_filepath')
        raw_store = FreesoundsDataset(zip_filepath)

    audio_store = raw_store[audio_key]
    annots_store = (
        raw_store[annots_key].set_index(annots_table_key_col)[annots_table_tag_col]
    ).to_dict()
    dacc = KvDataSource(
        kv_store=audio_store,
        key_to_tag=annots_store.get,
    )
    try:
        dacc.tag_counts = dict(Counter(annots_store.values()).most_common())
        dacc.tags = list(dacc.tag_counts)
    except Exception:
        pass

    return dacc


mk_train_dacc = partial(
    _mk_dacc,
    audio_key='audio_train/',
    annots_key='train_post_competition.csv',
    annots_table_key_col='fname',
    annots_table_tag_col='label',
)

mk_test_dacc = partial(
    _mk_dacc,
    audio_key='audio_test/',
    annots_key='test_post_competition.csv',
    annots_table_key_col='fname',
    annots_table_tag_col='label',
)

mk_dacc = mk_train_dacc


_meta = dict(
    name="freesounds", description="A bunch of sounds from kaggle", mk_dacc=mk_dacc
)
