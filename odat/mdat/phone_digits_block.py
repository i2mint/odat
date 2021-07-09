from py2store.stores.local_store import RelativeDirPathFormatKeys, PickleStore
from hear.session_block_stores import BlockWfStore, ScoopAnnotationsStore
import numpy as np
from sklearn.preprocessing import normalize, RobustScaler


def mk_dacc(root_dir):
    return Dacc(root_dir=root_dir)


class DirStore(RelativeDirPathFormatKeys):
    def __getitem__(self, k):
        return self.__class__(self._id_of_key(k))

    def __repr__(self):
        return self._prefix


class Dacc:
    def __init__(self, root_dir):
        self.root_store = DirStore(root_dir)
        self.audio_dir = self.root_store['data/c/macosx_built-in-mic']
        self.annots_dir = self.root_store['annotations/s']
        self.snips_dir = self.root_store['data/snips']

        self.block_store = BlockWfStore(channel_data_dir=self.audio_dir._prefix)

        self.annots_store = ScoopAnnotationsStore(
            self.annots_dir._prefix,
            time_units_per_sec=int(1e6),
            csv_timestamp_time_units_per_sec=1000,
            sr=self.block_store.sr,
        )

        self.annots_df = self.annots_store[list(self.annots_store)[-1]]
        self.annots = self.annots_df.to_dict('records')

    def wf_tag_gen(self):
        for annot in self.annots:
            bt, tt, tag = annot['bt'], annot['tt'], annot['tag']
            for wf in self.block_store.block_search(bt=bt, tt=tt).values():
                normal_wf = normalize(np.float32(wf).reshape(1, -1))[0]
                yield normal_wf, tag
                # yield wf, tag

    def chk_tag_gen(self, chunker):
        for wf, tag in self.wf_tag_gen():
            for chk in chunker(wf):
                yield chk, tag

    def filtered_chk_tag_gen(
        self, chunker, snipper, score_of_snip, thresh_score_for_tag
    ):
        for wf, tag in self.wf_tag_gen():
            for chk in chunker(wf):
                snip = next(snipper.wf_to_snips(chk))
                if score_of_snip(tag, snip) > thresh_score_for_tag[tag]:
                    yield chk, tag

    def snips_tag_gen(self, snipper):
        for wf, tag in self.wf_tag_gen():
            yield snipper.wf_to_snips(wf), tag


_meta = dict(name='phone_digits', description='Phone digits data', mk_dacc=mk_dacc)
