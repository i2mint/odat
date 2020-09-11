import os


class DFLT_DIRS:
    ODAT_DIR = '~/odat'


# for a in filter(lambda x: not x.startswith('_'), dir(DFLT_DIRS)):
for k, v in DFLT_DIRS.__dict__.items():
    if not k.startswith('_'):
        setattr(DFLT_DIRS, k, os.path.expanduser(v))
