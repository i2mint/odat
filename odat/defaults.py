import os


class DFLT_DIRS:
    LOCAL_STORE = '~/odat'

# for a in filter(lambda x: not x.startswith('_'), dir(DFLT_DIRS)):
for k, v in DFLT_DIRS.__dict__.items():
    setattr(DFLT_DIRS, k, os.path.expanduser(v))


