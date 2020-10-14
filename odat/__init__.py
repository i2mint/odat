import os
import re
from importlib import import_module

from odat.defaults import DFLT_DIRS
from odat.util import ModuleNotFoundIgnore, IgnoreAllErrors, submodules_of

pjoin = os.path.join
ODAT_DIR = pjoin(os.environ.get('ODAT_DIR', DFLT_DIRS.ODAT_DIR))


def ojoin(*paths):
    return os.path.join(ODAT_DIR, *paths)


def ddir(obj):
    return [attr for attr in dir(obj) if not attr.startswith('_')]


spaces_re = re.compile('\s+')


def print_attrs(obj, with_doc_snippet=True, max_doc_length=73, sep='\n'):
    if not with_doc_snippet:
        print(*ddir(obj), sep=sep)
    else:
        def gen():
            for attr_name in ddir(obj):
                attr = getattr(obj, attr_name)
                attr_doc = (getattr(attr, '__doc__', '') or '').strip()
                if len(attr_doc) > max_doc_length:
                    attr_doc = spaces_re.sub(' ', attr_doc[:max_doc_length].strip()) + '...'
                if attr_doc:
                    yield "- " + f"{attr_name}:\n\t{attr_doc}"
                else:
                    yield "- " + attr_name

        print(*gen(), sep=sep)


if not os.path.isdir(ODAT_DIR):
    print(f"""
    Hey, you're not finished with the setup. 
    I need a rootdir to work with. 
    That was supposed to be:
        {ODAT_DIR}
    But I didn't find that directory.

    So...

    Either make that directory for me, or define one you want me to work with.

    You can specify where you want this to be by defining an environment variable called ODAT_DIR.
    That's the first place the system will look for, so will override my default    

    """)


def dacc_info_gen(on_error='ignore'):
    src = 'odat.mdat'
    for submodule_name in submodules_of(src, include_init=False):
        try:
            submodule_dotpath = src + '.' + submodule_name
            submodule = import_module(submodule_dotpath)
            d = getattr(submodule, '_meta')
            yield submodule_name, d
        except Exception as e:
            if on_error == 'raise':
                raise
            elif on_error == 'print':
                print(f"Error with {submodule_dotpath}: {e}")


daccs = dict(dacc_info_gen(on_error='ignore'))
