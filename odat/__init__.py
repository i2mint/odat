import os
from odat.defaults import DFLT_DIRS
from odat.util import ModuleNotFoundIgnore

daccs = {}

with ModuleNotFoundIgnore():
    from odat.mdat import xe

    daccs['xe'] = {
        'description': 'Fridge compressor data',
        'dacc': xe.dacc
    }

with ModuleNotFoundIgnore():
    from odat.mdat import sa

    daccs['sa'] = {
        'description': 'Loose screws in rotating car dashboard',
        'dacc': sa.Dacc()
    }

with ModuleNotFoundIgnore():
    from odat.mdat import iatis

    daccs['iatis'] = {
        'description': 'Over 5 million tagged sounds',
        'dacc': iatis.Dacc()
    }

pjoin = os.path.join
ODAT_DIR = pjoin(os.environ.get('ODAT_DIR', DFLT_DIRS.ODAT_DIR))


def ojoin(*paths):
    return os.path.join(ODAT_DIR, *paths)


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
