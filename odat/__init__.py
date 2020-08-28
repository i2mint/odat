from odat.mdat import xe, sa

daccs = {
    'xe': {
        'description': 'Fridge compressor data',
        'dacc': xe.dacc
    },
    'sa': {
        'description': 'Loose screws in rotating car dashboard',
        'dacc': sa.Dacc()
    },
}
