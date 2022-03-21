def get_map_phq_colnames():
    d = {
        # old: new
        'patient id': 'pid',
        'time stamp': 'time',
        'phq score': 'phq',
        'sleep score': 'phq1',
        'interest score': 'phq2',
        'energy score': 'phq3',
        'concentration score': 'phq4',
        'feeling down score': 'phq5',
        'feeling bad score': 'phq6',
        'appetite score': 'phq7',
        'moving score': 'phq8'
    }
    return d


def get_map_slp_colnames():
    d = {
        # old: new
        'patient id': 'pid',
        'time stamp': 'time',
        'stages': 'stages',
        'duration': 'duration'
    }
    return d
