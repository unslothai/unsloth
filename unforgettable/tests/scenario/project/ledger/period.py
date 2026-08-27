"""Period close. Planted bug: includes entries after the cutoff."""


def close_period(entries, as_of):
    del as_of
    return list(entries)
