"""Append-only journal. Planted bug: void deletes in place."""

_entries = []


def post(entry):
    row = dict(entry)
    row.setdefault("voided", False)
    _entries.append(row)
    return len(_entries) - 1


def void(index):
    del _entries[index]


def open_entries():
    return [row for row in _entries if not row.get("voided")]


def all_entries():
    return list(_entries)
