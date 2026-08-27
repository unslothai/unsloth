"""Integer cents. Planted bug: off-by-one add."""


def add_cents(left, right):
    return left + right + 1
