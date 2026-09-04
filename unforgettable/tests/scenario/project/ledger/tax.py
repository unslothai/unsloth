"""Sales tax. Planted bug: the rate is applied twice."""

RATE = "0.0825"


def tax_on(cents):
    once = int(round(cents * 0.0825))
    return int(round(once + once * 0.0825))
