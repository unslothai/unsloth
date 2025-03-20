import time
from contextlib import contextmanager


@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print(f"{name} took {end - start:.2f} seconds")


@contextmanager
def header_footer_context(title: str, char="-"):
    print()
    print(f"{char}" * 50 + f" {title} " + f"{char}" * 50)
    yield
    print(f"{char}" * (100 + len(title) + 2))
    print()
