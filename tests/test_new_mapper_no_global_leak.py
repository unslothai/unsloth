"""Regression test for ``_get_new_mapper`` leaking into ``loader_utils`` globals.

``get_model_name`` calls ``_get_new_mapper()`` whenever a name misses the local
tables, purely to answer "would a newer Unsloth support this?". It fetches
``mapper.py`` from GitHub main, prefixes the three mappers it wants with
``NEW_``, and ``exec``s the result into ``globals()``.

The slice starts at ``__INT_TO_FLOAT_MAPPER``, so it also carries
``FLOAT_TO_FP8_BLOCK_MAPPER``/``FLOAT_TO_FP8_ROW_MAPPER`` and the two
``_add_*`` helpers, and those names are NOT renamed. Exec'ing into
``globals()`` therefore rebinds the FP8 tables that ``loader_utils`` imported
from the installed ``mapper``, so every later ``get_model_name(...,
load_in_fp8 = ...)`` in the process resolves through GitHub main's table
instead of the installed one. The probe is supposed to read, not to swap the
installed mappings out from under the caller.

``loader_utils`` imports torch, so ast-extract ``_get_new_mapper`` and run it
against a stubbed ``requests`` rather than importing unsloth (which needs a GPU).
"""

import ast
import os
import sys
import time
import types

_MODELS = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "models")


def _mapper_source():
    with open(os.path.join(_MODELS, "mapper.py"), encoding = "utf-8") as f:
        return f.read()


def _extract_get_new_mapper(namespace):
    # loader_utils imports this from .mapper, so the stand-in module globals need it
    # too. Without it the probe raises NameError into its own bare except and hands
    # back empty tables, which looks exactly like "the fetch did not run".
    from unsloth.models.mapper import build_mappers
    import unsloth.models.loader_utils as loader_utils

    namespace.setdefault("build_mappers", build_mappers)
    namespace.setdefault("_MAPPER_HELPERS", loader_utils._MAPPER_HELPERS)
    with open(os.path.join(_MODELS, "loader_utils.py"), encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_get_new_mapper":
            exec(compile(ast.Module([node], []), node.name, "exec"), namespace)
            return namespace["_get_new_mapper"]
    raise AssertionError("_get_new_mapper not found in loader_utils.py")


class _FakeRaw:
    """`read1` over a fixed list of chunks, returning b"" at the end."""

    def __init__(self, chunks):
        self._chunks = iter(chunks)

    decode_content = False

    def read1(self, amount = -1):
        # The probe must ASK for decoding; `requests` only enables it inside
        # `iter_content`, so a raw read of a gzip response would hand compressed bytes
        # to `ast.parse`.
        if not self.decode_content:
            return b"not-decoded"
        return next(self._chunks, b"")


class _FakeResponse:
    """The streaming half of `requests.Response`, which is all the probe uses.

    `_get_new_mapper` reads the body in chunks and stops at its byte cap while
    reading, because `requests.get` would otherwise buffer and decode the whole
    response before any length check could run. It also follows redirects by hand,
    since `requests` drains each intermediate body inside `get`, so the fake has to
    carry a status and headers as well as `iter_content`.
    """

    def __init__(
        self,
        text,
        chunks = None,
        status_code = 200,
        headers = None,
        raw = None,
    ):
        self.encoding = "utf-8"
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks if chunks is not None else [text.encode("utf-8")]
        self._raw = raw

    def iter_content(self, chunk_size = 1):
        yield from self._chunks

    @property
    def raw(self):
        """The urllib3 response the probe reads through.

        It calls `read1`, which returns whatever ONE socket read produced, so that a
        deadline check sits between every read rather than only between whole chunks.
        `iter_content` is kept because it is what `requests` exposes and the fake would
        otherwise drift from the real object.
        """
        if self._raw is None:
            self._raw = _FakeRaw(self._chunks)
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install_fake_requests(
    monkeypatch,
    text,
    chunks = None,
):
    module = types.ModuleType("requests")
    module.compat = types.SimpleNamespace(urljoin = lambda base, url: url)
    module.get = lambda url, timeout = None, stream = False, allow_redirects = True: (
        _FakeResponse(text, chunks)
    )
    monkeypatch.setitem(sys.modules, "requests", module)


def test_get_new_mapper_does_not_rebind_the_installed_fp8_tables(monkeypatch):
    _install_fake_requests(monkeypatch, _mapper_source())

    installed = {}
    exec(compile(_mapper_source(), "mapper.py", "exec"), installed)
    block = installed["FLOAT_TO_FP8_BLOCK_MAPPER"]
    row = installed["FLOAT_TO_FP8_ROW_MAPPER"]
    assert block and row, "the installed FP8 tables should not be empty"

    # Stand in for loader_utils' module globals, which import the FP8 tables.
    namespace = {"FLOAT_TO_FP8_BLOCK_MAPPER": block, "FLOAT_TO_FP8_ROW_MAPPER": row}
    get_new_mapper = _extract_get_new_mapper(namespace)

    int_to_float, float_to_int, map_to_16bit, fp8_block, fp8_row = get_new_mapper()

    # _get_new_mapper swallows every exception and returns empty dicts, so assert
    # it actually ran before trusting anything below.
    assert int_to_float and float_to_int and map_to_16bit, "the fetch/exec path did not run"

    # the probe has to hand the FETCHED fp8 tables back, or a newly added fp8 repo would
    # miss both the installed tables and the probe and skip the upgrade message
    assert fp8_block and fp8_row
    assert fp8_block is not block and fp8_row is not row

    assert namespace["FLOAT_TO_FP8_BLOCK_MAPPER"] is block
    assert namespace["FLOAT_TO_FP8_ROW_MAPPER"] is row


def test_get_new_mapper_leaves_no_helpers_behind(monkeypatch):
    _install_fake_requests(monkeypatch, _mapper_source())

    namespace = {}
    get_new_mapper = _extract_get_new_mapper(namespace)
    before = set(namespace)

    assert all(get_new_mapper()), "the fetch/exec path did not run"

    leaked = set(namespace) - before
    assert not leaked, f"_get_new_mapper leaked {sorted(leaked)} into its module globals"


def test_the_byte_cap_stops_the_read_instead_of_measuring_it_afterwards(monkeypatch):
    """A cap applied after `requests.get` returns cannot prevent what it exists for.

    `requests.get` buffers and decodes the whole body before handing it over, so
    `len(text) > cap` only ever measures something already in memory. An endpoint - or
    anything that can answer for it - serving an endless body would be read to the end
    first. This pins the cap to the READ: the probe must stop pulling chunks, and must
    return the empty tables it returns for every other failure.
    """
    served = []

    def endless():
        while True:
            served.append(1)
            if len(served) > 5_000:  # the probe should have stopped long before this
                raise AssertionError("the probe kept reading past its cap")
            yield b"x" * 65_536

    _install_fake_requests(monkeypatch, "", chunks = endless())
    get_new_mapper = _extract_get_new_mapper({})

    assert get_new_mapper() == ({}, {}, {}, {}, {})
    # The cap at 64KB a chunk is a few dozen chunks; anything near the guard above means the
    # cap is not being enforced while reading.
    assert len(served) < 200, len(served)


def test_a_redirect_body_is_bounded_too(monkeypatch):
    """`requests` drains an intermediate 3xx body inside `get`, before `stream=True`
    hands anything to the caller, so a redirect was a way around the cap and the
    deadline. The probe follows redirects itself for that reason; this pins it."""
    served = []

    def endless():
        while True:
            served.append(1)
            if len(served) > 5_000:
                raise AssertionError("the probe kept reading a redirect body past its cap")
            yield b"x" * 65_536

    # A redirect body must not be read at all, so `served` should stay empty.

    module = types.ModuleType("requests")
    module.compat = types.SimpleNamespace(urljoin = lambda base, url: url)
    module.get = lambda url, timeout = None, stream = False, allow_redirects = True: (
        _FakeResponse(
            "",
            chunks = endless(),
            status_code = 302,
            headers = {"location": "https://example.invalid/next"},
        )
    )
    monkeypatch.setitem(sys.modules, "requests", module)

    get_new_mapper = _extract_get_new_mapper({})
    assert get_new_mapper() == ({}, {}, {}, {}, {})
    assert not served, len(served)


def test_a_redirect_loop_ends(monkeypatch):
    """A peer that redirects forever must not keep the probe going forever."""
    hops = []

    module = types.ModuleType("requests")
    module.compat = types.SimpleNamespace(urljoin = lambda base, url: url)

    def get(
        url,
        timeout = None,
        stream = False,
        allow_redirects = True,
    ):
        hops.append(url)
        assert len(hops) < 50, "the probe followed redirects without a hop limit"
        return _FakeResponse("", status_code = 302, headers = {"location": url})

    module.get = get
    monkeypatch.setitem(sys.modules, "requests", module)

    get_new_mapper = _extract_get_new_mapper({})
    assert get_new_mapper() == ({}, {}, {}, {}, {})


def test_a_trickled_body_ends_at_the_deadline_not_at_the_chunk_size(monkeypatch):
    """The per-read check, which is what makes the deadline reachable at all.

    `iter_content(chunk_size = n)` yields only once n bytes have ARRIVED, and the
    socket timeout is per read, not per chunk. An endpoint sending one byte at a time
    resets that timeout on every read while the underlying call blocks for the whole
    chunk, so a deadline checked between chunks is not reached until 65_536 reads have
    gone by. The probe reads through `read1`, which returns what one read produced.

    The fake models exactly that difference: two seconds per byte either way, but
    `iter_content` only returns after a whole chunk of them. Time is faked rather than
    slept, so this is deterministic and instant. Against the previous chunked loop the
    clock reaches 131_072 seconds before the first check.
    """
    clock = {"now": 0.0}

    class _Trickling:
        """One byte per read at two seconds each, endlessly."""

        encoding = "utf-8"
        status_code = 200
        headers: dict = {}

        def iter_content(self, chunk_size = 1):
            while True:
                clock["now"] += 2.0 * chunk_size
                yield b"x" * chunk_size

        @property
        def raw(self):
            return self

        def read1(self, amount = -1):
            clock["now"] += 2.0
            return b"x"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    module = types.ModuleType("requests")
    module.compat = types.SimpleNamespace(urljoin = lambda base, url: url)
    module.get = lambda url, timeout = None, stream = False, allow_redirects = True: (_Trickling())
    monkeypatch.setitem(sys.modules, "requests", module)
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

    get_new_mapper = _extract_get_new_mapper({})
    assert get_new_mapper() == ({}, {}, {}, {}, {})
    # A 10s deadline at 2s a read is a handful of reads. A chunked loop would not look
    # at the clock until 65_536 of them had gone by.
    assert clock["now"] < 60, clock["now"]
