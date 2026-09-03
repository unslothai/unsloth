# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic tests for the FLUX.2 size-pairing preflight.

A FLUX.2 GGUF only carries the transformer, so its ``inner_dim`` has to match the companion base
repo the loader assembles around it. The loader's own guard opens the DOWNLOADED checkpoint, so it
fires after the base shards were pulled and after the resident pipeline was torn down; these tests
pin the cheap version that runs first, off a range-read header.

Every synthetic checkpoint here is a real GGUF header written by ``gguf.GGUFWriter`` -- the tensor
table and nothing else, which is exactly what the range request brings back. The HTTP layer is
stubbed at ``huggingface_hub.utils.get_session``, so nothing in this module touches the network,
and a stubbed ``hf_hub_download`` fails the test if anything tries to fetch a whole file.
"""

from __future__ import annotations

import inspect
import threading
import time
import types

import pytest

from core.inference import diffusion_compat
from core.inference.diffusion import DiffusionBackend, _LoadingState
from core.inference.diffusion_families import (
    detect_family,
    gguf_flux2_inner_dim_from_header,
    sd_cpp_text_encoders_for,
)

KLEIN_4B_BASE = "black-forest-labs/FLUX.2-klein-4B"
KLEIN_9B_BASE = "black-forest-labs/FLUX.2-klein-9B"
KLEIN_4B_GGUF = "unsloth/FLUX.2-klein-4B-GGUF"
KLEIN_4B_FILE = "flux-2-klein-4b-Q4_K_M.gguf"

FLUX2_FAMILY = types.SimpleNamespace(name = "flux.2-klein", single_file_is_pipeline = False)


def _gguf_header(
    inner_dim: int,
    tmp_path,
    *,
    name = "double_stream_modulation_img.lin.weight",
    siblings = 4,
    probe_last = False,
):
    """A real GGUF header (magic + kv + tensor table, no tensor DATA) for a FLUX.2 of this size.

    Written with the shipped writer rather than hand-rolled bytes, so a format change breaks the
    test the same way it would break production. ``write_tensor_data`` is deliberately never
    called: the bytes that come back are precisely the prefix a range request returns."""
    import numpy as np
    from gguf import GGMLQuantizationType, GGUFWriter

    path = tmp_path / f"header-{inner_dim}-{siblings}-{int(probe_last)}.gguf"
    writer = GGUFWriter(str(path), "flux")

    def _probe():
        # FLUX.2 sizes this projection as (6 * inner_dim, inner_dim); GGUF stores dims reversed.
        writer.add_tensor_info(
            name,
            [6 * inner_dim, inner_dim],
            np.dtype(np.float16),
            6 * inner_dim * inner_dim * 2,
            raw_dtype = GGMLQuantizationType.F16,
        )

    if not probe_last:
        _probe()
    # Siblings so the probe tensor is not the whole table, as in a real checkpoint.
    for i in range(siblings):
        writer.add_tensor_info(
            f"blk.{i}.weight",
            [64, 64],
            np.dtype(np.float16),
            64 * 64 * 2,
            raw_dtype = GGMLQuantizationType.F16,
        )
    if probe_last:
        _probe()
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    writer.close()
    return path.read_bytes()


class _FakeResponse:
    """The slice of ``requests.Response`` the header read uses."""

    def __init__(
        self,
        status_code,
        body = b"",
    ):
        self.status_code = status_code
        self._body = body

    def iter_content(self, chunk_size = 1):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _stub_range_reads(
    monkeypatch,
    bodies,
    *,
    status = 206,
):
    """Serve ``{filename: header_bytes}`` over the stubbed Hub session; returns the request log.

    Also arms the negative assertions this whole module rests on: nothing may call
    ``hf_hub_download`` (that is the multi-GB pull), and nothing may read the ambient cache."""
    requests: list[tuple[str, str]] = []

    class _Session:
        def get(
            self,
            url,
            headers = None,
            timeout = None,
            stream = False,
        ):
            requests.append((url, (headers or {}).get("Range", "")))
            body = next((b for name, b in bodies.items() if url.endswith(name)), None)
            if body is None:
                return _FakeResponse(404)
            return _FakeResponse(status, body)

    monkeypatch.setattr("huggingface_hub.utils.get_session", lambda: _Session())
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda *a, **k: pytest.fail("the compatibility preflight must not download the file"),
    )
    diffusion_compat._reset_inner_dim_cache()
    return requests


@pytest.fixture(autouse = True)
def _clean_probe_cache():
    # The memo is process-global by design; a leak between tests would hide a missing probe.
    diffusion_compat._reset_inner_dim_cache()
    yield
    diffusion_compat._reset_inner_dim_cache()


# ── the header parser ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("inner_dim", [3072, 4096, 6144])
def test_the_inner_dim_is_read_from_a_header_with_no_tensor_data(inner_dim, tmp_path):
    # The whole premise: ~400 bytes of tensor table answer a question the 19 GB body would.
    assert gguf_flux2_inner_dim_from_header(_gguf_header(inner_dim, tmp_path)) == inner_dim


def test_a_plain_reader_cannot_read_the_same_prefix(tmp_path):
    # Why the header-only reader exists at all: GGUFReader also builds numpy views over every
    # tensor's DATA, so on a prefix it raises -- feed it one and the guard silently never fires.
    from gguf import GGUFReader
    from core.inference.diffusion_families import gguf_flux2_inner_dim

    prefix = tmp_path / "prefix-only.gguf"
    prefix.write_bytes(_gguf_header(3072, tmp_path))

    with pytest.raises(Exception):
        GGUFReader(str(prefix))
    assert gguf_flux2_inner_dim(prefix) is None
    assert gguf_flux2_inner_dim_from_header(prefix.read_bytes()) == 3072


@pytest.mark.parametrize(
    "header",
    [b"", b"not a gguf at all", b"GGUF" + b"\x00" * 64],
    ids = ["empty", "garbage", "truncated"],
)
def test_an_unreadable_header_yields_no_opinion(header):
    assert gguf_flux2_inner_dim_from_header(header) is None


def test_a_header_without_the_probe_tensor_yields_no_opinion(tmp_path):
    # A non-FLUX.2 checkpoint parses fine and simply has nothing to say.
    header = _gguf_header(3072, tmp_path, name = "blk.0.attn.weight")
    assert gguf_flux2_inner_dim_from_header(header) is None


@pytest.mark.parametrize("probe_last", [False, True], ids = ["probe-first", "probe-last"])
def test_no_truncation_of_a_valid_header_can_invent_a_dim(probe_last, tmp_path):
    # The sharp edge. The table is read field by field, so a cut BETWEEN a tensor's name and its
    # dims used to leave the name matching and the shape zero-filled -- "inner_dim 0", which is a
    # wrong answer rather than a missing one, and refuses a perfectly valid pick. A 206 body a few
    # bytes short, or a partially written On Device file, lands exactly there. Every prefix of a
    # real header must read as the real dim or as nothing at all.
    header = _gguf_header(4096, tmp_path, probe_last = probe_last)
    assert gguf_flux2_inner_dim_from_header(header) == 4096

    verdicts = {gguf_flux2_inner_dim_from_header(header[:cut]) for cut in range(1, len(header))}
    assert verdicts <= {None, 4096}, f"a truncated header invented {verdicts - {None, 4096}}"


def test_a_header_declaring_a_huge_checkpoint_is_cheap(tmp_path):
    # The base reader also builds a numpy view over every tensor's DATA. Satisfying those from a
    # prefix would allocate the whole DECLARED size -- tens of GiB, which commits pagefile on
    # Windows and turns the preflight into a permanent no-op there. 1200 tensors declaring ~37 GiB.
    import time

    header = _gguf_header(4096, tmp_path, siblings = 1200)
    started = time.monotonic()
    assert gguf_flux2_inner_dim_from_header(header) == 4096
    assert time.monotonic() - started < 5.0


# ── the preflight verdict ──────────────────────────────────────────────────────


def test_a_mismatched_pair_is_refused_from_the_header_alone(monkeypatch, tmp_path):
    # P1-5: a 4B GGUF against the 9B base. Refused off one range request, with no base metadata
    # call and no download at all -- the 19.17 GB pull never starts.
    requests = _stub_range_reads(monkeypatch, {KLEIN_4B_FILE: _gguf_header(3072, tmp_path)})

    with pytest.raises(ValueError) as excinfo:
        diffusion_compat.assert_flux2_pick_compatible(
            FLUX2_FAMILY, KLEIN_4B_GGUF, KLEIN_4B_FILE, KLEIN_9B_BASE
        )

    detail = str(excinfo.value)
    assert KLEIN_4B_FILE in detail and KLEIN_9B_BASE in detail
    assert "klein-4B" in detail and "klein-9B" in detail
    assert len(requests) == 1
    url, byte_range = requests[0]
    assert url.endswith(f"{KLEIN_4B_GGUF}/resolve/main/{KLEIN_4B_FILE}")
    # Bounded: the request must name an end offset, or a mis-set header streams the checkpoint.
    assert byte_range == f"bytes=0-{diffusion_compat._GGUF_HEADER_BYTES - 1}"


def test_a_matching_pair_passes(monkeypatch, tmp_path):
    _stub_range_reads(monkeypatch, {KLEIN_4B_FILE: _gguf_header(3072, tmp_path)})

    assert (
        diffusion_compat.flux2_pick_mismatch(
            FLUX2_FAMILY, KLEIN_4B_GGUF, KLEIN_4B_FILE, KLEIN_4B_BASE
        )
        is None
    )


def test_an_unreadable_header_fails_open(monkeypatch):
    # The contract: a false positive is worse than the bug it prevents, so a header we cannot
    # parse leaves the load exactly as it was, with the loader's own guard as the backstop.
    _stub_range_reads(monkeypatch, {KLEIN_4B_FILE: b"not a gguf"})

    assert (
        diffusion_compat.flux2_pick_mismatch(
            FLUX2_FAMILY, KLEIN_4B_GGUF, KLEIN_4B_FILE, KLEIN_9B_BASE
        )
        is None
    )


def test_an_offline_host_fails_open(monkeypatch):
    def _boom(*_a, **_k):
        raise OSError("no route to host")

    monkeypatch.setattr(
        "huggingface_hub.utils.get_session", lambda: types.SimpleNamespace(get = _boom)
    )
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)

    assert (
        diffusion_compat.flux2_pick_mismatch(
            FLUX2_FAMILY, KLEIN_4B_GGUF, KLEIN_4B_FILE, KLEIN_9B_BASE
        )
        is None
    )


def test_a_trickling_server_cannot_hold_the_picker_open(monkeypatch):
    # A pick is blocked on this read in the UI, and the pre-eviction preflight runs it on the
    # route's thread, so an unbounded read is a load request that never answers.
    #
    # Testing the deadline BETWEEN chunks is not enough, which is what this pins. iter_content
    # blocks inside urllib3 until a whole 64 KiB chunk has arrived, and requests' timeout is per
    # socket read, so a server dribbling a byte at a time resets it forever: measured against a
    # real loopback socket, one chunk at a byte a second is 18 hours and the loop never comes
    # back to look at the clock. Half-closing the socket is what ends it -- response.close() does
    # not, it drops the file object and leaves the socket readable -- and requests surfaces that
    # as a broken-connection error, which reads as "no opinion" like any other transport failure.
    interrupted = threading.Event()

    class _Trickle:
        status_code = 206

        def __init__(self):
            self.raw = types.SimpleNamespace(shutdown = interrupted.set)

        def iter_content(self, chunk_size = 1):
            if not interrupted.wait(10):
                raise AssertionError("the header read was never interrupted")
            raise OSError("Connection broken: IncompleteRead(12 bytes read)")

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(diffusion_compat, "_HEADER_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(
        "huggingface_hub.utils.get_session",
        lambda: types.SimpleNamespace(get = lambda *a, **k: _Trickle()),
    )
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)

    out: list[bytes] = []
    reader = threading.Thread(
        target = lambda: out.append(
            diffusion_compat._read_gguf_header(KLEIN_4B_GGUF, KLEIN_4B_FILE, None)
        ),
        daemon = True,
    )
    reader.start()
    reader.join(8)

    assert not reader.is_alive(), "the header read outlived its own deadline"
    assert out == [b""]


def test_a_large_custom_prefix_keeps_the_interrupting_deadline(monkeypatch):
    interrupted = threading.Event()
    ranges = []

    class _Trickle:
        status_code = 206

        def __init__(self):
            self.raw = types.SimpleNamespace(shutdown = interrupted.set)

        def iter_content(self, chunk_size = 1):
            if not interrupted.wait(5):
                raise AssertionError("the custom prefix read was never interrupted")
            raise OSError("connection interrupted")

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def _get(
        _url,
        headers = None,
        **_kwargs,
    ):
        ranges.append((headers or {}).get("Range"))
        return _Trickle()

    monkeypatch.setattr(
        "huggingface_hub.utils.get_session", lambda: types.SimpleNamespace(get = _get)
    )

    started = time.monotonic()
    assert (
        diffusion_compat._read_gguf_header(
            KLEIN_4B_GGUF,
            KLEIN_4B_FILE,
            None,
            revision = "pinned-sha",
            max_bytes = 32 * 1024**2,
            timeout_seconds = 0.1,
        )
        == b""
    )

    assert time.monotonic() - started < 2
    assert interrupted.is_set()
    assert ranges == [f"bytes=0-{32 * 1024**2 - 1}"]


def test_an_old_urllib3_with_no_shutdown_still_cannot_hold_the_picker_open(monkeypatch):
    # HTTPResponse.shutdown landed in urllib3 2.3.0. requirements/studio.txt floors it there, but
    # an install that resolved its environment BEFORE that floor keeps whatever it already has,
    # and nothing re-resolves a transitive pin on upgrade. On that urllib3 the watchdog degrades
    # to Response.close(), which leaves the socket readable and the read parked -- measured
    # against a real trickling loopback socket, the reader was still alive after 40 s.
    #
    # So the bound cannot be the interrupt working. This pins the other half: the drain runs on a
    # worker the caller ABANDONS, so the picker answers on time no matter what is underneath.
    still_reading = threading.Event()

    class _Unwakeable:
        status_code = 206
        # No `raw`, exactly like a urllib3 predating shutdown: _interrupt_read falls through to
        # close(), and close() does not end a read already inside iter_content.
        raw = None

        def iter_content(self, chunk_size = 1):
            still_reading.set()
            time.sleep(30)
            yield b"never arrives"

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(diffusion_compat, "_HEADER_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(
        "huggingface_hub.utils.get_session",
        lambda: types.SimpleNamespace(get = lambda *a, **k: _Unwakeable()),
    )
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)

    started = time.monotonic()
    assert diffusion_compat._read_gguf_header(KLEIN_4B_GGUF, KLEIN_4B_FILE, None) == b""
    elapsed = time.monotonic() - started

    assert still_reading.is_set(), "the fixture never got as far as blocking"
    # Deadline plus the abandon grace, with room for a loaded CI box. The point is that this is
    # bounded at all: without the worker it is the fixture's 30 s, and in the field it is forever.
    assert elapsed < 5, f"the picker waited {elapsed:.1f}s on a read it cannot interrupt"


def test_a_server_that_ignores_the_range_header_is_abandoned(monkeypatch, tmp_path):
    # A 200 means the whole multi-GB checkpoint is on the wire. Reading it here would BE the
    # download this preflight exists to avoid, so the body is dropped unread and the check
    # fails open.
    body_reads: list[int] = []

    class _WholeFile(_FakeResponse):
        def iter_content(self, chunk_size = 1):
            body_reads.append(chunk_size)
            yield b""

    class _Session:
        def get(
            self,
            url,
            headers = None,
            timeout = None,
            stream = False,
        ):
            return _WholeFile(200, _gguf_header(3072, tmp_path))

    monkeypatch.setattr("huggingface_hub.utils.get_session", lambda: _Session())
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)

    assert (
        diffusion_compat.flux2_pick_mismatch(
            FLUX2_FAMILY, KLEIN_4B_GGUF, KLEIN_4B_FILE, KLEIN_9B_BASE
        )
        is None
    )
    assert body_reads == []


@pytest.mark.parametrize(
    "base",
    ["someone/a-base-we-do-not-ship", "/models/my-local-flux2"],
    ids = ["unknown-repo", "local-path"],
)
def test_a_base_outside_the_size_table_costs_no_request_at_all(base, monkeypatch, tmp_path):
    # Nothing to compare the header against, so the check must notice BEFORE a round trip.
    requests = _stub_range_reads(monkeypatch, {KLEIN_4B_FILE: _gguf_header(3072, tmp_path)})

    assert (
        diffusion_compat.flux2_pick_mismatch(FLUX2_FAMILY, KLEIN_4B_GGUF, KLEIN_4B_FILE, base)
        is None
    )
    assert requests == []


def test_a_known_ungated_mirror_is_checked_like_its_upstream(monkeypatch, tmp_path):
    # Not an exception to the rule above: an unsloth mirror is a byte-identical copy, canonical_base
    # maps it back, and skipping it would leave the mirror picks -- the ones an anonymous user
    # actually gets -- as the only unguarded path.
    _stub_range_reads(monkeypatch, {KLEIN_4B_FILE: _gguf_header(3072, tmp_path)})
    from core.inference.diffusion_families import canonical_base

    mirror = "unsloth/FLUX.2-klein-9B"
    assert canonical_base(mirror).lower() == KLEIN_9B_BASE.lower(), "mirror table changed"

    reason = diffusion_compat.flux2_pick_mismatch(
        FLUX2_FAMILY, KLEIN_4B_GGUF, KLEIN_4B_FILE, mirror
    )

    assert reason is not None and mirror in reason


def test_a_non_flux2_family_costs_no_request_at_all(monkeypatch, tmp_path):
    # The base is a mapped FLUX.2 one, so only the FAMILY gate can stop this.
    requests = _stub_range_reads(monkeypatch, {"z.gguf": _gguf_header(3072, tmp_path)})

    assert (
        diffusion_compat.flux2_pick_mismatch(
            types.SimpleNamespace(name = "z-image"),
            "unsloth/Z-Image-Turbo-GGUF",
            "z.gguf",
            KLEIN_9B_BASE,
        )
        is None
    )
    assert requests == []


def test_a_non_gguf_single_file_costs_no_request_at_all(monkeypatch, tmp_path):
    # A single_file load names a .safetensors, which has no GGUF header to read.
    requests = _stub_range_reads(monkeypatch, {"model.safetensors": _gguf_header(3072, tmp_path)})

    assert (
        diffusion_compat.flux2_pick_mismatch(
            FLUX2_FAMILY, KLEIN_4B_GGUF, "model.safetensors", KLEIN_9B_BASE
        )
        is None
    )
    assert requests == []


def test_pasting_a_token_re_probes_a_pick_that_missed_anonymously(monkeypatch, tmp_path):
    # The memo keeps a MISS so an unreachable Hub is not asked three times per load, which would
    # otherwise leave a gated GGUF permanently unguarded for the session once it 401d anonymously.
    bodies: dict[str, bytes] = {}
    requests = _stub_range_reads(monkeypatch, bodies)

    assert (
        diffusion_compat.flux2_pick_mismatch(
            FLUX2_FAMILY, KLEIN_4B_GGUF, KLEIN_4B_FILE, KLEIN_9B_BASE
        )
        is None
    )
    assert len(requests) == 1
    # Same anonymous pick again: the miss is remembered, so no second round trip.
    diffusion_compat.flux2_pick_mismatch(FLUX2_FAMILY, KLEIN_4B_GGUF, KLEIN_4B_FILE, KLEIN_9B_BASE)
    assert len(requests) == 1

    bodies[KLEIN_4B_FILE] = _gguf_header(3072, tmp_path)
    reason = diffusion_compat.flux2_pick_mismatch(
        FLUX2_FAMILY, KLEIN_4B_GGUF, KLEIN_4B_FILE, KLEIN_9B_BASE, "hf_token_value"
    )

    assert len(requests) == 2
    assert reason is not None and KLEIN_9B_BASE in reason


def test_the_header_probe_is_memoised_across_the_three_checks(monkeypatch, tmp_path):
    # The plan, the pre-eviction preflight and the loader all ask about the same pick; one probe.
    requests = _stub_range_reads(monkeypatch, {KLEIN_4B_FILE: _gguf_header(3072, tmp_path)})

    for _ in range(3):
        with pytest.raises(ValueError):
            diffusion_compat.assert_flux2_pick_compatible(
                FLUX2_FAMILY, KLEIN_4B_GGUF, KLEIN_4B_FILE, KLEIN_9B_BASE
            )

    assert len(requests) == 1


def test_a_remembered_hub_failure_does_not_outlive_the_download(monkeypatch, tmp_path):
    # The memo exists so three checks cost one probe, and a "no opinion" has to be remembered too
    # or a genuinely unreadable pick re-probes the Hub forever. But the ordinary sequence is:
    # plan probes while the file is NOT yet on disk (a blip, or simply an unfinished download) ->
    # download completes -> the pre-eviction preflight asks again. If the negative shadows the
    # local read, the guard is silent for the rest of the process on a file sitting right there,
    # which is exactly the mismatch-after-a-19-GB-pull this preflight exists to prevent.
    requests = _stub_range_reads(monkeypatch, {})  # the Hub has nothing to say: negative memoised

    assert diffusion_compat.flux2_inner_dim_for_pick(KLEIN_4B_GGUF, KLEIN_4B_FILE) is None
    assert len(requests) == 1

    # The download lands the blob in the cache, which is what try_to_load_from_cache reports.
    staged = tmp_path / "blobs" / KLEIN_4B_FILE
    staged.parent.mkdir(parents = True, exist_ok = True)
    staged.write_bytes(_gguf_header(4096, tmp_path))
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: str(staged))

    assert diffusion_compat.flux2_inner_dim_for_pick(KLEIN_4B_GGUF, KLEIN_4B_FILE) == 4096
    # ...and the retry is free: the negative still suppresses a second range request.
    assert len(requests) == 1


def test_a_remembered_hub_failure_still_suppresses_a_re_probe_while_nothing_is_on_disk(
    monkeypatch, tmp_path
):
    # The other half of the memo's job, which the fix above must not cost: with no local file to
    # re-read, a remembered negative answers without touching the network again.
    requests = _stub_range_reads(monkeypatch, {})

    for _ in range(3):
        assert diffusion_compat.flux2_inner_dim_for_pick(KLEIN_4B_GGUF, KLEIN_4B_FILE) is None

    assert len(requests) == 1


def test_a_checkpoint_swapped_in_place_is_read_again(monkeypatch, tmp_path):
    """Same directory, same filename, different checkpoint. Keying the memo on the name alone
    answers the new file with the old one's dim, which refuses a valid 9B pairing and hands the
    native backend the 4B text encoders until the process restarts."""
    local = tmp_path / "on-device"
    local.mkdir()
    target = local / KLEIN_4B_FILE
    target.write_bytes(_gguf_header(3072, tmp_path))
    _stub_range_reads(monkeypatch, {})

    assert diffusion_compat.flux2_inner_dim_for_pick(str(local), KLEIN_4B_FILE) == 3072

    # Rewritten in place. os.stat resolution is coarse enough that a same-size overwrite inside
    # one tick could tie, so the two headers differ in length as well -- which is also what a real
    # 4B-for-9B swap looks like.
    target.write_bytes(_gguf_header(4096, tmp_path, siblings = 9))

    assert (
        diffusion_compat.flux2_inner_dim_for_pick(str(local), KLEIN_4B_FILE) == 4096
    ), "the memo answered for the file that used to be at this path"


def test_a_bad_token_does_not_poison_the_good_one_that_replaces_it(monkeypatch, tmp_path):
    """Keying on the token's mere PRESENCE made every non-empty credential one key. A first probe
    with an expired token cached its miss there, and pasting a working token afterwards inherited
    it for the rest of the process -- the preflight silently off for that pick."""
    served: dict[str, bytes] = {}
    requests: list[tuple[str, str]] = []

    class _GatedSession:
        def get(
            self,
            url,
            headers = None,
            timeout = None,
            stream = False,
        ):
            token = (headers or {}).get("authorization", "")
            requests.append((url, token))
            if "good" not in token:
                return _FakeResponse(401)
            return _FakeResponse(206, served[KLEIN_4B_FILE])

    served[KLEIN_4B_FILE] = _gguf_header(4096, tmp_path)
    monkeypatch.setattr("huggingface_hub.utils.get_session", lambda: _GatedSession())
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    diffusion_compat._reset_inner_dim_cache()

    assert (
        diffusion_compat.flux2_inner_dim_for_pick(KLEIN_4B_GGUF, KLEIN_4B_FILE, "expired") is None
    )
    assert diffusion_compat.flux2_inner_dim_for_pick(KLEIN_4B_GGUF, KLEIN_4B_FILE, "good") == 4096
    # ...and each token still memoises on its own: the second good probe costs nothing.
    before = len(requests)
    assert diffusion_compat.flux2_inner_dim_for_pick(KLEIN_4B_GGUF, KLEIN_4B_FILE, "good") == 4096
    assert len(requests) == before


def test_the_memo_never_holds_the_token_itself():
    """It is a process-global dict; a traceback or a heap dump renders it."""
    diffusion_compat._reset_inner_dim_cache()
    secret = "hf_averyrealsecrettoken"
    assert secret not in diffusion_compat._token_fingerprint(secret)
    assert diffusion_compat._token_fingerprint(None) == ""
    assert diffusion_compat._token_fingerprint("a") != diffusion_compat._token_fingerprint("b")


def test_a_gguf_already_on_disk_is_read_instead_of_fetched(monkeypatch, tmp_path):
    # A cached or On Device checkpoint answers for free, and it is the same file the loader opens.
    local = tmp_path / "local-repo"
    local.mkdir()
    (local / KLEIN_4B_FILE).write_bytes(_gguf_header(4096, tmp_path))
    requests = _stub_range_reads(monkeypatch, {})

    reason = diffusion_compat.flux2_pick_mismatch(
        FLUX2_FAMILY, str(local), KLEIN_4B_FILE, KLEIN_4B_BASE
    )

    assert reason is not None and "klein-9B" in reason
    assert requests == []


def test_a_local_header_past_the_prefix_cap_falls_back_to_the_full_reader(monkeypatch, tmp_path):
    # The prefix parse is an optimisation, not a limit: a table longer than the 256 KiB cap must
    # still be answered off the complete file on disk, not silently give up.
    import numpy as np
    from gguf import GGUFWriter
    from core.inference.diffusion_families import gguf_flux2_inner_dim_from_header

    local = tmp_path / "big-header-repo"
    local.mkdir()
    path = local / "wide.gguf"
    writer = GGUFWriter(str(path), "flux")
    # Tiny weights, enormous NAMES: the table has to exceed the cap while the file stays small.
    writer.add_tensor(
        "double_stream_modulation_img.lin.weight", np.zeros((48, 8), dtype = np.float16)
    )
    for i in range(1200):
        writer.add_tensor(f"blk.{i}.{'x' * 220}.weight", np.zeros((8, 8), dtype = np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    prefix = path.read_bytes()[: diffusion_compat._GGUF_HEADER_BYTES]
    assert gguf_flux2_inner_dim_from_header(prefix) is None, "fixture must overflow the cap"
    _stub_range_reads(monkeypatch, {})

    assert diffusion_compat.flux2_inner_dim_for_pick(str(local), "wide.gguf") == 8


# ── wiring: the plan reports, the load refuses, nothing is torn down ───────────


def test_the_download_plan_reports_the_mismatch_instead_of_raising(monkeypatch, tmp_path):
    # Reported, not raised: the images page falls back to /images/load on ANY plan failure, so a
    # 400 here would start the download the verdict is meant to prevent.
    _stub_range_reads(monkeypatch, {KLEIN_4B_FILE: _gguf_header(3072, tmp_path)})
    monkeypatch.setattr(
        "core.inference.diffusion._resolve_base_repo", lambda *a, **k: KLEIN_9B_BASE
    )
    monkeypatch.setattr(
        DiffusionBackend, "_te_prequant_plan_files", staticmethod(lambda *a, **k: {})
    )
    monkeypatch.setattr(
        DiffusionBackend, "_estimate_download_bytes", staticmethod(lambda *a, **k: (0, []))
    )
    monkeypatch.setattr(
        "core.inference.diffusion._assert_base_repo_accessible", lambda *a, **k: None
    )

    plan = DiffusionBackend().download_plan(KLEIN_4B_GGUF, gguf_filename = KLEIN_4B_FILE)

    assert KLEIN_9B_BASE in (plan["incompatible_reason"] or "")
    # And it survives the envelope the route returns, or the picker never sees it.
    from models.inference import DiffusionDownloadPlanResponse

    assert KLEIN_9B_BASE in (DiffusionDownloadPlanResponse(**plan).incompatible_reason or "")


def test_a_planner_that_omits_the_field_still_answers():
    # The native and video planners share this envelope and have no base pairing to check, so the
    # response model must default the field rather than 500 on its absence.
    from models.inference import DiffusionDownloadPlanResponse
    assert DiffusionDownloadPlanResponse(entries = [], total_bytes = 0).incompatible_reason is None


def test_a_compatible_plan_reports_nothing(monkeypatch, tmp_path):
    _stub_range_reads(monkeypatch, {KLEIN_4B_FILE: _gguf_header(3072, tmp_path)})
    monkeypatch.setattr(
        "core.inference.diffusion._resolve_base_repo", lambda *a, **k: KLEIN_4B_BASE
    )
    monkeypatch.setattr(
        DiffusionBackend, "_te_prequant_plan_files", staticmethod(lambda *a, **k: {})
    )
    monkeypatch.setattr(
        DiffusionBackend, "_estimate_download_bytes", staticmethod(lambda *a, **k: (0, []))
    )
    monkeypatch.setattr(
        "core.inference.diffusion._assert_base_repo_accessible", lambda *a, **k: None
    )

    plan = DiffusionBackend().download_plan(KLEIN_4B_GGUF, gguf_filename = KLEIN_4B_FILE)

    assert plan["incompatible_reason"] is None


def test_the_pre_eviction_preflight_refuses_the_mismatch(monkeypatch, tmp_path):
    # The route's last refusal before it takes the GPU from chat.
    _stub_range_reads(monkeypatch, {KLEIN_4B_FILE: _gguf_header(3072, tmp_path)})
    monkeypatch.setattr(
        "core.inference.diffusion._resolve_base_repo", lambda *a, **k: KLEIN_9B_BASE
    )
    monkeypatch.setattr(
        "core.inference.diffusion._assert_base_repo_accessible", lambda *a, **k: None
    )

    with pytest.raises(ValueError, match = "klein"):
        DiffusionBackend().preflight_base_access(
            KLEIN_4B_GGUF,
            FLUX2_FAMILY,
            gguf_filename = KLEIN_4B_FILE,
            model_kind = "gguf",
        )


def test_the_load_refuses_before_prefetching_or_unloading_anything(monkeypatch, tmp_path):
    # The regression this whole change is about: the old order downloaded the base AND freed the
    # resident pipeline, then raised. Nothing below may run.
    _stub_range_reads(monkeypatch, {KLEIN_4B_FILE: _gguf_header(3072, tmp_path)})
    monkeypatch.setattr(
        "core.inference.diffusion._resolve_base_repo", lambda *a, **k: KLEIN_9B_BASE
    )
    monkeypatch.setattr(
        "core.inference.diffusion._assert_base_repo_accessible", lambda *a, **k: None
    )
    monkeypatch.setattr(
        DiffusionBackend, "_te_prequant_plan_files", staticmethod(lambda *a, **k: {})
    )
    monkeypatch.setattr(
        DiffusionBackend, "_estimate_download_bytes", staticmethod(lambda *a, **k: (0, []))
    )
    monkeypatch.setattr(
        DiffusionBackend,
        "_prefetch_files",
        lambda self, *a, **k: pytest.fail("a rejected pick must not stage a single byte"),
    )
    monkeypatch.setattr(
        DiffusionBackend,
        "load_pipeline",
        lambda self, **k: pytest.fail("a rejected pick must not reach the load"),
    )
    backend = DiffusionBackend()
    monkeypatch.setattr(
        backend,
        "_unload_locked",
        lambda: pytest.fail("a rejected pick must not free the resident pipeline"),
    )
    # The model the user is already working with, and the marker begin_load would have set.
    resident = object()
    backend._state = resident
    backend._load_token = 7
    backend._loading = _LoadingState(repo_id = KLEIN_4B_GGUF, base_repo = "")

    backend._run_load(
        repo_id = KLEIN_4B_GGUF,
        gguf_filename = KLEIN_4B_FILE,
        model_kind = "gguf",
        _load_token = 7,
    )

    # _run_load swallows into load_progress rather than raising; the refusal lands there.
    progress = backend.load_progress()
    assert progress["phase"] == "error"
    assert KLEIN_9B_BASE in progress["error"]
    assert backend._state is resident


# ── the sibling hole: the native text-encoder pick ─────────────────────────────


def test_the_native_encoder_follows_the_header_over_the_filename():
    # A renamed or hand-picked 9B checkpoint says nothing in its id, and the 4B encoder fails deep
    # inside sd-cli. The header knows.
    fam = detect_family("unsloth/FLUX.2-klein-4B-GGUF")
    assert fam is not None and fam.name == "flux.2-klein"

    by_name = sd_cpp_text_encoders_for(fam, "unsloth/FLUX.2-klein-4B-GGUF", "my-checkpoint.gguf")
    by_header = sd_cpp_text_encoders_for(
        fam, "unsloth/FLUX.2-klein-4B-GGUF", "my-checkpoint.gguf", inner_dim = 4096
    )

    assert by_name == fam.sd_cpp_text_encoders  # today's answer: the 4B encoder
    assert by_header != by_name
    assert any("9B" in repo for repo, _f, _k in by_header)


def test_the_header_also_overrides_a_misleading_9b_filename():
    fam = detect_family("unsloth/FLUX.2-klein-9B-GGUF")
    assert fam is not None

    picked = sd_cpp_text_encoders_for(
        fam, "unsloth/FLUX.2-klein-9B-GGUF", "flux-2-klein-9b-Q4_K_M.gguf", inner_dim = 3072
    )

    assert picked == fam.sd_cpp_text_encoders  # the file really is 4B


def test_an_unmapped_or_absent_inner_dim_keeps_the_filename_rule():
    fam = detect_family("unsloth/FLUX.2-klein-9B-GGUF")
    assert fam is not None
    nine_b = sd_cpp_text_encoders_for(fam, "unsloth/FLUX.2-klein-9B-GGUF", "klein-9b.gguf")

    for inner_dim in (None, 1234):
        assert (
            sd_cpp_text_encoders_for(
                fam, "unsloth/FLUX.2-klein-9B-GGUF", "klein-9b.gguf", inner_dim = inner_dim
            )
            == nine_b
        )
    assert any("9B" in repo for repo, _f, _k in nine_b)


def test_a_stalled_response_header_cannot_hold_the_picker_open(monkeypatch):
    """requests' timeout is an INACTIVITY timeout, not a wall clock.

    A peer that trickles response headers a byte at a time resets it forever, so a deadline
    armed only once get() has returned never gets armed at all -- and this call sits on the
    /images/load route thread and on the download-plan path, both of which promise to fail
    open in seconds. The request itself has to be on the abandonable worker.
    """
    import threading
    import types

    released = threading.Event()

    def _never_returns(*_args, **_kwargs):
        # Still inside connect / the header wait: no response object exists to interrupt.
        released.wait(30)
        raise AssertionError("the request should have been abandoned, not awaited")

    monkeypatch.setattr(diffusion_compat, "_HEADER_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(
        "huggingface_hub.utils.get_session",
        lambda: types.SimpleNamespace(get = _never_returns),
    )
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)

    out: list[bytes] = []
    caller = threading.Thread(
        target = lambda: out.append(
            diffusion_compat._read_gguf_header(KLEIN_4B_GGUF, KLEIN_4B_FILE, None)
        ),
        daemon = True,
    )
    caller.start()
    caller.join(8)
    released.set()

    assert not caller.is_alive(), "the caller waited on a request that never returned headers"
    assert out == [b""]


def test_the_offline_caller_still_gets_a_memoised_remote_answer(monkeypatch, tmp_path):
    """begin_load probes with allow_network=False right after a plan-time probe answered.

    Returning None there anyway sent it to the filename heuristic, which publishes the 4B
    encoder repos for a renamed 9B checkpoint -- so the delete-cached guard did not cover the
    companion repo the load was about to use.
    """
    monkeypatch.setattr(diffusion_compat, "_INNER_DIM_CACHE", {})
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        diffusion_compat,
        "_read_gguf_header",
        lambda *a, **k: _gguf_header(4096, tmp_path),
    )

    online = diffusion_compat.flux2_inner_dim_for_pick(KLEIN_4B_GGUF, "renamed-4b.gguf")
    assert online == 4096

    # Same pick, same token, no network allowed: the memo answers.
    monkeypatch.setattr(
        diffusion_compat,
        "_read_gguf_header",
        lambda *a, **k: pytest.fail("the offline caller made a range request"),
    )
    assert (
        diffusion_compat.flux2_inner_dim_for_pick(
            KLEIN_4B_GGUF, "renamed-4b.gguf", allow_network = False
        )
        == 4096
    )


# ── a cached copy the Hub has moved past ──────────────────────────────────────


def _cached_snapshot(
    tmp_path,
    sha,
    body,
    filename = KLEIN_4B_FILE,
):
    """A file where huggingface_hub puts it: ``models--org--repo/snapshots/<sha>/<file>``."""
    path = tmp_path / f"models--{KLEIN_4B_GGUF.replace('/', '--')}" / "snapshots" / sha / filename
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(body)
    return path


def _stub_revision(monkeypatch, sha):
    def _meta(*_args, **_kwargs):
        if sha is None:
            raise OSError("offline")
        return types.SimpleNamespace(commit_hash = sha, etag = sha)

    monkeypatch.setattr("huggingface_hub.get_hf_file_metadata", _meta)


def test_a_republished_checkpoint_is_not_refused_from_the_stale_cache(monkeypatch, tmp_path):
    """try_to_load_from_cache resolves the LOCAL refs/main, so a 4B file republished at the same
    name as a 9B one would refuse a pick the loader's own hf_hub_download refreshes and loads."""
    requests = _stub_range_reads(monkeypatch, {KLEIN_4B_FILE: _gguf_header(4096, tmp_path)})
    cached = _cached_snapshot(tmp_path, "oldcommit", _gguf_header(3072, tmp_path))
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: str(cached))
    _stub_revision(monkeypatch, "newcommit")

    reason = diffusion_compat.flux2_pick_mismatch(
        FLUX2_FAMILY, KLEIN_4B_GGUF, KLEIN_4B_FILE, KLEIN_9B_BASE
    )

    assert reason is None, "a stale cached header refused a pick the loader would have loaded"
    assert requests, "the live header was never read"


def test_a_cached_checkpoint_at_the_current_revision_is_still_refused(monkeypatch, tmp_path):
    # The refusal this preflight exists for: the cache is current, so its header is the verdict
    # and nothing has to be fetched to say so.
    requests = _stub_range_reads(monkeypatch, {KLEIN_4B_FILE: _gguf_header(4096, tmp_path)})
    cached = _cached_snapshot(tmp_path, "oldcommit", _gguf_header(3072, tmp_path))
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: str(cached))
    _stub_revision(monkeypatch, "oldcommit")

    reason = diffusion_compat.flux2_pick_mismatch(
        FLUX2_FAMILY, KLEIN_4B_GGUF, KLEIN_4B_FILE, KLEIN_9B_BASE
    )

    assert reason is not None and "klein-9B" in reason
    assert requests == []


def test_a_revision_check_that_cannot_run_keeps_the_cached_refusal(monkeypatch, tmp_path):
    # Offline is the case the cached copy answers best: unable to ask, the preflight keeps the
    # verdict it has rather than fetching or going silent.
    requests = _stub_range_reads(monkeypatch, {KLEIN_4B_FILE: _gguf_header(4096, tmp_path)})
    cached = _cached_snapshot(tmp_path, "oldcommit", _gguf_header(3072, tmp_path))
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: str(cached))
    _stub_revision(monkeypatch, None)

    reason = diffusion_compat.flux2_pick_mismatch(
        FLUX2_FAMILY, KLEIN_4B_GGUF, KLEIN_4B_FILE, KLEIN_9B_BASE
    )

    assert reason is not None and "klein-9B" in reason
    assert requests == []


def test_an_on_device_checkpoint_is_never_revalidated(monkeypatch, tmp_path):
    # An On Device file IS the file the loader opens, so there is no Hub revision to be behind
    # and a mismatch must be refused without a single network call.
    local = tmp_path / "on-device"
    local.mkdir()
    (local / KLEIN_4B_FILE).write_bytes(_gguf_header(3072, tmp_path))
    requests = _stub_range_reads(monkeypatch, {})
    monkeypatch.setattr(
        diffusion_compat,
        "_hub_revision",
        lambda *a, **k: pytest.fail("an On Device pick asked the Hub for a revision"),
    )

    reason = diffusion_compat.flux2_pick_mismatch(
        FLUX2_FAMILY, str(local), KLEIN_4B_FILE, KLEIN_9B_BASE
    )

    assert reason is not None and "klein-9B" in reason
    assert requests == []


# ── Speech picks ───────────────────────────────────────────────────────────────
# The variant listing drops the speech quants whose bytes are on disk, but an UNDOWNLOADED one
# has none to read and stays offered -- and detect_family_for_pick answers from the folder name,
# so a csm file beside a FLUX denoiser resolves to flux.1 and reaches this loader as its own.

CSM_REPO = "someone/mixed-media-GGUF"
CSM_FILE = "csm-1b-Q4_0.gguf"
DENOISER_FILE = "flux1-dev-Q4_K_M.gguf"


def _arch_header(architecture: str) -> bytes:
    """A minimal GGUF prefix carrying just ``general.architecture``.

    Hand-rolled rather than via ``GGUFWriter`` like the size-pairing fixtures: this probe reads a
    single KV pair, and writing it by hand is what ``test_cached_gguf_routes.py`` does for the
    listing-side gate, so both ends of this feature are pinned against the same bytes."""
    import struct

    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    return (
        struct.pack("<IIQQ", 0x46554747, 3, 0, 1)
        + string("general.architecture")
        + struct.pack("<I", 8)
        + string(architecture)
    )


def test_an_undownloaded_speech_pick_is_refused_before_any_download(monkeypatch):
    """The whole point: the refusal lands off one range request, so the checkpoint is never
    pulled and the resident pipeline never torn down for a file that cannot decode."""
    requests = _stub_range_reads(monkeypatch, {CSM_FILE: _arch_header("llama-csm")})

    with pytest.raises(ValueError) as excinfo:
        diffusion_compat.assert_pick_is_not_speech(CSM_REPO, CSM_FILE)

    detail = str(excinfo.value)
    assert CSM_FILE in detail and "llama-csm" in detail
    assert len(requests) == 1
    url, byte_range = requests[0]
    assert url.endswith(f"{CSM_REPO}/resolve/main/{CSM_FILE}")
    # Bounded: the request must name an end offset, or a mis-set header streams the checkpoint.
    assert byte_range == f"bytes=0-{diffusion_compat._GGUF_HEADER_BYTES - 1}"


def test_a_runnable_media_pick_in_the_same_repo_still_loads(monkeypatch):
    """The sibling this refusal exists to protect: catching it too would hide the only checkpoint
    in the folder the media backends CAN run."""
    _stub_range_reads(monkeypatch, {DENOISER_FILE: _arch_header("flux")})

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, DENOISER_FILE) is None
    diffusion_compat.assert_pick_is_not_speech(CSM_REPO, DENOISER_FILE)


def test_an_unreadable_speech_header_fails_open(monkeypatch):
    """Fail-open throughout: refusing a pick that works is worse than the download this saves."""
    _stub_range_reads(monkeypatch, {}, status = 200)

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is None
    diffusion_compat.assert_pick_is_not_speech(CSM_REPO, CSM_FILE)


def test_a_pick_with_no_gguf_filename_is_not_probed(monkeypatch):
    """A pipeline or single_file pick names no GGUF, so there is no header to ask."""
    requests = _stub_range_reads(monkeypatch, {CSM_FILE: _arch_header("llama-csm")})

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, None) is None
    assert requests == []


def test_the_speech_verdict_is_memoised_per_pick(monkeypatch):
    """Expanding and re-picking a row must not re-spend the range request."""
    requests = _stub_range_reads(monkeypatch, {CSM_FILE: _arch_header("llama-csm")})

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None
    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None
    assert len(requests) == 1


def test_a_failed_probe_is_not_reused_for_a_retry_with_a_working_token(monkeypatch):
    """A probe that failed on an expired credential memoises "no verdict". Keyed without the
    token, the retry with a working one reads that back and the speech file reaches the download
    this preflight exists to stop."""
    bodies = {CSM_FILE: _arch_header("llama-csm")}
    requests: list = []

    class _Session:
        def get(
            self,
            url,
            headers = None,
            timeout = None,
            stream = False,
        ):
            header_map = headers or {}
            requests.append(header_map.get("authorization") or header_map.get("Authorization"))
            # The expired credential is refused; the working one is served.
            if not any("good" in str(v) for v in header_map.values()):
                return _FakeResponse(401)
            body = next((b for name, b in bodies.items() if url.endswith(name)), None)
            return _FakeResponse(206, body) if body is not None else _FakeResponse(404)

    monkeypatch.setattr("huggingface_hub.utils.get_session", lambda: _Session())
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(diffusion_compat, "_local_gguf_path", lambda *a, **k: None)

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE, "expired") is None
    # Same pick, different credential: it must probe again rather than answer from that miss.
    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE, "good-token") is not None
    assert len(requests) == 2


def test_a_checkpoint_that_lands_after_a_miss_is_probed_again(monkeypatch, tmp_path):
    """A file replaced under the same name is a different checkpoint; keyed without its identity
    the verdict never revisits it."""
    _stub_range_reads(monkeypatch, {})
    landed = tmp_path / CSM_FILE
    seen: dict = {"path": None}
    monkeypatch.setattr(diffusion_compat, "_local_gguf_path", lambda *a, **k: seen["path"])

    # Nothing on disk and nothing served: no opinion, and that miss is memoised.
    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is None

    landed.write_bytes(_arch_header("llama-csm"))
    seen["path"] = str(landed)

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None


def _cache_entry(tmp_path, revision, arch):
    """A GGUF sitting in an HF cache snapshot dir, so ``_snapshot_revision`` can read its commit."""
    snapshot = tmp_path / "models--someone--mixed-media-GGUF" / "snapshots" / revision
    snapshot.mkdir(parents = True, exist_ok = True)
    path = snapshot / CSM_FILE
    path.write_bytes(_arch_header(arch))
    return str(path)


def test_a_republished_gguf_is_not_refused_from_the_stale_cached_copy(monkeypatch, tmp_path):
    """``try_to_load_from_cache`` resolves the LOCAL refs/main, so a republished filename would
    be refused off the csm bytes still on disk while ``hf_hub_download`` refreshes and loads the
    new ones."""
    cached = _cache_entry(tmp_path, "oldsha", "llama-csm")
    monkeypatch.setattr(diffusion_compat, "_local_gguf_path", lambda *a, **k: cached)
    monkeypatch.setattr(diffusion_compat, "_hub_revision", lambda *a, **k: "newsha")
    # The Hub now serves a denoiser at the same name.
    _stub_range_reads(monkeypatch, {CSM_FILE: _arch_header("flux")})

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is None


def test_a_cached_gguf_at_the_current_revision_is_still_refused(monkeypatch, tmp_path):
    """The revalidation must not become an escape hatch: same commit, same verdict."""
    cached = _cache_entry(tmp_path, "samesha", "llama-csm")
    monkeypatch.setattr(diffusion_compat, "_local_gguf_path", lambda *a, **k: cached)
    monkeypatch.setattr(diffusion_compat, "_hub_revision", lambda *a, **k: "samesha")
    _stub_range_reads(monkeypatch, {})

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None


def test_a_revision_check_that_cannot_run_keeps_the_refusal(monkeypatch, tmp_path):
    """An offline or erroring host leaves today's verdict alone rather than opening the gate."""
    cached = _cache_entry(tmp_path, "oldsha", "llama-csm")
    monkeypatch.setattr(diffusion_compat, "_local_gguf_path", lambda *a, **k: cached)
    monkeypatch.setattr(diffusion_compat, "_hub_revision", lambda *a, **k: None)
    _stub_range_reads(monkeypatch, {})

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None


def test_an_on_device_speech_checkpoint_is_never_revalidated(monkeypatch, tmp_path):
    """An On Device file is the one the loader opens, so there is no revision to be behind."""
    on_device = tmp_path / CSM_FILE
    on_device.write_bytes(_arch_header("llama-csm"))
    monkeypatch.setattr(diffusion_compat, "_local_gguf_path", lambda *a, **k: str(on_device))
    asked: list = []

    def _never(*a, **k):
        asked.append(a)
        return "newsha"

    monkeypatch.setattr(diffusion_compat, "_hub_revision", _never)
    _stub_range_reads(monkeypatch, {})

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None
    assert asked == []


def test_a_pick_that_names_the_checkpoint_outright_is_probed_like_the_loader_opens_it(tmp_path):
    """A file-valued ``repo_id`` is a pick the video loader accepts, so the preflight must read it.

    ``VideoBackend._resolve_checkpoint_path`` answers ``if root.is_file(): return root`` and
    ignores ``gguf_filename``, and ``VideoBackend.validate_load_request`` admits exactly that
    pick, so ``POST /video/load`` with ``model_path`` naming a .gguf reaches the loader today.

    Resolving it as a repo ROOT instead appends the filename under the file, raising
    ``FileNotFoundError`` -- an ``OSError``, hence swallowed as "remote id". The pick then has no
    local path: an offline preflight allows it outright, an online one range-requests a
    filesystem path off the Hub and fails open. Either way the csm checkpoint survives the gate,
    the resident pipeline is evicted, and the file reaches the media loader."""
    direct = tmp_path / "wan-models" / CSM_FILE
    direct.parent.mkdir(parents = True)
    direct.write_bytes(_arch_header("llama-csm"))

    # Offline, so a probe that found no local path has nothing left to fall back on and must let
    # the pick through: this asserts the file was actually read, not that the network saved us.
    reason = diffusion_compat.speech_pick_refusal(str(direct), CSM_FILE, None, False)
    assert reason is not None and "llama-csm" in reason

    # And it resolves to the very file the loader would open, rather than merely to something.
    assert diffusion_compat._local_gguf_path(str(direct), CSM_FILE) == str(direct)

    # The runnable sibling named the same way still loads: this must not refuse every direct file.
    denoiser = direct.parent / DENOISER_FILE
    denoiser.write_bytes(_arch_header("flux"))
    assert diffusion_compat.speech_pick_refusal(str(denoiser), DENOISER_FILE, None, False) is None

    # The folder-valued pick keeps resolving through the containment check, unchanged.
    assert diffusion_compat._local_gguf_path(str(direct.parent), CSM_FILE) == str(direct)


# ── Every engine and both stages ───────────────────────────────────────────────
# The Images and Video pages stage and download BEFORE they call load, so a load-only gate
# arrives after the bytes. And on all three engines: a mixed VIDEO repo goes through
# VideoBackend and a CPU/MPS or sd.cpp-forced image pick through SdCppDiffusionBackend, neither
# of which shares DiffusionBackend's preflight.


def test_the_shared_refusal_is_wired_into_every_plan_and_load_path():
    """Structural, deliberately: the tests above pin the verdict, but what keeps regressing is
    where it is CALLED, and exercising each engine's plan/load needs a live backend and a Hub."""
    import inspect

    from core.inference import diffusion, sd_cpp_backend, video

    # Images: folded into incompatible_reason, which the page renders instead of staging entries.
    plan = inspect.getsource(diffusion.DiffusionBackend.download_plan)
    assert "speech_pick_refusal" in plan

    # Video: plan and worker both, since a direct begin_load reaches no plan.
    assert "_assert_pick_is_not_speech" in inspect.getsource(video.VideoBackend.download_plan)
    assert "_assert_pick_is_not_speech" in inspect.getsource(video.VideoBackend._run_load)

    # sd.cpp: plan and worker. NOT begin_load, which is offline-only and cannot afford the bound.
    sd_plan = inspect.getsource(sd_cpp_backend.SdCppDiffusionBackend.download_plan)
    assert "_assert_pick_is_not_speech" in sd_plan
    assert "_assert_pick_is_not_speech" not in inspect.getsource(
        sd_cpp_backend.SdCppDiffusionBackend.begin_load
    )


def test_the_video_and_sd_cpp_helpers_delegate_to_the_one_verdict(monkeypatch):
    """Three engines, one refusal. A second copy of the speech arch set is how they drift."""
    from core.inference import sd_cpp_backend, video

    seen: list = []
    monkeypatch.setattr(
        diffusion_compat,
        "assert_pick_is_not_speech",
        lambda *a, **k: seen.append(a),
    )

    video._assert_pick_is_not_speech(CSM_REPO, CSM_FILE, "tok", allow_network = False)
    sd_cpp_backend._assert_pick_is_not_speech(CSM_REPO, CSM_FILE, "tok", allow_network = False)

    # The cache-only flag rides through: an offline promise must not drop at the delegation edge.
    assert seen == [(CSM_REPO, CSM_FILE, "tok", False), (CSM_REPO, CSM_FILE, "tok", False)]


def test_a_media_gguf_republished_as_speech_is_refused(monkeypatch, tmp_path):
    """The direction that matters more: a stale refusal only costs a pick, but a stale ALLOW
    hands csm bytes to a media loader after the download and the teardown. So the revalidation
    cannot be the cheap side of an asymmetry the way the size pairing's is."""
    snapshot = tmp_path / "models--someone--mixed-media-GGUF" / "snapshots" / "oldsha"
    snapshot.mkdir(parents = True)
    cached = snapshot / CSM_FILE
    # On disk this is still the runnable denoiser the row was offering.
    cached.write_bytes(_arch_header("flux"))

    monkeypatch.setattr(diffusion_compat, "_local_gguf_path", lambda *a, **k: str(cached))
    monkeypatch.setattr(diffusion_compat, "_hub_revision", lambda *a, **k: "newsha")
    # The Hub has since replaced it with a speech checkpoint at the same filename.
    _stub_range_reads(monkeypatch, {CSM_FILE: _arch_header("llama-csm")})

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None


def test_an_uncached_pick_never_spends_a_revision_head(monkeypatch):
    """No cached copy means no revision to be behind, so the symmetric revalidation must not cost
    every remote pick an extra Hub round trip."""
    monkeypatch.setattr(diffusion_compat, "_local_gguf_path", lambda *a, **k: None)
    asked: list = []
    monkeypatch.setattr(
        diffusion_compat, "_hub_revision", lambda *a, **k: asked.append(a) or "newsha"
    )
    _stub_range_reads(monkeypatch, {CSM_FILE: _arch_header("llama-csm")})

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None
    assert asked == []


def test_a_cache_only_load_never_range_reads_an_uncached_pick(monkeypatch):
    """An automatic load sets local_files_only, and that contract covers the metadata probes too:
    an uncached pick gives up rather than spend the range request and its 15s bound."""
    requests = _stub_range_reads(monkeypatch, {CSM_FILE: _arch_header("llama-csm")})
    monkeypatch.setattr(diffusion_compat, "_local_gguf_path", lambda *a, **k: None)

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE, allow_network = False) is None
    assert requests == []

    # And nothing was memoised, so the next caller that CAN wait still gets a real answer.
    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None
    assert len(requests) == 1


def test_a_cache_only_load_still_reads_a_checkpoint_already_on_disk(monkeypatch, tmp_path):
    """Offline does not mean blind: a copy already on disk answers with no request at all."""
    on_device = tmp_path / CSM_FILE
    on_device.write_bytes(_arch_header("llama-csm"))
    requests = _stub_range_reads(monkeypatch, {})
    monkeypatch.setattr(diffusion_compat, "_local_gguf_path", lambda *a, **k: str(on_device))

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE, allow_network = False) is not None
    assert requests == []


def test_a_cache_only_probe_does_not_memoise_a_skipped_revision_check(monkeypatch, tmp_path):
    """A cached copy read WITHOUT its revision check is half an answer; memoising it would let
    the network-allowed caller behind it read that back and never revalidate."""
    snapshot = tmp_path / "models--someone--mixed-media-GGUF" / "snapshots" / "oldsha"
    snapshot.mkdir(parents = True)
    cached = snapshot / CSM_FILE
    cached.write_bytes(_arch_header("flux"))
    monkeypatch.setattr(diffusion_compat, "_local_gguf_path", lambda *a, **k: str(cached))

    heads: list = []
    monkeypatch.setattr(
        diffusion_compat, "_hub_revision", lambda *a, **k: heads.append(a) or "newsha"
    )
    _stub_range_reads(monkeypatch, {CSM_FILE: _arch_header("llama-csm")})

    # Offline: reads the stale local bytes, asks no revision, allows the pick.
    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE, allow_network = False) is None
    assert heads == []

    # The next caller that can reach the Hub must still catch the republish.
    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None
    assert len(heads) == 1


def test_an_uncached_remote_verdict_is_not_memoised_forever(monkeypatch):
    """An uncached remote pick keys on a local identity of None, so a republish under the same
    filename changes nothing about the key. Without a TTL the first verdict would outlive the
    process and keep refusing a checkpoint that is runnable now."""
    monkeypatch.setattr(diffusion_compat, "_local_gguf_path", lambda *a, **k: None)
    bodies = {CSM_FILE: _arch_header("llama-csm")}
    _stub_range_reads(monkeypatch, bodies)
    # Driven, not real: the entry expires relative to whatever monotonic said when it was written.
    clock = [1000.0]
    monkeypatch.setattr(diffusion_compat.time, "monotonic", lambda: clock[0])

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None
    # Inside the window: the memo answers and the Hub is not asked again.
    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None

    # The repo republishes it as a runnable denoiser and the window lapses.
    bodies[CSM_FILE] = _arch_header("flux")
    clock[0] += diffusion_compat._SPEECH_REMOTE_TTL_SECONDS + 1.0

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is None


def test_a_snapshot_backed_verdict_keeps_its_entry_across_the_window(monkeypatch, tmp_path):
    """A cached entry needs no TTL: its key carries the file identity and the revision check asks
    the Hub. Expiring it too would spend a range read per pick for nothing."""
    snapshot = tmp_path / "models--someone--mixed-media-GGUF" / "snapshots" / "samesha"
    snapshot.mkdir(parents = True)
    cached = snapshot / CSM_FILE
    cached.write_bytes(_arch_header("llama-csm"))
    monkeypatch.setattr(diffusion_compat, "_local_gguf_path", lambda *a, **k: str(cached))
    monkeypatch.setattr(diffusion_compat, "_hub_revision", lambda *a, **k: "samesha")
    requests = _stub_range_reads(monkeypatch, {})

    clock = [1000.0]
    monkeypatch.setattr(diffusion_compat.time, "monotonic", lambda: clock[0])

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None
    clock[0] += diffusion_compat._SPEECH_REMOTE_TTL_SECONDS * 10
    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None
    assert requests == []


def test_both_media_routes_refuse_a_speech_pick_before_taking_the_gpu():
    """The backends assert this too, but on the load worker, INSIDE acquire_for, so a refusal
    there arrives having already evicted the chat model the gate exists to preserve -- and on the
    image route after an engine switch unloaded the resident pipeline. Both routes must refuse
    before they reach the arbiter."""
    import inspect

    from routes import inference as inference_route
    from routes import video as video_route

    # The CALL, not the import line at the top of each route, which names acquire_for far earlier.
    for source, acquire, label in (
        (inspect.getsource(video_route.load_video_model_gated), "acquire_for(VIDEO", "video"),
        (
            inspect.getsource(inference_route.load_diffusion_model_gated),
            "acquire_for(DIFFUSION",
            "images",
        ),
    ):
        assert "assert_pick_is_not_speech" in source, label
        assert acquire in source, label
        assert source.index("assert_pick_is_not_speech") < source.index(acquire), label


def test_an_automatic_image_load_keeps_the_pre_eviction_preflight_offline():
    """The route's own speech check already honours user_initiated, but it then calls
    preflight_base_access, which reaches the same assertion again. Left network-enabled that
    second call spends a revision HEAD (or an uncached range request and its 15s bound) on the
    one path that promised to stay off the Hub."""
    import inspect

    from core.inference import diffusion, sd_cpp_backend
    from routes import inference as inference_route

    preflight = inspect.getsource(diffusion.DiffusionBackend.preflight_base_access)
    assert "assert_pick_is_not_speech(repo_id, gguf_filename, hf_token, allow_network)" in preflight

    # The route hands its own locality flag down rather than letting the default win.
    route = inspect.getsource(inference_route.load_diffusion_model_gated)
    assert "allow_network = user_initiated" in route

    # Both engines keep one signature, since the route preflights whichever one it picked.
    for backend in (diffusion.DiffusionBackend, sd_cpp_backend.SdCppDiffusionBackend):
        assert "allow_network" in inspect.signature(backend.preflight_base_access).parameters


# The Mimi vocoder in ggml-org/sesame-csm-1b-GGUF writes a SENTENCE where the architecture
# identifier belongs. Read off the live repo, not invented.
_VOCODER_ARCH = "this model cannot be used as LLM, use it via --model-vocoder in TTS examples"


# ── the HTTP client huggingface_hub actually hands us ──────────────────────────


class _HttpxLikeClient:
    """An httpx.Client's surface, which is what ``get_session`` returns on huggingface_hub 1.x.

    Deliberately NOT a mock of the requests API: `get` rejects `stream`, there is no
    `iter_content`, and there is no `raw`. requirements/studio.txt floors 1.23 on python >= 3.10,
    so this is the client every supported install has."""

    def __init__(self, body):
        self.body = body
        self.calls: list[tuple[str, str]] = []

    def get(
        self,
        url,
        params = None,
        headers = None,
        timeout = None,
        **kwargs,
    ):
        if "stream" in kwargs:
            raise TypeError("Client.get() got an unexpected keyword argument 'stream'")
        raise AssertionError("a ranged header read must use the streaming API")

    def stream(
        self,
        method,
        url,
        headers = None,
        timeout = None,
        follow_redirects = False,
    ):
        self.calls.append((method, (headers or {}).get("Range", "")))
        assert follow_redirects, "the Hub answers a resolve URL with a 302 to the CDN"
        return _HttpxLikeStream(self.body)


class _HttpxLikeStream:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def status_code(self):
        return 206

    def iter_bytes(self, chunk_size = 65536):
        for start in range(0, len(self.body), chunk_size):
            yield self.body[start : start + chunk_size]

    def close(self):
        pass


def test_the_ranged_read_works_on_the_httpx_client_hub_1_x_returns(monkeypatch):
    """huggingface_hub 1.0 swapped requests for httpx, and httpx has no ``stream = True``
    keyword. Asking for one raised TypeError inside the worker's blanket except, so the probe
    read nothing and EVERY uncached remote pick failed open: a preflight that refused nothing."""
    client = _HttpxLikeClient(_arch_header("llama-csm"))
    monkeypatch.setattr("huggingface_hub.utils.get_session", lambda: client)
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    diffusion_compat._reset_inner_dim_cache()

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None
    # Still one bounded ranged request, exactly as on the requests client.
    assert client.calls == [("GET", f"bytes=0-{diffusion_compat._GGUF_HEADER_BYTES - 1}")]


def test_the_ranged_read_still_works_on_the_requests_session_hub_0_x_returns(monkeypatch):
    """The other half of the floor: python < 3.10 pins 0.36, whose session is requests'."""
    requests_log = _stub_range_reads(monkeypatch, {CSM_FILE: _arch_header("llama-csm")})

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None
    assert len(requests_log) == 1


def test_a_requests_session_is_not_mistaken_for_an_httpx_client(monkeypatch):
    """``requests.Session.stream`` is a plain bool attribute and ``httpx.Client.stream`` is a
    method, so the branch has to test for a CALLABLE. Testing for the name alone sends a requests
    session down the httpx path and breaks the client that works today."""
    import requests

    assert hasattr(requests.Session(), "stream")
    assert not callable(getattr(requests.Session(), "stream", None))


# ── verdict freshness ──────────────────────────────────────────────────────────


def test_a_cached_snapshot_verdict_does_not_outlive_its_revision_check(monkeypatch, tmp_path):
    """A snapshot-backed entry memoises a revision check. Holding it forever meant the Hub was
    asked once per file per process, so a checkpoint republished later in the session kept the
    verdict read off bytes that had since been replaced."""
    cached = _cache_entry(tmp_path, "oldsha", "llama-csm")
    monkeypatch.setattr(diffusion_compat, "_local_gguf_path", lambda *a, **k: cached)
    heads: list = []

    def _head(*a, **k):
        heads.append(a)
        return "oldsha"

    monkeypatch.setattr(diffusion_compat, "_hub_revision", _head)
    _stub_range_reads(monkeypatch, {})

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None
    assert len(heads) == 1
    # Inside the window the memo answers, as before.
    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None
    assert len(heads) == 1
    # Past it, the Hub is asked again.
    monkeypatch.setattr(
        diffusion_compat.time,
        "monotonic",
        lambda: 1e9 + diffusion_compat._SPEECH_REMOTE_TTL_SECONDS,
    )
    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None
    assert len(heads) == 2


def test_an_on_device_verdict_is_still_cached_permanently(monkeypatch, tmp_path):
    """The other half: an On Device file has no revision to be behind, so its entry is keyed on
    the file's identity and never needs to age out."""
    on_device = tmp_path / CSM_FILE
    on_device.write_bytes(_arch_header("llama-csm"))
    monkeypatch.setattr(diffusion_compat, "_local_gguf_path", lambda *a, **k: str(on_device))
    _stub_range_reads(monkeypatch, {})

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None
    key = next(iter(diffusion_compat._SPEECH_ARCH_CACHE))
    assert diffusion_compat._SPEECH_ARCH_CACHE[key][1] is None


def test_a_failed_refresh_keeps_the_verdict_it_already_had(monkeypatch, tmp_path):
    """A HEAD that reports a new revision and a re-read that then fails used to replace a known
    llama-csm verdict with None, opening the gate on nothing worse than a dropped connection.
    Failing open on an UNKNOWN pick is the contract; discarding a known one is not."""
    cached = _cache_entry(tmp_path, "oldsha", "llama-csm")
    monkeypatch.setattr(diffusion_compat, "_local_gguf_path", lambda *a, **k: cached)
    monkeypatch.setattr(diffusion_compat, "_hub_revision", lambda *a, **k: "newsha")
    # The re-read of the new revision returns nothing at all.
    _stub_range_reads(monkeypatch, {})

    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None

    # A re-read that SUCCEEDS still replaces it, in both directions.
    diffusion_compat._reset_inner_dim_cache()
    _stub_range_reads(monkeypatch, {CSM_FILE: _arch_header("flux")})
    monkeypatch.setattr(diffusion_compat, "_local_gguf_path", lambda *a, **k: cached)
    monkeypatch.setattr(diffusion_compat, "_hub_revision", lambda *a, **k: "newsha")
    assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is None


def test_every_published_csm_spelling_is_refused_by_the_media_preflight(monkeypatch):
    """The media half of the same set the chat gate uses: a bundle's vocoder reaches a media
    loader exactly like its backbone does."""
    for arch in ("llama-csm", "csm", "csm-tts", "mimi"):
        _stub_range_reads(monkeypatch, {CSM_FILE: _arch_header(arch)})
        assert diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE) is not None, arch
    # The vocoder's sentence is quoted back to nobody, but it still refuses.
    _stub_range_reads(monkeypatch, {CSM_FILE: _arch_header(_VOCODER_ARCH)})
    reason = diffusion_compat.speech_pick_refusal(CSM_REPO, CSM_FILE)
    assert reason is not None and "--model-vocoder" not in reason


def test_one_pick_range_reads_the_header_once_for_both_probes(monkeypatch, tmp_path):
    """The size pairing and the speech verdict read the SAME prefix of the SAME file.

    `/images/download-plan` runs them back to back on every hub pick, and `or` only skips the
    second when the first refuses -- so the ordinary case, a flux.2 GGUF that pairs correctly,
    made two range requests. Each carries its own _HEADER_TIMEOUT_SECONDS, so a picker the user
    is sitting in front of could wear twice the bound this module documents. One read now.
    """
    requests = _stub_range_reads(monkeypatch, {KLEIN_4B_FILE: _gguf_header(3072, tmp_path)})

    assert (
        diffusion_compat.flux2_pick_mismatch(
            FLUX2_FAMILY, KLEIN_4B_GGUF, KLEIN_4B_FILE, KLEIN_4B_BASE
        )
        is None
    ), "the fixture pairs correctly, so the speech probe behind it really does run"
    assert diffusion_compat.speech_pick_refusal(KLEIN_4B_GGUF, KLEIN_4B_FILE) is None

    assert len(requests) == 1, f"one pick, one header read; got {len(requests)}"


def test_a_revalidation_still_re_reads_rather_than_answering_from_the_shared_prefix(
    monkeypatch, tmp_path
):
    """The shared read must not reach the paths whose whole job is to re-read.

    A checkpoint republished at the same filename is caught by re-reading its header off the
    Hub. Serving that from the prefix memo would hand the revalidation the very bytes it is
    trying to get past, and a media GGUF republished as speech would load.
    """
    source = inspect.getsource(diffusion_compat)
    for fn in ("_revalidated_inner_dim", "_revalidated_speech_arch"):
        body = source.split(f"def {fn}(", 1)[1].split("\ndef ", 1)[0]
        assert "_shared_gguf_header" not in body, f"{fn} must re-read, not consult the memo"
        assert "_read_gguf_header" in body


def test_the_chat_backend_does_not_import_pyyaml_to_learn_the_speech_verdict():
    """`core.inference.llama_cpp` must not drag the models package in at import time.

    The speech verdict is shared, and reaching it through `utils.models.gguf_metadata` runs
    `utils.models.__init__`, which imports `model_config`, which imports `yaml`. That made
    PyYAML a hard import dependency of the chat backend and took the repo's own Source lint
    job red, where `tests/studio/load_freeze/test_load_orchestrator.py` imports the backend
    without PyYAML installed. The constants live in the leaf module `utils.gguf_archs`, and
    every caller imports them from there.
    """
    import importlib
    import subprocess
    import sys
    from pathlib import Path

    backend = Path(diffusion_compat.__file__).resolve().parents[2]
    probe = (
        "import sys\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'yaml' or name.startswith('yaml.'):\n"
        "            raise ModuleNotFoundError(\"No module named 'yaml'\")\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        f"sys.path.insert(0, {str(backend)!r})\n"
        "import core.inference.llama_cpp as m\n"
        "from utils.gguf_archs import SPEECH_GGUF_ARCHS\n"
        # Identity, not equality: one definition, re-exported, never copied.
        "assert m.LlamaCppBackend._SPEECH_ARCHES is SPEECH_GGUF_ARCHS\n"
        "print('ok')\n"
    )
    # A subprocess because this pytest session has already imported yaml, and a meta_path
    # blocker cannot un-import it.
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output = True, text = True, timeout = 300
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    # And nothing re-exports them from inside the heavy package: that would let an importer
    # reach them by the very path this test exists to keep out of the chain, and
    # scripts/verify_import_hoist.py blocks one outside a package __init__ anyway.
    gguf_metadata = importlib.import_module("utils.models.gguf_metadata")
    for name in ("SPEECH_GGUF_ARCHS", "is_speech_gguf_architecture"):
        assert not hasattr(
            gguf_metadata, name
        ), f"{name} must be imported from utils.gguf_archs, not through utils.models"
