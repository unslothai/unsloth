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

    assert diffusion_compat.flux2_inner_dim_for_pick(str(local), KLEIN_4B_FILE) == 4096, (
        "the memo answered for the file that used to be at this path"
    )


def test_a_bad_token_does_not_poison_the_good_one_that_replaces_it(monkeypatch, tmp_path):
    """Keying on the token's mere PRESENCE made every non-empty credential one key. A first probe
    with an expired token cached its miss there, and pasting a working token afterwards inherited
    it for the rest of the process -- the preflight silently off for that pick."""
    served: dict[str, bytes] = {}
    requests: list[tuple[str, str]] = []

    class _GatedSession:
        def get(self, url, headers = None, timeout = None, stream = False):
            token = (headers or {}).get("authorization", "")
            requests.append((url, token))
            if "good" not in token:
                return _FakeResponse(401)
            return _FakeResponse(206, served[KLEIN_4B_FILE])

    served[KLEIN_4B_FILE] = _gguf_header(4096, tmp_path)
    monkeypatch.setattr("huggingface_hub.utils.get_session", lambda: _GatedSession())
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    diffusion_compat._reset_inner_dim_cache()

    assert diffusion_compat.flux2_inner_dim_for_pick(KLEIN_4B_GGUF, KLEIN_4B_FILE, "expired") is None
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
