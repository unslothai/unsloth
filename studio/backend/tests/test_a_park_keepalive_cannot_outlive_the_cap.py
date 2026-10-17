# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A park before the first token is waited out on its keepalives, but only up to the park cap."""

import asyncio
import inspect
import itertools
import time

import httpx
import pytest

import core.inference.llama_cpp as llama_mod
from core.inference.llama_cpp import LlamaCppBackend


class _Stream:
    def __init__(self, chunks):
        self._chunks = chunks

    def iter_text(self):
        for chunk in self._chunks:
            time.sleep(0.01)
            yield chunk

    def close(self):
        pass


@pytest.fixture(autouse = True)
def _short_windows(monkeypatch):
    monkeypatch.setattr(llama_mod, "_SERVER_PARK_STALL_CAP_S", 0.3)
    monkeypatch.setattr(llama_mod, "_DEFAULT_FIRST_TOKEN_TIMEOUT_S", 30.0)
    monkeypatch.setattr(
        LlamaCppBackend, "_set_stream_read_timeout", staticmethod(lambda *a, **k: None)
    )


def _read(chunks):
    return LlamaCppBackend._iter_text_cancellable(
        _Stream(chunks), first_token_deadline = time.monotonic() + 30.0
    )


def test_a_park_that_never_ends_still_times_out():
    forever = itertools.chain(
        [": preempted 1\n\n"], itertools.repeat(": preempt-keepalive\n\n", 300)
    )
    started = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        for _ in _read(forever):
            pass
    assert time.monotonic() - started < 5, "every keepalive pushed the deadline out again"


def test_a_park_that_resumes_inside_the_cap_still_answers(monkeypatch):
    # A cap well above the park itself, so a slow runner's sleeps cannot outlast it.
    monkeypatch.setattr(llama_mod, "_SERVER_PARK_STALL_CAP_S", 5.0)
    data = 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
    chunks = [": preempted 1\n\n"] + [": preempt-keepalive\n\n"] * 5 + [": resumed 1\n\n", data]
    assert "".join(_read(chunks)).endswith(data)


class TestARawRelayParkedAfterItsFirstToken:
    """The async relay sees each keepalive as an item; only the park cap may renew on them."""

    @staticmethod
    async def _relay(items, monkeypatch, cap):
        import routes.inference as inference

        monkeypatch.setattr(inference, "_RAW_PARK_STALL_CAP_S", cap)

        async def upstream():
            for item in items:
                await asyncio.sleep(0.01)
                yield item

        seen = []
        async for item in inference._aiter_llama_stream_items(
            upstream(),
            first_token_deadline = time.monotonic() + 30.0,
            post_first_item_read_timeout_s = 0.1,
        ):
            seen.append(item)
        return seen

    def test_a_wedged_park_times_out(self, monkeypatch):
        items = ['data: {"choices":[{"delta":{"content":"a"}}]}', ": preempted"]
        items += [": preempt-keepalive"] * 300 + ["never sent"]

        with pytest.raises(httpx.ReadTimeout):
            asyncio.run(self._relay(items, monkeypatch, cap = 0.3))

    def test_a_park_inside_the_cap_still_answers(self, monkeypatch):
        items = ['data: {"choices":[{"delta":{"content":"a"}}]}', ": preempted"]
        items += [": preempt-keepalive"] * 5 + [
            ": resumed",
            'data: {"choices":[{"delta":{"content":"b"}}]}',
        ]
        seen = asyncio.run(self._relay(items, monkeypatch, cap = 5.0))
        assert seen[-1].endswith('"b"}}]}')

    def test_the_cap_holds_with_the_park_aware_read_installed(self, monkeypatch):
        import routes.inference as inference

        monkeypatch.setattr(inference, "_RAW_PARK_STALL_CAP_S", 1.0)
        monkeypatch.setattr(inference, "_install_park_aware_read", lambda response, grace: True)
        monkeypatch.setattr(inference, "_set_stream_response_read_timeout", lambda *a, **k: None)
        items = ['data: {"choices":[{"delta":{"content":"a"}}]}', ": preempted"]
        items += [": preempt-keepalive"] * 400

        async def relay():
            async def upstream():
                for item in items:
                    await asyncio.sleep(0.01)
                    yield item

            async for _ in inference._aiter_llama_stream_items(
                upstream(),
                first_token_deadline = time.monotonic() + 30.0,
                post_first_item_read_timeout_s = 0.1,
                response = object(),
                stall_grace = lambda: True,
            ):
                pass

        started = time.monotonic()
        with pytest.raises(httpx.ReadTimeout):
            asyncio.run(relay())
        assert time.monotonic() - started < 1.7, "the relay stayed open a second park cap"


class TestARawRelayParkedBeforeItsFirstToken:
    """Keepalives before any token renew the first-token window, never past the park cap."""

    @staticmethod
    async def _relay(
        items,
        monkeypatch,
        cap,
        window,
        step = 0.01,
    ):
        import routes.inference as inference

        monkeypatch.setattr(inference, "_RAW_PARK_STALL_CAP_S", cap)
        monkeypatch.setattr(inference, "_DEFAULT_FIRST_TOKEN_TIMEOUT_S", window)
        monkeypatch.setattr(inference, "configured_first_token_timeout_s", lambda: window)

        async def upstream():
            for item in items:
                await asyncio.sleep(step)
                yield item

        seen = []
        async for item in inference._aiter_llama_stream_items(
            upstream(), first_token_deadline = time.monotonic() + window
        ):
            seen.append(item)
        return seen

    def test_the_cap_is_counted_from_the_park_not_from_the_last_keepalive(self, monkeypatch):
        items = [": preempted"] + [": preempt-keepalive"] * 400
        started = time.monotonic()
        with pytest.raises(httpx.ReadTimeout):
            asyncio.run(self._relay(items, monkeypatch, cap = 1.0, window = 0.8))
        assert time.monotonic() - started < 1.5, "a renewal reached a whole window past the cap"

    def test_a_park_that_resumes_inside_the_cap_answers(self, monkeypatch):
        items = (
            [": preempted"]
            + [": preempt-keepalive"] * 5
            + [": resumed", 'data: {"choices":[{"delta":{"content":"a"}}]}']
        )
        seen = asyncio.run(self._relay(items, monkeypatch, cap = 5.0, window = 30.0))
        assert seen[-1].startswith("data:")


def test_a_second_park_before_the_first_token_gets_its_own_cap(monkeypatch):
    """Two parks during prefill, each inside the cap, together over it: the resume between them
    starts the second one's clock afresh."""
    # Each park is six 50 ms steps, 0.3 s, with the cap just under the two together: a few long
    # sleeps stretch far less on a loaded runner or a coarse timer than sixty short ones.
    items = [": preempted"] + [": preempt-keepalive"] * 6 + [": resumed", ": preempted"]
    items += [": preempt-keepalive"] * 6 + [
        ": resumed",
        'data: {"choices":[{"delta":{"content":"a"}}]}',
    ]
    seen = asyncio.run(
        TestARawRelayParkedBeforeItsFirstToken._relay(
            items, monkeypatch, cap = 0.58, window = 0.4, step = 0.05
        )
    )
    assert seen[-1].startswith("data:")


class TestAChatParkedAfterItsFirstToken:
    """The synchronous reader renews its stall window on keepalives only up to the park cap."""

    @staticmethod
    def _read(chunks):
        return LlamaCppBackend._iter_text_cancellable(
            _Stream(chunks),
            stall_timeout_s = 0.1,
            first_token_deadline = time.monotonic() + 30.0,
        )

    def test_a_wedged_park_times_out(self):
        data = 'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        chunks = [data, ": preempted\n\n"] + [": preempt-keepalive\n\n"] * 300
        started = time.monotonic()
        with pytest.raises(httpx.ReadTimeout, match = "stopped producing tokens"):
            for _ in self._read(chunks):
                pass
        assert time.monotonic() - started < 2, "every keepalive renewed the stall window"

    def test_a_park_inside_the_cap_still_answers(self):
        data = 'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        chunks = [data, ": preempted\n\n"] + [": preempt-keepalive\n\n"] * 5
        chunks += [": resumed\n\n", data.replace('"a"', '"b"')]
        assert "".join(self._read(chunks)).endswith('"b"}}]}\n\n')

    def test_a_wedged_park_in_split_chunks_times_out(self):
        data = 'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        chunks = [data, ": preempted\n\n"]
        chunks += [": preempt-", "keepalive\n\n: preempt-keepalive\n\n"] * 200
        started = time.monotonic()
        with pytest.raises(httpx.ReadTimeout, match = "stopped producing tokens"):
            for _ in self._read(chunks):
                pass
        assert time.monotonic() - started < 3, "split keepalives renewed the stall window"

    def test_a_split_park_that_resumes_still_answers(self):
        data = 'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        chunks = [data, ": preempted\n\n: preempt-"] + ["keepalive\n\n"] * 5
        chunks += [": resumed\n\n", data.replace('"a"', '"b"')]
        assert "".join(self._read(chunks)).endswith('"b"}}]}\n\n')


class TestAByteStreamParkedAfterItsFirstToken:
    """`/v1/completions` relays raw bytes, so a notice may be split or share a chunk."""

    @staticmethod
    async def _relay(items, monkeypatch, cap):
        import routes.inference as inference

        monkeypatch.setattr(inference, "_RAW_PARK_STALL_CAP_S", cap)

        async def upstream():
            for item in items:
                await asyncio.sleep(0.01)
                yield item

        seen = []
        async for item in inference._aiter_llama_stream_items(
            upstream(),
            first_token_deadline = time.monotonic() + 30.0,
            post_first_item_read_timeout_s = 0.1,
        ):
            seen.append(item)
        return seen

    def test_a_wedged_park_in_split_chunks_times_out(self, monkeypatch):
        items = [b'data: {"choices":[{"text":"a"}]}\n\n', b": preempted\n\n"]
        items += [b": preempt-", b"keepalive\n\n: preempt-keepalive\n\n"] * 200
        with pytest.raises(httpx.ReadTimeout):
            asyncio.run(self._relay(items, monkeypatch, cap = 0.3))

    def test_a_park_inside_the_cap_still_answers(self, monkeypatch):
        items = [b'data: {"choices":[{"text":"a"}]}\n\n', b": preempted\n\n: preempt-"]
        items += [b"keepalive\n\n"] * 5 + [
            b": resumed\n\n",
            b'data: {"choices":[{"text":"b"}]}\n\n',
        ]
        seen = asyncio.run(self._relay(items, monkeypatch, cap = 5.0))
        assert seen[-1].endswith(b'"b"}]}\n\n')


class TestALineSourceAndABatch:
    """`aiter_lines` yields a blank separator after every event, and a batch parks per prompt."""

    @staticmethod
    async def _relay(
        items,
        monkeypatch,
        cap,
        window = 30.0,
    ):
        import routes.inference as inference

        monkeypatch.setattr(inference, "_RAW_PARK_STALL_CAP_S", cap)
        monkeypatch.setattr(inference, "_DEFAULT_FIRST_TOKEN_TIMEOUT_S", window)
        monkeypatch.setattr(inference, "configured_first_token_timeout_s", lambda: window)

        async def upstream():
            for item in items:
                await asyncio.sleep(0.01)
                yield item

        seen = []
        async for item in inference._aiter_llama_stream_items(
            upstream(),
            first_token_deadline = time.monotonic() + window,
            post_first_item_read_timeout_s = 0.1,
        ):
            seen.append(item)
        return seen

    def test_blank_separators_do_not_renew_a_wedged_park(self, monkeypatch):
        items = ['data: {"choices":[{"delta":{"content":"a"}}]}', "", ": preempted", ""]
        items += [": preempt-keepalive", ""] * 200
        with pytest.raises(httpx.ReadTimeout):
            asyncio.run(self._relay(items, monkeypatch, cap = 0.3))

    def test_a_blank_before_the_first_token_is_not_the_first_token(self, monkeypatch):
        # The blank after each notice must not end the prefill and swap in the post-token clock.
        items = [": preempted", ""] + [": preempt-keepalive", ""] * 200
        with pytest.raises(httpx.ReadTimeout, match = "first token"):
            asyncio.run(self._relay(items, monkeypatch, cap = 0.3, window = 0.2))

    def test_a_sibling_resuming_does_not_reset_the_cap(self, monkeypatch):
        # Each half fits the cap but the park as a whole does not.
        items = [b'data: {"choices":[{"text":"a"}]}\n\n', b": preempted 0\n\n: preempted 1\n\n"]
        items += [b": preempt-keepalive\n\n"] * 30 + [b": resumed 0\n\n"]
        items += [b": preempt-keepalive\n\n"] * 40 + [b'data: {"choices":[{"text":"b"}]}\n\n']
        with pytest.raises(httpx.ReadTimeout):
            asyncio.run(self._relay(items, monkeypatch, cap = 0.5))

    def test_a_park_in_the_first_tokens_chunk_is_read(self, monkeypatch):
        # Output and park in one chunk: unread, every later keepalive looked like progress.
        items = [b'data: {"choices":[{"text":"a"}]}\n\n: preempted\n\n']
        items += [b": preempt-keepalive\n\n"] * 200
        with pytest.raises(httpx.ReadTimeout):
            asyncio.run(self._relay(items, monkeypatch, cap = 0.3))

    def test_the_last_prompt_resuming_ends_the_park(self, monkeypatch):
        items = [b'data: {"choices":[{"text":"a"}]}\n\n', b": preempted 0\n\n: preempted 1\n\n"]
        items += [b": resumed 0\n\n", b": resumed 1\n\n", b'data: {"choices":[{"text":"b"}]}\n\n']
        seen = asyncio.run(self._relay(items, monkeypatch, cap = 5.0))
        assert seen[-1].endswith(b'"b"}]}\n\n')

    def test_the_completions_relay_holds_a_resume_while_a_prompt_is_parked(self):
        import routes.inference as inference

        source = inspect.getsource(inference.openai_completions)
        assert "_cmpl_parks = ServerParkNotices()" in source
        assert "if _park == _OPENAI_PREEMPT_SSE_RESUMED and _cmpl_parks.parked:" in source


def test_a_resume_renews_the_configured_first_token_window(monkeypatch):
    """An operator who raised the first-token timeout for a slow model keeps it across a park:
    the resume used to hand the prefill the 20-minute default instead."""
    import routes.inference as inference

    source = inspect.getsource(inference._aiter_llama_stream_items)
    start = source.index("def _park_renews_first_window(")
    window = source[start : source.index("\n    def ", start + 10)]
    assert "_DEFAULT_FIRST_TOKEN_TIMEOUT_S" not in window
    assert window.count("_first_token_timeout_s()") == 2


def test_a_probe_excused_deadline_renews_the_configured_window():
    """The aggregate-probe excuse for a build without notices renews the first-token window the
    operator configured, as the notice path does, not the 20-minute default."""
    import routes.inference as inference

    source = inspect.getsource(inference._aiter_llama_stream_items)
    excused = source[
        source.index("if not await _probe_off_the_loop(park_above.excuses_silence):") :
    ]
    excused = excused[: excused.index("return True")]
    assert "first_token_deadline = now + _first_token_timeout_s()" in excused
    assert "_DEFAULT_FIRST_TOKEN_TIMEOUT_S" not in excused


class TestAnUnboundedReadGetsNoParkGraceWithoutAPark:
    """A prefill-progress read runs with no socket timeout, so the grace below never asks."""

    @staticmethod
    def _stalled(monkeypatch, parked: bool) -> float:
        import routes.inference as inference

        monkeypatch.setattr(inference, "_RAW_PARK_STALL_CAP_S", 1.0)
        monkeypatch.setattr(inference, "_DEFAULT_FIRST_TOKEN_TIMEOUT_S", 0.1)
        # The window now comes from the saved first-token limit, not the default.
        monkeypatch.setattr(inference, "configured_first_token_timeout_s", lambda: 0.1)
        monkeypatch.setattr(inference, "_install_park_aware_read", lambda response, grace: True)
        monkeypatch.setattr(inference, "_set_stream_response_read_timeout", lambda *a, **k: None)
        monkeypatch.setattr(inference, "_live_read_timeout", lambda response: None)

        async def relay():
            async def upstream():
                await asyncio.sleep(30)
                yield "never sent"

            async for _ in inference._aiter_llama_stream_items(
                upstream(),
                first_token_deadline = time.monotonic() + 0.05,
                response = object(),
                track_prefill_progress = True,
                stall_grace = lambda: parked,
            ):
                pass

        started = time.monotonic()
        with pytest.raises(httpx.ReadTimeout):
            asyncio.run(relay())
        return time.monotonic() - started

    def test_a_stalled_prefill_times_out_on_its_own_window(self, monkeypatch):
        assert self._stalled(monkeypatch, parked = False) < 0.5, "a stall got the park cap"

    def test_a_parked_prefill_still_waits_up_to_the_cap(self, monkeypatch):
        assert 0.5 < self._stalled(monkeypatch, parked = True) < 5
