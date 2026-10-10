# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Opt-in encoding through the ``tokenizers`` 1.0 release candidate.

transformers requires ``tokenizers<0.24`` and builds its tokenizers from the 0.x component API,
which the RC drops, so the RC cannot replace 0.x. Instead it is installed beside it:

    pip install --pre --no-deps --target ~/.cache/unsloth/tokenizers_v1 "tokenizers>=1.0.0rc2"
    UNSLOTH_TOKENIZERS_V1=1 python train.py   # UNSLOTH_TOKENIZERS_V1_PATH overrides the directory

and loaded under another module name. The fast tokenizer's 0.x backend is then wrapped: plain
encodes (str input, no pair, no pre-split words; padding and truncation included) run on the RC,
everything else goes to 0.x unchanged. Each RC copy is built from the backend's own JSON and must
match 0.x on a probe set first, else the backend stays on 0.x. Fields the RC's ``Encoding`` lacks
(offsets, tokens, word ids, special tokens mask, ...) are recomputed by 0.x on first access.
"""

import contextlib
import copy
import functools
import glob
import hashlib
import importlib.machinery
import importlib.util
import os
import sys
import tempfile
import threading
import weakref

from transformers.utils import logging

logger = logging.get_logger(__name__)

__all__ = ["enable_tokenizers_v1", "tokenizers_v1_enabled", "TokenizersV1Backend"]

_ENV = "UNSLOTH_TOKENIZERS_V1"
_PATH_ENV = "UNSLOTH_TOKENIZERS_V1_PATH"
_VERIFY_ENV = "UNSLOTH_TOKENIZERS_V1_VERIFY"
_DEFAULT_DIR = os.path.join(os.path.expanduser("~"), ".cache", "unsloth", "tokenizers_v1")
# The last component must stay "tokenizers": CPython calls PyInit_<last component>.
_MODULE_NAME = "unsloth_tokenizers_v1.tokenizers"
# Mutations the per-encode fingerprint (vocab size + component states) cannot see.
_MUTATING = frozenset(("model", "add_tokens", "add_special_tokens", "train", "train_from_iterator"))
# A backend whose JSON keeps changing between encodes would pay a rebuild per call.
_MAX_REBUILDS = 16
_PROBES = (
    "Hello world! How are you?",
    "  leading and trailing spaces  ",
    "Tabs\tand\nnewlines\r\n\n\nend",
    "Numbers 0 12 345 6789 3.14159 -42 1,000,000",
    "Emoji 😁🚀👍🏽 and accents: café naïve Zürich",
    "中文测试，日本語のテキスト、한국어 텍스트",
    "def f(x):\n    return x ** 2  # code\n",
    "",
)

_v1_module = None
_v1_tried = False
_v1_lock = threading.Lock()
# RC from_file deadlocks in a child forked after any RC build (encoding inherited ones is fine):
# snapshots are keyed by JSON, refreshed before fork, and such children only reuse them.
_MAX_SNAPSHOTS = 8
_snapshots = {}
_snapshots_lock = threading.Lock()
_built_here = False
_no_build = False
_live = weakref.WeakSet()
_all_snapshots = weakref.WeakSet()
_fork_hooks = False


def _fatal(error):
    # PyO3 panics derive from BaseException; only these must propagate.
    return isinstance(error, (KeyboardInterrupt, SystemExit, GeneratorExit))


def tokenizers_v1_enabled():
    return os.environ.get(_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def _load_v1():
    global _v1_module, _v1_tried
    if _v1_tried:
        return _v1_module
    with _v1_lock:
        if _v1_tried:
            return _v1_module
        directory = os.path.expanduser(os.environ.get(_PATH_ENV) or _DEFAULT_DIR)
        candidates = [
            path
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
            for path in glob.glob(os.path.join(directory, "tokenizers", "tokenizers*" + suffix))
            if os.path.basename(path) == "tokenizers" + suffix
        ]
        module = None
        loaded = sys.modules.get(_MODULE_NAME)
        loaded_file = os.path.realpath(getattr(loaded, "__file__", None) or "")
        if loaded is not None and any(os.path.realpath(path) == loaded_file for path in candidates):
            # An extension initialises once per process; only tests reset the cache.
            candidates = []
            module = loaded
        elif not candidates:
            logger.warning_once(
                f"Unsloth: {_ENV}=1 but no tokenizers release candidate was found in {directory}. "
                f'Install it with `pip install --pre --no-deps --target {directory} "tokenizers>=1.0.0rc2"`. '
                "Using tokenizers 0.x."
            )
        else:
            try:
                loader = importlib.machinery.ExtensionFileLoader(_MODULE_NAME, candidates[0])
                spec = importlib.util.spec_from_file_location(
                    _MODULE_NAME, candidates[0], loader = loader
                )
                module = importlib.util.module_from_spec(spec)
                loader.exec_module(module)
                major = int(str(module.__version__).split(".")[0])
                if major < 1:
                    raise ImportError(f"found tokenizers {module.__version__}, need >= 1.0")
                sys.modules[_MODULE_NAME] = module
            except BaseException as error:
                if _fatal(error):
                    raise
                logger.warning_once(
                    f"Unsloth: could not load the tokenizers release candidate from {candidates[0]}: {error}. "
                    "Using tokenizers 0.x."
                )
                module = None
        _v1_module = module
        _v1_tried = True
        return module


def _same(a, b):
    return a.ids == b.ids and a.type_ids == b.type_ids and a.attention_mask == b.attention_mask


class _Snapshot:
    """An RC tokenizer and a 0.x twin, both built from one backend JSON (no padding / truncation)."""

    __slots__ = ("v1", "reference", "lock", "option_cache", "n_special", "__weakref__")

    def __init__(self, v1, reference):
        self.v1 = v1
        self.reference = reference
        self.lock = threading.Lock()
        self.option_cache = {}
        self.n_special = reference.num_special_tokens_to_add(False)
        _all_snapshots.add(self)

    @contextlib.contextmanager
    def configured(
        self,
        padding,
        truncation,
        encode_special_tokens = False,
    ):
        with self.lock:
            reference = self.reference
            try:
                reference.encode_special_tokens = encode_special_tokens
                if truncation is not None:
                    reference.enable_truncation(**truncation)
                if padding is not None:
                    reference.enable_padding(**padding)
                yield reference
            finally:
                reference.no_padding()
                reference.no_truncation()
                reference.encode_special_tokens = False

    def reference_encode(
        self,
        text,
        add_special_tokens,
        encode_special_tokens,
        padding = None,
        truncation = None,
    ):
        with self.configured(padding, truncation, encode_special_tokens) as reference:
            return reference.encode(text, add_special_tokens = add_special_tokens)

    def options(self, padding, truncation):
        """0.x padding / truncation dicts -> RC objects; None if the RC cannot express them."""
        key = (
            None if padding is None else tuple(sorted(padding.items())),
            None if truncation is None else tuple(sorted(truncation.items())),
        )
        cache = self.option_cache
        if key in cache:
            return cache[key]
        module = sys.modules.get(_MODULE_NAME)
        try:
            v1_padding = (
                None
                if padding is None
                else module.Padding(
                    direction = padding["direction"],
                    pad_id = padding["pad_id"],
                    pad_type_id = padding["pad_type_id"],
                    pad_token = padding["pad_token"],
                    length = padding["length"],
                    pad_to_multiple_of = padding["pad_to_multiple_of"],
                )
            )
            # stride only shapes 0.x overflow, which the RC lacks and 0.x recomputes on demand.
            v1_truncation = (
                None
                if truncation is None
                else module.Truncation(
                    max_length = truncation["max_length"],
                    strategy = truncation["strategy"],
                    direction = truncation["direction"],
                )
            )
            result = (v1_padding, v1_truncation)
        except Exception:
            result = None
        if len(cache) < 64:
            cache[key] = result
        return result


class _State:
    __slots__ = (
        "dirty",
        "generation",
        "fingerprint",
        "digest",
        "snapshot",
        "disabled",
        "rebuilds",
        "lock",
    )

    def __init__(self):
        self.dirty = True
        self.generation = 0
        self.fingerprint = None
        self.digest = None
        self.snapshot = None
        self.disabled = False
        self.rebuilds = 0
        self.lock = threading.Lock()


def _build_snapshot(base, text):
    v1_module = _load_v1()
    # The RC loads only from a file or the Hub.
    fd, path = tempfile.mkstemp(suffix = ".json", prefix = "unsloth_tokenizers_v1_")
    try:
        with os.fdopen(fd, "w", encoding = "utf-8") as file:
            file.write(text)
        global _built_here
        _built_here = True
        v1 = v1_module.Tokenizer.from_file(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    reference = type(base).from_str(text)
    added = [token.content for token in base.get_added_tokens_decoder().values()][:64]
    probes = list(_PROBES) + ["".join(added), " ".join(added) + " Hello"]
    for encode_special_tokens in (False, True):
        reference.encode_special_tokens = encode_special_tokens
        for add_special_tokens in (True, False):
            want = reference.encode_batch(probes, add_special_tokens = add_special_tokens)
            got = v1.encode_batch(
                probes,
                add_special_tokens = add_special_tokens,
                encode_special_tokens = encode_special_tokens,
            )
            if len(want) != len(got) or not all(_same(a, b) for a, b in zip(want, got)):
                raise ValueError("token ids differ from tokenizers 0.x on the probe set")
    snapshot = _Snapshot(v1, reference)
    pad_id = (base.token_to_id(added[0]) if added else None) or 0
    for direction in ("right", "left"):
        padding = dict(
            length = None,
            pad_to_multiple_of = None,
            pad_id = pad_id,
            pad_type_id = 0,
            pad_token = added[0] if added else "[PAD]",
            direction = direction,
        )
        truncation = dict(max_length = 5, stride = 0, strategy = "longest_first", direction = direction)
        for pad, trunc in ((padding, None), (None, truncation), (padding, truncation)):
            v1_padding, v1_truncation = snapshot.options(pad, trunc)
            with snapshot.configured(pad, trunc) as configured:
                want = configured.encode_batch(probes)
            got = v1.encode_batch(probes, padding = v1_padding, truncation = v1_truncation)
            if len(want) != len(got) or not all(_same(a, b) for a, b in zip(want, got)):
                raise ValueError(f"padded / truncated ids differ from tokenizers 0.x ({direction})")
    return snapshot


def _rebuild_encoding(encoding):
    return encoding


class _CompatEncoding:
    """An RC ``Encoding``; any field the RC lacks is answered by an exact 0.x re-encode."""

    __slots__ = ("_v1", "_text", "_add", "_split", "_snapshot", "_padding", "_truncation", "_real")

    def __init__(
        self, v1, text, add_special_tokens, encode_special_tokens, snapshot, padding, truncation
    ):
        self._v1 = v1
        self._text = text
        self._add = add_special_tokens
        self._split = encode_special_tokens
        self._snapshot = snapshot
        self._padding = padding
        self._truncation = truncation
        self._real = None

    # Once materialized the 0.x encoding answers everything: Encoding.pad / truncate mutate it.
    @property
    def ids(self):
        return (self._v1 if self._real is None else self._real).ids

    @property
    def type_ids(self):
        return (self._v1 if self._real is None else self._real).type_ids

    @property
    def attention_mask(self):
        return (self._v1 if self._real is None else self._real).attention_mask

    @property
    def n_sequences(self):
        # Only single sequences are routed here; BatchEncoding reads this on every call.
        return 1 if self._real is None else self._real.n_sequences

    def __len__(self):
        return len(self._v1 if self._real is None else self._real)

    def _materialize(self):
        if self._real is None:
            padding = self._padding
            if padding is not None:
                # Pad-to-longest depends on the batch: replay the length this row got.
                padding = dict(padding, length = len(self._v1), pad_to_multiple_of = None)
            self._real = self._snapshot.reference_encode(
                self._text, self._add, self._split, padding, self._truncation
            )
        return self._real

    def __getattr__(self, name):
        return getattr(self._materialize(), name)

    def __reduce__(self):
        return (_rebuild_encoding, (self._materialize(),))

    def __repr__(self):
        return repr(self._materialize())


def _rewrap(base):
    return enable_tokenizers_v1(base) if tokenizers_v1_enabled() else base


class TokenizersV1Backend:
    """Stands in for a ``tokenizers`` 0.x ``Tokenizer``; only plain encodes take the RC."""

    __slots__ = ("_unsloth_base", "_unsloth_state", "__weakref__")

    def __init__(self, base):
        object.__setattr__(self, "_unsloth_base", base)
        object.__setattr__(self, "_unsloth_state", _State())
        _live.add(self)
        _register_fork_hooks()

    def __getattr__(self, name):
        if name in _MUTATING:
            self._unsloth_mark()
        return getattr(self._unsloth_base, name)

    def __setattr__(self, name, value):
        setattr(self._unsloth_base, name, value)
        if name != "encode_special_tokens":
            self._unsloth_mark()

    def _unsloth_mark(self):
        state = self._unsloth_state
        state.generation += 1
        state.dirty = True

    def _unsloth_fingerprint(self):
        base = self._unsloth_base
        return (base.get_vocab_size(True),) + tuple(
            None if part is None else part.__getstate__()
            for part in (base.normalizer, base.pre_tokenizer, base.post_processor, base.decoder)
        )

    def __dir__(self):
        return dir(self._unsloth_base)

    def __repr__(self):
        return repr(self._unsloth_base)

    def __reduce__(self):
        return (_rewrap, (self._unsloth_base,))

    def __copy__(self):
        return _rewrap(copy.copy(self._unsloth_base))

    def __deepcopy__(self, memo):
        return _rewrap(copy.deepcopy(self._unsloth_base, memo))

    def _unsloth_refresh(self):
        state = self._unsloth_state
        if state.disabled:
            return None
        # Catches in-place component edits and edits through other references to the backend.
        try:
            fingerprint = self._unsloth_fingerprint()
        except BaseException as error:
            if _fatal(error):
                raise
            # Custom Python components cannot be serialized, so the RC cannot copy them either.
            state.disabled = True
            state.snapshot = None
            return None
        if not state.dirty and fingerprint == state.fingerprint:
            return state.snapshot
        with state.lock:
            if state.disabled:
                return None
            generation = state.generation
            base = self._unsloth_base
            try:
                text = base.to_str()
                if base.padding is not None or base.truncation is not None:
                    # Padding / truncation change per call and are passed per encode instead.
                    clone = type(base).from_str(text)
                    clone.no_padding()
                    clone.no_truncation()
                    text = clone.to_str()
            except BaseException as error:
                if _fatal(error):
                    raise
                # e.g. custom Python components, which 0.x cannot serialize either.
                state.disabled = True
                state.snapshot = None
                return None
            digest = hashlib.blake2b(text.encode("utf-8"), digest_size = 16).digest()
            if digest != state.digest:
                snapshot = self._unsloth_lookup_or_build(state, base, text, digest)
                if snapshot is None:
                    state.disabled = True
                    state.snapshot = None
                    return None
                state.snapshot = snapshot
                state.digest = digest
            state.fingerprint = fingerprint
            if state.generation == generation:
                state.dirty = False
            return state.snapshot

    @staticmethod
    def _unsloth_lookup_or_build(state, base, text, digest):
        with _snapshots_lock:
            snapshot = _snapshots.get(digest)
        if snapshot is not None:
            return snapshot
        if _no_build:
            logger.warning_once(
                "Unsloth: a tokenizer changed inside a forked worker, so that worker stays on tokenizers 0.x."
            )
            return None
        state.rebuilds += 1
        if state.rebuilds > _MAX_REBUILDS:
            logger.warning_once(
                "Unsloth: the tokenizer keeps changing between encodes, so it stays on tokenizers 0.x."
            )
            return None
        try:
            snapshot = _build_snapshot(base, text)
        except BaseException as error:
            if _fatal(error):
                raise
            logger.warning_once(
                f"Unsloth: this tokenizer stays on tokenizers 0.x, the release candidate cannot match it: {error}"
            )
            return None
        with _snapshots_lock:
            _snapshots[digest] = snapshot
            while len(_snapshots) > _MAX_SNAPSHOTS:
                _snapshots.pop(next(iter(_snapshots)))
        return snapshot

    def _unsloth_encode(self, snapshot, texts, add_special_tokens):
        base = self._unsloth_base
        split = base.encode_special_tokens
        padding, truncation = base.padding, base.truncation
        # 0.x skips truncation when max_length cannot fit the added special tokens; the RC drops content.
        if (
            truncation is not None
            and add_special_tokens
            and truncation["max_length"] < snapshot.n_special
        ):
            return None
        options = snapshot.options(padding, truncation)
        if options is None:
            return None
        try:
            encodings = snapshot.v1.encode_batch(
                texts,
                add_special_tokens = add_special_tokens,
                encode_special_tokens = split,
                padding = options[0],
                truncation = options[1],
            )
        except BaseException as error:
            if _fatal(error):
                raise
            return None
        out = [
            _CompatEncoding(
                encoding, text, add_special_tokens, split, snapshot, padding, truncation
            )
            for encoding, text in zip(encodings, texts)
        ]
        if os.environ.get(_VERIFY_ENV, "0") == "1":
            for text, encoding in zip(texts, out):
                if not _same(encoding._materialize(), encoding):
                    raise RuntimeError(
                        f"Unsloth: tokenizers release candidate differs from 0.x on {text[:200]!r}"
                    )
        return out

    def _unsloth_try(self, texts, add_special_tokens, is_pretokenized, kwargs):
        if kwargs or is_pretokenized or not isinstance(texts, (list, tuple)):
            return None
        if not all(type(text) is str for text in texts):
            return None
        snapshot = self._unsloth_refresh()
        return (
            None
            if snapshot is None
            else self._unsloth_encode(snapshot, list(texts), add_special_tokens)
        )

    # Signatures match tokenizers 0.x. encode_batch_fast is not routed: 0.x returns zero offsets there.
    def encode_batch(
        self,
        input,
        is_pretokenized = False,
        add_special_tokens = True,
        **kwargs,
    ):
        out = self._unsloth_try(input, add_special_tokens, is_pretokenized, kwargs)
        if out is not None:
            return out
        return self._unsloth_base.encode_batch(
            input, is_pretokenized = is_pretokenized, add_special_tokens = add_special_tokens, **kwargs
        )

    def encode(
        self,
        sequence,
        pair = None,
        is_pretokenized = False,
        add_special_tokens = True,
        **kwargs,
    ):
        if pair is None and type(sequence) is str:
            out = self._unsloth_try([sequence], add_special_tokens, is_pretokenized, kwargs)
            if out is not None:
                return out[0]
        return self._unsloth_base.encode(
            sequence,
            pair,
            is_pretokenized = is_pretokenized,
            add_special_tokens = add_special_tokens,
            **kwargs,
        )


def _refresh_before_fork():
    if _no_build:
        return
    for proxy in list(_live):
        try:
            snapshot = proxy._unsloth_refresh()
            if snapshot is not None:
                with _snapshots_lock:
                    _snapshots.setdefault(proxy._unsloth_state.digest, snapshot)
        except BaseException as error:
            if _fatal(error):
                raise


def _after_fork_in_child():
    global _no_build, _v1_lock, _snapshots_lock
    _no_build = _no_build or _built_here
    # A lock another thread held at fork time stays locked forever in the child.
    _v1_lock = threading.Lock()
    _snapshots_lock = threading.Lock()
    for proxy in list(_live):
        proxy._unsloth_state.lock = threading.Lock()
    for snapshot in list(_all_snapshots):
        snapshot.lock = threading.Lock()


def _register_fork_hooks():
    global _fork_hooks
    if not _fork_hooks and hasattr(os, "register_at_fork"):
        _fork_hooks = True
        os.register_at_fork(before = _refresh_before_fork, after_in_child = _after_fork_in_child)


def _is_tokenizers_0x(backend):
    try:
        import tokenizers
    except ImportError:
        return False
    # Under tokenizers >= 1.0 the backend is already the fast engine.
    return isinstance(backend, tokenizers.Tokenizer) and str(tokenizers.__version__).startswith(
        "0."
    )


def enable_tokenizers_v1(tokenizer):
    """Route a fast tokenizer's (or processor's inner tokenizer's) plain encodes to the RC.

    Accepts a raw 0.x backend too, returning the wrapper. Returns its argument unchanged when
    ``UNSLOTH_TOKENIZERS_V1`` is off, the RC is missing, or there is no 0.x backend; safe to repeat.
    """
    if not tokenizers_v1_enabled() or tokenizer is None:
        return tokenizer
    if isinstance(tokenizer, TokenizersV1Backend):
        return tokenizer
    if _is_tokenizers_0x(tokenizer):
        return TokenizersV1Backend(tokenizer) if _load_v1() is not None else tokenizer
    for obj in (tokenizer, getattr(tokenizer, "tokenizer", None)):
        backend = getattr(obj, "_tokenizer", None) if obj is not None else None
        if (
            backend is None
            or isinstance(backend, TokenizersV1Backend)
            or not _is_tokenizers_0x(backend)
        ):
            continue
        if _load_v1() is None:
            break
        obj._tokenizer = TokenizersV1Backend(backend)
    return tokenizer


def tokenizers_v1_on_return(fn):
    """Decorate a ``from_pretrained`` returning ``(model, tokenizer)``."""

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if not tokenizers_v1_enabled():
            return result
        if isinstance(result, tuple) and len(result) == 2:
            try:
                enable_tokenizers_v1(result[1])
            except Exception as error:
                logger.warning_once(
                    f"Unsloth: could not enable the tokenizers release candidate: {error}"
                )
        return result

    return _wrapper
