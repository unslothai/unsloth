# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the native_context_length feature (PR #4746).

Verifies the `native_context_length` property on LlamaCppBackend and the
matching Pydantic fields. The raw GGUF `_context_length` must never be
overwritten by VRAM-capping logic.

Needs no GPU, network, or libraries beyond pytest and pydantic.
"""

import ast
import io
import json
import struct
import sys
import types as _types
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Stub heavy / unavailable deps before importing the module under test.
# Same pattern as test_kv_cache_estimation.py.
# ---------------------------------------------------------------------------

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# loggers
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

# structlog
_structlog_stub = _types.ModuleType("structlog")
sys.modules.setdefault("structlog", _structlog_stub)

# httpx -- stub only names referenced at import / class-definition time
_httpx_stub = _types.ModuleType("httpx")
for _exc_name in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
):
    setattr(_httpx_stub, _exc_name, type(_exc_name, (Exception,), {}))


class _FakeTimeout:
    def __init__(self, *a, **kw):
        pass


_httpx_stub.Timeout = _FakeTimeout
_httpx_stub.Client = type(
    "Client",
    (),
    {
        "__init__": lambda self, **kw: None,
        "__enter__": lambda self: self,
        "__exit__": lambda self, *a: None,
    },
)
sys.modules.setdefault("httpx", _httpx_stub)

from core.inference.llama_cpp import LlamaCppBackend
from models.inference import LoadResponse, InferenceStatusResponse


# ── Helpers ──────────────────────────────────────────────────────────


def _write_kv(buf: io.BytesIO, key: str, value, vtype: int) -> None:
    """Append a single GGUF KV pair to *buf*."""
    key_bytes = key.encode("utf-8")
    buf.write(struct.pack("<Q", len(key_bytes)))
    buf.write(key_bytes)
    buf.write(struct.pack("<I", vtype))
    if vtype == 4:  # UINT32
        buf.write(struct.pack("<I", value))
    elif vtype == 10:  # UINT64
        buf.write(struct.pack("<Q", value))
    elif vtype == 8:  # STRING
        val_bytes = value.encode("utf-8")
        buf.write(struct.pack("<Q", len(val_bytes)))
        buf.write(val_bytes)
    else:
        raise ValueError(f"Unsupported vtype in test helper: {vtype}")


def make_gguf(
    tmp_path: Path,
    arch: str,
    kvs: list,
    *,
    arch_first: bool = True,
    filename: str = "test.gguf",
) -> str:
    """Create a minimal valid GGUF v3 binary in *tmp_path*."""
    buf = io.BytesIO()
    buf.write(struct.pack("<I", 0x46554747))  # GGUF magic
    buf.write(struct.pack("<I", 3))  # version 3
    buf.write(struct.pack("<Q", 0))  # tensor count = 0

    ordered = []
    arch_entry = ("general.architecture", arch, 8)

    if arch_first:
        ordered.append(arch_entry)
    for suffix, val, vt in kvs:
        ordered.append((f"{arch}.{suffix}", val, vt))
    if not arch_first:
        ordered.append(arch_entry)

    buf.write(struct.pack("<Q", len(ordered)))
    for key, val, vt in ordered:
        _write_kv(buf, key, val, vt)

    path = tmp_path / filename
    path.write_bytes(buf.getvalue())
    return str(path)


@pytest.fixture
def backend():
    """Create a fresh LlamaCppBackend with side effects disabled."""
    with patch.object(LlamaCppBackend, "_kill_orphaned_servers"):
        with patch("atexit.register"):
            return LlamaCppBackend()


# =====================================================================
# A. TestNativeContextLengthProperty -- the new property
# =====================================================================


class TestNativeContextLengthProperty:
    """Tests the new `native_context_length` property on LlamaCppBackend."""

    def test_none_on_fresh_backend(self, backend):
        """Returns None when no model loaded."""
        assert backend.native_context_length is None

    def test_returns_raw_gguf_value(self, backend):
        """Directly returns _context_length when set."""
        backend._context_length = 131072
        assert backend.native_context_length == 131072

    def test_not_capped_by_effective(self, backend):
        """native_context_length ignores _effective_context_length."""
        backend._context_length = 131072
        backend._effective_context_length = 32768
        assert backend.native_context_length == 131072

    def test_not_capped_by_max(self, backend):
        """native_context_length ignores _max_context_length."""
        backend._context_length = 131072
        backend._max_context_length = 65536
        assert backend.native_context_length == 131072

    def test_none_after_unload(self, backend):
        """After unload_model(), returns None."""
        backend._context_length = 131072
        assert backend.native_context_length == 131072
        backend.unload_model()
        assert backend.native_context_length is None

    def test_after_gguf_parse(self, tmp_path, backend):
        """Synthetic GGUF with context_length=16384 populates the property."""
        path = make_gguf(
            tmp_path,
            "llama",
            [("context_length", 16384, 4)],
        )
        backend._read_gguf_metadata(path)
        assert backend.native_context_length == 16384

    def test_resets_between_parses(self, tmp_path, backend):
        """Second GGUF without context_length resets native to None."""
        path_a = make_gguf(
            tmp_path,
            "llama",
            [("context_length", 16384, 4)],
            filename = "a.gguf",
        )
        backend._read_gguf_metadata(path_a)
        assert backend.native_context_length == 16384

        path_b = make_gguf(
            tmp_path,
            "gpt2",
            [("block_count", 12, 4)],
            filename = "b.gguf",
        )
        backend._read_gguf_metadata(path_b)
        assert backend.native_context_length is None


# =====================================================================
# B. TestContextValueSeparation -- core invariant
# =====================================================================


class TestContextValueSeparation:
    """_context_length is never overwritten by VRAM logic."""

    def test_preserved_after_effective_set(self, backend):
        """Setting _effective_context_length does not change _context_length."""
        backend._context_length = 131072
        backend._effective_context_length = 32768
        assert backend._context_length == 131072
        assert backend.native_context_length == 131072

    def test_ordering_when_capped(self, backend):
        """native >= max >= effective holds when VRAM-capped."""
        backend._context_length = 131072
        backend._max_context_length = 65536
        backend._effective_context_length = 32768
        assert backend.native_context_length >= backend.max_context_length
        assert backend.max_context_length >= backend.context_length

    def test_all_equal_when_uncapped(self, backend):
        """All three equal when no VRAM constraint."""
        backend._context_length = 8192
        # No effective/max set -- properties fall back to _context_length.
        assert backend.native_context_length == 8192
        assert backend.max_context_length == 8192
        assert backend.context_length == 8192

    def test_fit_context_does_not_modify(self, backend):
        """_fit_context_to_vram() does not touch _context_length."""
        backend._context_length = 131072
        backend._n_layers = 32
        backend._n_kv_heads = 8
        backend._n_heads = 32
        backend._embedding_length = 4096
        original = backend._context_length

        # Tiny VRAM budget forces capping.
        result = backend._fit_context_to_vram(
            requested_ctx = 131072,
            available_mib = 512,  # very small
            model_size_bytes = 0,
        )
        # Returns the capped value without modifying _context_length.
        assert backend._context_length == original
        assert backend.native_context_length == original
        # Capped value must be <= requested.
        assert result <= 131072

    def test_native_gt_context_when_capped(self, backend):
        """native_context_length > context_length after VRAM capping."""
        backend._context_length = 131072
        backend._effective_context_length = 16384
        assert backend.native_context_length > backend.context_length


# =====================================================================
# C. TestPydanticModels -- LoadResponse & InferenceStatusResponse
# =====================================================================


class TestPydanticModels:
    """Tests native_context_length field on Pydantic models."""

    def test_load_response_has_field(self):
        """Field exists in LoadResponse.model_fields."""
        assert "native_context_length" in LoadResponse.model_fields
        assert "context_length" in LoadResponse.model_fields

    def test_load_response_defaults_none(self):
        """Omitting native_context_length defaults to None."""
        resp = LoadResponse(
            status = "loaded",
            model = "test",
            display_name = "Test",
            inference = {},
        )
        assert resp.native_context_length is None

    def test_load_response_accepts_int(self):
        """native_context_length=131072 stores correctly."""
        resp = LoadResponse(
            status = "loaded",
            model = "test",
            display_name = "Test",
            inference = {},
            native_context_length = 131072,
        )
        assert resp.native_context_length == 131072

    def test_load_response_json_null(self):
        """None serializes to JSON null."""
        resp = LoadResponse(
            status = "loaded",
            model = "test",
            display_name = "Test",
            inference = {},
        )
        data = json.loads(resp.model_dump_json())
        assert data["native_context_length"] is None

    def test_load_response_json_int(self):
        """131072 serializes to JSON number."""
        resp = LoadResponse(
            status = "loaded",
            model = "test",
            display_name = "Test",
            inference = {},
            native_context_length = 131072,
        )
        data = json.loads(resp.model_dump_json())
        assert data["native_context_length"] == 131072

    def test_status_response_has_field(self):
        """Field exists in InferenceStatusResponse.model_fields."""
        assert "native_context_length" in InferenceStatusResponse.model_fields
        assert "context_length" in InferenceStatusResponse.model_fields

    def test_status_response_has_chat_template_field(self):
        """Status includes chat_template so the UI can rehydrate after refresh."""
        assert "chat_template" in InferenceStatusResponse.model_fields

    def test_status_response_defaults_none(self):
        """Omitting native_context_length defaults to None."""
        resp = InferenceStatusResponse()
        assert resp.native_context_length is None

    def test_status_response_chat_template_roundtrip(self):
        """chat_template serializes and validates as part of status."""
        resp = InferenceStatusResponse(chat_template = "{{ messages }}")
        roundtripped = InferenceStatusResponse.model_validate_json(resp.model_dump_json())
        assert roundtripped.chat_template == "{{ messages }}"

    def test_roundtrip_preserves_value(self):
        """model_validate_json(model_dump_json()) round-trips."""
        resp = LoadResponse(
            status = "loaded",
            model = "test",
            display_name = "Test",
            inference = {},
            native_context_length = 131072,
        )
        roundtripped = LoadResponse.model_validate_json(resp.model_dump_json())
        assert roundtripped.native_context_length == 131072

    def test_context_length_roundtrip(self):
        """Runtime context_length serializes for non-GGUF/hub models."""
        resp = LoadResponse(
            status = "loaded",
            model = "test",
            display_name = "Test",
            inference = {},
            context_length = 8192,
        )
        roundtripped = LoadResponse.model_validate_json(resp.model_dump_json())
        assert roundtripped.context_length == 8192


# =====================================================================
# D. TestRouteCompleteness -- source-level verification
# =====================================================================


_HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete", "head", "options"})

# `match` capture nodes, which bind a name without an ast.Name store. Looked up rather than named
# directly so this file still parses on a Python without them.
_MATCH_CAPTURES = tuple(
    node
    for node in (getattr(ast, attr, None) for attr in ("MatchAs", "MatchStar", "MatchMapping"))
    if node is not None
) or (type(None),)


class TestRouteCompleteness:
    """All response construction sites in routes/inference.py include native_context_length."""

    @pytest.fixture(autouse = True)
    def _load_source(self):
        """Read routes/inference.py source once."""
        routes_path = Path(__file__).resolve().parent.parent / "routes" / "inference.py"
        self._source = routes_path.read_text(encoding = "utf-8")

    def _find_construction_blocks(self, class_name: str) -> list[str]:
        """Extract all code blocks that construct a given response class."""
        blocks = []
        idx = 0
        while True:
            start = self._source.find(f"{class_name}(", idx)
            if start == -1:
                break
            # Find the matching closing paren via a depth counter.
            depth = 0
            end = start
            for i, ch in enumerate(self._source[start:], start):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            blocks.append(self._source[start:end])
            idx = end
        return blocks

    _NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)

    @staticmethod
    def _is_runtime_fields_call(node) -> bool:
        """``_llama_runtime_fields(llama_backend)``, argument included.

        The argument is half the contract: a same-named helper called on something else does not
        prove the GGUF ``/status`` response is fed from the llama backend.
        """
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_llama_runtime_fields"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "llama_backend"
        )

    def _bindings_in_own_scope(self, scope, name: str) -> list:
        """Every rebinding of ``name`` in ``scope``'s OWN body, as (node, value) pairs.

        Nested functions, lambdas and classes are separate scopes, so a binding inside one says
        nothing about the outer name and is skipped. A binding form this does not model (a for
        target, a ``with ... as``) yields a None value, which the caller reads as "not the helper"
        so an unrecognised rebind fails closed rather than passing silently.

        Not every binding is an ``ast.Name`` store, though, and the ones that are not would slip
        past a Name-only walk entirely rather than fail closed. ``except E as fields`` is the sharp
        one: Python DELETES the target when the handler ends, so a handled-exception path can reach
        the constructor with the name unbound. ``import x as fields`` and the capture patterns of a
        ``match`` statement bind through their own node types too. All are recorded with a None
        value.
        """
        found: list = []
        handled: set = set()

        def walk(node) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, self._NESTED_SCOPES):
                    # The BODY is a separate scope, but a `def`/`class` also binds its own
                    # name out here, so `def fields(): ...` after the hoist leaves the splat
                    # receiving a function. Record the declaration, skip the body.
                    if getattr(child, "name", None) == name:
                        found.append((child, None))
                    continue
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name) and target.id == name:
                            found.append((child, child.value))
                            handled.add(id(target))
                elif isinstance(child, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
                    target = child.target
                    if isinstance(target, ast.Name) and target.id == name:
                        # AnnAssign may be a bare declaration with no value; AugAssign rebinds to
                        # something derived, never the helper call itself.
                        value = (
                            child.value
                            if isinstance(child, (ast.AnnAssign, ast.NamedExpr))
                            else None
                        )
                        found.append((child, value))
                        handled.add(id(target))
                elif (
                    isinstance(child, ast.Name)
                    and child.id == name
                    and isinstance(child.ctx, ast.Store)
                    and id(child) not in handled
                ):
                    found.append((child, None))
                elif isinstance(child, ast.ExceptHandler) and child.name == name:
                    found.append((child, None))
                elif isinstance(child, ast.alias) and name in (
                    child.asname,
                    # `import a.b` with no asname binds only the TOP package, and the alias
                    # node spells the whole dotted path, so comparing the raw name misses it.
                    None if child.asname else child.name.split(".")[0],
                ):
                    found.append((child, None))
                elif isinstance(child, (ast.Global, ast.Nonlocal)) and name in child.names:
                    found.append((child, None))
                elif isinstance(child, _MATCH_CAPTURES) and name in (
                    getattr(child, "name", None),
                    getattr(child, "rest", None),  # MatchMapping spells its capture `rest`
                ):
                    found.append((child, None))
                walk(child)

        walk(scope)
        return found

    @staticmethod
    def _parameter_names(scope) -> set:
        """Every name ``scope`` binds through its signature.

        A parameter is a binding that no assignment statement records, so without this a function
        taking ``fields`` and reassigning it to the helper AFTER the constructor would look like a
        single clean hoist while the call actually splatted the incoming argument.
        """
        args = getattr(scope, "args", None)
        if args is None:
            return set()
        names = {
            a.arg
            for a in (
                list(getattr(args, "posonlyargs", []) or [])
                + list(args.args or [])
                + list(args.kwonlyargs or [])
            )
        }
        for extra in (args.vararg, args.kwarg):
            if extra is not None:
                names.add(extra.arg)
        return names

    @staticmethod
    def _statement_chain(scope, call) -> list:
        """``[(block, index), ...]`` from ``scope``'s body down to the statement holding ``call``.

        Each entry is a statement list and the position within it of the statement containing the
        call, which is what lets the caller ask "did a binding run before this" without a real
        control-flow graph: an earlier sibling in a shared block always executes first.
        """
        chain: list = []

        def contains(node) -> bool:
            return any(n is call for n in ast.walk(node))

        def descend(block) -> None:
            for i, stmt in enumerate(block):
                if not contains(stmt):
                    continue
                chain.append((block, i))
                for _field, value in ast.iter_fields(stmt):
                    if (
                        isinstance(value, list)
                        and value
                        and isinstance(value[0], ast.stmt)
                        and any(contains(s) for s in value)
                    ):
                        descend(value)
                        break
                return

        descend(scope.body)
        return chain

    def _binding_dominates(self, scope, name: str, call) -> bool:
        """True when a helper binding of ``name`` provably runs before ``call``.

        Sibling order in one statement list is the only ordering Python guarantees without
        modelling control flow, so that is all this claims: the binding has to be an earlier
        statement in a block that also (transitively) contains the call. A binding tucked inside an
        earlier ``if`` is not a sibling and does not count, which is the point -- a conditional
        hoist can be skipped, leaving the constructor to splat a stale or unbound name.
        """
        for block, index in self._statement_chain(scope, call):
            for stmt in block[:index]:
                if isinstance(stmt, ast.Assign):
                    targets = stmt.targets
                elif isinstance(stmt, ast.AnnAssign):
                    targets = [stmt.target]
                else:
                    continue
                if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
                    continue
                if self._is_runtime_fields_call(stmt.value):
                    return True
        return False

    # The fields this contract asserts reach the response. Overwriting one is as fatal as deleting
    # it: the route answers, so nothing raises, and /status quietly reports a value the backend
    # never produced.
    _PROTECTED_RUNTIME_KEYS = frozenset({"native_context_length"})

    def _mutations_are_safe(self, scope, name: str, call) -> bool:
        """True when nothing in ``scope`` can remove a protected key from ``name``.

        The real call site edits one entry of the hoisted dict before splatting it, so mutation has
        to be allowed -- but only the additive kind. A subscript write to an unprotected key adds or
        replaces something this contract makes no claim about.

        Everything else about the name fails closed, and the rule is stated positively for that
        reason: the ONLY reads of the name allowed anywhere in the scope are the splat itself and
        the object of an allowed subscript write. Enumerating forbidden mutations instead
        (``del``, ``pop``, ``clear``) misses the one that reaches them all -- ``alias = fields``,
        after which every check aimed at ``fields`` looks at the wrong name while the same dict
        loses keys through ``alias``. Passing the dict to a function is the same hole with the
        mutation offsite. A legitimate new pattern will trip this and get a human to re-read the
        contract, which is the cheaper mistake.
        """
        allowed_reads: set = {id(call_kw.value) for call_kw in call.keywords if call_kw.arg is None}
        ok = True

        def walk(node) -> None:
            nonlocal ok
            for child in ast.iter_child_nodes(node):
                if isinstance(child, self._NESTED_SCOPES):
                    continue
                if isinstance(child, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                    if len(targets) > 1 and any(
                        isinstance(t, ast.Name) and t.id == name for t in targets
                    ):
                        # `fields = alias = helper(...)` binds the same dict twice. The binding
                        # check sees one clean helper assignment and every other check is aimed at
                        # `fields`, so `alias.pop(...)` would pass unseen.
                        ok = False
                    for target in targets:
                        if not isinstance(target, ast.Subscript):
                            continue
                        obj = target.value
                        if not (isinstance(obj, ast.Name) and obj.id == name):
                            continue
                        key = target.slice
                        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                            ok = False  # a computed key could be any of the protected ones
                        elif key.value in self._PROTECTED_RUNTIME_KEYS:
                            ok = False
                        else:
                            allowed_reads.add(id(obj))
                walk(child)

        walk(scope)
        if not ok:
            return False

        # Nested scopes ARE walked here, unlike the allow-scan above. A closure that
        # names the dict can do anything to it, and its call site reads nothing
        # syntactically: `def scrub(): fields.pop("native_context_length")` followed by
        # `scrub()` is invisible to both a mutation list and a call-site check. Reading
        # the name inside a closure is therefore itself the disqualifier.
        def check_reads(node) -> None:
            nonlocal ok
            for child in ast.iter_child_nodes(node):
                if (
                    isinstance(child, ast.Name)
                    and child.id == name
                    and not isinstance(child.ctx, ast.Store)
                    and id(child) not in allowed_reads
                ):
                    ok = False
                check_reads(child)

        check_reads(scope)
        return ok

    @staticmethod
    def _route_handler(tree, path: str):
        """The function registered at ``path``, found through its router decorator.

        The decorator is the definition of the endpoint, so matching on it rather than on the
        function's name keeps this pointed at whatever serves ``/status`` after a rename.
        """
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not (isinstance(decorator, ast.Call) and decorator.args):
                    continue
                func = decorator.func
                if not (isinstance(func, ast.Attribute) and func.attr in _HTTP_METHODS):
                    continue
                first = decorator.args[0]
                if isinstance(first, ast.Constant) and first.value == path:
                    return node
        return None

    @staticmethod
    def _returned_calls(scope, class_name: str) -> list:
        """``return ClassName(...)`` in ``scope``'s OWN body, nested scopes excluded.

        Two narrowings, both because the caller only asks whether SOME call qualifies. A nested
        helper is a separate scope the handler may never call, so a hoist left behind in one
        certifies nothing; and a construction that is not returned is a value the caller never
        sees, so it cannot stand in for the response that goes out.
        """
        calls: list = []

        def walk(node) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                    continue
                if isinstance(child, ast.Return) and isinstance(child.value, ast.Call):
                    call = child.value
                    if isinstance(call.func, ast.Name) and call.func.id == class_name:
                        calls.append(call)
                walk(child)

        walk(scope)
        return calls

    def _status_calls_getting_runtime_fields(self) -> list:
        """``InferenceStatusResponse(...)`` calls that actually receive the llama runtime fields.

        A call site may hoist the helper call to overwrite one entry before splatting, since
        passing an overriding keyword alongside ``**fields`` is a TypeError. That is a valid way
        to receive the fields, so accept it -- under three conditions, because a hoist turns one
        expression into a name whose value depends on everything the function does to it:

        1. EVERY binding of the name in the function's own scope is the helper call. Stricter than
           reaching definitions and needs no flow analysis: one hoist passes, any rebind to
           anything else fails, conditional or not. Parameters count as bindings, so a name that
           arrives as an argument and is reassigned later cannot pass on the later assignment.
        2. One of those bindings DOMINATES the call, as an earlier sibling statement. Condition 1
           alone accepts a binding that is conditional or sits after the constructor, where some
           executions reach the splat with a different dict or an unbound name.
        3. No mutation between them can remove a protected key. The splatted dict is deliberately
           edited at the real call site, and a future ``pop`` or ``del`` of
           ``native_context_length`` would leave this contract green while /status stopped
           reporting the field.

        Only the ``/status`` handler is searched, and only what it RETURNS. The caller asserts
        that SOME call qualifies, so anything looser certifies the wrong code: over the whole
        module, an unrelated route hoisting the same helper; inside the handler but in a nested
        helper, one the handler may never call; anywhere but a return, a constructed response that
        is discarded while the real answer goes out without the fields.
        """
        tree = ast.parse(self._source)
        funcs = [
            n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        handler = self._route_handler(tree, "/status")
        if handler is None:
            return []  # no handler to check is not a passing contract
        found = []
        for call in self._returned_calls(handler, "InferenceStatusResponse"):
            for kw in call.keywords:
                if kw.arg is not None:
                    continue
                if self._is_runtime_fields_call(kw.value):
                    found.append(call)
                    break
                if not isinstance(kw.value, ast.Name):
                    continue
                # Innermost function containing this call.
                enclosing = [
                    f for f in funcs if f.lineno <= call.lineno <= (f.end_lineno or call.lineno)
                ]
                if not enclosing:
                    continue
                scope = max(enclosing, key = lambda f: f.lineno)
                name = kw.value.id
                if name in self._parameter_names(scope):
                    continue
                bindings = self._bindings_in_own_scope(scope, name)
                if not bindings or not all(
                    self._is_runtime_fields_call(value) for _node, value in bindings
                ):
                    continue
                if not self._binding_dominates(scope, name, call):
                    continue
                if not self._mutations_are_safe(scope, name, call):
                    continue
                found.append(call)
                break
        return found

    def test_gguf_load_responses_have_field(self):
        """Every GGUF LoadResponse (is_gguf = True) includes native_context_length."""
        blocks = self._find_construction_blocks("LoadResponse")
        gguf_blocks = [b for b in blocks if "is_gguf = True" in b or "is_gguf=True" in b]
        assert (
            len(gguf_blocks) == 1
        ), f"Expected one shared GGUF LoadResponse block, found {len(gguf_blocks)}"
        for i, block in enumerate(gguf_blocks):
            assert (
                "_llama_runtime_fields(llama_backend)" in block
            ), f"GGUF LoadResponse block #{i} missing runtime fields:\n{block[:200]}"
        assert "for name in _InferenceRuntimeFields.model_fields" in self._source

    def test_non_gguf_load_responses_omit_field(self):
        """Non-GGUF LoadResponse blocks do not set native_context_length (defaults to None)."""
        blocks = self._find_construction_blocks("LoadResponse")
        non_gguf = [b for b in blocks if "is_gguf = True" not in b and "is_gguf=True" not in b]
        # Non-GGUF paths shouldn't reference native_context_length
        # (Pydantic defaults it to None, so omitting it is correct).
        for block in non_gguf:
            assert (
                "native_context_length" not in block
            ), f"Non-GGUF LoadResponse should not set native_context_length:\n{block[:200]}"

    def test_non_gguf_load_responses_set_runtime_context_length(self):
        """Non-GGUF LoadResponse blocks report runtime context_length."""
        blocks = self._find_construction_blocks("LoadResponse")
        non_gguf = [b for b in blocks if "is_gguf = True" not in b and "is_gguf=True" not in b]
        assert non_gguf, "Expected at least one non-GGUF LoadResponse block"
        for block in non_gguf:
            assert (
                "context_length" in block
            ), f"Non-GGUF LoadResponse should set context_length:\n{block[:200]}"

    def test_status_path(self):
        """InferenceStatusResponse construction with llama_backend has the field."""
        calls = self._status_calls_getting_runtime_fields()
        assert calls, "No InferenceStatusResponse block with llama_backend has runtime fields"
        assert "for name in _InferenceRuntimeFields.model_fields" in self._source

    # ── the hoist analysis itself, on synthetic sources ──────────────────────
    def _accepts(self, body: str) -> bool:
        """Run the analyser over a synthetic route function instead of the real file."""
        import textwrap

        self._source = (
            '@router.get("/status", response_model = InferenceStatusResponse)\n'
            "async def get_status(llama_backend):\n"
        ) + textwrap.indent(textwrap.dedent(body).strip("\n"), "    ")
        return bool(self._status_calls_getting_runtime_fields())

    def test_the_hoist_analysis_accepts_the_shape_the_route_actually_uses(self):
        # Sibling hoist, then one edit to a field this contract makes no claim about. The three
        # rejection tests below are only meaningful if this stays accepted.
        assert self._accepts(
            """
            if llama_backend is not None:
                fields = _llama_runtime_fields(llama_backend)
                fields["chat_template_override"] = None
                return InferenceStatusResponse(is_gguf = True, **fields)
            """
        )
        # The unhoisted form needs no analysis at all.
        assert self._accepts(
            "return InferenceStatusResponse(**_llama_runtime_fields(llama_backend))"
        )

    def test_the_hoist_analysis_rejects_a_binding_that_can_be_skipped(self):
        """A conditional hoist is not a hoist: the constructor can run without it.

        Every binding is the helper call, so a bindings-only check passes this, and the route
        raises UnboundLocalError on the branch that skipped the assignment. Requiring the binding
        to be an earlier sibling of the call rejects it.
        """
        assert not self._accepts(
            """
            if llama_backend.is_gguf:
                fields = _llama_runtime_fields(llama_backend)
            return InferenceStatusResponse(is_gguf = True, **fields)
            """
        )

    def test_the_hoist_analysis_rejects_a_binding_that_lands_after_the_call(self):
        """A name that arrives as an argument and is rebound later never reaches the splat."""
        import textwrap

        self._source = textwrap.dedent(
            """
            @router.get("/status")
            async def get_status(llama_backend, fields):
                response = InferenceStatusResponse(is_gguf = True, **fields)
                fields = _llama_runtime_fields(llama_backend)
                return response
            """
        )
        assert not self._status_calls_getting_runtime_fields()

    def test_the_hoist_analysis_rejects_dropping_a_protected_field(self):
        """The dict is edited before the splat, so removal has to be checked, not just binding."""
        for destructive in (
            'del fields["native_context_length"]',
            'fields.pop("native_context_length", None)',
            "fields.clear()",
            "fields[_key] = None",  # a computed key could be the protected one
            'fields["native_context_length"] = None',
        ):
            assert not self._accepts(
                f"""
                fields = _llama_runtime_fields(llama_backend)
                {destructive}
                return InferenceStatusResponse(is_gguf = True, **fields)
                """
            ), f"{destructive} left the /status contract green"

    def test_the_hoist_analysis_rejects_reaching_the_dict_by_another_name(self):
        """Naming the dict twice routes every mutation past a check aimed at the first name.

        `alias = fields` then `alias.pop(...)` drops a protected key while nothing touches
        `fields`, and handing the dict to a function moves the same mutation offsite. So the rule
        is positive: the only reads of the splatted name are the splat and the object of an
        allowed subscript write.
        """
        for leak in (
            "alias = fields\n                alias.pop('native_context_length')",
            "_scrub(fields)",
            "return InferenceStatusResponse(is_gguf = True, **fields, extra = fields)",
        ):
            assert not self._accepts(
                f"""
                fields = _llama_runtime_fields(llama_backend)
                {leak}
                return InferenceStatusResponse(is_gguf = True, **fields)
                """
            ), f"{leak!r} left the /status contract green"

    def test_the_hoist_analysis_rejects_a_binding_form_with_no_name_node(self):
        """`except E as fields` deletes the name when the handler ends.

        The exception target is not an `ast.Name` store, so a Name-only walk misses the rebind
        entirely rather than failing closed on it, and a handled-exception path then reaches the
        constructor with `fields` unbound.
        """
        assert not self._accepts(
            """
            fields = _llama_runtime_fields(llama_backend)
            try:
                _probe()
            except Exception as fields:
                pass
            return InferenceStatusResponse(is_gguf = True, **fields)
            """
        )
        # Same for the other name-binding forms that carry no Name node.
        assert not self._accepts(
            """
            fields = _llama_runtime_fields(llama_backend)
            import json as fields
            return InferenceStatusResponse(is_gguf = True, **fields)
            """
        )

    def test_the_hoist_analysis_rejects_a_chained_assignment(self):
        """`fields = alias = helper(...)` binds the same dict under two names at once.

        The binding check sees one clean helper assignment, and every other check is aimed at
        `fields`, so the protected key can leave through `alias` unobserved.
        """
        assert not self._accepts(
            """
            fields = alias = _llama_runtime_fields(llama_backend)
            alias.pop("native_context_length")
            return InferenceStatusResponse(is_gguf = True, **fields)
            """
        )

    def test_the_contract_is_tied_to_the_status_route(self):
        """`test_status_path` asserts only that SOME call qualifies.

        Searched over the whole module, an unrelated route or helper that hoists the same dict
        would satisfy it while the GGUF branch of the real handler dropped the fields. The handler
        is found through its router decorator, so a rename of the function still matches.
        """
        import textwrap

        self._source = textwrap.dedent(
            """
            def _some_helper(llama_backend):
                fields = _llama_runtime_fields(llama_backend)
                return InferenceStatusResponse(is_gguf = True, **fields)

            @router.get("/status", response_model = InferenceStatusResponse)
            async def get_status(llama_backend):
                return InferenceStatusResponse(is_gguf = True)
            """
        )
        assert not self._status_calls_getting_runtime_fields()

        # And with no /status handler at all there is nothing to certify.
        self._source = textwrap.dedent(
            """
            def _some_helper(llama_backend):
                fields = _llama_runtime_fields(llama_backend)
                return InferenceStatusResponse(is_gguf = True, **fields)
            """
        )
        assert not self._status_calls_getting_runtime_fields()

    def test_the_hoist_analysis_rejects_a_qualifying_call_the_route_never_returns(self):
        """A nested helper is a scope the handler may never call, and a discarded construction
        is a value the caller never sees. Either one standing in for the response means the
        real `/status` answer can lose the fields with this contract still green."""
        import textwrap

        # A qualifying hoist inside a nested helper, and a bare response returned.
        self._source = textwrap.dedent(
            """
            @router.get("/status")
            async def get_status(llama_backend):
                def _unused():
                    fields = _llama_runtime_fields(llama_backend)
                    return InferenceStatusResponse(is_gguf = True, **fields)
                return InferenceStatusResponse(is_gguf = True)
            """
        )
        assert not self._status_calls_getting_runtime_fields()

        # A qualifying construction that is built and thrown away.
        self._source = textwrap.dedent(
            """
            @router.get("/status")
            async def get_status(llama_backend):
                fields = _llama_runtime_fields(llama_backend)
                InferenceStatusResponse(is_gguf = True, **fields)
                return InferenceStatusResponse(is_gguf = True)
            """
        )
        assert not self._status_calls_getting_runtime_fields()

    def test_the_hoist_analysis_rejects_a_closure_that_touches_the_dict(self):
        """A closure's call site reads nothing syntactically.

        `def scrub(): fields.pop("native_context_length")` then `scrub()` removes the field
        while no statement in the handler names `fields` at all, so naming it inside a nested
        scope has to be the disqualifier.
        """
        assert not self._accepts(
            """
            fields = _llama_runtime_fields(llama_backend)
            def _scrub():
                fields.pop("native_context_length")
            _scrub()
            return InferenceStatusResponse(is_gguf = True, **fields)
            """
        )

    def test_the_hoist_analysis_rejects_a_declaration_that_shadows_the_dict(self):
        """`def`/`class` bind their own name in the enclosing scope, and `import a.b` binds `a`.

        The body of a `def` is a separate scope, but the name it creates is not, so skipping
        the whole node let a later declaration shadow the hoisted dict unnoticed.
        """
        for shadow in (
            "def fields():\n                    pass",
            "class fields:\n                    pass",
            "import fields.submodule",
        ):
            assert not self._accepts(
                f"""
                fields = _llama_runtime_fields(llama_backend)
                {shadow}
                return InferenceStatusResponse(is_gguf = True, **fields)
                """
            ), f"{shadow!r} left the /status contract green"

    def test_non_gguf_status_path_reports_runtime_context_length(self):
        """Non-GGUF InferenceStatusResponse reports context_length from model_info."""
        blocks = self._find_construction_blocks("InferenceStatusResponse")
        found = False
        for block in blocks:
            if "is_gguf = False" in block and "context_length" in block:
                found = True
                break
        assert found, "No non-GGUF InferenceStatusResponse block with context_length"

    def test_openai_models_listing_reports_context_length(self):
        """/v1/models includes context_length when the backend knows it."""
        assert 'entry["context_length"]' in self._source
        assert 'model_info.get("context_length")' in self._source


# =====================================================================
# E. TestEdgeCases
# =====================================================================


class TestNativeContextEdgeCases:
    """Edge cases for native_context_length."""

    def test_context_length_zero(self, tmp_path, backend):
        """GGUF context_length=0 returns 0, not None."""
        path = make_gguf(tmp_path, "llama", [("context_length", 0, 4)])
        backend._read_gguf_metadata(path)
        assert backend.native_context_length == 0

    def test_context_length_uint32_max(self, tmp_path, backend):
        """2^32 - 1 survives without truncation."""
        val = 2**32 - 1
        path = make_gguf(tmp_path, "llama", [("context_length", val, 4)])
        backend._read_gguf_metadata(path)
        assert backend.native_context_length == val

    def test_context_length_uint64(self, tmp_path, backend):
        """UINT64 type context_length parsed correctly."""
        val = 2**33  # exceeds UINT32 range
        path = make_gguf(tmp_path, "llama", [("context_length", val, 10)])
        backend._read_gguf_metadata(path)
        assert backend.native_context_length == val

    def test_no_context_length_in_gguf(self, tmp_path, backend):
        """GGUF without context_length key yields None."""
        path = make_gguf(tmp_path, "llama", [("block_count", 32, 4)])
        backend._read_gguf_metadata(path)
        assert backend.native_context_length is None

    def test_native_equals_context_when_uncapped(self, backend):
        """Both equal when no VRAM cap applied."""
        backend._context_length = 8192
        assert backend.native_context_length == backend.context_length

    def test_native_survives_parse_then_cap(self, tmp_path, backend):
        """Parse then set effective cap: native unchanged."""
        path = make_gguf(
            tmp_path,
            "llama",
            [
                ("context_length", 131072, 4),
                ("block_count", 32, 4),
                ("attention.head_count", 32, 4),
                ("attention.head_count_kv", 8, 4),
                ("embedding_length", 4096, 4),
            ],
        )
        backend._read_gguf_metadata(path)
        assert backend.native_context_length == 131072

        # Simulate VRAM capping via effective and max.
        backend._effective_context_length = 16384
        backend._max_context_length = 32768
        assert backend.native_context_length == 131072


# =====================================================================
# F. TestCrossPlatform -- binary I/O and serialization
# =====================================================================


class TestCrossPlatform:
    """Binary I/O and serialization correctness across platforms."""

    def test_le_uint32_context_length(self, tmp_path, backend):
        """Little-endian UINT32 parsed correctly."""
        path = make_gguf(tmp_path, "llama", [("context_length", 16384, 4)])
        backend._read_gguf_metadata(path)
        assert backend.native_context_length == 16384

    def test_le_uint64_context_length(self, tmp_path, backend):
        """Little-endian UINT64 parsed correctly."""
        path = make_gguf(tmp_path, "llama", [("context_length", 16384, 10)])
        backend._read_gguf_metadata(path)
        assert backend.native_context_length == 16384

    def test_gguf_magic_le_byte_order(self, tmp_path):
        """Magic 0x46554747 matches GGUF spec (little-endian 'GGUF')."""
        path = tmp_path / "magic_check.gguf"
        buf = io.BytesIO()
        buf.write(struct.pack("<I", 0x46554747))
        raw = buf.getvalue()
        # 'G' = 0x47, 'G' = 0x47, 'U' = 0x55, 'F' = 0x46
        assert raw == b"GGUF"

    def test_json_serialization_deterministic(self):
        """model_dump_json() is consistent across calls."""
        resp = LoadResponse(
            status = "loaded",
            model = "test",
            display_name = "Test",
            inference = {},
            native_context_length = 131072,
        )
        json1 = resp.model_dump_json()
        json2 = resp.model_dump_json()
        assert json1 == json2
        assert '"native_context_length":131072' in json1
