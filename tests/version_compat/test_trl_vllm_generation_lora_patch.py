# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Behavioural check on `vllm_generation_init_patch()` across TRL `generate` shapes.

`fast_inference = True` GRPO hands its own vLLM engine to TRL's
`VLLMGeneration`, and the adapter only reaches vLLM if `lora_request=` is
passed on the rollout call. That injection used to be a source rewrite of
TRL's `VLLMGeneration.generate`, anchored on a
`self.llm.collective_rpc("reload_weights")` line. TRL 1.10.0 deleted that
call (it invokes `self.sync_weights()` instead), so the anchor matched
nothing and raised, the `lora_request` injection that ran after it in the
same function never happened, and `_init_vllm` / `sync_weights` were already
installed -- rollouts came from the BASE model, exit 0, finite losses, no
warning a user would connect to it.

So these tests are shape-driven, not version-driven: they build a synthetic
`trl.generation.vllm_generation` module whose `_init_vllm` / `sync_weights`
are TRL-shaped (those two are still source-patched) and whose `generate`
reaches the engine the way a given TRL era reaches it, then run the real
patch over it and watch what arrives at a fake engine. No vLLM, no GPU, no
network, and no dependence on which TRL happens to be installed.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import linecache
import os
import sys
import textwrap
import types

import pytest


# Collection must stay cheap and safe where the runtime is absent (the daily-fresh-fetch job collects
# tests/version_compat/ with only pytest).
if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed; this test drives the real patch", allow_module_level = True)


# TRL-shaped method sources ------------------------------------------------ `_init_vllm` and `sync_weights` are still
# --- TRL-shaped method sources ------------------------------------------------
_INIT_VLLM = """
def _init_vllm(self, model):
    if self.mode == "colocate":
        self.llm = LLM(
            model = model.name_or_path,
            enable_sleep_mode = self.enable_sleep_mode,
        )
    else:
        self.llm = None
"""

_SYNC_WEIGHTS = """
def sync_weights(self):
    if self._llm_weights_sleeping:
        self.llm.wake_up()
        self._llm_weights_sleeping = False
    self.llm.collective_rpc("update_weights")
"""

# TRL >= 1.10.0: no `collective_rpc("reload_weights")` anywhere in `generate`.
_GENERATE_TRL_1_10 = """
def generate(self, prompts, **kwargs):
    self.sync_weights()
    return self.llm.generate(prompts, sampling_params = self.sampling_params, use_tqdm = False)
"""

# TRL 0.22.2 era: reloads the checkpoint into the engine before sampling.
_GENERATE_TRL_0_22 = """
def generate(self, prompts, **kwargs):
    if self.enable_sleep_mode:
        self.llm.wake_up()
    self.llm.collective_rpc("reload_weights")
    return self.llm.generate(prompts, sampling_params = self.sampling_params, use_tqdm = False)
"""

_GENERATE_CHAT = """
def generate(self, prompts, **kwargs):
    self.sync_weights()
    return self.llm.chat(prompts, sampling_params = self.sampling_params)
"""

# Server mode: TRL talks to a remote vLLM over HTTP and `self.llm` is None.
_GENERATE_SERVER = """
def generate(self, prompts, **kwargs):
    return self.client.generate(prompts)
"""

# A `sync_weights` the source patch cannot anchor on -- stands in for any future
_SYNC_WEIGHTS_UNPATCHABLE = """
def sync_weights(self, tags = None):
    self.llm.collective_rpc("update_weights")
"""


class FakeEngine:
    """Stand-in for the vLLM `LLM` object unsloth hands to TRL.

    `shared_weights = True` is what marks an engine as unsloth's own, i.e. one
    that already holds the live training weights and therefore needs the LoRA
    passed explicitly on every call.
    """

    shared_weights = True

    def __init__(
        self,
        log,
        shared_weights = True,
    ):
        self.log = log
        self.shared_weights = shared_weights
        self.woken = 0

    def generate(self, *args, **kwargs):
        self.log.append(("generate", kwargs.get("lora_request", "ABSENT")))
        return ["generated"]

    def chat(self, *args, **kwargs):
        self.log.append(("chat", kwargs.get("lora_request", "ABSENT")))
        return ["chatted"]

    def collective_rpc(self, method, *args, **kwargs):
        self.log.append(("collective_rpc", method))
        return "rpc_ran"

    def wake_up(self, *args, **kwargs):
        self.woken += 1
        self.log.append(("wake_up", kwargs.get("tags", None)))


def _rl_replacements():
    from unsloth.models import rl_replacements
    return rl_replacements


def _build_fake_trl(
    monkeypatch,
    generate_src,
    sync_src = _SYNC_WEIGHTS,
    version = "1.10.0",
):
    """Install a synthetic `trl.generation.vllm_generation` and return its class.

    Everything goes through `monkeypatch`, so sys.modules is exactly as it was
    once the test ends no matter which order tests run in.
    """
    class_src = (
        "class VLLMGeneration:\n"
        + "\n".join(
            textwrap.indent(textwrap.dedent(src), "    ")
            for src in (_INIT_VLLM, sync_src, generate_src)
        )
        + "\n"
    )

    # `inspect.getsource` is how the patch reads these methods back out.
    filename = f"<fake_trl_{version}_vllm_generation>"
    monkeypatch.setitem(
        linecache.cache,
        filename,
        (len(class_src), None, [line + "\n" for line in class_src.splitlines()], filename),
    )

    vllm_generation = types.ModuleType("trl.generation.vllm_generation")
    vllm_generation.__file__ = filename
    vllm_generation.__spec__ = importlib.machinery.ModuleSpec(
        name = "trl.generation.vllm_generation",
        loader = None,
        origin = filename,
    )
    # Referenced by the unpatched `_init_vllm` branch; never actually called here.
    vllm_generation.LLM = FakeEngine
    exec(compile(class_src, filename, "exec"), vllm_generation.__dict__)

    generation = types.ModuleType("trl.generation")
    generation.__spec__ = importlib.machinery.ModuleSpec(
        name = "trl.generation",
        loader = None,
        origin = "<fake trl>",
    )
    generation.vllm_generation = vllm_generation

    # A fake `trl` package too, so nothing here depends on a real TRL install: the patch gates on
    # `importlib.util.find_spec("trl")`, which resolves out of sys.modules when the name is already there.
    trl = types.ModuleType("trl")
    trl.__spec__ = importlib.machinery.ModuleSpec(name = "trl", loader = None, origin = "<fake trl>")
    trl.__spec__.submodule_search_locations = []
    trl.__path__ = []
    trl.__version__ = version
    trl.generation = generation

    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "trl.generation", generation)
    monkeypatch.setitem(sys.modules, "trl.generation.vllm_generation", vllm_generation)
    monkeypatch.setattr(
        _rl_replacements(),
        "importlib_version",
        lambda name: version if name == "trl" else "0",
    )
    return vllm_generation.VLLMGeneration


def _make_generation(
    cls,
    log,
    shared_weights = True,
    with_lora = True,
    sleeping = True,
):
    """A `VLLMGeneration` instance shaped the way unsloth's `_init_vllm` leaves it."""
    self = cls.__new__(cls)
    self.llm = FakeEngine(log, shared_weights = shared_weights)
    self.mode = "colocate"
    self.enable_sleep_mode = True
    self._llm_weights_sleeping = sleeping
    self.sampling_params = {"n": 1}
    if with_lora:
        self.unsloth_fast_inference_lora = True
        self._unsloth_load_lora = lambda name, load_tensors = False: f"LORA[{name}|{load_tensors}]"
    return self


def _lora_requests(log, kind = "generate"):
    return [entry[1] for entry in log if entry[0] == kind]


def _lora_name():
    """The adapter directory the patch derives, device suffix and all.

    Recomputed rather than hardcoded: the suffix depends on CUDA_VISIBLE_DEVICES,
    which is set on any CI runner that has a GPU and unset on the ones that do not.
    """
    name = "vllm_gen_lora"
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        name += "_" + os.environ.get("CUDA_VISIBLE_DEVICES", "0").replace(",", "")
    return name


def test_lora_reaches_engine_without_reload_weights_anchor(monkeypatch):
    """TRL >= 1.10.0 `generate`: no reload_weights line, adapter still passed.

    This is the reported bug. Before the fix the anchor regex matched nothing,
    raised, and took the `lora_request` injection down with it, so vLLM sampled
    the base model.
    """
    cls = _build_fake_trl(monkeypatch, _GENERATE_TRL_1_10)
    _rl_replacements().vllm_generation_init_patch()

    log = []
    self = _make_generation(cls, log)
    assert cls.generate(self, ["hello"]) == ["generated"]

    requests = _lora_requests(log)
    assert requests, f"engine.generate was never called: {log}"
    assert all(isinstance(r, str) and r.startswith("LORA[vllm_gen_lora") for r in requests), (
        f"vLLM was asked to sample without the adapter -> rollouts come from the "
        f"BASE model. lora_request values: {requests}"
    )
    assert all(
        "|True" in r for r in requests
    ), f"the adapter tensors were not loaded (load_tensors=True): {requests}"


def test_trl_0_22_shape_keeps_working(monkeypatch):
    """The older `generate`, with the reload_weights call, must not regress."""
    cls = _build_fake_trl(monkeypatch, _GENERATE_TRL_0_22, version = "0.28.0")
    _rl_replacements().vllm_generation_init_patch()

    log = []
    self = _make_generation(cls, log)
    assert cls.generate(self, ["hello"]) == ["generated"]

    requests = _lora_requests(log)
    assert requests and all(
        str(r).startswith("LORA[vllm_gen_lora") for r in requests
    ), f"adapter missing on the 0.22.2-era shape: {log}"
    # The shared engine already holds the live training weights, so a reload_weights would drag the original checkpoint
    assert (
        ("collective_rpc", "reload_weights") not in log
    ), f"reload_weights reached the shared engine and clobbered the trained weights: {log}"


def test_chat_rollouts_get_the_adapter_too(monkeypatch):
    """Conversational rollouts go through `LLM.chat`, same adapter requirement."""
    cls = _build_fake_trl(monkeypatch, _GENERATE_CHAT)
    _rl_replacements().vllm_generation_init_patch()

    log = []
    self = _make_generation(cls, log)
    assert cls.generate(self, [[{"role": "user", "content": "hi"}]]) == ["chatted"]

    requests = _lora_requests(log, kind = "chat")
    assert requests and all(
        str(r).startswith("LORA[vllm_gen_lora") for r in requests
    ), f"adapter missing on llm.chat: {log}"


def test_engine_is_restored_after_the_call(monkeypatch):
    """The override lasts one `generate` call and no longer.

    The same engine object backs `model.fast_generate`, so a leaked override
    would silently attach the training adapter to unrelated user generations.
    """
    cls = _build_fake_trl(monkeypatch, _GENERATE_TRL_1_10)
    _rl_replacements().vllm_generation_init_patch()

    log = []
    self = _make_generation(cls, log)
    cls.generate(self, ["hello"])

    for name in ("generate", "chat", "collective_rpc"):
        assert name not in self.llm.__dict__, f"{name} override outlived the call"

    log.clear()
    self.llm.generate(["direct call, e.g. model.fast_generate"])
    assert _lora_requests(log) == [
        "ABSENT"
    ], f"a direct engine call picked up an injected lora_request: {log}"


def test_pre_existing_instance_attribute_is_put_back(monkeypatch):
    """Restoring must not delete an override the engine already carried."""
    cls = _build_fake_trl(monkeypatch, _GENERATE_TRL_1_10)
    _rl_replacements().vllm_generation_init_patch()

    log = []
    self = _make_generation(cls, log)

    def sentinel(*args, **kwargs):
        log.append(("generate", kwargs.get("lora_request", "ABSENT")))
        return ["generated"]

    self.llm.generate = sentinel
    cls.generate(self, ["hello"])

    assert (
        self.llm.__dict__.get("generate", None) is sentinel
    ), "the engine's own generate was deleted instead of restored"
    assert all(str(r).startswith("LORA[") for r in _lora_requests(log)), log


def test_engine_trl_created_itself_is_left_alone(monkeypatch):
    """No shared weights and no unsloth LoRA -> upstream behaviour, untouched."""
    cls = _build_fake_trl(monkeypatch, _GENERATE_TRL_0_22, version = "0.28.0")
    _rl_replacements().vllm_generation_init_patch()

    log = []
    self = _make_generation(cls, log, shared_weights = False, with_lora = False)
    assert cls.generate(self, ["hello"]) == ["generated"]

    assert _lora_requests(log) == [
        "ABSENT"
    ], f"unsloth injected an adapter into an engine TRL owns: {log}"
    assert ("collective_rpc", "reload_weights") in log, (
        f"TRL's own reload_weights was suppressed on an engine unsloth does not "
        f"share weights with: {log}"
    )


def test_server_mode_falls_through(monkeypatch):
    """Server mode has no `self.llm` at all; the wrapper must just delegate."""
    cls = _build_fake_trl(monkeypatch, _GENERATE_SERVER)
    _rl_replacements().vllm_generation_init_patch()

    log = []
    self = cls.__new__(cls)
    self.llm = None
    self.mode = "server"
    self.client = types.SimpleNamespace(
        generate = lambda prompts, **kwargs: log.append(("client", prompts)) or ["served"],
    )
    assert cls.generate(self, ["hello"]) == ["served"]
    assert log == [("client", ["hello"])]


def test_sleeping_engine_is_woken_before_sync_weights_returns(monkeypatch):
    """TRL >= 1.x only wakes the engine inside `sync_weights`.

    The shared-weights guard returns early from that method, so without an
    explicit wake-up the next rollout runs against a sleeping engine.
    """
    cls = _build_fake_trl(monkeypatch, _GENERATE_TRL_1_10)
    _rl_replacements().vllm_generation_init_patch()

    log = []
    self = _make_generation(cls, log, sleeping = True)
    cls.generate(self, ["hello"])

    assert self.llm.woken == 1, f"engine was never woken: {log}"
    assert self._llm_weights_sleeping is False
    # ...and the sleep flag means it is woken once, not once per rollout.
    cls.generate(self, ["again"])
    assert self.llm.woken == 1, f"engine woken again while already awake: {log}"


def test_failed_patch_rolls_all_three_methods_back(monkeypatch):
    """Half-patched is worse than unpatched, so it must be unreachable.

    `_init_vllm` + `sync_weights` without the adapter injection means no weight
    sync AND no LoRA, which is exactly the silent base-model sampling this fix
    exists to stop.
    """
    cls = _build_fake_trl(monkeypatch, _GENERATE_TRL_1_10, sync_src = _SYNC_WEIGHTS_UNPATCHABLE)
    originals = {name: getattr(cls, name) for name in ("_init_vllm", "sync_weights", "generate")}

    _rl_replacements().vllm_generation_init_patch()

    for name, original in originals.items():
        assert getattr(cls, name) is original, (
            f"{name} stayed patched after a sibling patch failed; VLLMGeneration "
            f"is now half-patched"
        )


def test_patching_twice_does_not_double_wrap(monkeypatch):
    """`vllm_generation_init_patch` runs once per RL trainer patch, so it repeats."""
    cls = _build_fake_trl(monkeypatch, _GENERATE_TRL_1_10)
    rl_replacements = _rl_replacements()

    rl_replacements.vllm_generation_init_patch()
    after_first = {name: getattr(cls, name) for name in ("_init_vllm", "sync_weights", "generate")}
    rl_replacements.vllm_generation_init_patch()

    for name, method in after_first.items():
        assert getattr(cls, name) is method, f"{name} was re-patched on the second call"

    # One layer of wrapping, and it unwraps to TRL's own function so that `inspect.getsource` / `inspect.signature`
    wrapped = getattr(cls.generate, "__wrapped__", None)
    assert wrapped is not None, "the wrapper did not set __wrapped__"
    assert getattr(wrapped, "__wrapped__", None) is None, "generate was wrapped twice"

    log = []
    self = _make_generation(cls, log)
    cls.generate(self, ["hello"])
    assert len(_lora_requests(log)) == 1, f"generate reached the engine twice: {log}"


# --- vLLM signature fidelity -------------------------------------------------- The tests above use an engine whose
# methods take `*args, **kwargs`, so they say nothing about how the injection behaves against vLLM's ACTUAL parameter
# lists.
# Those differ between the two entry points, and that difference matters: LLM.generate(self, prompts, sampling_params =
# None, *, use_tqdm, lora_request, ...) LLM.chat(self, messages, sampling_params = None, use_tqdm = True, lora_request =
# None, ...) `lora_request` is KEYWORD-ONLY on `generate` in every vLLM release from 0.11.0 to 0.27.1, so nothing can
# reach it positionally there.
# On `chat` it is an ordinary positional parameter, and its index has already moved once (`tokenization_kwargs` was
# inserted in 0.18.0).
# --- vLLM signature fidelity --------------------------------------------------
class VLLMSignatureEngine(FakeEngine):
    """`FakeEngine` with vLLM 0.27.1's real parameter lists on both entry points."""

    def generate(
        self,
        prompts,
        sampling_params = None,
        *,
        use_tqdm = True,
        lora_request = None,
        priority = None,
        tokenization_kwargs = None,
        mm_processor_kwargs = None,
    ):
        self.log.append(("generate", lora_request if lora_request is not None else "ABSENT"))
        return ["generated"]

    def chat(
        self,
        messages,
        sampling_params = None,
        use_tqdm = True,
        lora_request = None,
        chat_template = None,
        chat_template_content_format = "auto",
        add_generation_prompt = True,
        continue_final_message = False,
        tools = None,
        chat_template_kwargs = None,
        tokenization_kwargs = None,
        mm_processor_kwargs = None,
    ):
        self.log.append(("chat", lora_request if lora_request is not None else "ABSENT"))
        return ["chatted"]


# TRL reaching `chat` with `lora_request` as the fourth POSITIONAL argument.
_GENERATE_CHAT_POSITIONAL = """
def generate(self, prompts, **kwargs):
    self.sync_weights()
    return self.llm.chat(prompts, self.sampling_params, False, self.caller_lora)
"""


def test_the_adapter_still_reaches_a_signature_accurate_engine(monkeypatch):
    """`generate`'s keyword-only `lora_request` accepts the injected keyword."""
    cls = _build_fake_trl(monkeypatch, _GENERATE_TRL_1_10)
    _rl_replacements().vllm_generation_init_patch()

    log = []
    self = _make_generation(cls, log)
    self.llm = VLLMSignatureEngine(log)

    assert cls.generate(self, ["hello"]) == ["generated"]
    assert _lora_requests(log) == ["LORA[%s|True]" % _lora_name()], log


def test_a_positionally_supplied_adapter_is_not_injected_over(monkeypatch):
    """The caller already filled `lora_request`; a keyword on top is a TypeError.

    Not hypothetical arithmetic: `chat` takes `lora_request` positionally in every
    vLLM release checked, so this is reachable from any TRL that spells the call
    that way. Before the signature check the wrapper raised here.
    """
    cls = _build_fake_trl(monkeypatch, _GENERATE_CHAT_POSITIONAL)
    _rl_replacements().vllm_generation_init_patch()

    log = []
    self = _make_generation(cls, log)
    self.llm = VLLMSignatureEngine(log)
    self.caller_lora = "CALLER_LORA"

    assert cls.generate(self, [[{"role": "user", "content": "hi"}]]) == ["chatted"]
    # The caller's own adapter, untouched, and exactly one call.
    assert _lora_requests(log, kind = "chat") == ["CALLER_LORA"], log


def test_an_explicit_none_adapter_is_overridden(monkeypatch):
    """`lora_request = None` by keyword means base-model rollouts, which is the bug.

    A shared-weights engine holds the BASE weights, so honouring the None is how
    the adapter goes missing in the first place. Only a non-None value from the
    caller is treated as a choice worth keeping.
    """
    cls = _build_fake_trl(
        monkeypatch,
        """
def generate(self, prompts, **kwargs):
    self.sync_weights()
    return self.llm.generate(prompts, sampling_params = self.sampling_params, lora_request = None)
""",
    )
    _rl_replacements().vllm_generation_init_patch()

    log = []
    self = _make_generation(cls, log)
    self.llm = VLLMSignatureEngine(log)

    assert cls.generate(self, ["hello"]) == ["generated"]
    assert _lora_requests(log) == ["LORA[%s|True]" % _lora_name()], log
