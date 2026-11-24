# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Load-time output a user cannot act on stays out of the default log.

* `import unsloth` on transformers 5 printed `rope_scaling is deprecated`: Unsloth's own
  rope probe writes the alias on purpose. The probe is quiet, the user's writes are not.
* A preinstalled xformers built for another torch (every Kaggle image) printed three
  lines; it is one line now, the fix and the error sit behind UNSLOTH_ENABLE_LOGGING=1.
* A multi-GPU load printed the planner's whole budget table; it is one line now.
"""

import ast
import inspect
import os
import re
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _extract(
    relative_path,
    *names,
    namespace = None,
):
    """Exec just the named top-level functions / assignments of a module (no package import)."""
    tree = ast.parse((ROOT / relative_path).read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            nodes.append(node)
    found = {getattr(n, "name", None) or n.targets[0].id for n in nodes}
    missing = set(names) - found
    assert not missing, f"{relative_path} no longer defines {sorted(missing)}"
    namespace = {"os": os, "re": re, **(namespace or {})}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(relative_path), "exec"), namespace)
    return namespace


# ---------------------------------------------------------------- rope_scaling


def _rope_warnings(record):
    return [
        w
        for w in record
        if issubclass(w.category, FutureWarning) and "rope_scaling" in str(w.message)
    ]


def test_rope_scaling_probe_does_not_warn():
    from unsloth import import_fixes

    with warnings.catch_warnings(record = True) as record:
        warnings.simplefilter("always")
        quiet = import_fixes._transformers_rope_scaling_assignment_drops_theta()
    assert _rope_warnings(record) == [], [str(w.message) for w in record]
    # Same answer as the raw probe: only the message changed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert quiet == import_fixes._probe_rope_scaling_assignment_drops_theta()


def test_user_rope_scaling_write_still_warns_after_probe():
    import transformers
    from transformers import LlamaConfig
    from unsloth import import_fixes

    import_fixes._transformers_rope_scaling_assignment_drops_theta()
    config = LlamaConfig(hidden_size = 64, num_attention_heads = 2, num_key_value_heads = 2)
    with warnings.catch_warnings(record = True) as record:
        warnings.simplefilter("always")
        config.rope_scaling = {"rope_type": "linear", "factor": 2.0}
    # transformers 5 deprecates the alias (a property); 4.x stores a plain attribute and never warned.
    if isinstance(inspect.getattr_static(LlamaConfig, "rope_scaling", None), property):
        assert _rope_warnings(
            record
        ), f"transformers {transformers.__version__}: user write lost its warning"
    else:
        assert _rope_warnings(record) == []


# ---------------------------------------------------------------- xformers


class _XformersLoadError(Exception):
    pass


_CU128_ERROR = (
    "xFormers can't load C++/CUDA extensions. xFormers was built for:\n"
    "    PyTorch 2.10.0+cu128 with CUDA 1208 (you have 2.11.0+cu130)\n"
    "    Python  3.12.12 (you have 3.12.12)"
)


def _notice():
    return _extract("unsloth/models/_utils.py", "_broken_xformers_notice")[
        "_broken_xformers_notice"
    ]


def test_broken_xformers_is_one_line_by_default():
    out = _notice()(_XformersLoadError(_CU128_ERROR), "2.11.0+cu130", device_type = "cuda")
    assert "\n" not in out
    assert out.startswith("Unsloth: xformers does not load with torch 2.11.0+cu130")
    assert "PyTorch attention" in out and "UNSLOTH_ENABLE_LOGGING=1" in out
    assert "install-kernels" not in out and "xFormers can't load" not in out
    assert len(out) < 140


def test_broken_xformers_verbose_keeps_fix_and_error():
    out = _notice()(
        _XformersLoadError(_CU128_ERROR), "2.11.0+cu130", device_type = "cuda", verbose = True
    )
    assert "it was built for 2.10.0+cu128" in out
    assert "unsloth install-kernels xformers" in out
    assert "xFormers can't load C++/CUDA extensions" in out


def test_broken_xformers_silent_when_flash_attention_wins():
    notice = _notice()
    assert (
        notice(
            _XformersLoadError(_CU128_ERROR),
            "2.11.0+cu130",
            device_type = "cuda",
            has_flash_attention = True,
        )
        == ""
    )
    # Not CUDA: nothing to say unless asked.
    assert notice(_XformersLoadError(_CU128_ERROR), "2.11.0", device_type = "xpu") == ""


def test_unsloth_version_guard_message_is_kept():
    guard = NotImplementedError(
        "Unsloth: Xformers 0.0.32.post2 has a broken FA3 dispatch on SM 12.0 GPUs."
    )
    notice = _notice()
    assert notice(guard, "2.11.0", device_type = "cuda") == str(guard)
    assert notice(guard, "2.11.0", device_type = "cuda", has_flash_attention = True) == str(guard)
    verbose = notice(guard, "2.11.0", device_type = "cuda", verbose = True)
    assert str(guard) in verbose and "install-kernels" not in verbose


# ---------------------------------------------------------------- placement report


_GiB = 1024**3


class _Plan:
    device_map = {"model.embed_tokens": 0, "lm_head": 1}

    def summary(self):
        return "Unsloth: placed the model over 2 GPUs: cuda:0 7.26 GiB, cuda:1 6.86 GiB (lm_head on cuda:1)."

    def describe(self):
        return "total weights      : 14.115 GiB\nhead headroom      : 1.000 GiB\nactivation reserve : 2.000 GiB"


class _OldZooPlan(_Plan):
    summary = None


class _FakeCuda:
    def device_count(self):
        return 2

    def mem_get_info(self, index):
        return (14 * _GiB, 16 * _GiB)


def _resolver(plan, monkeypatch):
    import sys
    import types

    module = types.ModuleType("unsloth_zoo.device_map_planner")
    module.plan_device_map_for_pretrained = lambda name, **kwargs: plan
    monkeypatch.setitem(sys.modules, "unsloth_zoo.device_map_planner", module)
    ns = _extract(
        "unsloth/models/loader_utils.py",
        "resolve_unsloth_device_map",
        "_as_bytes",
        "_SIZE_UNITS",
        "_PLANNED_DEVICE_MAPS",
        "UNSLOTH_DEVICE_MAP",
        "UNSLOTH_BALANCED_DEVICE_MAP",
        namespace = {
            "inspect": inspect,
            "torch": types.SimpleNamespace(cuda = _FakeCuda()),
            "DEVICE_TYPE_TORCH": "cuda",
            "is_distributed": lambda: False,
        },
    )
    # The balanced map always splits; "unsloth" would keep a model that fits on one card.
    return lambda: ns["resolve_unsloth_device_map"](ns["UNSLOTH_BALANCED_DEVICE_MAP"], "some/model")


def test_placement_is_one_line_by_default(monkeypatch, capsys):
    monkeypatch.delenv("UNSLOTH_ENABLE_LOGGING", raising = False)
    assert _resolver(_Plan(), monkeypatch)() == _Plan.device_map
    out = capsys.readouterr().out
    assert out.strip() == _Plan().summary()
    for detail in ("head headroom", "activation reserve", "total weights"):
        assert detail not in out


def test_placement_full_table_with_logging(monkeypatch, capsys):
    monkeypatch.setenv("UNSLOTH_ENABLE_LOGGING", "1")
    _resolver(_Plan(), monkeypatch)()
    assert capsys.readouterr().out.strip() == _Plan().describe()


def test_placement_older_zoo_keeps_its_table(monkeypatch, capsys):
    monkeypatch.delenv("UNSLOTH_ENABLE_LOGGING", raising = False)
    _resolver(_OldZooPlan(), monkeypatch)()
    assert capsys.readouterr().out.strip() == _Plan().describe()


def test_loader_prints_the_summary_not_the_table():
    source = (ROOT / "unsloth/models/loader_utils.py").read_text()
    assert "print(plan.describe())" not in source
    assert "print(plan.device_plan.describe())" not in source
    assert source.count("else summary())") == 2
