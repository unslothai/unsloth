# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""A Hub adapter's base_model_name_or_path must not send the load to a Windows network share."""

import ast
import os
import pathlib
import types

import pytest

LOADER = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "loader.py"


def _helper(name = "_refuse_network_share_adapter_base"):
    tree = ast.parse(LOADER.read_text(encoding = "utf-8"))
    fns = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_refuse_network_share_")
    ]
    ns = {"os": os}
    exec(compile(ast.Module(body = fns, type_ignores = []), str(LOADER), "exec"), ns)
    return ns[name]


@pytest.fixture
def windows(monkeypatch):
    # unsloth_zoo's own _is_network_share with the Windows branch on; read from its source so the
    # package __init__ (which needs an accelerator) never runs.
    import importlib.util
    import re
    import sys

    spec = importlib.util.find_spec("unsloth_zoo")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("unsloth_zoo not installed")
    source = pathlib.Path(list(spec.submodule_search_locations)[0], "vision_utils.py").read_text()
    tree = ast.parse(source)
    wanted = {"_is_network_share", "_LOCAL_SHARE_HOSTS"}
    body = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in wanted)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in wanted for t in n.targets))
    ]
    if not any(isinstance(n, ast.FunctionDef) for n in body):
        pytest.skip("unsloth_zoo without _is_network_share")
    ns = {"re": re, "_WINDOWS": True}
    exec(compile(ast.Module(body = body, type_ignores = []), "vision_utils.py", "exec"), ns)
    monkeypatch.setitem(sys.modules, "unsloth_zoo", types.ModuleType("unsloth_zoo"))
    monkeypatch.setitem(
        sys.modules,
        "unsloth_zoo.vision_utils",
        types.SimpleNamespace(_is_network_share = ns["_is_network_share"]),
    )


def _config(base):
    return types.SimpleNamespace(base_model_name_or_path = base)


@pytest.mark.parametrize("base", [r"\\evil.example\share\model", "//evil.example/share/model"])
def test_hub_adapter_with_unc_base_is_refused(windows, base):
    with pytest.raises(ValueError, match = "network share"):
        _helper()("someone/adapter", _config(base))


def test_hub_adapter_with_hub_base_and_local_adapter_are_unchanged(windows, tmp_path):
    guard = _helper()
    guard("someone/adapter", _config("unsloth/Llama-3.2-1B-Instruct"))
    # A local adapter folder is the user's own; a base on their own fileserver keeps working.
    guard(str(tmp_path), _config(r"\\fileserver\models\base"))


def test_both_loaders_check_before_the_base_is_used():
    source = LOADER.read_text(encoding = "utf-8")
    calls = [
        i
        for i in range(len(source))
        if source.startswith(
            "    _refuse_network_share_adapter_base(modelscope_repo_id or model_name, peft_config)",
            i,
        )
    ]
    uses = [
        i
        for i in range(len(source))
        if source.startswith("_model_type_config = peft_config if peft_config is not None", i)
    ]
    assert len(calls) == 2 and len(uses) == 2
    assert all(c < u for c, u in zip(calls, uses))


@pytest.mark.parametrize("peft_type", ["LORA", "NOT_A_PEFT_TYPE"])
def test_raw_adapter_config_with_unc_base_is_refused(windows, monkeypatch, tmp_path, peft_type):
    # transformers swaps in base_model_name_or_path from the raw JSON, whether or not PEFT can parse it.
    import json

    utils = pytest.importorskip("transformers.utils")
    config = tmp_path / "adapter_config.json"
    monkeypatch.setattr(utils, "find_adapter_config_file", lambda name, **kwargs: str(config))
    guard = _helper("_refuse_network_share_raw_adapter_base")
    config.write_text(
        json.dumps({"peft_type": peft_type, "base_model_name_or_path": r"\\evil.example\share\m"})
    )
    with pytest.raises(ValueError, match = "network share"):
        guard("someone/model", token = None)
    config.write_text(json.dumps({"peft_type": peft_type, "base_model_name_or_path": "owner/base"}))
    guard("someone/model", token = None)
    monkeypatch.setattr(utils, "find_adapter_config_file", lambda name, **kwargs: None)
    guard("someone/model")
    monkeypatch.setattr(utils, "find_adapter_config_file", lambda name, **kwargs: 1 / 0)
    guard(str(tmp_path))


def test_routes_that_hand_a_hub_id_to_transformers_check_the_raw_config():
    source = LOADER.read_text(encoding = "utf-8")
    calls = [
        i
        for i in range(len(source))
        if source.startswith("_refuse_network_share_raw_adapter_base(\n", i)
    ]
    assert len(calls) == 2
    for c in calls:
        assert "revision = _resolved_revision(model_name, base_revision," in source[c : c + 300]
    st = LOADER.with_name("sentence_transformer.py").read_text(encoding = "utf-8")
    checks = [
        i for i in range(len(st)) if st.startswith("_refuse_network_share_raw_adapter_base(\n", i)
    ]
    # The third SentenceTransformer(...) is built from already loaded modules, not from a repo id.
    loads = [
        i
        for i in range(len(st))
        if st.startswith("st_model = SentenceTransformer(", i) and "model_name" in st[i : i + 70]
    ]
    assert len(checks) == 2 and len(loads) == 2
    for c in checks:
        assert "SENTENCE_TRANSFORMERS_HOME" in st[c : st.index("SentenceTransformer(", c)]
    assert all(c < l for c, l in zip(checks, loads))


def test_older_unsloth_zoo_still_refuses_plain_unc(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "unsloth_zoo", types.ModuleType("unsloth_zoo"))
    monkeypatch.setitem(
        sys.modules, "unsloth_zoo.vision_utils", types.ModuleType("unsloth_zoo.vision_utils")
    )
    monkeypatch.setattr(os, "name", "nt")
    guard = _helper()
    with pytest.raises(ValueError, match = "network share"):
        guard("someone/adapter", _config(r"\\evil.example\share\model"))
    guard("someone/adapter", _config("unsloth/Llama-3.2-1B-Instruct"))
    guard("someone/adapter", _config(r"\\wsl$\Ubuntu\models\base"))


def test_text_diffusion_loader_refuses_a_hub_adapter_with_unc_base(monkeypatch):
    # FastModel hands text-diffusion repos to FastDiffusionModel, which resolves the adapter itself.
    pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    import unsloth.models.diffusion as diffusion
    import unsloth_zoo.vision_utils as vision_utils

    monkeypatch.setattr(vision_utils, "_WINDOWS", True)
    adapter = types.SimpleNamespace(
        base_model_name_or_path = r"\\evil.example\share\model", revision = None
    )
    monkeypatch.setattr(peft.PeftConfig, "from_pretrained", lambda *args, **kwargs: adapter)
    opened = []

    def load_config(name, *args, **kwargs):
        opened.append(name)
        raise RuntimeError("base opened")

    monkeypatch.setattr(diffusion, "_load_diffusion_config", load_config)
    with pytest.raises(ValueError, match = "network share"):
        diffusion.FastDiffusionModel.from_pretrained("someone/adapter")
    assert opened == []
