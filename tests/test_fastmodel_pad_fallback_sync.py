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

"""FastModel's patch_tokenizer fallback must return a model whose config pad id is the returned tokenizer's pad.

An older unsloth_zoo fix_pad_token added a pad, grew the embedding, then raised; vision.py swallowed that, reloaded a
fresh tokenizer without the added pad and left config.pad_token_id naming the dropped token. Offline: the helper is
extracted from source, so no unsloth import (and no GPU) is needed.
"""

import ast
import os
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_UTILS = os.path.join(ROOT, "unsloth", "tokenizer_utils.py")
VISION = os.path.join(ROOT, "unsloth", "models", "vision.py")


def _load_helper():
    tree = ast.parse(open(TOKENIZER_UTILS, encoding = "utf-8").read())
    node = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_sync_pad_token_id"),
        None,
    )
    assert node is not None, "unsloth/tokenizer_utils.py has no _sync_pad_token_id"
    namespace = {}
    exec(compile(ast.Module(body = [node], type_ignores = []), TOKENIZER_UTILS, "exec"), namespace)
    return namespace["_sync_pad_token_id"]


class _Cfg(types.SimpleNamespace):
    def update(
        self,
        mapping = None,
        **kwargs,
    ):
        for key, value in {**(mapping or {}), **kwargs}.items():
            setattr(self, key, value)


class _Model:
    def __init__(self, rows, pad_token_id):
        self.rows = rows
        self.config = _Cfg(pad_token_id = pad_token_id)
        self.generation_config = _Cfg(pad_token_id = pad_token_id)

    def get_input_embeddings(self):
        return types.SimpleNamespace(weight = types.SimpleNamespace(shape = (self.rows, 8)))


def _tok(pad_token_id):
    return types.SimpleNamespace(pad_token_id = pad_token_id)


def test_stale_config_pad_follows_the_fallback_tokenizer():
    # The reported state: 258 rows, config pad 257 (the dropped <｜PAD▁TOKEN｜>), fallback tokenizer pads with id 256.
    sync = _load_helper()
    model = _Model(rows = 258, pad_token_id = 257)
    assert sync(model, _tok(256)) is model
    assert model.config.pad_token_id == 256 and model.generation_config.pad_token_id == 256


def test_processor_inner_tokenizer_is_used():
    sync = _load_helper()
    model = _Model(rows = 258, pad_token_id = 257)
    sync(model, types.SimpleNamespace(tokenizer = _tok(256), pad_token_id = None))
    assert model.config.pad_token_id == 256


def test_out_of_range_or_missing_pad_leaves_config_alone():
    sync = _load_helper()
    for pad in (300, None, -1, True):
        model = _Model(rows = 258, pad_token_id = 257)
        sync(model, _tok(pad))
        assert model.config.pad_token_id == 257 and model.generation_config.pad_token_id == 257
    assert sync(None, _tok(1)) is None


def test_vision_fallback_syncs_after_repatching():
    tree = ast.parse(open(VISION, encoding = "utf-8").read())
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            args = [a.id for a in node.args if isinstance(a, ast.Name)]
            if node.func.id == "patch_tokenizer" and args == ["model", "_fallback_tok"]:
                calls.append(("patch", node.lineno))
            elif node.func.id == "_sync_pad_token_id" and args == ["model", "_fallback_tok"]:
                calls.append(("sync", node.lineno))
    kinds = [kind for kind, _ in sorted(calls, key = lambda c: c[1])]
    assert kinds == ["patch", "sync"], kinds
