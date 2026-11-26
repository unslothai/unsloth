"""GRPO ScaleRL (arXiv 2510.13786): No-Positive-Resampling sampler + pass-rate counters, truncation warning."""

import ast
import inspect
import logging
import os
import textwrap
import types
from collections import defaultdict

import pytest
import torch

from unsloth.models import rl_replacements as R  # noqa: E402

# Optional: a checkout holding temp/upstream_rl/trl (TRL main) to check the anchors there too.
WS = os.environ.get("UNSLOTH_TEST_UPSTREAM_ROOT", "")


def _trl_sources():
    import importlib.util

    trl_dir = importlib.util.find_spec("trl").submodule_search_locations[0]
    out = {"installed": open(os.path.join(trl_dir, "trainer", "grpo_trainer.py")).read()}
    up = os.path.join(WS, "temp/upstream_rl/trl/trl/trainer/grpo_trainer.py")
    if os.path.exists(up):
        out["upstream"] = open(up).read()
    return out


def _method(src, name):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    lines = src.splitlines(keepends = True)
                    start = item.lineno - 1 - len(item.decorator_list)
                    return "".join(lines[start : item.end_lineno])
    return None


@pytest.mark.parametrize("which", ["installed", "upstream"])
def test_hooks_install_on_trl_sources(which):
    srcs = _trl_sources()
    if which not in srcs:
        pytest.skip("no upstream TRL clone")
    src = srcs[which]
    sampler = _method(src, "_get_train_sampler")
    out = R.grpo_trainer_scalerl_hooks("_get_train_sampler", sampler)
    assert out.count("return _unsloth_grpo_npr_sampler(self, RepeatSampler(") == 1
    ast.parse(textwrap.dedent(out))
    found_trunc = found_npr = False
    for name in ("_generate", "_generate_and_score_completions"):
        body = _method(src, name)
        new = R.grpo_trainer_scalerl_hooks(name, body)
        ast.parse(textwrap.dedent(new))
        found_trunc |= "_unsloth_grpo_truncation_warn(self, mode)" in new
        found_npr |= "_unsloth_grpo_npr_record(self, mode, inputs, rewards_per_func)" in new
    assert found_trunc and found_npr
    # Other methods pass through untouched.
    other = _method(src, "_get_eval_sampler")
    assert R.grpo_trainer_scalerl_hooks("_get_eval_sampler", other) == other


def _repeat_sampler(
    n,
    G,
    prompts_per_chunk,
    repeat,
    seed = 0,
):
    from trl.trainer.utils import RepeatSampler
    data = [{"prompt": [{"role": "user", "content": f"q{i}"}], "answer": str(i)} for i in range(n)]
    return data, RepeatSampler(
        data,
        mini_repeat_count = G,
        batch_size = prompts_per_chunk,
        repeat_count = repeat,
        shuffle = True,
        seed = seed,
    )


def _trainer(G):
    t = types.SimpleNamespace()
    t.num_generations = G
    t.reward_weights = torch.tensor([1.0, 0.5])
    t.accelerator = types.SimpleNamespace(num_processes = 1)
    t._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    return t


def test_off_by_default_identity(monkeypatch):
    monkeypatch.delenv("UNSLOTH_GRPO_NO_POSITIVE_RESAMPLING", raising = False)
    _, s = _repeat_sampler(12, 2, 3, 2)
    cls = type(s)
    assert R._unsloth_grpo_npr_sampler(_trainer(2), s) is s and type(s) is cls
    t = _trainer(2)
    R._unsloth_grpo_npr_record(t, "train", [{"prompt": "a"}] * 2, torch.ones(2, 2))
    assert not hasattr(t, "_unsloth_npr_stats")


def test_same_stream_without_solved(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_NO_POSITIVE_RESAMPLING", "1")
    _, a = _repeat_sampler(13, 4, 3, 2, seed = 5)
    _, b = _repeat_sampler(13, 4, 3, 2, seed = 5)
    R._unsloth_grpo_npr_sampler(_trainer(4), b)
    assert list(a) == list(b)
    assert len(b) == len(a)


def test_solved_prompts_dropped_next_epoch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_NO_POSITIVE_RESAMPLING", "1")
    G, per_chunk, repeat = 4, 3, 2
    data, s = _repeat_sampler(15, G, per_chunk, repeat, seed = 1)
    t = _trainer(G)
    s = R._unsloth_grpo_npr_sampler(t, s)
    epoch1 = list(s)
    # One generation batch: prompts 0..4, rewards [correct, format]; prompt 1 and 3 pass 4 / 4,
    # prompt 2 passes 3 / 4 (0.75 < 0.9), prompt 4 has one unscored completion and 3 / 3 passes.
    inputs, rows = [], []
    for i in range(5):
        for g in range(G):
            inputs.append(data[i])
            if i in (1, 3):
                rows.append([1.0, 0.0])
            elif i == 2:
                rows.append([1.0, 0.0] if g < 3 else [0.0, 1.0])
            elif i == 4:
                rows.append([float("nan"), float("nan")] if g == 0 else [1.0, 0.2])
            else:
                rows.append([0.0, 1.0])
    R._unsloth_grpo_npr_record(t, "train", inputs, torch.tensor(rows))
    solved_keys = {R._unsloth_grpo_npr_key(data[i]["prompt"]) for i in (1, 3, 4)}
    assert t._unsloth_npr_solved == solved_keys
    assert t._metrics["train"]["npr/solved_prompts"] == [3.0]
    epoch2 = list(s)
    seen = set(epoch2)
    assert seen.isdisjoint({1, 3, 4}) and seen
    # Layout: each prompt repeated G times consecutively; chunks of per_chunk prompts repeated `repeat` times.
    assert len(epoch2) % (G * per_chunk * repeat) == 0
    for i in range(0, len(epoch2), G):
        assert len(set(epoch2[i : i + G])) == 1
    block = G * per_chunk
    for c in range(0, len(epoch2), block * repeat):
        assert epoch2[c : c + block] == epoch2[c + block : c + 2 * block]
    # Epoch 1 had every prompt (15 // 3 full chunks).
    assert set(epoch1) == set(range(15))


def test_pass_rate_accumulates_across_batches(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_NO_POSITIVE_RESAMPLING", "1")
    monkeypatch.setenv("UNSLOTH_GRPO_NPR_PASS_RATE", "0.75")
    t = _trainer(2)
    x = {"prompt": "p"}
    R._unsloth_grpo_npr_record(t, "train", [x, x], torch.tensor([[1.0, 0.0], [0.0, 0.0]]))  # 1 / 2
    assert not t._unsloth_npr_solved
    R._unsloth_grpo_npr_record(t, "train", [x, x], torch.tensor([[1.0, 0.0], [1.0, 0.0]]))  # 3 / 4
    assert t._unsloth_npr_solved == {R._unsloth_grpo_npr_key("p")}
    R._unsloth_grpo_npr_record(
        t, "eval", [x, x], torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    )  # eval ignored
    assert t._unsloth_npr_stats[R._unsloth_grpo_npr_key("p")] == (3.0, 4.0)


def _trunc_trainer(rate, step = 0):
    t = types.SimpleNamespace(
        args = types.SimpleNamespace(max_completion_length = 256),
        state = types.SimpleNamespace(global_step = step),
    )
    t._metrics = {"train": defaultdict(list)}
    t._metrics["train"]["completions/clipped_ratio"].append(rate)
    return t


def test_truncation_warning(monkeypatch, caplog):
    monkeypatch.delenv("UNSLOTH_GRPO_TRUNCATION_WARN", raising = False)
    from unsloth_zoo.log import logger as zl

    zl.propagate = True
    with caplog.at_level(logging.WARNING):
        t = _trunc_trainer(0.25, 3)
        R._unsloth_grpo_truncation_warn(t, "train")
        assert "25% of completions hit max_completion_length" in caplog.text
        caplog.clear()
        t.state.global_step = 10
        R._unsloth_grpo_truncation_warn(t, "train")  # rate limited (< 50 steps)
        assert caplog.text == ""
        t.state.global_step = 60
        R._unsloth_grpo_truncation_warn(t, "train")
        assert "hit max_completion_length" in caplog.text
        caplog.clear()
        R._unsloth_grpo_truncation_warn(_trunc_trainer(0.05), "train")
        R._unsloth_grpo_truncation_warn(_trunc_trainer(0.5), "eval")
        monkeypatch.setenv("UNSLOTH_GRPO_TRUNCATION_WARN", "0")
        R._unsloth_grpo_truncation_warn(_trunc_trainer(0.9), "train")
        assert caplog.text == ""


def test_pre_items_are_self_contained():
    # These are copied into the generated trainer: only names it defines may be used.
    for fn in (
        R._unsloth_grpo_npr_settings,
        R._unsloth_grpo_npr_key,
        R._unsloth_grpo_npr_record,
        R._unsloth_grpo_npr_sampler,
        R._unsloth_grpo_truncation_warn,
    ):
        src = inspect.getsource(fn)
        assert "_warn_once(" not in src and "logger.warning(" not in src.replace(
            "_unsloth_logger.warning(", ""
        )
