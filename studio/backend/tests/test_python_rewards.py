# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import gc
import json
import os
import signal
import sys
import threading
import time
import weakref
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.training import python_rewards, rewards, rl
from core.training.python_rewards import PythonRewardError, RewardWorker, make_python_reward_funcs
from routes.rewards import router
from utils.account_context import AccountContext, bind_account

NOTEBOOK = (Path(__file__).parent / "fixtures" / "notebook_rewards" / "qwen25_gsm8k.py").read_text(
    "utf-8"
)
NOTEBOOK_FUNCS = (
    "xmlcount_reward_func",
    "soft_format_reward_func",
    "strict_format_reward_func",
    "int_reward_func",
    "correctness_reward_func",
)
GOOD = "<reasoning>\n48 + 24 = 72\n</reasoning>\n<answer>\n72\n</answer>\n"
TEXTS = [
    GOOD,
    "<reasoning>\nsome work\n</reasoning>\n<answer>\n73\n</answer>\n",
    "<reasoning>x</reasoning> <answer>72</answer>",
    "<reasoning>\nwork\n</reasoning>\n<answer>\n72\n</answer>\ntrailing text after",
    "The answer is 72.",
    "",
    "<answer>\nseventy-two\n</answer>",
    "<reasoning>\néè unicode ✓\n</reasoning>\n<answer>\n72\n</answer>\n",
]


def _python_md(
    name: str,
    code: str,
    entry: str = "reward",
) -> str:
    return f"---\nname: {name}\nkind: python\nentry: {entry}\ndescription: test\n---\n```python\n{code}```\n"


def _spec(
    name: str,
    code: str,
    entry: str = "reward",
) -> dict:
    return rewards.parse_reward_markdown(_python_md(name, code, entry))


@pytest.fixture
def worker_factory():
    # "full" skips the OS sandbox so these tests check the protocol, not the host they run on.
    started = []

    def make(specs, **kwargs):
        worker = RewardWorker(specs, mode = "full", **kwargs).start()
        started.append(worker)
        return worker

    yield make
    for worker in started:
        worker.close()


def test_a_closed_worker_is_not_kept_alive_by_its_atexit_hook():
    # Studio is long-lived: a hook per Try it worker would keep every closed worker and its code.
    specs = [_spec("one", "def reward(completions, **kw):\n    return [1.0]\n")]
    worker = RewardWorker(specs, mode = "full").start()
    worker.close()
    ref = weakref.ref(worker)
    del worker
    deadline = time.monotonic() + 5
    while ref() is not None and time.monotonic() < deadline:
        gc.collect()
        time.sleep(0.05)
    assert ref() is None


def test_python_reward_round_trips_through_markdown():
    spec = _spec(
        "counter", "def reward(completions, **kwargs):\n    return [1.0] * len(completions)\n"
    )
    assert spec["kind"] == "python" and spec["entry"] == "reward" and spec["rule"] is None
    again = rewards.parse_reward_markdown(rewards.render_reward_markdown(spec))
    assert again == spec


def test_unfenced_and_crlf_python_bodies_parse():
    raw = "---\r\nname: t\r\nkind: python\r\n---\r\ndef reward(completions, **kw):\r\n    return [0.0]\r\n"
    assert "def reward" in rewards.parse_reward_markdown(raw)["code"]


@pytest.mark.parametrize(
    "code, entry, message",
    [
        ("def reward(:\n", "reward", "syntax error"),
        ("import os\n", "reward", "named 'reward'"),
        ("def other(completions): return []\n", "reward", "named 'reward'"),
        ("def reward(c): return []\n", "not an identifier", "entry must be"),
        ("async def reward(completions): return []\n", "reward", "is async"),
    ],
)
def test_bad_python_rewards_are_refused(code, entry, message):
    with pytest.raises(rewards.RewardError, match = message):
        rewards.parse_reward_markdown(_python_md("t", code, entry))


def test_notebook_rewards_score_the_same_in_the_worker(worker_factory):
    specs = [_spec(name.replace("_", "-"), NOTEBOOK, name) for name in NOTEBOOK_FUNCS]
    worker = worker_factory(specs)
    namespace: dict = {}
    exec(compile(NOTEBOOK, "notebook", "exec"), namespace)
    prompts = [[{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]] * len(TEXTS)
    completions = [[{"role": "assistant", "content": t}] for t in TEXTS]
    answer = ["72"] * len(TEXTS)
    names = [s["name"] for s in specs]
    got = worker.score(names, prompts, completions, {"answer": answer})
    for spec, func in zip(specs, NOTEBOOK_FUNCS):
        expected = namespace[func](prompts = prompts, completions = completions, answer = answer)
        assert got[spec["name"]] == pytest.approx(expected), func


def test_notebook_correctness_scores_a_raw_gsm8k_answer(worker_factory):
    from datasets import Dataset

    raw = Dataset.from_list(
        [{"question": "q", "answer": "48 + 24 = <<48+24=72>>72 clips.\n#### 72"}]
    )
    row = rl.format_rl_dataset(raw, "grpo", keep_columns = ("*",))[0][0]
    specs = [
        _spec("correctness", NOTEBOOK, "correctness_reward_func"),
        _spec(
            "raw",
            "def reward(completions, answer_raw, **kw):\n    return [float(len(answer_raw[0]))]\n",
        ),
    ]
    got = make_python_reward_funcs(specs, worker_factory(specs))
    prompts, completions = [row["prompt"]], [[{"role": "assistant", "content": GOOD}]]
    kwargs = {"answer": [row["answer"]]}
    assert got[0](prompts = prompts, completions = completions, **kwargs) == [2.0]
    assert got[1](prompts = prompts, completions = completions, **kwargs) == [float(len(row["answer"]))]
    # A bare answer passes through, and a real answer_raw column is not replaced.
    worker = worker_factory(specs)
    assert worker.score(
        ["correctness", "raw"], prompts, completions, {"answer": ["72"], "answer_raw": ["abc"]}
    ) == {"correctness": [2.0], "raw": [3.0]}


def test_answer_raw_only_reaches_a_reward_that_takes_it(worker_factory):
    code = "def reward(prompts, completions, answer):\n    return [float(answer[0])]\n"
    specs = [_spec("strict", code)]
    worker = worker_factory(specs)
    completions = [[{"role": "assistant", "content": GOOD}]]
    got = worker.score(["strict"], [[]], completions, {"answer": ["x\n#### 72"]})
    assert got == {"strict": [72.0]}


def test_a_rendered_prompt_reaches_python_rewards_as_its_turns():
    from datasets import Dataset

    from core.training.rewards import PREFILL_COLUMN, PROMPT_TURNS_COLUMN

    class Tok:
        chat_template = "{% if enable_thinking is defined %}{% endif %}"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
            return "".join(f"<{m['role']}>{m['content']}" for m in messages)

    turns = [{"role": "system", "content": "S"}, {"role": "user", "content": "2+2?"}]
    ds = Dataset.from_list([{"prompt": turns, "answer": "4"}])
    out, rendered = rl.render_grpo_prompts(ds, Tok(), False, keep_turns = True)
    assert rendered and isinstance(out[0]["prompt"], str)
    assert json.loads(out[0][PROMPT_TURNS_COLUMN]) == turns
    assert PROMPT_TURNS_COLUMN not in rl.render_grpo_prompts(ds, Tok(), False)[0].column_names

    seen = {}

    class Worker:
        def score(self, names, prompts, completions, kwargs):
            seen.update(prompts = prompts, completions = completions, kwargs = kwargs)
            return {names[0]: [1.0]}

    (fn,) = python_rewards.make_python_reward_funcs([{"name": "r"}], Worker())
    fn(
        prompts = [out[0]["prompt"]],
        completions = ["4"],
        answer = ["4"],
        **{PROMPT_TURNS_COLUMN: [turns], PREFILL_COLUMN: ["It is "]},
    )
    assert seen["prompts"] == [turns]
    assert seen["completions"] == [[{"role": "assistant", "content": "It is 4"}]]
    assert seen["kwargs"] == {"answer": ["4"]}
    # Arrow pads each message with every field another row's message has, as None.
    padded = [{**m, "tool_calls": None, "name": None} for m in turns]
    fn(prompts = ["x"], completions = ["4"], **{PROMPT_TURNS_COLUMN: [padded]})
    assert seen["prompts"] == [turns]
    fn(prompts = [padded], completions = [[{"role": "assistant", "content": "4"}]])
    assert seen["prompts"] == [turns]


def test_rendered_turns_drop_padded_tool_call_arguments():
    from datasets import Dataset

    from core.training.rewards import PROMPT_TURNS_COLUMN

    class Tok:
        chat_template = "{{ messages }}"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
            return str(messages)

    tool = {
        "type": "function",
        "function": {
            "name": "add",
            "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}},
        },
    }

    def call(args):
        fn = {"name": "add", "arguments": args}
        calls = [{"type": "function", "function": fn}]
        return [{"role": "assistant", "content": "", "tool_calls": calls}]

    rows = [
        {"prompt": [{"role": "user", "content": "q"}, *call({"a": 1})], "tools": [tool]},
        {"prompt": [{"role": "user", "content": "q"}, *call({"b": 2})], "tools": [tool]},
    ]
    out, rendered = rl.render_grpo_prompts(Dataset.from_list(rows), Tok(), None, keep_turns = True)
    assert rendered
    seen = {}

    class Worker:
        def score(self, names, prompts, completions, kwargs):
            seen["prompts"] = prompts
            return {names[0]: [1.0]}

    (fn,) = python_rewards.make_python_reward_funcs([{"name": "r"}], Worker())
    turns = {PROMPT_TURNS_COLUMN: [out[0][PROMPT_TURNS_COLUMN]]}
    fn(prompts = [out[0]["prompt"]], completions = ["x"], **turns)
    args = seen["prompts"][0][-1]["tool_calls"][0]["function"]["arguments"]
    assert args == {"a": 1}


def test_superseded_previews_are_bounded_and_not_kept_warm(monkeypatch):
    release, started = threading.Event(), []

    class Slow:
        isolation = {}

        def __init__(self):
            self.closed = False
            started.append(self)

        def score_each(self, names, *_):
            release.wait(10)
            return {names[0]: [1.0]}, {}

        def close(self):
            self.closed = True

    monkeypatch.setattr(python_rewards, "_start_preview_worker", lambda specs, mode: Slow())
    monkeypatch.setattr(python_rewards, "_previews", {})

    def spec(n):
        return [{"name": "r", "entry": "reward", "code": f"# {n}"}]

    def run(n):
        python_rewards.preview_python_scores(spec(n), "x", {}, owner = "acct")

    threads = [threading.Thread(target = run, args = (n,)) for n in range(2)]
    for t in threads:
        t.start()
        time.sleep(0.3)
    with pytest.raises(python_rewards.PythonRewardError, match = "still scoring"):
        run(2)
    release.set()
    for t in threads:
        t.join(10)
    assert len(started) == 2 and started[0].closed and not started[1].closed
    python_rewards._close_preview()


def test_resume_scores_with_the_stored_python_reward_code():
    from routes.training import _stored_reward_specs

    code = "def reward(**kw): ..."
    py = {"name": "len", "kind": "python", "entry": "reward", "code": code, "rule": None}
    stored = {"reward_specs": [py, {"name": "half", "kind": "python", "rule": None}]}
    assert _stored_reward_specs(stored) == {"len": py}


def test_python_rewards_get_the_row_tools_grpo_rendering_drops():
    from datasets import Dataset

    from core.training.rewards import TOOLS_COLUMN

    class Tok:
        chat_template = "{{ messages }}"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
            return str(messages)

    tool = {"type": "function", "function": {"name": "add", "parameters": {"type": "object"}}}
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": "q"}], "tools": [tool]}])
    out, rendered = rl.render_grpo_prompts(ds, Tok(), None, keep_turns = True)
    assert rendered and "tools" not in out.column_names
    seen = {}

    class Worker:
        def score(self, names, prompts, completions, kwargs):
            seen.update(kwargs)
            return {names[0]: [1.0]}

    (fn,) = python_rewards.make_python_reward_funcs([{"name": "r"}], Worker())
    fn(prompts = [out[0]["prompt"]], completions = ["x"], **{TOOLS_COLUMN: [out[0][TOOLS_COLUMN]]})
    assert seen == {"tools": [[tool]]}


def test_reward_funcs_keep_spec_order_and_accept_plain_strings(tmp_path, monkeypatch):
    monkeypatch.setattr(rewards, "_user_root", lambda: tmp_path / "rewards")
    specs = [
        {
            **_spec(
                "py-len",
                "def reward(completions, **kw):\n    return [len(c[0]['content']) for c in completions]\n",
            ),
            "weight": 1.0,
        },
        {**rewards.get_reward("strict-xml-format"), "weight": 2.0},
    ]
    created = []

    def unstarted(python, mode):
        created.append(RewardWorker(python, mode = "full"))
        return created[-1]

    try:
        funcs = rl._reward_funcs(
            specs, unstarted, python_rewards.make_python_reward_funcs, rewards.make_reward_func
        )
        assert [f.__name__ for f in funcs] == ["py_len", "strict_xml_format"]
        # Thinking rendered into the prompt text makes TRL pass strings; the notebook shape still works.
        assert funcs[0](prompts = ["p"], completions = ["abc", ""]) == [3.0, 0.0]
    finally:
        for worker in created:
            worker.close()


def test_a_worker_cannot_be_started_twice(worker_factory):
    worker = worker_factory([_spec("t", "def reward(completions, **kw):\n    return []\n")])
    with pytest.raises(PythonRewardError, match = "already started"):
        worker.start()


def test_an_orphaned_worker_exits_when_the_host_goes_quiet(worker_factory):
    import time

    worker = worker_factory([_spec("t", "def reward(completions, **kw):\n    return []\n")])
    worker._closed.set()  # the host "dies": no more heartbeats
    stale = time.time() - 3600
    os.utime(os.path.join(worker._workdir, "alive"), (stale, stale))
    worker._proc.wait(timeout = 10)
    assert worker._proc.returncode == 0


def test_columns_reach_the_reward_and_trainer_kwargs_do_not(worker_factory):
    code = (
        "def reward(completions, **kw):\n"
        "    assert sorted(kw) == ['level', 'prompts'], sorted(kw)\n"
        "    return [float(kw['level'][i]) + 1 for i in range(len(completions))]\n"
    )
    worker = worker_factory([_spec("cols", code)])
    scores = worker.score(
        ["cols"],
        [[{"role": "user", "content": "q"}]],
        [[{"role": "assistant", "content": "a"}]],
        {"level": [3], "trainer_state": object(), "completion_ids": [[1, 2]]},
    )
    assert scores["cols"] == [4.0]


def test_a_failing_reward_names_itself(worker_factory):
    worker = worker_factory(
        [_spec("boom", "def reward(completions, **kw):\n    raise KeyError('answer')\n")]
    )
    with pytest.raises(PythonRewardError, match = r"boom: KeyError: 'answer'"):
        worker.score(["boom"], [], [[{"content": "x"}]], {})


def test_wrong_number_of_scores_is_an_error(worker_factory):
    worker = worker_factory([_spec("short", "def reward(completions, **kw):\n    return [1.0]\n")])
    with pytest.raises(PythonRewardError, match = "returned 1 scores for 2 completions"):
        worker.score(["short"], [], [[{"content": "a"}], [{"content": "b"}]], {})


def test_nan_and_none_scores_become_none(worker_factory):
    code = "def reward(completions, **kw):\n    return [float('nan'), None, 1]\n"
    worker = worker_factory([_spec("odd", code)])
    assert worker.score(["odd"], [], [[{"content": ""}]] * 3, {})["odd"] == [None, None, 1.0]


def test_code_that_fails_to_load_stops_the_run_before_training():
    with pytest.raises(PythonRewardError, match = "bad-import: ModuleNotFoundError"):
        RewardWorker(
            [
                _spec(
                    "bad-import",
                    "import not_a_real_module_xyz\ndef reward(completions, **kw):\n    return []\n",
                )
            ],
            mode = "full",
        ).start()


def test_a_hanging_reward_times_out_and_the_worker_is_killed(worker_factory):
    code = "import time\ndef reward(completions, **kw):\n    time.sleep(60)\n"
    worker = worker_factory([_spec("slow", code)], batch_timeout = 2.0)
    with pytest.raises(PythonRewardError, match = "did not answer within 2 s"):
        worker.score(["slow"], [], [[{"content": ""}]], {})
    with pytest.raises(PythonRewardError, match = "not running"):
        worker.score(["slow"], [], [[{"content": ""}]], {})


def test_prints_do_not_block_the_worker(worker_factory):
    code = "def reward(completions, **kw):\n    print('x' * 1_000_000)\n    return [1.0] * len(completions)\n"
    worker = worker_factory([_spec("loud", code)])
    for _ in range(3):
        assert worker.score(["loud"], [], [[{"content": ""}]], {})["loud"] == [1.0]


def test_worker_files_are_removed_on_close(worker_factory):
    worker = worker_factory(
        [_spec("t", "def reward(completions, **kw):\n    return [0.0] * len(completions)\n")]
    )
    workdir = worker._workdir
    assert os.path.isdir(workdir)
    worker.close()
    assert not os.path.exists(workdir)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(rewards, "_user_root", lambda: tmp_path / "rewards")
    real = python_rewards.RewardWorker
    modes = []

    def worker(
        specs,
        *,
        mode = "auto",
        **kw,
    ):
        # The mode the preview asked for is recorded, then "full" so this runs on any host.
        modes.append(mode)
        return real(specs, mode = "full", **kw)

    monkeypatch.setattr(python_rewards, "RewardWorker", worker)
    app = FastAPI()
    app.include_router(router, prefix = "/api/rewards")
    app.dependency_overrides[get_current_subject] = lambda: "test"
    client = TestClient(app)
    client.worker_modes = modes
    yield client
    python_rewards._close_preview()


def test_import_and_preview_a_python_reward(client):
    md = _python_md("correctness", NOTEBOOK, "correctness_reward_func")
    created = client.post("/api/rewards", json = {"markdown": md})
    assert created.status_code == 201, created.text
    assert (
        created.json()["kind"] == "python"
        and "def correctness_reward_func" in created.json()["code"]
    )
    response = client.post(
        "/api/rewards/preview",
        json = {
            "rewards": [{"name": "correctness", "weight": 1.5}, {"name": "strict-xml-format"}],
            "completion": GOOD,
            "row": {"answer": "72", "question": "q"},
            "prompt": "q",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [s["score"] for s in body["scores"]] == [2.0, 0.5]
    assert body["total"] == 3.5
    assert body["isolation"] == {"backend": "none", "os_isolation": False}


def test_a_python_score_too_large_to_weight_fails_alone_in_the_preview(client):
    huge = _python_md("huge", "def reward(completions, **kw):\n    return [1e308]\n")
    response = client.post(
        "/api/rewards/preview",
        json = {
            "rewards": [{"markdown": huge, "weight": 10}, {"name": "strict-xml-format"}],
            "completion": GOOD,
            "python_reward_isolation": "auto",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    scores = {s["name"]: s for s in body["scores"]}
    assert scores["huge"]["weighted"] is None and "outside -1,000,000" in scores["huge"]["error"]
    assert body["total"] == 0.5


def test_a_broken_python_reward_fails_alone_in_the_preview(client):
    # One reward's error sits on its own chip; the rule and the working Python reward still score.
    boom = _python_md("boom", "def reward(completions, **kw):\n    return 1 / 0\n")
    broken_load = _python_md(
        "typo", "import no_such_module_xyz\n\ndef reward(completions, **kw):\n    return [1.0]\n"
    )
    payload = {
        "rewards": [
            {"name": "soft-xml-format"},
            {"markdown": boom},
            {"markdown": _python_md("one", ONE)},
            {"markdown": broken_load},
        ],
        "completion": "<answer>x</answer>",
        "python_reward_isolation": "auto",
    }
    response = client.post("/api/rewards/preview", json = payload)
    assert response.status_code == 200, response.text
    scores = {s["name"]: s for s in response.json()["scores"]}
    assert scores["soft-xml-format"]["score"] == 0.5 and scores["soft-xml-format"]["error"] is None
    assert scores["one"]["score"] == 1.0 and scores["one"]["error"] is None
    assert scores["boom"]["score"] is None and "ZeroDivisionError" in scores["boom"]["error"]
    assert scores["typo"]["score"] is None and "no_such_module_xyz" in scores["typo"]["error"]
    assert response.json()["total"] == 1.5


def _preview_as(client, account_id, payload):
    async def subject():
        # Async, as the real dependency is: the binding has to reach the route's thread.
        bind_account(AccountContext(account_id, "same-name"))
        return "same-name"

    client.app.dependency_overrides[get_current_subject] = subject
    response = client.post("/api/rewards/preview", json = payload)
    assert response.status_code == 200, response.text
    return response.json()["scores"][0]["score"]


def test_a_warm_preview_worker_is_not_shared_across_accounts(client):
    # Module state survives between previews of one account; another account gets a fresh
    # worker, even under the same username (a deleted and re-created user).
    md = _python_md(
        "calls",
        "seen = []\n\n"
        "def reward(completions, **kw):\n"
        "    seen.append(1)\n"
        "    return [float(len(seen))] * len(completions)\n",
    )
    payload = {"rewards": [{"markdown": md}], "completion": "x"}
    scores = [_preview_as(client, a, payload) for a in ("acct-a", "acct-a", "acct-b")]
    assert scores == [1.0, 2.0, 1.0]


def test_another_accounts_preview_does_not_close_a_worker_mid_score(client):
    slow = _python_md(
        "slow",
        "import time\n\ndef reward(completions, **kw):\n"
        "    time.sleep(3)\n    return [1.0] * len(completions)\n",
    )
    fast = _python_md(
        "fast", "def reward(completions, **kw):\n    return [2.0] * len(completions)\n"
    )
    first = {}
    thread = threading.Thread(
        target = lambda: first.update(
            score = _preview_as(
                client, "acct-a", {"rewards": [{"markdown": slow}], "completion": "x"}
            )
        )
    )
    thread.start()
    time.sleep(1.5)
    assert (
        _preview_as(client, "acct-b", {"rewards": [{"markdown": fast}], "completion": "x"}) == 2.0
    )
    thread.join(30)
    assert first == {"score": 1.0}


def test_a_preview_row_keeps_its_value_types(client):
    md = _python_md(
        "typed",
        "def reward(completions, level, meta, **kw):\n"
        "    return [float(level[0] + meta[0]['bonus'])] * len(completions)\n",
    )
    payload = {
        "rewards": [{"markdown": md}],
        "completion": "x",
        "row": {"level": 3, "meta": {"bonus": 2}},
    }
    assert _preview_as(client, "acct-a", payload) == 5.0


ONE = "def reward(completions, **kw):\n    return [1.0] * len(completions)\n"


def test_a_python_preview_needs_the_os_sandbox_unless_opted_out(client):
    md = _python_md("one", ONE)
    payload = {"rewards": [{"markdown": md}], "completion": "x"}
    _preview_as(client, "acct-a", payload)
    _preview_as(client, "acct-a", {**payload, "python_reward_isolation": "auto"})
    assert client.worker_modes == ["required", "auto"]


# What a stock Colab VM reports: os_sandbox's own wording, written for the chat tools.
COLAB_REASON = (
    "OS_ISOLATION_UNAVAILABLE: the bubblewrap live probe could not run: SandboxUnavailableError: "
    "bubblewrap (bwrap) is not installed on this host"
)
COLAB_REMEDIATION = (
    "bubblewrap (bwrap) is not installed. To isolate tool calls, run `sudo apt-get install -y "
    "bubblewrap`; Unsloth picks it up within a minute, no restart needed. Python and Terminal "
    "still run, with software safeguards only and no OS isolation."
)


def test_a_python_preview_without_the_os_sandbox_keeps_the_rule_scores(client, monkeypatch):
    from core.inference.os_sandbox import SandboxUnavailableError

    class NoSandbox:
        def __init__(self, specs, **kw):
            pass

        def start(self):
            raise SandboxUnavailableError(COLAB_REASON, remediation = COLAB_REMEDIATION)

    monkeypatch.setattr(python_rewards, "RewardWorker", NoSandbox)
    md = _python_md("one", ONE)
    response = client.post(
        "/api/rewards/preview",
        json = {
            "rewards": [{"name": "soft-xml-format"}, {"markdown": md}],
            "completion": "<answer>x</answer>",
        },
    )
    assert response.status_code == 200, response.text
    rule, python = response.json()["scores"]
    assert rule["score"] == 0.5 and rule["error"] is None
    assert python["score"] is None
    assert python["error"] == python_rewards.sandbox_refusal(COLAB_REASON, COLAB_REMEDIATION)


def test_try_it_start_and_training_refuse_in_the_same_plain_words(monkeypatch):
    from core.inference import os_sandbox

    text = python_rewards.sandbox_refusal(COLAB_REASON, COLAB_REMEDIATION)
    assert "OS_ISOLATION_UNAVAILABLE" not in text and "Error:" not in text
    assert text.count("not installed") == 1
    assert "still run" not in text and "tool calls" not in text and "within a minute" not in text
    assert "sudo apt-get install -y bubblewrap`, then try again." in text
    assert 'grpo_python_reward_isolation: "auto"' in text
    # The OS sandbox confines files and processes; neither mode confines the network.
    assert "using the network" not in text and "Neither setting limits the network." in text

    monkeypatch.setattr(
        os_sandbox,
        "capability_snapshot",
        lambda **kw: os_sandbox.SandboxCapability(
            backend = "bubblewrap",
            available = False,
            reason = COLAB_REASON,
            remediation = COLAB_REMEDIATION,
        ),
    )
    assert python_rewards.python_isolation_refusal("required") == text
    assert python_rewards.python_isolation_refusal("auto") is None
    specs = [{**_spec("py-len", LEN_REWARD), "weight": 1.0}]
    with pytest.raises(PythonRewardError) as err:
        rl._reward_funcs(
            specs, RewardWorker, python_rewards.make_python_reward_funcs, rewards.make_reward_func
        )
    assert str(err.value) == text


def test_a_refusal_without_remediation_drops_the_internal_codes():
    text = python_rewards.sandbox_refusal(
        "OS_ISOLATION_UNAVAILABLE: probe failed: SandboxUnavailableError: no user namespaces"
    )
    assert "No user namespaces." in text and "OS_ISOLATION" not in text and "Error" not in text


def _os_sandbox_available() -> bool:
    import sys

    from core.inference import os_sandbox
    try:
        return os_sandbox.capability_snapshot(
            execution_kind = "python", selected_executable = sys.executable
        ).available
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(
    os.environ.get("UNSLOTH_TEST_OS_SANDBOX") != "1" or not _os_sandbox_available(),
    reason = "set UNSLOTH_TEST_OS_SANDBOX=1 on a host with OS isolation (MXC, Landlock, Seatbelt)",
)
def test_rewards_run_os_isolated_and_cannot_read_home():
    secret = Path.home() / "unsloth-reward-sandbox-probe.txt"
    secret.write_text("host secret", "utf-8")
    code = (
        "def reward(completions, **kw):\n"
        "    try:\n"
        f"        open({str(secret)!r}).read()\n"
        "        return [1.0]\n"
        "    except OSError:\n"
        "        return [0.0]\n"
    )
    from core.inference.os_sandbox import SandboxUnavailableError

    try:
        worker = RewardWorker([_spec("probe", code)], mode = "required").start()
    except SandboxUnavailableError as exc:
        secret.unlink()
        pytest.skip(f"no OS isolation for this test process: {exc}")
    try:
        assert worker.isolation["os_isolation"] is True
        assert worker.score(["probe"], [], [[{"content": ""}]], {})["probe"] == [0.0]
    finally:
        worker.close()
        secret.unlink()


def test_python_rewards_keep_every_dataset_column():
    from datasets import Dataset

    raw = Dataset.from_list([{"question": "q", "solution": "72", "level": 3, "board": "x"}])
    shaped, _ = rl.format_rl_dataset(
        raw, "grpo", {"question": "prompt", "solution": "answer"}, keep_columns = ("*",)
    )
    assert sorted(shaped.column_names) == [
        "answer",
        "board",
        "level",
        "prompt",
        "question",
        "solution",
    ]


def test_kept_columns_skip_the_schema_keys_trl_reinterprets():
    from datasets import Dataset

    turns = [{"role": "user", "content": "q"}]
    raw = Dataset.from_list([{"messages": turns, "completion": "c", "label": True, "level": 3}])
    shaped, _ = rl.format_rl_dataset(raw, "grpo", {"messages": "prompt"}, keep_columns = ("*",))
    assert sorted(shaped.column_names) == ["answer", "level", "prompt"]


def test_a_preview_row_can_carry_a_wide_dataset(client):
    md = _python_md("wide", "def reward(completions, c99, **kw):\n    return [float(c99[0])]\n")
    row = {f"c{i}": i for i in range(100)}
    payload = {"rewards": [{"markdown": md}], "completion": "x", "row": row}
    assert _preview_as(client, "acct-a", payload) == 99.0


def test_a_rule_can_compare_to_the_column_mapped_as_answer():
    from datasets import Dataset

    raw = Dataset.from_list([{"question": "q", "solution": "72"}])
    shaped, _ = rl.format_rl_dataset(
        raw, "grpo", {"question": "prompt", "solution": "answer"}, keep_columns = ("solution",)
    )
    assert shaped[0]["solution"] == "72"
    assert shaped[0]["answer"] == "72"


@pytest.fixture
def no_os_sandbox(monkeypatch):
    # Stands in for a Linux host without bubblewrap (stock Colab, most containers), on any runner.
    from core.inference import os_sandbox
    monkeypatch.setattr(
        os_sandbox,
        "capability_snapshot",
        lambda **kw: os_sandbox.SandboxCapability(
            backend = "bubblewrap",
            available = False,
            reason = "bubblewrap is not installed",
            remediation = f"Install bubblewrap. {os_sandbox._FALLBACK_NOTE}",
        ),
    )


LEN_REWARD = (
    "def reward(completions, **kw):\n    return [len(c[0]['content']) for c in completions]\n"
)


def test_a_training_run_refuses_python_rewards_without_the_os_sandbox(no_os_sandbox):
    specs = [{**_spec("py-len", LEN_REWARD), "weight": 1.0}]
    with pytest.raises(PythonRewardError, match = "need the OS sandbox") as err:
        rl._reward_funcs(
            specs, RewardWorker, python_rewards.make_python_reward_funcs, rewards.make_reward_func
        )
    assert "Install bubblewrap." in str(err.value)
    assert '"auto"' in str(err.value)
    assert "still run" not in str(err.value)


def test_opting_into_software_safeguards_runs_and_reports_it(no_os_sandbox):
    specs = [{**_spec("py-len", LEN_REWARD), "weight": 1.0}]
    isolation, created = {}, []

    def tracked(python, mode):
        created.append(RewardWorker(python, mode = mode))
        return created[-1]

    funcs = rl._reward_funcs(
        specs,
        tracked,
        python_rewards.make_python_reward_funcs,
        rewards.make_reward_func,
        mode = "auto",
        isolation = isolation,
    )
    try:
        assert funcs[0](prompts = ["p"], completions = ["abc"]) == [3.0]
    finally:
        for worker in created:
            worker.close()
    assert isolation == {"backend": "software-safeguards", "os_isolation": False}
    metrics = rl.with_reward_isolation({"reward": 1.0}, isolation)
    assert metrics[rl.REWARD_ISOLATION_METRIC] == 0.0
    # A summary log with no RL keys stays empty rather than becoming an isolation-only point.
    assert rl.with_reward_isolation(None, isolation) is None
    assert rl.with_reward_isolation({"reward": 1.0}, None) == {"reward": 1.0}


def test_worker_log_stops_growing_at_its_cap(tmp_path):
    from core.training.reward_worker_main import _BoundedLog

    path = tmp_path / "worker.log"
    log = _BoundedLog(str(path), 10)
    assert log.write("x" * 25) == 25
    assert log.write("more") == 4
    log.flush()
    text = path.read_text("utf-8")
    assert text.startswith("x" * 10 + "\n[worker.log is full")
    assert "more" not in text


@pytest.mark.skipif(os.name == "nt", reason = "rlimits are POSIX")
def test_reward_workers_start_under_file_size_and_descriptor_limits():
    import subprocess
    import sys

    probe = (
        "import resource\n"
        "print(resource.getrlimit(resource.RLIMIT_FSIZE)[0], resource.getrlimit(resource.RLIMIT_NOFILE)[0])\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        preexec_fn = python_rewards._reward_worker_preexec,
        capture_output = True,
        text = True,
        check = True,
    ).stdout.split()
    assert int(out[0]) == python_rewards.REWARD_FSIZE_BYTES
    assert int(out[1]) <= python_rewards.REWARD_NOFILE


def test_a_preview_prompt_has_the_turns_training_gives_the_reward(client):
    # Training hands rewards the formatted prompt: the row's own turns, the system prompt first.
    md = _python_md(
        "roles",
        "def reward(prompts, completions, **kw):\n"
        "    return [float(len(prompts[0]) * 10 + (prompts[0][0]['role'] == 'system'))]\n",
    )
    turns = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    payload = {
        "rewards": [{"markdown": md}],
        "completion": "x",
        "prompt": turns,
        "system_prompt": "be brief",
    }
    trained = rl.format_rl_dataset(
        __import__("datasets").Dataset.from_list([{"question": turns, "answer": "a"}]),
        "grpo",
        system_prompt = "be brief",
    )[0][0]["prompt"]
    assert _preview_as(client, "acct-a", payload) == len(trained) * 10 + 1 == 31


def test_start_refuses_python_rewards_before_the_model_loads():
    # The admit step runs before any worker or model load; the refusal used to come ~2 min later.
    import inspect

    import routes.training as training_routes

    source = inspect.getsource(training_routes._admit_training_request)
    check = source.index("python_isolation_refusal, request.grpo_python_reward_isolation")
    assert check < source.index("training_kwargs = {")
    assert "raise HTTPException(status_code = 400, detail = refusal)" in source[check:]


def test_isolation_status_route_reports_what_start_would_say(client, monkeypatch):
    monkeypatch.setattr(python_rewards, "python_isolation_refusal", lambda mode: f"no ({mode})")
    assert client.get("/api/rewards/python-isolation").json() == {"refusal": "no (required)"}
    assert client.get("/api/rewards/python-isolation?mode=auto").json() == {"refusal": "no (auto)"}


@pytest.mark.skipif(os.name == "nt", reason = "process groups are POSIX")
def test_closing_a_worker_kills_what_its_rewards_started(worker_factory):
    # A reward that starts a child and returns: the worker exits cleanly, the child must not.
    code = (
        "import subprocess\n\n"
        "def reward(completions, **kw):\n"
        "    child = subprocess.Popen(['sleep', '300'])\n"
        "    return [float(child.pid)]\n"
    )
    worker = worker_factory([_spec("spawner", code)])
    pid = int(worker.score(["spawner"], [], [[{"content": "x"}]], {})["spawner"][0])
    os.kill(pid, 0)
    worker.close()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(pid, signal.SIGKILL)
        pytest.fail("the reward's child outlived its worker")


class _FakeJob:
    def __init__(self):
        self.calls = []

    def terminate(self):
        self.calls.append("terminate")
        return True

    def close(self):
        self.calls.append("close")


def test_closing_a_worker_ends_its_windows_job(worker_factory, monkeypatch):
    # Windows: the job is what reaches a reward's children once the worker itself has exited.
    job = _FakeJob()
    monkeypatch.setattr(python_rewards, "_windows_job", lambda proc: job)
    worker = worker_factory([_spec("one", "def reward(completions, **kw):\n    return [1.0]\n")])
    worker.score(["one"], [], [[{"content": "x"}]], {})
    worker.close()
    assert job.calls == ["terminate", "close"]


def test_the_windows_job_is_closed_even_if_terminating_it_fails():
    class _Stuck(_FakeJob):
        def terminate(self):
            raise OSError("access denied")

    job = _Stuck()
    python_rewards._kill_group(None, job)
    assert job.calls == ["close"]


def test_reward_code_waits_until_the_worker_is_in_its_job(worker_factory, monkeypatch):
    # A reward that starts a child at import must start it inside the job, so the setup the
    # worker loads rewards from is only written after the job holds the worker.
    seen = {}

    def capture(proc):
        workdir = Path(proc.args[-1]).parent
        assert (workdir / "worker.py").exists()
        seen["setup_written"] = (workdir / "rewards.json").exists()
        return None

    monkeypatch.setattr(python_rewards, "_windows_job", capture)
    worker = worker_factory([_spec("one", "def reward(completions, **kw):\n    return [1.0]\n")])
    assert worker.score(["one"], [], [[{"content": "x"}]], {})["one"] == [1.0]
    assert seen == {"setup_written": False}


def test_without_a_windows_job_close_kills_the_tree_before_the_worker_exits(monkeypatch, tmp_path):
    calls = []

    class _Proc:
        pid = 4321

        def wait(self, timeout = None):
            calls.append("wait")
            return 0

    monkeypatch.setattr(python_rewards, "_kill_tree", lambda proc: calls.append("kill_tree"))
    monkeypatch.setattr(python_rewards.os, "name", "nt")
    worker = RewardWorker([_spec("one", "def reward(completions, **kw):\n    return [1.0]\n")])
    worker._proc, worker._workdir = _Proc(), str(tmp_path / "w")
    os.makedirs(worker._workdir)
    worker.close()
    assert calls == ["kill_tree"]


def test_the_worker_bounds_scores_like_rule_rewards():
    from core.training import reward_worker_main
    assert reward_worker_main.MAX_SCORE == rewards.MAX_SCORE


@pytest.mark.parametrize("value", ["1e308", "float('inf')", "-float('inf')"])
def test_training_refuses_a_python_score_past_the_bound(worker_factory, value):
    huge = _spec("huge", f"def reward(completions, **kw):\n    return [{value}]\n")
    worker = worker_factory([huge])
    with pytest.raises(PythonRewardError, match = "outside -1,000,000 to 1,000,000"):
        worker.score(["huge"], [], [[{"content": "x"}]], {})


def test_a_reward_that_edits_its_inputs_does_not_change_the_next_ones(worker_factory):
    vandal = (
        "def reward(completions, prompts, **kw):\n"
        "    completions[0][0]['content'] = 'changed'\n"
        "    prompts[0][0]['content'] = 'changed'\n"
        "    kw['answer'][0] = 'changed'\n"
        "    return [0.0]\n"
    )
    reader = (
        "def reward(completions, prompts, answer, **kw):\n"
        "    seen = (completions[0][0]['content'], prompts[0][0]['content'], answer[0])\n"
        "    return [1.0 if seen == ('x', 'q', 'a') else -1.0]\n"
    )
    worker = worker_factory([_spec("a-vandal", vandal), _spec("b-reader", reader)])
    scores = worker.score(
        ["a-vandal", "b-reader"],
        [[{"role": "user", "content": "q"}]],
        [[{"role": "assistant", "content": "x"}]],
        {"answer": ["a"]},
    )
    assert scores["b-reader"] == [1.0]


class _RecordingKernel:
    def __init__(self):
        self.calls = []

    def SetInformationJobObject(self, handle, info_class, info_ref, size):
        info = info_ref._obj
        self.calls.append(
            (handle, info_class, info.BasicLimitInformation.LimitFlags, info.ProcessMemoryLimit)
        )
        return 1


def test_the_windows_reward_job_carries_the_memory_limit(monkeypatch):
    # Windows has no RLIMIT_AS; the job's per-process memory limit stands in for it.
    import types

    from core.inference import tools

    kernel32 = _RecordingKernel()
    job = types.SimpleNamespace(_kernel32 = kernel32, _handle = 77)
    monkeypatch.setattr(python_rewards, "os", types.SimpleNamespace(name = "nt"))
    monkeypatch.setattr(tools, "_windows_job_capture", lambda proc: job)
    monkeypatch.setattr(python_rewards, "_reward_as_bytes", 8 * 1024**3)
    assert python_rewards._windows_job(object()) is job
    assert kernel32.calls == [(77, 9, python_rewards._JOB_OBJECT_LIMIT_PROCESS_MEMORY, 8 * 1024**3)]


def test_no_memory_limit_set_means_no_windows_job_limit():
    kernel32 = _RecordingKernel()
    assert python_rewards._limit_job_memory(kernel32, 77, None) is False
    assert kernel32.calls == []


class _SelfJobKernel(_RecordingKernel):
    def __init__(self):
        super().__init__()
        self.assigned = None

    def CreateJobObjectW(self, attributes, name):
        return 55

    def GetCurrentProcess(self):
        return -1

    def AssignProcessToJobObject(self, job, process):
        self.assigned = (job, process)
        return 1


def test_the_reward_worker_puts_itself_in_a_capped_job():
    # Under MXC the host's job may hold only the launcher, so the worker caps itself as well.
    from core.training import reward_worker_main as worker_main

    kernel32 = _SelfJobKernel()
    assert worker_main._cap_own_memory(8 * 1024**3, kernel32 = kernel32) is True
    assert kernel32.calls == [(55, 9, worker_main._JOB_OBJECT_LIMIT_PROCESS_MEMORY, 8 * 1024**3)]
    assert kernel32.assigned == (55, -1)
    assert worker_main._cap_own_memory(None, kernel32 = _SelfJobKernel()) is False


def test_the_worker_caps_itself_before_loading_any_reward(monkeypatch, tmp_path):
    from core.training import reward_worker_main as worker_main

    order = []
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(worker_main, "_cap_own_memory", lambda limit: order.append(("cap", limit)))

    def load(specs):
        order.append(("load", len(specs)))
        (tmp_path / "stop").touch()
        return {}, {}

    monkeypatch.setattr(worker_main, "_load", load)
    (tmp_path / "alive").touch()
    (tmp_path / "rewards.json").write_text(
        json.dumps({"specs": [{}], "memory_limit_bytes": 1024}), encoding = "utf-8"
    )
    assert worker_main.main() == 0
    assert order == [("cap", 1024), ("load", 1)]


def test_the_host_hands_the_worker_its_memory_limit(worker_factory, monkeypatch):
    from utils import sandbox_memory_limit

    monkeypatch.setattr(sandbox_memory_limit, "memory_limit_bytes", lambda: 3 * 1024**3)
    worker = worker_factory([_spec("one", "def reward(completions, **kw):\n    return [1.0]\n")])
    setup = json.loads(Path(worker._workdir, "rewards.json").read_text(encoding = "utf-8"))
    assert setup["memory_limit_bytes"] == 3 * 1024**3


def test_a_windows_worker_that_cannot_cap_itself_loads_no_reward(monkeypatch, tmp_path):
    import types

    from core.training import reward_worker_main as worker_main

    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.chdir(tmp_path)
    windows_os = types.ModuleType("os")
    windows_os.__dict__.update(os.__dict__)
    windows_os.name = "nt"
    monkeypatch.setattr(worker_main, "os", windows_os)
    monkeypatch.setattr(worker_main, "_cap_own_memory", lambda limit: False)
    loaded = []

    def load(specs):
        loaded.extend(specs)
        (tmp_path / "stop").touch()
        return {}, {}

    monkeypatch.setattr(worker_main, "_load", load)
    (tmp_path / "alive").touch()
    (tmp_path / "rewards.json").write_text(
        json.dumps({"specs": [{"name": "greedy", "code": "x = 1"}], "memory_limit_bytes": 1024}),
        encoding = "utf-8",
    )
    assert worker_main.main() == 0
    ready = json.loads((tmp_path / "ready.json").read_text(encoding = "utf-8"))
    assert loaded == [] and ready["loaded"] == []
    assert "memory limit" in ready["errors"]["greedy"]
