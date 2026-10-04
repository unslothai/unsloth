# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.training import python_rewards, rewards, rl
from core.training.python_rewards import PythonRewardError, RewardWorker
from routes.rewards import router

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


def _python_md(name: str, code: str, entry: str = "reward") -> str:
    return f"---\nname: {name}\nkind: python\nentry: {entry}\ndescription: test\n---\n```python\n{code}```\n"


def _spec(name: str, code: str, entry: str = "reward") -> dict:
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


def test_python_reward_round_trips_through_markdown():
    spec = _spec("counter", "def reward(completions, **kwargs):\n    return [1.0] * len(completions)\n")
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


def test_reward_funcs_keep_spec_order_and_accept_plain_strings(tmp_path, monkeypatch):
    monkeypatch.setattr(rewards, "_user_root", lambda: tmp_path / "rewards")
    specs = [
        {**_spec("py-len", "def reward(completions, **kw):\n    return [len(c[0]['content']) for c in completions]\n"), "weight": 1.0},
        {**rewards.get_reward("strict-xml-format"), "weight": 2.0},
    ]
    created = []

    def unstarted(python):
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
    worker = worker_factory([_spec("boom", "def reward(completions, **kw):\n    raise KeyError('answer')\n")])
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
            [_spec("bad-import", "import not_a_real_module_xyz\ndef reward(completions, **kw):\n    return []\n")],
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
    worker = worker_factory([_spec("t", "def reward(completions, **kw):\n    return [0.0] * len(completions)\n")])
    workdir = worker._workdir
    assert os.path.isdir(workdir)
    worker.close()
    assert not os.path.exists(workdir)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(rewards, "_user_root", lambda: tmp_path / "rewards")
    real = python_rewards.RewardWorker
    monkeypatch.setattr(
        python_rewards, "RewardWorker", lambda specs, **kw: real(specs, mode = "full", **kw)
    )
    app = FastAPI()
    app.include_router(router, prefix = "/api/rewards")
    app.dependency_overrides[get_current_subject] = lambda: "test"
    yield TestClient(app)
    python_rewards._close_preview()


def test_import_and_preview_a_python_reward(client):
    md = _python_md("correctness", NOTEBOOK, "correctness_reward_func")
    created = client.post("/api/rewards", json = {"markdown": md})
    assert created.status_code == 201, created.text
    assert created.json()["kind"] == "python" and "def correctness_reward_func" in created.json()["code"]
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


def test_preview_reports_a_broken_python_reward_as_a_400(client):
    md = _python_md("boom", "def reward(completions, **kw):\n    return 1 / 0\n")
    response = client.post("/api/rewards/preview", json = {"rewards": [{"markdown": md}], "completion": "x"})
    assert response.status_code == 400
    assert "ZeroDivisionError" in response.json()["detail"]


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
    assert sorted(shaped.column_names) == ["answer", "board", "level", "prompt"]
