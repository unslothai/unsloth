# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

# Needs CUDA, llama.cpp >= b11443 at UNSLOTH_LLAMA_CPP_PATH and its llama-server at UNSLOTH_TEST_LLAMA_SERVER.

import json
import math
import os
import random
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest
from real_accelerator import has_real_cuda

torch = pytest.importorskip("torch")

SERVER = os.environ.get("UNSLOTH_TEST_LLAMA_SERVER", "")
pytestmark = pytest.mark.skipif(
    not (has_real_cuda() and SERVER and Path(SERVER).is_file()),
    reason = "needs CUDA and UNSLOTH_TEST_LLAMA_SERVER",
)
REPORT = os.environ.get("UNSLOTH_TEST_GGUF_REPORT")
TINY_QWEN3_5 = "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration"
TINY_MODERNBERT = "hf-internal-testing/tiny-random-ModernBertModel"
QUESTIONS = {
    "outage": {"type": "noul", "instructions": "Is a service down?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle it?",
        "criteria": {"billing": "payments", "tech": "bugs", "sales": "new orders"},
    },
    "mood": {
        "type": "score",
        "instructions": "How upset is the customer?",
        "criteria": ["calm", "annoyed", "angry", "furious"],
    },
}
STATES = (
    "the server is down again and nothing loads",
    "my invoice was paid twice please refund",
    "I want to order ten more licenses",
    "login page shows an error since this morning",
)
GOLD = (("true", "tech", 3), ("false", "billing", 1), ("false", "sales", 0), ("true", "tech", 2))


def _rows(
    n,
    seed,
    noise = 0.3,
):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        kind = rng.randrange(len(STATES))
        outage, team, mood = GOLD[kind]
        # Label noise, so calibration has something to fit instead of saturating.
        if rng.random() < noise:
            outage = rng.choice(["true", "false"])
        if rng.random() < noise:
            team = rng.choice(["billing", "tech", "sales"])
        if rng.random() < noise:
            mood = rng.randrange(4)
        rows.append(
            {
                "state": f"ticket {i}: {STATES[kind]}",
                "questions": QUESTIONS,
                "gold": {"outage": {"label": outage}, "team": {"label": team}, "mood": mood},
            }
        )
    return rows


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(url, payload):
    request = urllib.request.Request(
        url, data = json.dumps(payload).encode(), headers = {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout = 600) as response:
        return json.loads(response.read())


class _Server:
    def __init__(self, model, mmproj, log):
        self.port = _free_port()
        argv = [SERVER, "-m", str(model), "--host", "127.0.0.1", "--port", str(self.port)]
        argv += ["-c", "16384", "-b", "16384", "-ub", "16384", "-np", "1", "-ngl", "999"]
        if mmproj:
            argv += ["--mmproj", str(mmproj)]
        self.log = Path(log)
        env = {**os.environ, "LD_LIBRARY_PATH": str(Path(SERVER).resolve().parent)}
        self.process = subprocess.Popen(
            argv,
            stdout = open(self.log, "w"),
            stderr = subprocess.STDOUT,
            env = env,
            start_new_session = True,
        )
        deadline = time.time() + 300
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited: {self.log.read_text()[-3000:]}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout = 2) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(0.5)
        self.close()
        raise RuntimeError("llama-server did not become ready")

    def decide(self, state, questions):
        return _post(
            f"http://127.0.0.1:{self.port}/v1/systemone", {"state": state, "questions": questions}
        )["answers"]

    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, 15)
            try:
                self.process.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, 9)


def _probs(answer):
    if "noul" in answer and not isinstance(answer["noul"], bool):
        p = float(answer["noul"])
        return {"true": p, "false": 1.0 - p}
    return {str(k): float(v) for k, v in answer["probabilities"].items()}


def _gold(row, name):
    gold = row["gold"][name]
    return str(gold["label"] if isinstance(gold, dict) else gold)


# Ties (near-uniform answers) break by this fixed option order, not by dict order.
OPTIONS = {
    name: ["true", "false"]
    if q["type"] == "noul"
    else list(q["criteria"])
    if q["type"] == "choice"
    else [str(i) for i in range(len(q["criteria"]))]
    for name, q in QUESTIONS.items()
}


def _top(probs, name):
    return max(OPTIONS[name], key = probs.get)


def _scores(answers, rows):
    nll, bins = [], [[0, 0.0, 0.0] for _ in range(10)]
    for answer, row in zip(answers, rows):
        for name in QUESTIONS:
            probs = _probs(answer[name])
            gold = _gold(row, name)
            nll.append(-math.log(max(probs[gold], 1e-12)))
            top = _top(probs, name)
            conf = probs[top]
            b = bins[min(int(conf * 10), 9)]
            b[0] += 1
            b[1] += conf
            b[2] += float(top == gold)
    n = sum(b[0] for b in bins)
    ece = sum(abs(b[1] - b[2]) for b in bins if b[0]) / n
    return sum(nll) / len(nll), ece


def _max_diff(served, reference):
    worst, same = 0.0, True
    for got, want in zip(served, reference):
        for name in QUESTIONS:
            p, q = _probs(got[name]), _probs(want[name])
            worst = max(worst, max(abs(p[k] - q[k]) for k in q))
            # A flip between options PyTorch itself scores within 0.01 is a tie, not a different answer.
            same &= q[_top(q, name)] - q[_top(p, name)] <= 0.01
    return worst, same


def _serve_and_compare(export_dir, data, reference, rows, label, tmp_path):
    results = {}
    for quant in data["quantizations"]:
        entry = data["files"][quant]
        mmproj = export_dir / entry["mmproj"] if entry.get("mmproj") else None
        server = _Server(export_dir / entry["model"], mmproj, tmp_path / f"{label}_{quant}.log")
        try:
            served = [server.decide(row["state"], row["questions"]) for row in rows]
        finally:
            server.close()
        worst, same = _max_diff(served, reference)
        nll, ece = _scores(served, rows)
        results[quant] = {"max_abs_prob_diff": worst, "same_answers": same, "nll": nll, "ece": ece}
    nll, ece = _scores(reference, rows)
    results["pytorch"] = {"nll": nll, "ece": ece}
    if REPORT:
        path = Path(REPORT)
        report = json.loads(path.read_text()) if path.is_file() else {}
        report[label] = results
        path.write_text(json.dumps(report, indent = 2))
    print(label, json.dumps(results, indent = 2))
    return results


def _assert_parity(results, quant, tolerance):
    # ECE is reported only: it bins by the top answer, so it jumps on near-ties.
    got = results[quant]
    assert got["max_abs_prob_diff"] <= tolerance and got["same_answers"], got
    assert abs(got["nll"] - results["pytorch"]["nll"]) <= tolerance, got


def _train(model, tokenizer, rows, tmp_path, steps):
    from transformers import TrainingArguments
    from unsloth import DecisionTrainer, FastDecisionModel

    items, _ = FastDecisionModel.build_dataset(rows, tokenizer, model)
    held = FastDecisionModel.build_dataset(_rows(48, seed = 1), tokenizer, model)[0]
    DecisionTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = items,
        head_learning_rate = 5e-3,
        args = TrainingArguments(
            output_dir = str(tmp_path / "run"),
            per_device_train_batch_size = 8,
            max_steps = steps,
            learning_rate = 5e-3,
            report_to = "none",
            save_strategy = "no",
        ),
    ).train()
    FastDecisionModel.for_inference(model)
    return FastDecisionModel.calibrate(model, tokenizer, held)


def _predict(model, tokenizer, rows):
    from unsloth import FastDecisionModel
    return [
        FastDecisionModel.predict(model, tokenizer, row["state"], row["questions"]) for row in rows
    ]


def test_clef_fine_tune_exports_and_serves_like_pytorch(tmp_path):
    from transformers import AutoConfig
    from unsloth import FastDecisionModel
    from unsloth.models.decision_gguf import read_decision_temperatures, write_decision_temperatures

    hidden = AutoConfig.from_pretrained(TINY_QWEN3_5).text_config.hidden_size
    head = {
        "hidden_size": hidden,
        "width": 64,
        "routing_layers": 1,
        "layers": 1,
        "heads": 4,
        "feedforward": 128,
    }
    model, processor = FastDecisionModel.from_pretrained(
        TINY_QWEN3_5, decision_head = "clef", head_config = head, max_seq_length = 512
    )
    model = FastDecisionModel.get_peft_model(model, r = 8, lora_alpha = 16)
    calibration = _train(model, processor, _rows(96, seed = 0), tmp_path, steps = 30)
    rows = _rows(16, seed = 2)
    config = model.decision_config
    print("calibration", calibration, config.get("head_temperature"), config.get("temperature"))

    reference = _predict(model, processor, rows)
    data = model.save_pretrained_gguf(
        tmp_path / "clef", processor, quantization_method = ["q8_0", "q4_k_m"]
    )
    export_dir = tmp_path / "clef" / "gguf"
    assert data["quantizations"] == ["Q8_0", "Q4_K_M"]
    assert data["files"]["Q8_0"]["mmproj"] == "mmproj-Q8_0.gguf"
    results = _serve_and_compare(export_dir, data, reference, rows, "clef_calibrated", tmp_path)
    # Q8_0 moved a sharply calibrated Clef-Flash fine-tune by 0.045 (BF16: 0.0047).
    _assert_parity(results, "Q8_0", 0.05)

    # Just under the 0.05 fold floor: the head temperature lives only in the config, so the GGUF must carry it.
    with torch.no_grad():
        model.head.joint_logit_scale.fill_(math.log(5.5))
    config["head_temperature"] = 0.05
    # 0.05 x 5.0 = 0.25: under laya's 0.5 floor, so re-clamping the product would show.
    config["temperature"] = [5.0, 5.0, 5.0]
    reference = _predict(model, processor, rows)
    data = model.save_pretrained_gguf(
        tmp_path / "floor", processor, quantization_method = ["bf16", "q8_0"]
    )
    export_dir = tmp_path / "floor" / "gguf"
    floor = {"choice": 0.25, "score": 0.25, "noul": 0.25}
    assert read_decision_temperatures(export_dir / "model-BF16.gguf") == pytest.approx(floor)
    results = _serve_and_compare(export_dir, data, reference, rows, "clef_floor", tmp_path)
    # Temperature 0.25 multiplies logit rounding by 4 (measured 0.0087 and 0.0105); the control drifts > 0.1.
    _assert_parity(results, "BF16", 0.02)

    # Negative control: the same file without the keys, as the converter alone writes it.
    write_decision_temperatures(export_dir / "model-BF16.gguf", {})
    control = {**data, "quantizations": ["BF16"]}
    results = _serve_and_compare(
        export_dir, control, reference, rows, "clef_floor_without_keys", tmp_path
    )
    assert results["BF16"]["max_abs_prob_diff"] > 0.1


def _tiny_laya(folder: Path) -> Path:
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoTokenizer
    from unsloth.models.decision import _laya

    source = snapshot_download(TINY_MODERNBERT, allow_patterns = ["*.json", "*.txt"])
    (folder / "encoder").mkdir(parents = True)
    config = AutoConfig.from_pretrained(source)
    (folder / "encoder" / "config.json").write_text(config.to_json_string())
    AutoTokenizer.from_pretrained(source).save_pretrained(str(folder / "tokenizer"))
    decision_config = {
        "encoder": TINY_MODERNBERT,
        "head_layers": 1,
        "act_costs": {"escalate": 0.5},
        "max_len": 256,
        "head_max_len": 96,
        "temperature": [1.0, 1.0, 1.0],
        "temperature_by_options": {"choice:3-5": 0.1006, "score:3-5": 0.8},
    }
    torch.manual_seed(0)
    model = _laya().common.build_model(decision_config, encoder_dir = str(folder / "encoder"))
    save_file(
        {k: v.half().contiguous() for k, v in model.state_dict().items()},
        str(folder / "model.safetensors"),
    )
    (folder / "rl_agent_config.json").write_text(json.dumps(decision_config))
    return folder


def test_laya_fine_tune_exports_and_serves_like_pytorch(tmp_path):
    from unsloth import FastDecisionModel
    from unsloth.models.decision_gguf import read_decision_temperatures

    base = _tiny_laya(tmp_path / "laya_base")
    model, tokenizer = FastDecisionModel.from_pretrained(str(base), full_finetuning = True)
    _train(model, tokenizer, _rows(96, seed = 0), tmp_path, steps = 30)
    # One bucket survives calibration only if its type was not refitted; keep one to test buckets.
    model.decision_config["temperature_by_options"] = {"choice:3-5": 0.1006}
    rows = _rows(16, seed = 2)
    reference = _predict(model, tokenizer, rows)
    data = model.save_pretrained_gguf(
        tmp_path / "laya", tokenizer, quantization_method = ["f16", "q8_0"]
    )
    export_dir = tmp_path / "laya" / "gguf"
    temperatures = read_decision_temperatures(export_dir / "model-F16.gguf")
    assert temperatures["choice.3_5"] == pytest.approx(0.5)
    results = _serve_and_compare(export_dir, data, reference, rows, "laya", tmp_path)
    _assert_parity(results, "F16", 0.01)
    assert results["Q8_0"]["max_abs_prob_diff"] <= 0.02
