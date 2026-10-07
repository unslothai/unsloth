# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# Unsloth's batched Clef head against Cloudflare's per-record head (vendored unmodified).

import hashlib
import json
import random
from pathlib import Path

import pytest
from real_accelerator import has_real_cuda

torch = pytest.importorskip("torch")

from unsloth._vendor.clef import joint_schema_model as reference
from unsloth.models import clef

HIDDEN, WIDTH, VOCAB = 64, 32, 500
VENDOR = Path(reference.__file__).parent


def test_vendored_files_match_the_manifest():
    manifest = json.loads((VENDOR / "clef_manifest.json").read_text(encoding = "utf-8"))
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == digest, name


def _records(seed, batch = 3):
    rng = random.Random(seed)
    records = []
    for _ in range(batch):
        length = rng.randint(130, 220)
        ids = tuple(rng.randrange(VOCAB) for _ in range(length))
        questions, position = [], 5
        for q in range(rng.randint(1, 4)):
            span = (position, position + rng.randint(1, 4))
            position = span[1] + 1
            options = []
            for _ in range(2 if q % 3 == 0 else rng.randint(2, 6)):
                options.append((position, position + rng.randint(1, 5)))
                position = options[-1][1] + 1
            questions.append(
                reference.EncodedQuestion(
                    f"q{q}", q % 3, span, tuple(options), tuple(map(str, range(len(options))))
                )
            )
        records.append(reference.EncodedRecord(ids, tuple(questions), "r"))
    return records


def _inputs(
    seed,
    dtype,
    hidden_dtype = None,
    scale = 3.0,
):
    torch.manual_seed(seed)
    head = clef.JointSchemaHead(HIDDEN, WIDTH, 2, 2, 4, 64).to(dtype)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    records = _records(seed)
    length = max(len(r.input_ids) for r in records)
    hidden = (torch.randn(len(records), length, HIDDEN, dtype = dtype) * scale + 1).to(
        hidden_dtype or dtype
    )
    ids = torch.zeros(len(records), length, dtype = torch.long)
    mask = torch.zeros(len(records), length, dtype = torch.long)
    for b, record in enumerate(records):
        ids[b, : len(record.input_ids)] = torch.tensor(record.input_ids)
        mask[b, : len(record.input_ids)] = 1
    embedding = torch.randn(VOCAB, HIDDEN, dtype = dtype)
    return head, hidden, ids, mask, records, embedding


def _padded(per_record, dtype):
    flat = [question for record in per_record for question in record]
    padded = torch.full((len(flat), max(len(z) for z in flat)), -1e4, dtype = dtype)
    for i, z in enumerate(flat):
        padded[i, : len(z)] = z
    return padded


def _run(head, forward, hidden, ids, mask, records, embedding, grad_out):
    head.zero_grad()
    hidden = hidden.clone().requires_grad_(True)
    out = forward(hidden, ids, mask, records, embedding)
    logits = out.flat_padded if hasattr(out, "flat_padded") else _padded(out, grad_out.dtype)
    valid = logits > -1e3
    (logits.to(grad_out.dtype) * grad_out * valid).sum().backward()
    grads = {n: p.grad.clone() for n, p in head.named_parameters() if p.grad is not None}
    return logits.detach(), hidden.grad.detach(), grads, valid


def _max_error(a, b):
    logits = (a[0] - b[0]).abs().max().item()
    hidden = (a[1] - b[1]).abs().max().item()
    grads = max(((a[2][n] - b[2][n]).abs().max()).item() for n in b[2])
    return logits, hidden, grads


@pytest.mark.parametrize("checkpoint", ["1", "0"])
@pytest.mark.parametrize("seed", [0, 1])
def test_batched_head_matches_cloudflares_head(monkeypatch, seed, checkpoint):
    monkeypatch.setenv("UNSLOTH_CLEF_CHECKPOINT", checkpoint)
    monkeypatch.setenv("UNSLOTH_CLEF_CHUNK", "16")  # several chunks per row
    results = {}
    for dtype in (torch.float64, torch.float32):
        head, hidden, ids, mask, records, embedding = _inputs(seed, dtype)
        generator = torch.Generator().manual_seed(seed)
        grad_out = torch.randn(
            sum(len(r.questions) for r in records), _o(records), generator = generator
        ).to(dtype)
        ours = _run(head, head.forward, hidden, ids, mask, records, embedding, grad_out)
        theirs = _run(
            head,
            lambda *a: reference.JointSchemaHead.forward(head, *a),
            hidden,
            ids,
            mask,
            records,
            embedding,
            grad_out,
        )
        assert torch.equal(ours[3], theirs[3])
        results[dtype] = (ours, theirs)
    # fp64: the same math up to summation order.
    logits, hidden_grad, grads = _max_error(*results[torch.float64])
    assert logits < 1e-12 and hidden_grad < 1e-12 and grads < 1e-10
    # fp32: no further from the fp64 oracle than Cloudflare's own fp32 head, plus one rounding step.
    oracle = results[torch.float64][1]
    for index, tolerance in ((0, 1e-6), (1, 1e-7)):
        ours_error = (results[torch.float32][0][index].double() - oracle[index]).abs().max().item()
        theirs_error = (
            (results[torch.float32][1][index].double() - oracle[index]).abs().max().item()
        )
        assert ours_error <= 1.5 * theirs_error + tolerance
    assert torch.equal(
        results[torch.float32][0][0].argmax(-1), results[torch.float32][1][0].argmax(-1)
    )


def _o(records):
    return max(len(q.option_spans) for r in records for q in r.questions)


def test_negative_control_is_caught():
    head, hidden, ids, mask, records, embedding = _inputs(0, torch.float64)
    grad_out = torch.randn(sum(len(r.questions) for r in records), _o(records), dtype = torch.float64)
    ours = _run(head, head.forward, hidden, ids, mask, records, embedding, grad_out)
    theirs = _run(
        head,
        lambda *a: reference.JointSchemaHead.forward(head, *a),
        hidden,
        ids,
        mask,
        records,
        embedding,
        grad_out,
    )
    perturbed = (ours[0] * (1 + 2**-7), *ours[1:])
    assert _max_error(perturbed, theirs)[0] > 1e-6


@pytest.mark.parametrize("hidden_dtype", [torch.float16, torch.bfloat16])
def test_16bit_hidden_states_into_fp32_head(monkeypatch, hidden_dtype):
    monkeypatch.setenv("UNSLOTH_CLEF_CHUNK", "16")
    # Large but finite fp16 activations, as a Qwen3.5 backbone produces.
    head, hidden, ids, mask, records, embedding = _inputs(
        2, torch.float32, hidden_dtype, scale = 1e4 if hidden_dtype == torch.float16 else 3.0
    )
    embedding = embedding.to(hidden_dtype)
    assert torch.isfinite(hidden).all() and hidden.abs().max() > 1e3 ** (
        hidden_dtype == torch.float16
    )
    grad_out = torch.randn(sum(len(r.questions) for r in records), _o(records))
    ours = _run(head, head.forward, hidden, ids, mask, records, embedding, grad_out)
    theirs = _run(head, head.forward_per_record, hidden, ids, mask, records, embedding, grad_out)
    assert torch.isfinite(ours[0]).all() and torch.isfinite(ours[1].float()).all()
    logits, _, grads = _max_error(ours, theirs)
    assert logits < 1e-4 and grads < 1e-3
    assert ours[1].dtype == hidden_dtype
    assert torch.equal(ours[0].argmax(-1), theirs[0].argmax(-1))


def test_kill_switch_runs_the_per_record_head(monkeypatch):
    head, hidden, ids, mask, records, embedding = _inputs(0, torch.float32)
    fast = head(hidden, ids, mask, records, embedding)
    monkeypatch.setenv("UNSLOTH_CLEF_FAST", "0")
    slow = head(hidden, ids, mask, records, embedding)
    assert hasattr(fast, "flat_padded") and not hasattr(slow, "flat_padded")
    for a, b in zip(fast, slow):
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol = 1e-5, atol = 1e-6)


def test_compile_gating(monkeypatch):
    import torch.utils._triton as triton_utils

    monkeypatch.setattr(triton_utils, "has_triton", lambda: True)
    assert not clef._compile_supported(torch.device("cpu"))
    assert clef._compile_supported(torch.device("cuda"))
    # ROCm reports cuda devices: compiled when its Triton is present.
    monkeypatch.setattr(torch.version, "hip", "6.4", raising = False)
    assert clef._compile_supported(torch.device("cuda"))
    # Windows ROCm / CUDA wheels without Triton: eager.
    monkeypatch.setattr(triton_utils, "has_triton", lambda: False)
    assert not clef._compile_supported(torch.device("cuda"))
    monkeypatch.setattr(triton_utils, "has_triton", lambda: True)
    monkeypatch.setenv("UNSLOTH_CLEF_COMPILE", "0")
    assert not clef._compile_supported(torch.device("cuda"))


def test_a_failed_compile_falls_back_to_eager(monkeypatch):
    head, hidden, ids, mask, records, embedding = _inputs(1, torch.float32)
    expected = head(hidden, ids, mask, records, embedding).flat_padded

    def broken(*args):
        raise RuntimeError("inductor failed")

    monkeypatch.setattr(clef, "_compiled_logits", lambda device: broken)
    monkeypatch.setenv("UNSLOTH_CLEF_COMPILE", "1")
    head.train()
    with pytest.warns(UserWarning, match = "running it eagerly"):
        got = head(hidden, ids, mask, records, embedding).flat_padded
    torch.testing.assert_close(got, expected)
    assert clef.os.environ["UNSLOTH_CLEF_COMPILE"] == "0"


@pytest.mark.skipif(
    not has_real_cuda(), reason = "Inductor CPU codegen for the head takes over 5 minutes"
)
def test_compiled_head_has_no_graph_breaks_and_matches(monkeypatch):
    from torch._dynamo.utils import counters

    cases = []
    # As the head's forward does: once Unsloth has patched torch's checkpoint (any model load in
    # this process), compiling needs torch's own one back.
    with clef._torch_checkpoint(True):
        for seed in (0, 1, 2):
            head, hidden, ids, mask, records, embedding = _inputs(seed, torch.float32)
            layout = clef.build_layout(records, hidden.device)
            eager = clef.batched_logits(head, hidden, embedding, layout, True, 16)
            cases.append((head, hidden, embedding, layout, eager))
        # Eager references first: after a model load Unsloth's patched layer_norm compiles itself
        # when called eagerly, which would add its own graphs to the count.
        torch._dynamo.reset()
        counters.clear()
        compiled = torch.compile(clef.batched_logits, fullgraph = True, dynamic = True)
        for head, hidden, embedding, layout, eager in cases:
            got = compiled(head, hidden, embedding, layout, True, 16)
            torch.testing.assert_close(got, eager, rtol = 1e-5, atol = 1e-5)
    assert not counters["graph_break"]
    # Three batches of different shapes, one graph.
    assert counters["stats"]["unique_graphs"] == 1


def test_autocast_matches_the_per_record_head(monkeypatch):
    # Training runs the head under bf16 autocast: the chunked memory projection must run in bf16
    # like the Linear it replaces, the norm and span means in fp32.
    monkeypatch.setenv("UNSLOTH_CLEF_CHUNK", "16")
    head, hidden, ids, mask, records, embedding = _inputs(3, torch.float32, torch.bfloat16)
    embedding = embedding.to(torch.bfloat16)
    grad_out = torch.randn(sum(len(r.questions) for r in records), _o(records))
    with torch.autocast("cpu", dtype = torch.bfloat16):
        ours = _run(head, head.forward, hidden, ids, mask, records, embedding, grad_out)
        theirs = _run(
            head, head.forward_per_record, hidden, ids, mask, records, embedding, grad_out
        )
    logits, _, grads = _max_error(ours, theirs)
    scale = theirs[0][theirs[3]].abs().max().item()
    assert logits <= 2e-2 * scale and grads < 5e-2
    assert torch.equal(ours[0].argmax(-1), theirs[0].argmax(-1))
