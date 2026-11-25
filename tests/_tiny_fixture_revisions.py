# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pinned Hub revisions for the tiny configs the gloo tensor / pipeline parallel tests build from.

Those tests take a tiny repo's config.json and build random weights from it, so an upstream re-upload
changes every number they compare: trl-internal-testing/tiny-Qwen3ForCausalLM went from head_dim 128
to 16 in its Nov 2026 re-upload and moved parity values past their tolerance. A pinned revision keeps
the fixture fixed; moving a pin is a deliberate edit here.
"""

REVISIONS = {
    "hf-internal-testing/tiny-random-LlamaForCausalLM": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
    # The revision before the Nov 2026 re-upload (head_dim 128).
    "trl-internal-testing/tiny-Qwen3ForCausalLM": "25bb9b8963688237e94750f6f3e48f1947d3c7a3",
    "trl-internal-testing/tiny-Gemma2ForCausalLM": "5b25f835b5ed059c0c282e79f7227f4a929ab235",
    "hf-internal-testing/tiny-random-Gemma3ForCausalLM": "04b93a1a5f97c57286261b2d14a38a98d1e0c280",
    "hf-internal-testing/tiny-random-MistralForCausalLM": "7517176934d50d00707900b92ef6a95c782e51a2",
    "trl-internal-testing/tiny-Phi3ForCausalLM-3": "e82e6bc7a65dc695e68b5c80fc92df026948b290",
    "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5-Coder": "fb30b6ef8f125eaea368e40f402311f96bcd6908",
    "hf-internal-testing/tiny-random-StableLmForCausalLM": "c300fb0fb9aa5127fbec3cdda59b30db8326a89d",
    "hf-internal-testing/tiny-random-Olmo2ForCausalLM": "dcdf4af956c38843cac650c839e282349b92382c",
    # The revision before its Nov 2026 re-upload.
    "trl-internal-testing/tiny-Qwen3MoeForCausalLM": "6db57163fa56fa0e00d4b820169908ac0d3a219d",
    "trl-internal-testing/tiny-DeepseekV3ForCausalLM": "2fb366d060ad5cea8917909f28ec1219ccb1ad2d",
    "hf-internal-testing/tiny-random-OlmoeForCausalLM": "99e4bc8eb1a024e46da438c410d7e707ed49d15a",
    "trl-internal-testing/tiny-BloomForCausalLM": "fb6f154da5e58e70d7a62bafe35c0f0da99691cb",
}


def revision(repo):
    """The pinned revision for a Hub repo id; None for a local path (a saved checkpoint)."""
    return REVISIONS.get(repo)
