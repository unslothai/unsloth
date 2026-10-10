"""Speculative-decoding acceptance harness: maths, sampling, loop and reporting.

Pure-tensor and stand-in-model tests -- no downloads, no accelerator required.
"""

import json
import math
import pytest
import torch

from types import SimpleNamespace

from unsloth.spec_decoding.acceptance import (
    acceptance_prob,
    estimate_speedup,
    expected_accepted_length,
    residual_distribution,
)
from unsloth.spec_decoding.cli import _build_parser
from unsloth.spec_decoding.data import (
    _to_ids,
    build_sequences,
    builtin_prompts,
    load_texts,
)
from unsloth.spec_decoding.measure import AcceptanceReport, measure_acceptance
from unsloth.spec_decoding.report import to_json, to_text
from unsloth.spec_decoding.sampling import (
    SamplingParams,
    align_vocab,
    sampling_distribution,
    total_variation,
)
from unsloth.spec_decoding.simulate import SimulationResult, simulate_speculative

# ---------------------------------------------------------------- test_sampling


def test_params_validation():
    with pytest.raises(ValueError):
        SamplingParams(temperature = -1)
    with pytest.raises(ValueError):
        SamplingParams(top_p = 0.0)
    with pytest.raises(ValueError):
        SamplingParams(top_p = 1.5)
    assert SamplingParams(temperature = 0).is_greedy


def test_softmax_is_default():
    logits = torch.tensor([1.0, 2.0, 3.0])
    p = sampling_distribution(logits, SamplingParams(temperature = 1.0))
    torch.testing.assert_close(p, torch.softmax(logits, dim = -1))
    assert torch.isclose(p.sum(), torch.tensor(1.0))


def test_greedy_is_one_hot():
    logits = torch.tensor([[0.1, 5.0, 0.2], [3.0, 1.0, 2.0]])
    p = sampling_distribution(logits, SamplingParams(temperature = 0))
    assert torch.equal(p, torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))


def test_temperature_sharpens_and_flattens():
    logits = torch.tensor([1.0, 2.0, 3.0])
    cold = sampling_distribution(logits, SamplingParams(temperature = 0.1))
    hot = sampling_distribution(logits, SamplingParams(temperature = 10.0))
    assert cold.max() > sampling_distribution(logits, SamplingParams()).max() > hot.max()


def test_top_k_keeps_k_tokens():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    p = sampling_distribution(logits, SamplingParams(temperature = 1.0, top_k = 2))
    assert (p > 0).sum() == 2
    assert torch.isclose(p.sum(), torch.tensor(1.0))
    assert p[0] == 0 and p[1] == 0  # the three smallest logits dropped


def test_top_p_nucleus():
    # probs after softmax(temp=1): heavily peaked so top_p=0.5 keeps only the top token
    logits = torch.tensor([10.0, 1.0, 1.0, 1.0])
    p = sampling_distribution(logits, SamplingParams(temperature = 1.0, top_p = 0.5))
    assert (p > 0).sum() == 1
    assert p.argmax() == 0
    assert torch.isclose(p.sum(), torch.tensor(1.0))


def test_top_p_keeps_crossing_token():
    # uniform-ish: top_p just above 1/n must keep 2 tokens (the one that crosses included)
    logits = torch.zeros(4)
    p = sampling_distribution(logits, SamplingParams(temperature = 1.0, top_p = 0.30))
    assert (p > 0).sum() == 2


def test_batched_leading_dims_preserved():
    logits = torch.randn(3, 5, 100)
    p = sampling_distribution(logits, SamplingParams(temperature = 0.8, top_k = 10, top_p = 0.9))
    assert p.shape == (3, 5, 100)
    torch.testing.assert_close(p.sum(-1), torch.ones(3, 5), atol = 1e-5, rtol = 0)


def test_total_variation_bounds():
    p = torch.tensor([0.5, 0.5, 0.0])
    q = torch.tensor([0.0, 0.5, 0.5])
    assert torch.isclose(total_variation(p, p), torch.tensor(0.0))
    assert torch.isclose(total_variation(p, q), torch.tensor(0.5))
    assert torch.isclose(
        total_variation(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])), torch.tensor(1.0)
    )


def test_align_vocab_truncates_to_min():
    t = torch.randn(4, 32001)
    d = torch.randn(4, 32000)
    ta, da = align_vocab(t, d)
    assert ta.shape[-1] == da.shape[-1] == 32000
    # equal sizes pass through untouched
    ta2, da2 = align_vocab(t, t)
    assert ta2.shape[-1] == 32001


# ---------------------------------------------------------------- test_acceptance


def test_identical_distributions_accept_with_prob_one():
    p = torch.softmax(torch.randn(50), dim = -1)
    assert torch.isclose(acceptance_prob(p, p), torch.tensor(1.0), atol = 1e-6)


def test_disjoint_support_never_accepts():
    p = torch.tensor([0.5, 0.5, 0.0, 0.0])
    q = torch.tensor([0.0, 0.0, 0.5, 0.5])
    assert torch.isclose(acceptance_prob(p, q), torch.tensor(0.0), atol = 1e-6)


def test_acceptance_equals_one_minus_tv():
    torch.manual_seed(0)
    for _ in range(20):
        p = torch.softmax(torch.randn(64), dim = -1)
        q = torch.softmax(torch.randn(64), dim = -1)
        alpha = acceptance_prob(p, q)
        assert torch.isclose(alpha, 1.0 - total_variation(p, q), atol = 1e-6)


def test_acceptance_matches_monte_carlo_rejection_rule():
    # alpha should equal E_{x~q}[min(1, p(x)/q(x))]
    torch.manual_seed(1)
    p = torch.softmax(torch.randn(16), dim = -1)
    q = torch.softmax(torch.randn(16), dim = -1)
    analytic = float(acceptance_prob(p, q))

    g = torch.Generator().manual_seed(123)
    draws = torch.multinomial(q, 200_000, replacement = True, generator = g)
    ratio = torch.clamp(p[draws] / q[draws], max = 1.0)
    assert abs(ratio.mean().item() - analytic) < 0.01


def test_acceptance_prob_batched():
    p = torch.softmax(torch.randn(7, 3, 20), dim = -1)
    q = torch.softmax(torch.randn(7, 3, 20), dim = -1)
    a = acceptance_prob(p, q)
    assert a.shape == (7, 3)
    assert (a >= 0).all() and (a <= 1).all()


def test_acceptance_prob_shape_mismatch_raises():
    with pytest.raises(ValueError):
        acceptance_prob(torch.rand(4, 10), torch.rand(4, 11))


def test_residual_is_valid_distribution():
    p = torch.tensor([0.6, 0.3, 0.1])
    q = torch.tensor([0.2, 0.3, 0.5])
    r = residual_distribution(p, q)
    assert torch.isclose(r.sum(), torch.tensor(1.0), atol = 1e-6)
    assert (r >= 0).all()
    # mass only where p exceeded q
    assert r[2] == 0
    assert torch.isclose(r[1], torch.tensor(0.0), atol = 1e-6)
    assert torch.isclose(r[0], torch.tensor(1.0), atol = 1e-6)


def test_residual_falls_back_to_p_when_equal():
    p = torch.tensor([0.5, 0.5])
    r = residual_distribution(p, p)
    torch.testing.assert_close(r, p)


def test_expected_accepted_length_alpha_one():
    got = expected_accepted_length(1.0, gamma = 5)
    assert got.accepted_draft_tokens == 5.0
    assert got.tokens_per_cycle == 6.0


def test_expected_accepted_length_alpha_zero():
    got = expected_accepted_length(0.0, gamma = 5)
    assert got.accepted_draft_tokens == 0.0
    assert got.tokens_per_cycle == 1.0


def test_expected_accepted_length_matches_geometric_sum():
    alpha, gamma = 0.8, 4
    got = expected_accepted_length(alpha, gamma)
    want = sum(alpha**k for k in range(1, gamma + 1))
    assert math.isclose(got.accepted_draft_tokens, want, rel_tol = 1e-9)
    assert math.isclose(got.tokens_per_cycle, want + 1.0, rel_tol = 1e-9)


def test_expected_accepted_length_monte_carlo():
    alpha, gamma = 0.7, 6
    g = torch.Generator().manual_seed(7)
    trials = 100_000
    draws = torch.rand(trials, gamma, generator = g) < alpha
    # accepted prefix length = index of first False (or gamma if all True)
    first_reject = torch.where(
        draws.all(dim = 1),
        torch.full((trials,), gamma),
        (~draws).float().argmax(dim = 1),
    )
    got = expected_accepted_length(alpha, gamma)
    assert abs(first_reject.float().mean().item() - got.accepted_draft_tokens) < 0.03


def test_estimate_speedup_monotonic_and_sane():
    fast = estimate_speedup(tokens_per_cycle = 4.0, gamma = 4, cost_ratio = 0.1)
    slow = estimate_speedup(tokens_per_cycle = 4.0, gamma = 4, cost_ratio = 0.9)
    assert fast > slow > 0
    # a free draft that gets everything accepted approaches tokens_per_cycle
    assert math.isclose(estimate_speedup(5.0, 4, 0.0), 5.0)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        expected_accepted_length(1.5, 4)
    with pytest.raises(ValueError):
        expected_accepted_length(0.5, 0)
    with pytest.raises(ValueError):
        estimate_speedup(4.0, 4, -0.1)


# ---------------------------------------------------------------- test_simulate
"""Speculative-decoding loop tests using stand-in models (no `transformers`)."""


def const_logits_fn(dist: torch.Tensor):
    """A 'model' that returns the same next-token logits for any context."""
    logl = torch.log(dist.clamp_min(1e-9))

    def fn(ids: torch.Tensor) -> torch.Tensor:
        return logl.unsqueeze(0).repeat(ids.shape[0], 1)  # [T, V], as a real model returns

    return fn


PROMPT = torch.tensor([1, 2, 3, 4], dtype = torch.long)


def test_identical_models_accept_everything_greedy():
    d = torch.tensor([0.1, 0.7, 0.2])
    fn = const_logits_fn(d)
    r = simulate_speculative(
        fn, fn, PROMPT, gamma = 4, max_new_tokens = 40, sampling = SamplingParams(temperature = 0)
    )
    assert r.empirical_alpha == 1.0
    assert all(n == 4 for n in r.accepted_lengths)
    assert r.tokens_per_cycle == 5.0
    assert r.generated_tokens == 40


def test_identical_models_accept_everything_sampled():
    d = torch.softmax(torch.randn(20), dim = -1)
    fn = const_logits_fn(d)
    gen = torch.Generator().manual_seed(0)
    r = simulate_speculative(
        fn,
        fn,
        PROMPT,
        gamma = 3,
        max_new_tokens = 30,
        sampling = SamplingParams(temperature = 1.0),
        generator = gen,
    )
    # p == q  =>  ratio p/q == 1  =>  u < 1 always  =>  every examined token accepted
    assert r.empirical_alpha == 1.0


def test_fully_disagreeing_models_accept_nothing():
    target = const_logits_fn(torch.tensor([0.98, 0.01, 0.01]))
    draft = const_logits_fn(torch.tensor([0.01, 0.01, 0.98]))
    r = simulate_speculative(
        target, draft, PROMPT, gamma = 4, max_new_tokens = 20, sampling = SamplingParams(temperature = 0)
    )
    assert r.accepted_draft_tokens == 0
    assert all(n == 0 for n in r.accepted_lengths)
    # each cycle emits exactly the one corrected token
    assert r.generated_tokens == r.cycles
    assert r.tokens_per_cycle == 1.0


def test_empirical_alpha_matches_theory_gamma_one():
    # alpha = sum min(p, q) = min(.7,.5) + min(.3,.5) = .8
    target = const_logits_fn(torch.tensor([0.7, 0.3]))
    draft = const_logits_fn(torch.tensor([0.5, 0.5]))
    gen = torch.Generator().manual_seed(42)
    r = simulate_speculative(
        target,
        draft,
        PROMPT,
        gamma = 1,
        max_new_tokens = 4000,
        sampling = SamplingParams(temperature = 1.0),
        generator = gen,
    )
    assert abs(r.empirical_alpha - 0.8) < 0.03


def test_max_new_tokens_is_respected():
    fn = const_logits_fn(torch.tensor([0.2, 0.8]))
    r = simulate_speculative(
        fn, fn, PROMPT, gamma = 8, max_new_tokens = 10, sampling = SamplingParams(temperature = 0)
    )
    assert r.generated_tokens == 10


def test_censored_cycle_excluded_from_accepted_lengths():
    """A cycle cut short by max_new_tokens must not drag the mean down (bug 3)."""
    fn = const_logits_fn(torch.tensor([0.1, 0.9]))
    r = simulate_speculative(
        fn, fn, PROMPT, gamma = 4, max_new_tokens = 12, sampling = SamplingParams(temperature = 0)
    )
    # 2 full cycles (5 tokens each) then a censored 3rd contributing 2 tokens
    assert r.generated_tokens == 12
    assert r.cycles == 3
    assert r.accepted_lengths == [4, 4]  # censored cycle omitted
    assert r.mean_accepted_length == 4.0
    # but its examined tokens still count toward alpha
    assert r.proposed_draft_tokens == 10
    assert r.accepted_draft_tokens == 10


def test_eos_stops_generation():
    fn = const_logits_fn(torch.tensor([0.99, 0.01]))
    r = simulate_speculative(
        fn,
        fn,
        PROMPT,
        gamma = 4,
        max_new_tokens = 100,
        sampling = SamplingParams(temperature = 0),
        eos_token_id = 0,
    )
    assert r.generated_tokens == 1  # stops the moment the first EOS is emitted
    assert r.cycles == 1
    assert r.verified_per_cycle == [1] and r.accepted_per_cycle == [1]


def test_draft_may_return_1d_logits():
    dist = torch.tensor([0.3, 0.3, 0.4])

    def draft_1d(ids):
        return torch.log(dist)  # [V]

    target = const_logits_fn(dist)
    r = simulate_speculative(
        target, draft_1d, PROMPT, gamma = 2, max_new_tokens = 12, sampling = SamplingParams(temperature = 0)
    )
    assert r.generated_tokens == 12


def test_draft_never_proposes_token_outside_shared_vocab():
    """Bug 2: the draft's extra vocab entries must be dropped before sampling."""
    target = const_logits_fn(torch.tensor([0.5, 0.5]))  # V = 2
    # draft puts almost all mass on token 2, which the target does not have
    draft = const_logits_fn(torch.tensor([0.005, 0.005, 0.99]))  # V = 3
    r = simulate_speculative(
        target,
        draft,
        PROMPT,
        gamma = 3,
        max_new_tokens = 20,
        sampling = SamplingParams(temperature = 1.0),
        generator = torch.Generator().manual_seed(0),
    )
    assert r.generated_tokens == 20  # would have raised IndexError before the fix


def test_alpha_by_offset_is_conditional_and_flat_for_identical_models():
    fn = const_logits_fn(torch.softmax(torch.randn(10), dim = -1))
    r = simulate_speculative(
        fn,
        fn,
        PROMPT,
        gamma = 4,
        max_new_tokens = 100,
        sampling = SamplingParams(temperature = 1.0),
        generator = torch.Generator().manual_seed(3),
    )
    assert all(x == 1.0 for x in r.alpha_by_offset)


def test_alpha_by_offset_decays_when_draft_is_poor():
    """Later offsets are conditioned on earlier acceptances, so a weak draft that keeps
    getting rejected reaches them rarely; where reached, acceptance must still be a
    valid probability."""
    target = const_logits_fn(torch.tensor([0.6, 0.4]))
    draft = const_logits_fn(torch.tensor([0.35, 0.65]))
    r = simulate_speculative(
        target,
        draft,
        PROMPT,
        gamma = 5,
        max_new_tokens = 3000,
        sampling = SamplingParams(temperature = 1.0),
        generator = torch.Generator().manual_seed(11),
    )
    offs = r.alpha_by_offset
    assert len(offs) == 5
    assert all(0.0 <= x <= 1.0 for x in offs if not math.isnan(x))
    # offset 0 is always reached; the last offset is reached far less often
    reached0 = sum(1 for v in r.verified_per_cycle if v > 0)
    reached4 = sum(1 for v in r.verified_per_cycle if v > 4)
    assert reached0 > reached4


def test_alpha_by_offset_nan_when_offset_never_reached():
    target = const_logits_fn(torch.tensor([0.99, 0.01]))
    draft = const_logits_fn(torch.tensor([0.01, 0.99]))
    r = simulate_speculative(
        target, draft, PROMPT, gamma = 3, max_new_tokens = 10, sampling = SamplingParams(temperature = 0)
    )
    offs = r.alpha_by_offset
    assert offs[0] == 0.0  # always reached, never accepted
    assert math.isnan(offs[1]) and math.isnan(offs[2])


def test_stats_derive_consistently_from_cycle_lists():
    fn = const_logits_fn(torch.tensor([0.3, 0.7]))
    r = simulate_speculative(
        fn, fn, PROMPT, gamma = 3, max_new_tokens = 25, sampling = SamplingParams(temperature = 0)
    )
    assert len(r.verified_per_cycle) == len(r.accepted_per_cycle) == r.cycles
    assert r.proposed_draft_tokens == sum(r.verified_per_cycle)
    assert r.accepted_draft_tokens == sum(r.accepted_per_cycle)
    assert all(a <= v for a, v in zip(r.accepted_per_cycle, r.verified_per_cycle))


def test_extend_merges_runs():
    a = SimulationResult(
        gamma = 2, cycles = 1, generated_tokens = 3, verified_per_cycle = [2], accepted_per_cycle = [2]
    )
    b = SimulationResult(
        gamma = 2, cycles = 1, generated_tokens = 1, verified_per_cycle = [1], accepted_per_cycle = [0]
    )
    a.extend(b)
    assert a.cycles == 2 and a.generated_tokens == 4
    assert a.proposed_draft_tokens == 3 and a.accepted_draft_tokens == 2


# ---------------------------------------------------------------- test_measure
"""End-to-end `measure_acceptance` tests with stand-in models."""


class StubLM:
    """Returns the same next-token logits at every position, ignoring context.

    Deliberately has no ``generate``, so ``generate_continuation`` exercises its
    plain-forward fallback path.
    """

    def __init__(self, dist: torch.Tensor):
        self._logits = torch.log(dist.clamp_min(1e-9))
        self.config = SimpleNamespace(vocab_size = dist.numel())

    def __call__(self, ids: torch.Tensor):
        t = ids.shape[1]
        return SimpleNamespace(logits = self._logits.expand(1, t, -1).clone())

    def eval(self):
        return self

    def parameters(self):
        return iter(())

    def to(self, _device):
        return self


SEQS = [torch.arange(2, 12), torch.arange(1, 9), torch.arange(3, 20)]


def test_identical_models_alpha_is_one():
    m = StubLM(torch.tensor([0.1, 0.2, 0.3, 0.4]))
    r = measure_acceptance(m, m, SEQS, sampling = SamplingParams(temperature = 1.0), gamma = 4)
    assert r.alpha == pytest.approx(1.0, abs = 1e-5)
    assert r.tokens_per_cycle == pytest.approx(5.0, abs = 1e-4)
    assert r.n_sequences == 3
    assert r.scored_positions == sum(s.numel() - 1 for s in SEQS)
    assert r.scored_span == "prompt"


def test_known_alpha_exact():
    # every scored position is identical: alpha == sum min(p, q) == 0.8
    target = StubLM(torch.tensor([0.7, 0.3]))
    draft = StubLM(torch.tensor([0.5, 0.5]))
    r = measure_acceptance(target, draft, SEQS, sampling = SamplingParams(temperature = 1.0), gamma = 3)
    assert r.alpha == pytest.approx(0.8, abs = 1e-5)
    assert r.alpha_std == pytest.approx(0.0, abs = 1e-5)


def test_greedy_alpha_is_top1_agreement():
    target = StubLM(torch.tensor([0.6, 0.4]))  # argmax = 0
    draft = StubLM(torch.tensor([0.3, 0.7]))  # argmax = 1
    r = measure_acceptance(target, draft, SEQS, sampling = SamplingParams(temperature = 0), gamma = 2)
    assert r.alpha == pytest.approx(0.0, abs = 1e-6)
    assert any("greedy" in n for n in r.notes)


def test_prompt_scoring_is_flagged_in_notes():
    m = StubLM(torch.tensor([0.5, 0.5]))
    r = measure_acceptance(m, m, SEQS, gamma = 2)
    assert any("prompt tokens" in n for n in r.notes)


def test_continuation_scoring_changes_span_and_positions():
    """Bug 7: scoring should be able to target the model's own output, not the prompt."""
    m = StubLM(torch.tensor([0.25, 0.25, 0.25, 0.25]))
    r = measure_acceptance(
        m,
        m,
        SEQS,
        sampling = SamplingParams(temperature = 1.0),
        gamma = 2,
        continuation_tokens = 12,
        seed = 1,
    )
    assert r.scored_span == "target continuation"
    # 12 generated tokens per sequence, last position of each is unscorable
    assert r.scored_positions == 3 * 12
    assert not any("prompt tokens" in n for n in r.notes)


def test_no_alpha_by_position_without_simulation():
    """Bug 1: the bogus teacher-forced by-offset column is gone; only the simulation
    can produce a real one."""
    m = StubLM(torch.tensor([0.5, 0.5]))
    r = measure_acceptance(m, m, SEQS, gamma = 4)
    assert r.alpha_by_draft_offset is None
    assert not hasattr(r, "alpha_by_position")


def test_simulation_populates_offsets_and_speedup():
    target = StubLM(torch.tensor([0.7, 0.3]))
    draft = StubLM(torch.tensor([0.5, 0.5]))
    r = measure_acceptance(
        target,
        draft,
        SEQS,
        sampling = SamplingParams(temperature = 1.0),
        gamma = 1,
        simulate = True,
        simulate_max_new_tokens = 400,
        seed = 0,
    )
    assert r.simulated
    assert r.empirical_alpha == pytest.approx(0.8, abs = 0.05)
    assert r.alpha_by_draft_offset is not None and len(r.alpha_by_draft_offset) == 1
    assert r.empirical_est_speedup is not None


def test_simulation_is_reproducible():
    target = StubLM(torch.tensor([0.6, 0.4]))
    draft = StubLM(torch.tensor([0.45, 0.55]))
    kw = dict(
        sampling = SamplingParams(temperature = 1.0),
        gamma = 3,
        simulate = True,
        simulate_max_new_tokens = 120,
        seed = 7,
    )
    a = measure_acceptance(target, draft, SEQS, **kw)
    b = measure_acceptance(target, draft, SEQS, **kw)
    assert a.empirical_alpha == b.empirical_alpha
    assert a.empirical_mean_accepted_length == b.empirical_mean_accepted_length


def test_vocab_mismatch_is_noted():
    target = StubLM(torch.tensor([0.25, 0.25, 0.25, 0.25]))
    draft = StubLM(torch.tensor([0.3, 0.3, 0.4]))
    r = measure_acceptance(target, draft, SEQS, sampling = SamplingParams(temperature = 1.0), gamma = 2)
    assert r.vocab_truncated_to == 3
    assert any("truncated" in n for n in r.notes)


def test_vocab_mismatch_survives_simulation():
    """Bug 2 at the measure layer: a bigger-vocab draft must not crash the sim."""
    target = StubLM(torch.tensor([0.5, 0.5]))
    draft = StubLM(torch.tensor([0.005, 0.005, 0.99]))
    r = measure_acceptance(
        target,
        draft,
        SEQS,
        sampling = SamplingParams(temperature = 1.0),
        gamma = 2,
        simulate = True,
        simulate_max_new_tokens = 40,
    )
    assert r.simulated and r.empirical_alpha is not None


def test_gamma_validation():
    m = StubLM(torch.tensor([0.5, 0.5]))
    with pytest.raises(ValueError):
        measure_acceptance(m, m, SEQS, gamma = 0)


def test_report_round_trips_to_dict():
    m = StubLM(torch.tensor([0.5, 0.5]))
    r = measure_acceptance(m, m, SEQS, gamma = 2)
    d = r.to_dict()
    assert d["alpha"] == r.alpha and "notes" in d and d["scored_span"] == "prompt"


# ---------------------------------------------------------------- test_data


class WordTokenizer:
    """Whitespace tokenizer, enough to exercise build_sequences without transformers."""

    chat_template = None

    def __call__(
        self,
        text,
        return_tensors = None,
    ):
        ids = [hash(w) % 1000 for w in text.split()]
        return {"input_ids": torch.tensor([ids], dtype = torch.long)}


def test_builtin_prompts_nonempty_and_clean():
    prompts = builtin_prompts()
    assert len(prompts) >= 10
    assert all(p and not p.startswith("#") for p in prompts)


def test_load_texts_builtin_alias():
    assert load_texts(None) == load_texts("builtin") == builtin_prompts()


def test_load_texts_from_file_line_separated(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("first prompt\nsecond prompt\n\n")
    assert load_texts(str(f)) == ["first prompt", "second prompt"]


def test_load_texts_from_file_blank_line_separated(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("para one\nstill one\n\npara two\n")
    assert load_texts(str(f)) == ["para one\nstill one", "para two"]


def test_load_texts_unknown_source():
    with pytest.raises(ValueError):
        load_texts("nonexistent-path-xyz")


def test_build_sequences_filters_and_caps():
    tok = WordTokenizer()
    texts = ["one two three four five six", "too short", "a b c d e f g h i j"]
    seqs = build_sequences(texts, tok, max_samples = 5, max_length = 8, min_length = 4)
    assert all(isinstance(s, torch.Tensor) and s.dtype == torch.long for s in seqs)
    assert all(4 <= s.numel() <= 8 for s in seqs)
    assert len(seqs) == 2  # "too short" (2 tokens) dropped


def test_build_sequences_raises_when_all_filtered():
    tok = WordTokenizer()
    with pytest.raises(ValueError):
        build_sequences(["a", "b"], tok, min_length = 50)


class ChatTokenizer:
    """Mimics transformers>=5, where apply_chat_template returns a BatchEncoding whose
    [0] is a tokenizers.Encoding, not a tensor (the shape that broke --chat-template)."""

    chat_template = "{{ messages }}"

    def __call__(
        self,
        text,
        return_tensors = None,
    ):
        return {"input_ids": torch.tensor([[7] * len(text.split())], dtype = torch.long)}

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt = False,
        return_tensors = None,
    ):
        n = len(messages[0]["content"].split()) + 3  # +3 for the template wrapper
        return {
            "input_ids": torch.tensor([[5] * n], dtype = torch.long),
            "attention_mask": torch.ones(1, n, dtype = torch.long),
        }


def test_to_ids_accepts_every_shape():
    want = torch.tensor([1, 2, 3])
    for src in (
        torch.tensor([1, 2, 3]),  # 1-D tensor
        torch.tensor([[1, 2, 3]]),  # 2-D batch of one
        {"input_ids": torch.tensor([[1, 2, 3]])},  # BatchEncoding-like, 2-D
        {"input_ids": torch.tensor([1, 2, 3])},  # BatchEncoding-like, 1-D
        [1, 2, 3],  # plain list
        [[1, 2, 3]],  # nested list
    ):
        got = _to_ids(src)
        assert got.dtype == torch.long
        assert torch.equal(got, want), src


def test_to_ids_rejects_higher_rank():
    with pytest.raises(ValueError):
        _to_ids(torch.zeros(2, 3, 4))


def test_build_sequences_with_chat_template():
    """Regression: --chat-template was never exercised and broke on transformers 5."""
    tok = ChatTokenizer()
    seqs = build_sequences(
        ["one two three", "four five"],
        tok,
        max_samples = 5,
        max_length = 64,
        min_length = 1,
        chat_template = True,
    )
    assert len(seqs) == 2
    assert all(s.dim() == 1 and s.dtype == torch.long for s in seqs)
    assert seqs[0].numel() == 6 and seqs[1].numel() == 5  # words + template wrapper
    assert (seqs[0] == 5).all()  # took the chat-template path


def test_build_sequences_skips_chat_template_when_absent():
    tok = ChatTokenizer()
    tok.chat_template = None
    seqs = build_sequences(["one two three"], tok, min_length = 1, chat_template = True)
    assert (seqs[0] == 7).all()  # fell back to plain tokenisation


# ---------------------------------------------------------------- test_report_cli


def _report(**over):
    base = dict(
        alpha = 0.78,
        alpha_std = 0.05,
        scored_positions = 6000,
        n_sequences = 32,
        scored_span = "target continuation",
        gamma = 4,
        accepted_draft_tokens = 2.3,
        tokens_per_cycle = 3.3,
        est_speedup = 2.1,
        cost_ratio = 0.15,
        sampling = "temperature=0.7",
    )
    base.update(over)
    return AcceptanceReport(**base)


def test_to_text_contains_key_numbers():
    txt = to_text(_report(), target = "big", draft = "small")
    assert "small" in txt and "big" in txt
    assert "0.780" in txt
    assert "2.10x" in txt
    assert "target continuation" in txt


def test_to_text_hides_simulation_block_when_absent():
    txt = to_text(_report())
    assert "generative" not in txt
    assert "alpha by draft offset" not in txt


def test_to_text_shows_simulation_block_when_present():
    r = _report(
        simulated = True,
        empirical_alpha = 0.77,
        empirical_tokens_per_cycle = 3.2,
        empirical_mean_accepted_length = 2.2,
        empirical_est_speedup = 2.0,
        alpha_by_draft_offset = [0.9, 0.8, 0.7, 0.6],
    )
    txt = to_text(r)
    assert "generative" in txt
    assert "0.770" in txt
    assert "alpha by draft offset" in txt
    assert "0.90  0.80  0.70  0.60" in txt


def test_to_text_renders_nan_offsets():
    r = _report(
        simulated = True,
        empirical_alpha = 0.1,
        empirical_tokens_per_cycle = 1.1,
        empirical_mean_accepted_length = 0.1,
        empirical_est_speedup = 0.7,
        alpha_by_draft_offset = [0.1, math.nan, math.nan, math.nan],
    )
    txt = to_text(r)
    assert "0.10" in txt and "-" in txt


def test_to_json_round_trips():
    r = _report(notes = ["hello"])
    d = json.loads(to_json(r))
    assert d["alpha"] == 0.78
    assert d["notes"] == ["hello"]
    assert d["alpha_by_draft_offset"] is None


def test_cli_parser_requires_target_and_draft():
    p = _build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["measure", "--target", "x"])  # missing --draft
    ns = p.parse_args(["measure", "--target", "a", "--draft", "b"])
    assert ns.command == "measure"
    assert ns.gamma == 4 and ns.temperature == 1.0 and ns.data == "builtin"
    assert ns.continuation_tokens == 0


def test_cli_parser_flags():
    ns = _build_parser().parse_args(
        [
            "measure",
            "--target",
            "a",
            "--draft",
            "b",
            "--temperature",
            "0",
            "--gamma",
            "6",
            "--continuation-tokens",
            "64",
            "--simulate",
            "--use-unsloth",
            "--json",
        ]
    )
    assert ns.temperature == 0.0 and ns.gamma == 6
    assert ns.continuation_tokens == 64
    assert ns.simulate and ns.use_unsloth and ns.json
