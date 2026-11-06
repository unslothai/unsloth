"""GPU-free test for the generate-kwarg gate in vision.py
(_unsloth_generate_accepts_kwarg), covering both logits_to_keep injection and mm_token_type_ids
stripping, AST-extracted so no unsloth/CUDA import is needed."""

import ast, inspect, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION = os.path.join(HERE, "unsloth", "models", "vision.py")


def _load_helper():
    src = open(VISION, encoding = "utf-8").read()
    mod = ast.parse(src)
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_unsloth_generate_accepts_kwarg":
            ns = {"inspect": inspect}
            exec(ast.get_source_segment(src, node), ns)
            return ns["_unsloth_generate_accepts_kwarg"]
    raise AssertionError("_unsloth_generate_accepts_kwarg not found in vision.py")


accepts = _load_helper()


class PrepHasKwargs_ForwardHasKey:
    # **kwargs on prepare unions forward params; key in forward -> ACCEPTED.
    def prepare_inputs_for_generation(self, input_ids, **kwargs): ...
    def forward(
        self,
        input_ids,
        logits_to_keep = 0,
        **kwargs,
    ): ...


class PrepNoKwargs_ForwardHasKey:
    # no **kwargs -> forward not unioned; key only in forward -> REJECTED (gpt-oss shape).
    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask = None,
    ): ...
    def forward(
        self,
        input_ids,
        logits_to_keep = 0,
    ): ...


class PrepHasKeyDirectly:
    def prepare_inputs_for_generation(
        self,
        input_ids,
        logits_to_keep = 0,
    ): ...
    def forward(self, input_ids): ...


class NoPrepare:
    # no prepare -> empty args, no union -> REJECTED.
    def forward(
        self,
        input_ids,
        logits_to_keep = 0,
        **kwargs,
    ): ...


class VisionRejectsMM:
    # Qwen3-VL shape: neither prepare nor forward names mm_token_type_ids -> REJECTED.
    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask = None,
    ): ...
    def forward(
        self,
        input_ids,
        pixel_values = None,
    ): ...


class VisionAcceptsMM:
    def prepare_inputs_for_generation(self, input_ids, **kwargs): ...
    def forward(
        self,
        input_ids,
        mm_token_type_ids = None,
        **kwargs,
    ): ...


CASES = [
    (
        "prep(**kwargs)+forward(key)  -> accept",
        PrepHasKwargs_ForwardHasKey(),
        "logits_to_keep",
        True,
    ),
    (
        "prep(no kwargs)+forward(key) -> reject",
        PrepNoKwargs_ForwardHasKey(),
        "logits_to_keep",
        False,
    ),
    ("prep(key) direct             -> accept", PrepHasKeyDirectly(), "logits_to_keep", True),
    ("no prepare_inputs_for_gen    -> reject", NoPrepare(), "logits_to_keep", False),
    (
        "num_logits_to_keep variant   -> reject",
        PrepNoKwargs_ForwardHasKey(),
        "num_logits_to_keep",
        False,
    ),
    (
        "mm_token_type_ids not accepted -> reject (strip)",
        VisionRejectsMM(),
        "mm_token_type_ids",
        False,
    ),
    ("mm_token_type_ids accepted     -> keep", VisionAcceptsMM(), "mm_token_type_ids", True),
]


def test_generate_kwarg_gate():
    for name, model, key, expected in CASES:
        got = accepts(model, key)
        assert got is expected, f"{name}: got {got}, expected {expected}"


# transformers >= 5 injects logits_to_keep=1 only as a default, so popping unconditionally
# turns an explicit 0 into 1. Strip only values the strict validator rejects.


def _filter_logits_kwargs(model, kwargs):
    """The v5 branch of unsloth_base_fast_generate, as a testable function."""
    for key in ("logits_to_keep", "num_logits_to_keep"):
        if key in kwargs and not accepts(model, key):
            kwargs.pop(key, None)
    return kwargs


def test_v5_preserves_a_supported_caller_value():
    model = PrepHasKwargs_ForwardHasKey()
    # 0 means "all logits": rewriting it to 1 changes the output shape
    assert _filter_logits_kwargs(model, {"logits_to_keep": 0}) == {"logits_to_keep": 0}
    assert _filter_logits_kwargs(model, {"logits_to_keep": 5}) == {"logits_to_keep": 5}


def test_v5_strips_a_value_the_model_would_reject():
    model = PrepHasKwargs_ForwardHasKey()
    assert _filter_logits_kwargs(model, {"num_logits_to_keep": 1}) == {}
    assert _filter_logits_kwargs(NoPrepare(), {"logits_to_keep": 1}) == {}


def test_v5_leaves_other_kwargs_alone():
    model = PrepHasKwargs_ForwardHasKey()
    out = _filter_logits_kwargs(model, {"logits_to_keep": 2, "max_new_tokens": 8})
    assert out == {"logits_to_keep": 2, "max_new_tokens": 8}


def test_source_has_no_unconditional_pop():
    src = open(VISION, encoding = "utf-8").read()
    assert (
        'kwargs.pop("logits_to_keep", None)\n        kwargs.pop("num_logits_to_keep", None)'
        not in src
    ), "the v5 branch must not drop caller-supplied logits_to_keep unconditionally"


def test_the_v5_gate_uses_the_plain_release_sentinel():
    # unsloth_zoo's Version() maps any suffix to ".1", so a "5.0.0.dev0" sentinel sorts above 5.0.0 final.
    src = open(VISION, encoding = "utf-8").read()
    assert 'Version(transformers_version) < Version("5.0.0")' in src
    assert 'Version("5.0.0.dev0")' not in src


if __name__ == "__main__":
    test_generate_kwarg_gate()
    for name, _, _, _ in CASES:
        print(f"  [PASS] {name}")
    print("OK: generate-kwarg gate behaves like transformers _validate_model_kwargs")
