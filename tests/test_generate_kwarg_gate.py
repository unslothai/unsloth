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
    # key directly on prepare -> ACCEPTED.
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
    # Qwen3-VL shape: neither prepare nor forward names mm_token_type_ids -> REJECTED (stripped).
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
    # forward names mm_token_type_ids and prepare unions it via **kwargs -> ACCEPTED (kept).
    def prepare_inputs_for_generation(self, input_ids, **kwargs): ...
    def forward(
        self,
        input_ids,
        mm_token_type_ids = None,
        **kwargs,
    ): ...


# (model, key, expected) per gate case.
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


# --- v5 logits-to-keep filtering ------------------------------------------
# transformers >= 5 injects logits_to_keep=1 in generate() itself, but the
# injection is guarded by `"logits_to_keep" not in model_kwargs`, so it is a
# DEFAULT. An explicit caller value must survive: popping unconditionally turns
# logits_to_keep=0 (give me the full sequence) into 1 without telling anyone.
# The only values that must be stripped are the ones the strict validator would
# raise on, which is exactly what the gate above predicts.


def _filter_logits_kwargs(model, kwargs):
    """The v5 branch of unsloth_base_fast_generate, as a testable function."""
    for key in ("logits_to_keep", "num_logits_to_keep"):
        if key in kwargs and not accepts(model, key):
            kwargs.pop(key, None)
    return kwargs


def test_v5_preserves_a_supported_caller_value():
    model = PrepHasKwargs_ForwardHasKey()
    # 0 means "all logits"; silently rewriting it to 1 changes the output shape.
    assert _filter_logits_kwargs(model, {"logits_to_keep": 0}) == {"logits_to_keep": 0}
    assert _filter_logits_kwargs(model, {"logits_to_keep": 5}) == {"logits_to_keep": 5}


def test_v5_strips_a_value_the_model_would_reject():
    # num_logits_to_keep was renamed away in v5, so the validator raises on it.
    model = PrepHasKwargs_ForwardHasKey()
    assert _filter_logits_kwargs(model, {"num_logits_to_keep": 1}) == {}
    # A VLM whose top-level forward has no logits_to_keep at all.
    assert _filter_logits_kwargs(NoPrepare(), {"logits_to_keep": 1}) == {}


def test_v5_leaves_other_kwargs_alone():
    model = PrepHasKwargs_ForwardHasKey()
    out = _filter_logits_kwargs(model, {"logits_to_keep": 2, "max_new_tokens": 8})
    assert out == {"logits_to_keep": 2, "max_new_tokens": 8}


def test_source_has_no_unconditional_pop():
    src = open(VISION).read()
    assert (
        'kwargs.pop("logits_to_keep", None)\n        kwargs.pop("num_logits_to_keep", None)'
        not in src
    ), "the v5 branch must not drop caller-supplied logits_to_keep unconditionally"


if __name__ == "__main__":
    test_generate_kwarg_gate()
    for name, _, _, _ in CASES:
        print(f"  [PASS] {name}")
    print("OK: generate-kwarg gate behaves like transformers _validate_model_kwargs")
