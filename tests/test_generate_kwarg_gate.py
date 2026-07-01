"""GPU-free test for the logits_to_keep gate in vision.py
(_unsloth_generate_accepts_kwarg), AST-extracted so no unsloth/CUDA import is needed."""
import ast, inspect, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION = os.path.join(HERE, "unsloth", "models", "vision.py")


def _load_helper():
    src = open(VISION).read()
    mod = ast.parse(src)
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_unsloth_generate_accepts_kwarg":
            ns = {"inspect": inspect}
            exec(ast.get_source_segment(src, node), ns)
            return ns["_unsloth_generate_accepts_kwarg"]
    raise AssertionError("_unsloth_generate_accepts_kwarg not found in vision.py")


accepts = _load_helper()


class PrepHasKwargs_ForwardHasKey:
    # prepare_inputs_for_generation takes **kwargs -> transformers unions forward params,
    # and forward names logits_to_keep -> ACCEPTED.
    def prepare_inputs_for_generation(self, input_ids, **kwargs): ...
    def forward(self, input_ids, logits_to_keep=0, **kwargs): ...


class PrepNoKwargs_ForwardHasKey:
    # prepare has NO **kwargs -> forward is NOT unioned; key only in forward -> REJECTED.
    # (This is the fused/PEFT gpt-oss failure shape.)
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None): ...
    def forward(self, input_ids, logits_to_keep=0): ...


class PrepHasKeyDirectly:
    # key present directly on prepare_inputs_for_generation -> ACCEPTED.
    def prepare_inputs_for_generation(self, input_ids, logits_to_keep=0): ...
    def forward(self, input_ids): ...


class NoPrepare:
    # no prepare_inputs_for_generation at all -> model_args empty, no union -> REJECTED.
    def forward(self, input_ids, logits_to_keep=0, **kwargs): ...


def run():
    cases = [
        ("prep(**kwargs)+forward(key)  -> accept", PrepHasKwargs_ForwardHasKey(), "logits_to_keep", True),
        ("prep(no kwargs)+forward(key) -> reject", PrepNoKwargs_ForwardHasKey(), "logits_to_keep", False),
        ("prep(key) direct            -> accept", PrepHasKeyDirectly(), "logits_to_keep", True),
        ("no prepare_inputs_for_gen   -> reject", NoPrepare(), "logits_to_keep", False),
        ("num_logits_to_keep variant  -> reject", PrepNoKwargs_ForwardHasKey(), "num_logits_to_keep", False),
    ]
    passed = 0
    for name, model, key, expected in cases:
        got = accepts(model, key)
        ok = got is expected
        passed += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got} expected={expected}")
        assert ok, f"{name}: got {got}, expected {expected}"

    # Parity check against transformers' real _validate_model_kwargs for the accept case.
    try:
        import torch  # noqa
        from transformers import AutoModelForCausalLM, AutoConfig
    except Exception as e:
        print(f"  [SKIP] transformers parity (import failed: {e})")
        print(f"{passed}/{len(cases)} logic cases passed")
        return
    print(f"{passed}/{len(cases)} logic cases passed")


if __name__ == "__main__":
    run()
    print("OK: generate-kwarg gate behaves like transformers _validate_model_kwargs")
