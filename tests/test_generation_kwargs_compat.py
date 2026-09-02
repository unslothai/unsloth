import ast
import inspect
from pathlib import Path


LLAMA_PATH = Path(__file__).parents[1] / "unsloth" / "models" / "llama.py"


def _load_generation_logits_kwarg_name():
    tree = ast.parse(LLAMA_PATH.read_text(encoding = "utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_generation_logits_kwarg_name"
    )
    module = ast.Module(body = [function], type_ignores = [])
    namespace = {"inspect": inspect}
    exec(compile(module, str(LLAMA_PATH), "exec"), namespace)
    return namespace["_generation_logits_kwarg_name"]


def test_generation_logits_kwarg_name_supports_transformers_spellings():
    select = _load_generation_logits_kwarg_name()

    class NewModel:
        def forward(
            self,
            input_ids = None,
            logits_to_keep = 0,
        ):
            pass

    class LegacyModel:
        def forward(
            self,
            input_ids = None,
            num_logits_to_keep = 0,
        ):
            pass

    assert select(NewModel()) == "logits_to_keep"
    assert select(LegacyModel()) == "num_logits_to_keep"


def test_generation_logits_kwarg_name_omits_unsupported_optimization():
    select = _load_generation_logits_kwarg_name()

    class PhiLikeModel:
        def forward(
            self,
            input_ids = None,
            attention_mask = None,
        ):
            pass

    class OpaqueModel:
        forward = None

    assert select(PhiLikeModel()) is None
    assert select(OpaqueModel()) is None


def test_generation_logits_kwarg_name_inspects_peft_base_model():
    select = _load_generation_logits_kwarg_name()

    class PhiLikeModel:
        def forward(self, input_ids = None):
            pass

    class PeftLikeWrapper:
        def __init__(self):
            self.base = PhiLikeModel()

        def get_base_model(self):
            return self.base

        def forward(
            self,
            input_ids = None,
            num_logits_to_keep = 0,
            logits_to_keep = 0,
        ):
            pass

    assert select(PeftLikeWrapper()) is None


def test_generation_paths_use_signature_helper():
    tree = ast.parse(LLAMA_PATH.read_text(encoding = "utf-8"))
    callers = set()
    for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
        if any(
            isinstance(call.func, ast.Name) and call.func.id == "_generation_logits_kwarg_name"
            for call in ast.walk(function)
            if isinstance(call, ast.Call)
        ):
            callers.add(function.name)

    assert callers == {"PeftModel_fast_forward", "unsloth_fast_generate"}
