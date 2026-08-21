import ast
from pathlib import Path


def _for_training():
    path = Path(__file__).parents[1] / "unsloth" / "models" / "llama.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    model_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FastLlamaModel"
    )
    method = next(
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "for_training"
    )
    module = ast.Module(body = [method], type_ignores = [])
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["for_training"]


class _Model:
    training = False
    gradient_checkpointing = False

    def __init__(self):
        self._flag_for_generation = True

    def parameters(self):
        return ()

    def modules(self):
        return ()

    def train(self):
        self.training = True


class _PeftProxy:
    training = False

    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def parameters(self):
        return ()

    def modules(self):
        return ()

    def train(self):
        self.training = True


def test_for_training_deletes_a_generation_flag_delegated_by_a_peft_wrapper():
    model = _Model()
    proxy = _PeftProxy(model)
    assert hasattr(proxy, "_flag_for_generation")
    assert "_flag_for_generation" not in vars(proxy)

    _for_training()(proxy)

    assert not hasattr(model, "_flag_for_generation")
