"""FastModel config passthrough and nested task config handling."""

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOADER_PATH = REPO_ROOT / "unsloth" / "models" / "loader.py"
VISION_PATH = REPO_ROOT / "unsloth" / "models" / "vision.py"
UTILS_PATH = REPO_ROOT / "unsloth" / "models" / "_utils.py"
LLAMA_PATH = REPO_ROOT / "unsloth" / "models" / "llama.py"


def _source(path):
    return path.read_text()


def _class_method(tree, class_name, method_name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{class_name}.{method_name} not found")


def _assigns_from_kwargs_pop(method, target_name, key_name):
    for node in ast.walk(method):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == target_name for target in node.targets
        ):
            continue
        value = node.value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "pop"
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "kwargs"
            and value.args
            and isinstance(value.args[0], ast.Constant)
            and value.args[0].value == key_name
        ):
            continue
        return True
    return False


def _calls_name(method, name):
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
        for node in ast.walk(method)
    )


def _load_task_attr_helper():
    source = _source(UTILS_PATH)
    funcs = {
        node.name: ast.get_source_segment(source, node)
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }
    ns = {}
    for name in ("_config_set", "set_task_config_attr"):
        exec(funcs[name], ns)
    return ns["set_task_config_attr"]


def _load_loader_task_helpers():
    source = _source(LOADER_PATH)
    funcs = {
        node.name: ast.get_source_segment(source, node)
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }
    ns = {}
    for name in (
        "_config_get",
        "_config_diff",
        "_has_sequence_classification_architecture",
        "_get_user_task_config_attrs",
    ):
        exec(funcs[name], ns)
    return ns["_get_user_task_config_attrs"]


def test_fast_model_consumes_user_config_kwarg():
    tree = ast.parse(_source(LOADER_PATH))
    method = _class_method(tree, "FastModel", "from_pretrained")

    assert _assigns_from_kwargs_pop(method, "user_config", "config")


def test_fast_base_model_consumes_user_config_kwarg():
    tree = ast.parse(_source(VISION_PATH))
    method = _class_method(tree, "FastBaseModel", "from_pretrained")

    assert _assigns_from_kwargs_pop(method, "user_config", "config")


def test_fast_llama_model_consumes_user_config_kwarg():
    tree = ast.parse(_source(LLAMA_PATH))
    method = _class_method(tree, "FastLlamaModel", "from_pretrained")

    assert _assigns_from_kwargs_pop(method, "user_config", "config")


def test_fast_base_model_sets_task_attrs_on_nested_text_config():
    tree = ast.parse(_source(VISION_PATH))
    method = _class_method(tree, "FastBaseModel", "from_pretrained")

    assert _calls_name(method, "set_task_config_attr")


def test_fast_base_model_pops_problem_type_as_config_attr():
    source = _source(VISION_PATH)

    assert '("id2label", "label2id", "problem_type")' in source


def test_fast_model_uses_user_config_num_labels_for_task_model_selection():
    tree = ast.parse(_source(LOADER_PATH))
    method = _class_method(tree, "FastModel", "from_pretrained")

    assert _calls_name(method, "_get_user_task_config_attrs")


def test_fast_model_captures_user_config_num_labels_before_text_only_switch():
    source = _source(LOADER_PATH)

    fallback = source.index("task_config_attrs = _get_user_task_config_attrs(user_config)")
    text_only_switch = source.index("model_config = text_config")

    assert fallback < text_only_switch


def test_user_task_config_attrs_ignore_default_num_labels():
    get_user_task_config_attrs = _load_loader_task_helpers()

    class Config:
        num_labels = 2
        id2label = {0: "LABEL_0", 1: "LABEL_1"}
        label2id = {"LABEL_0": 0, "LABEL_1": 1}

        def to_diff_dict(self):
            return {}

    assert get_user_task_config_attrs(Config()) == {}


def test_user_task_config_attrs_preserve_custom_label_maps():
    get_user_task_config_attrs = _load_loader_task_helpers()

    class Config:
        num_labels = 2
        id2label = {0: "negative", 1: "positive"}
        label2id = {"negative": 0, "positive": 1}

        def to_diff_dict(self):
            return {"id2label": self.id2label, "label2id": self.label2id}

    attrs = get_user_task_config_attrs(Config())

    assert attrs["num_labels"] == 2
    assert attrs["id2label"] == {0: "negative", 1: "positive"}
    assert attrs["label2id"] == {"negative": 0, "positive": 1}


def test_user_task_config_attrs_preserve_explicit_dict_num_labels():
    get_user_task_config_attrs = _load_loader_task_helpers()

    assert get_user_task_config_attrs({"num_labels": 2}) == {"num_labels": 2}


def test_task_config_attr_updates_parent_and_text_config_objects():
    set_task_config_attr = _load_task_attr_helper()

    class TextConfig:
        pass

    class ParentConfig:
        def __init__(self):
            self.text_config = TextConfig()

        def get_text_config(self):
            return self.text_config

    config = ParentConfig()

    set_task_config_attr(config, "num_labels", 3)

    assert config.num_labels == 3
    assert config.text_config.num_labels == 3


def test_task_config_attr_updates_parent_and_text_config_dicts():
    set_task_config_attr = _load_task_attr_helper()
    config = {"text_config": {}}

    set_task_config_attr(config, "label2id", {"negative": 0, "positive": 1})

    assert config["label2id"] == {"negative": 0, "positive": 1}
    assert config["text_config"]["label2id"] == {"negative": 0, "positive": 1}


def test_task_config_attr_ignores_primitive_text_config():
    set_task_config_attr = _load_task_attr_helper()
    config = {"text_config": "not-a-config"}

    set_task_config_attr(config, "num_labels", 2)

    assert config["num_labels"] == 2
    assert config["text_config"] == "not-a-config"
