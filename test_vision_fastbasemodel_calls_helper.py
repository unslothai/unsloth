import ast
from pathlib import Path


def _find_vision():
    for p in [
        Path(__file__).resolve().parent / "unsloth" / "models" / "vision.py",
        Path(__file__).resolve().parents[1] / "unsloth" / "models" / "vision.py",
        Path("/mnt/disks/unslothai/ubuntu/workspace_25/github_review/unsloth-pr-5053-staging-3/unsloth/models/vision.py"),
    ]:
        if p.exists():
            return p
    raise FileNotFoundError("vision.py not found")


def test_vision_fastbasemodel_from_pretrained_calls_helper():
    """FastBaseModel.from_pretrained must invoke _attach_bnb_multidevice_hooks
    after the underlying model load so the inference hook path is reachable
    via the base vision loader, not only via llama."""
    src = _find_vision().read_text()
    tree = ast.parse(src)
    found = False
    for cls in ast.walk(tree):
        if not (isinstance(cls, ast.ClassDef) and cls.name == "FastBaseModel"):
            continue
        for fn in ast.walk(cls):
            if not (isinstance(fn, ast.FunctionDef) and fn.name == "from_pretrained"):
                continue
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "_attach_bnb_multidevice_hooks"
                ):
                    found = True
                    break
    assert found, "FastBaseModel.from_pretrained must call _attach_bnb_multidevice_hooks"
