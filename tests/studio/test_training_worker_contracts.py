# SPDX-License-Identifier: AGPL-3.0-only

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
WORKER = REPO / "studio/backend/core/training/worker.py"


class _WorkerScopeVisitor(ast.NodeVisitor):
    def __init__(self, root: ast.FunctionDef):
        self.root = root
        self.eval_steps_bindings: list[ast.Assign] = []
        self.trainer_calls: list[ast.Call] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node is self.root:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return

    def visit_Assign(self, node: ast.Assign):
        if any(
            isinstance(target, ast.Name) and target.id == "eval_steps" for target in node.targets
        ):
            self.eval_steps_bindings.append(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "_train_worker":
            self.trainer_calls.append(node)
        self.generic_visit(node)


def test_training_worker_binds_eval_steps_before_forwarding_it():
    tree = ast.parse(WORKER.read_text(encoding = "utf-8"))
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_training_process"
    )
    visitor = _WorkerScopeVisitor(worker)
    visitor.visit(worker)

    assert len(visitor.trainer_calls) == 1
    trainer_call = visitor.trainer_calls[0]
    eval_steps_keyword = next(
        keyword for keyword in trainer_call.keywords if keyword.arg == "eval_steps"
    )

    assert isinstance(eval_steps_keyword.value, ast.Name)
    assert eval_steps_keyword.value.id == "eval_steps"
    assert len(visitor.eval_steps_bindings) == 1
    binding = visitor.eval_steps_bindings[0]
    assert binding.lineno < trainer_call.lineno
    assert isinstance(binding.value, ast.Call)
    assert isinstance(binding.value.func, ast.Attribute)
    assert isinstance(binding.value.func.value, ast.Name)
    assert binding.value.func.value.id == "config"
    assert binding.value.func.attr == "get"
    assert [argument.value for argument in binding.value.args] == ["eval_steps", 0.0]
