import ast
import sys
import types
from pathlib import Path


_BACKEND = Path(__file__).resolve().parent / "studio" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def test_no_dead_results_list_init():
    # After Fix C the empty-configs branch short-circuits before the executor
    # runs, so `results: list = []` initializer is no longer needed.
    src = (
        Path(__file__).resolve().parent
        / "studio"
        / "backend"
        / "routes"
        / "datasets.py"
    ).read_text()
    assert "results: list = []" not in src


def test_results_referenced_only_after_pool_map():
    src = (
        Path(__file__).resolve().parent
        / "studio"
        / "backend"
        / "routes"
        / "datasets.py"
    ).read_text()
    tree = ast.parse(src)

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_dataset_splits":
            target = node
            break
    assert target is not None

    # Ensure results is always assigned (Name with Store ctx) before any read.
    first_store = None
    first_load = None
    for n in ast.walk(target):
        if isinstance(n, ast.Name) and n.id == "results":
            if isinstance(n.ctx, ast.Store) and first_store is None:
                first_store = n.lineno
            if isinstance(n.ctx, ast.Load) and first_load is None:
                first_load = n.lineno
    assert first_store is not None, "results must be assigned in function"
    assert first_load is not None, "results must be read in function"
    assert first_store <= first_load, (
        "results must be assigned before any read; Fix C relies on the "
        "empty-configs guard raising before the executor path, but every "
        "remaining path must still initialize results via pool.map."
    )


def test_empty_configs_guard_before_executor_call():
    src = (
        Path(__file__).resolve().parent
        / "studio"
        / "backend"
        / "routes"
        / "datasets.py"
    ).read_text()
    # Find the order of key landmarks
    i_guard = src.find("has no registered configs or")
    i_pool = src.find("ThreadPoolExecutor(max_workers")
    assert i_guard != -1
    assert i_pool != -1
    assert (
        i_guard < i_pool
    ), "empty-configs guard must appear before ThreadPoolExecutor setup"


def test_no_redundant_if_configs_wrappers():
    src = (
        Path(__file__).resolve().parent
        / "studio"
        / "backend"
        / "routes"
        / "datasets.py"
    ).read_text()
    # After Fix C we should not wrap the executor block in `if configs:`;
    # the guard above raises early for the empty case.
    assert "        if configs:\n            max_workers" not in src
    # And the aggregate-failure guard no longer keys on `configs and`.
    assert "if configs and not all_splits" not in src
