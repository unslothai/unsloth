"""The declared-polls probe counts only requests this Studio serves.

``playwright_declared_polls.py`` fails CI on any /api/ path the app polls that the log budget
does not declare. It keyed on the path alone, so the Hub's direct calls to
``https://huggingface.co/api/models`` (from @huggingface/hub) counted as a Studio poll of
``/api/models``, a route Studio does not have, and failed the lane on whichever run caught
enough Hub search traffic in one dwell.
"""

import ast
from pathlib import Path
from urllib.parse import urlsplit

_SCRIPT = Path(__file__).resolve().parent / "playwright_declared_polls.py"


def _helpers():
    tree = ast.parse(_SCRIPT.read_text(encoding = "utf-8"))
    wanted = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("normalize", "studio_path")
    ]
    assert {node.name for node in wanted} == {"normalize", "studio_path"}
    namespace = {"urlsplit": urlsplit}
    exec(compile(ast.Module(body = wanted, type_ignores = []), str(_SCRIPT), "exec"), namespace)
    return namespace["studio_path"]


BASE = "http://127.0.0.1:18894"


def test_a_studio_request_keeps_its_path_without_the_query():
    studio_path = _helpers()
    assert studio_path(f"{BASE}/api/engines?_=1", BASE) == "/api/engines"


def test_hugging_face_api_models_is_not_a_studio_poll():
    studio_path = _helpers()
    assert studio_path("https://huggingface.co/api/models?search=gemma&limit=20", BASE) is None
    assert studio_path("https://huggingface.co/api/models/unsloth/gemma-3-270m-it", BASE) is None


def test_another_port_or_scheme_on_the_same_host_is_not_this_studio():
    studio_path = _helpers()
    assert studio_path("http://127.0.0.1:18895/api/engines", BASE) is None
    assert studio_path("https://127.0.0.1:18894/api/engines", BASE) is None


def test_the_request_handler_counts_through_studio_path():
    source = _SCRIPT.read_text(encoding = "utf-8")
    handler = source[
        source.index("def on_request(") : source.index('page.on("request", on_request)')
    ]
    assert (
        "studio_path(request.url, BASE)" in handler
    ), "on_request must filter by origin before counting, or cross-origin /api/ paths read as Studio polls"
