import json
from pathlib import Path


WORKDIR = Path(__file__).resolve().parents[2]
FRONTEND_PACKAGE_JSON = WORKDIR / "studio" / "frontend" / "package.json"


def test_assistant_ui_dependencies_are_pinned():
    package = json.loads(FRONTEND_PACKAGE_JSON.read_text(encoding = "utf-8"))
    dependencies = package["dependencies"]

    assert dependencies["@assistant-ui/core"] == "0.1.17"
    assert dependencies["@assistant-ui/react"] == "0.12.19"
    assert dependencies["@assistant-ui/react-markdown"] == "0.12.3"
    assert dependencies["@assistant-ui/react-streamdown"] == "0.1.2"
