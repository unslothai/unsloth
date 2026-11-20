"""Check Word run tokens without importing RAG, Torch or Unsloth."""

import ast
import os
from pathlib import Path

import pytest


parse_xml = pytest.importorskip("docx.oxml").parse_xml
ROOT = Path(os.environ.get("UNSLOTH_REPO_ROOT", Path(__file__).resolve().parents[1]))
SOURCE = ROOT / "studio/backend/core/rag/parsers.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
FUNCTIONS = {node.name: node for node in TREE.body if isinstance(node, ast.FunctionDef)}
W_ASSIGN = next(
    node
    for node in TREE.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "_W" for target in node.targets)
)
W = ast.literal_eval(W_ASSIGN.value)
LEGACY = "_docx_run_text" not in FUNCTIONS
if LEGACY:
    assert "_docx_checkbox" not in FUNCTIONS
    expression = next(
        node.elt.orelse
        for node in ast.walk(FUNCTIONS["_docx_paragraph_text"])
        if isinstance(node, ast.GeneratorExp) and isinstance(node.elt, ast.IfExp)
    )
    assert ast.unparse(expression) == "node.text", "Unknown legacy DOCX extraction path"
    LEGACY_CODE = compile(ast.Expression(body=expression), str(SOURCE), "eval")

    def extract(run):
        return eval(LEGACY_CODE, {"node": run})
else:
    namespace = {"_W": W}
    helpers = [FUNCTIONS[name] for name in ("_docx_checkbox", "_docx_run_text")]
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(SOURCE), "exec"), namespace)
    extract = namespace["_docx_run_text"]


def box(state=""):
    return (
        '<w:fldChar w:fldCharType="begin"><w:ffData><w:checkBox>'
        + state
        + "</w:checkBox></w:ffData></w:fldChar>"
        '<w:instrText xml:space="preserve"> FORMCHECKBOX </w:instrText>'
        '<w:fldChar w:fldCharType="separate"/><w:fldChar w:fldCharType="end"/>'
    )


missing_glyph = pytest.mark.xfail(
    condition=LEGACY,
    strict=True,
    raises=AssertionError,
    reason="Legacy paragraph node.text drops form checkbox glyphs",
)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param(
            "<w:t>Before</w:t><w:tab/><w:t>After</w:t><w:br/><w:t>End</w:t>",
            "Before\tAfter\nEnd",
            id="plain-tokens",
        ),
        pytest.param(
            '<w:t>Before</w:t><w:fldChar w:fldCharType="begin"><w:ffData>'
            "<w:textInput/></w:ffData></w:fldChar><w:t>After</w:t>",
            "BeforeAfter",
            id="non-checkbox-field",
        ),
        pytest.param(
            box() + '<w:t xml:space="preserve"> First</w:t>',
            "\u2610 First",
            id="unchecked-before-text",
            marks=missing_glyph,
        ),
        pytest.param(
            '<w:t xml:space="preserve">First </w:t>' + box("<w:checked/>"),
            "First \u2612",
            id="checked-after-text",
            marks=missing_glyph,
        ),
        pytest.param(
            '<w:t xml:space="preserve">A </w:t>'
            + box()
            + '<w:t xml:space="preserve"> B </w:t>'
            + box("<w:checked/>")
            + '<w:t xml:space="preserve"> C</w:t>',
            "A \u2610 B \u2612 C",
            id="two-interleaved-boxes",
            marks=missing_glyph,
        ),
        pytest.param(
            "<w:t>Before</w:t><w:tab/>" + box("<w:checked/>") + "<w:br/><w:t>After</w:t>",
            "Before\t\u2612\nAfter",
            id="box-with-tab-break",
            marks=missing_glyph,
        ),
        pytest.param(
            box('<w:default w:val="1"/><w:checked w:val="off"/>')
            + '<w:t xml:space="preserve"> First</w:t>',
            "\u2610 First",
            id="explicit-off-overrides-default",
            marks=missing_glyph,
        ),
    ],
)
def test_legacy_checkbox_run_tokens(body, expected):
    run = parse_xml(f'<w:r xmlns:w="{W[1:-1]}">{body}</w:r>')
    before = bytes(run.xml, "utf-8")
    result = extract(run)
    print(repr(result))
    assert bytes(run.xml, "utf-8") == before
    assert result == expected
