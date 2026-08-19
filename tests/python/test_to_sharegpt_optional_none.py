import ast
import re
from pathlib import Path


def _load_formatter_builders():
    # Extract _parse_combined_prompt and _create_formatter without importing
    # unsloth (importing unsloth needs unsloth_zoo / a GPU). Both are pure
    # Python and only use the `re` module.
    source = Path(__file__).parents[2] / "unsloth" / "chat_templates.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    wanted = {"_parse_combined_prompt", "_create_formatter"}
    funcs = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"re": re}
    module = ast.Module(body = funcs, type_ignores = [])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["_parse_combined_prompt"], namespace["_create_formatter"]


class _StubDataset:
    def __init__(self, column_names):
        self.column_names = column_names


def _render(merged_prompt, columns, batch):
    parse, create = _load_formatter_builders()
    possible_columns, final_optional_prompts = parse(merged_prompt, _StubDataset(columns))
    processor = create(possible_columns, final_optional_prompts, "text")
    return processor(batch)["text"]


def test_optional_block_missing_second_column_does_not_render_none():
    # A [[...]] block may reference several columns; only the first gates the
    # block. A later column that is None must not render as the literal "None".
    merged_prompt = "Location: [[{city}, {country}]] end"
    out = _render(
        merged_prompt,
        ["city", "country"],
        {"city": ["Paris"], "country": [None]},
    )
    assert out[0] == "Location: Paris,  end"
    assert "None" not in out[0]


def test_optional_block_all_columns_present_unchanged():
    merged_prompt = "Location: [[{city}, {country}]] end"
    out = _render(
        merged_prompt,
        ["city", "country"],
        {"city": ["Paris"], "country": ["France"]},
    )
    assert out[0] == "Location: Paris, France end"


def test_optional_block_gating_column_empty_is_dropped():
    # When the gating (first) column is empty the whole block is omitted; this
    # behaviour is unchanged by the None coercion.
    merged_prompt = "Location: [[{city}, {country}]] end"
    out = _render(
        merged_prompt,
        ["city", "country"],
        {"city": [""], "country": ["France"]},
    )
    assert out[0] == "Location:  end"


def test_single_column_optional_block_gated_out_on_none():
    # Single-column blocks were already gated correctly (the sole column is the
    # gate); confirm they stay unaffected.
    merged_prompt = "Name: [[{name}]]!"
    out = _render(merged_prompt, ["name"], {"name": [None, "Bob"]})
    assert out == ["Name: !", "Name: Bob!"]


def test_required_column_none_does_not_render_none():
    # A required (non-[[...]]) column that is None must not render as the
    # literal "None" either; coercion happens at the row source, so both the
    # required and optional branches are covered.
    merged_prompt = "Location: {city}, {country} end"
    out = _render(
        merged_prompt,
        ["city", "country"],
        {"city": ["Paris"], "country": [None]},
    )
    assert out[0] == "Location: Paris,  end"
    assert "None" not in out[0]


def test_optional_block_falsy_but_present_gating_value_still_renders():
    # The gate keeps a block whenever the first column is not "". A falsy but
    # real value (0) must not be treated as absent, so the block still renders.
    merged_prompt = "Count: [[{n}]]!"
    out = _render(merged_prompt, ["n"], {"n": [0]})
    assert out[0] == "Count: 0!"
