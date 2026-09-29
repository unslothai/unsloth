import ast
import re
from pathlib import Path


def _load_formatter_builders():
    # Extract _parse_combined_prompt and _create_formatter without importing unsloth (importing unsloth needs
    # unsloth_zoo / a GPU).
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
    # When the gating (first) column is empty the whole block is omitted; this behaviour is unchanged by the None
    # coercion.
    merged_prompt = "Location: [[{city}, {country}]] end"
    out = _render(
        merged_prompt,
        ["city", "country"],
        {"city": [""], "country": ["France"]},
    )
    assert out[0] == "Location:  end"


def test_single_column_optional_block_gated_out_on_none():
    merged_prompt = "Name: [[{name}]]!"
    out = _render(merged_prompt, ["name"], {"name": [None, "Bob"]})
    assert out == ["Name: !", "Name: Bob!"]


def test_required_column_none_does_not_render_none():
    # A required (non-[[...]]) column that is None must not render as the literal "None" either; coercion happens at the
    # row source, so both the required and optional branches are covered.
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


def _load_to_sharegpt():
    # Same trick as above: pull to_sharegpt and the two helpers it calls out of the source without importing unsloth.
    source = Path(__file__).parents[2] / "unsloth" / "chat_templates.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    wanted = {"_parse_combined_prompt", "_create_formatter", "to_sharegpt"}
    funcs = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"re": re}
    module = ast.Module(body = funcs, type_ignores = [])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["to_sharegpt"]


def _alpaca():
    from datasets import Dataset
    return Dataset.from_dict(
        {
            "instruction": ["What is 2+2?", "Capital of France?"],
            "output": ["4", "Paris"],
        }
    )


def test_default_merged_prompt_keeps_the_input_column():
    to_sharegpt = _load_to_sharegpt()
    converted = to_sharegpt(_alpaca())
    users = [row["conversations"][0]["value"] for row in converted]
    assert users == ["What is 2+2?", "Capital of France?"]


def test_default_merged_prompt_with_renamed_columns():
    from datasets import Dataset

    # merged_prompt is optional: without one, merged_column_name names a column that is already there.
    to_sharegpt = _load_to_sharegpt()
    dataset = Dataset.from_dict({"Query": ["123?"], "Answer": ["456"]})
    converted = to_sharegpt(
        dataset,
        merged_column_name = "Query",
        output_column_name = "Answer",
    )
    assert converted[0]["conversations"] == [
        {"from": "human", "value": "123?"},
        {"from": "gpt", "value": "456"},
    ]


def test_explicit_merged_prompt_still_merges():
    from datasets import Dataset

    to_sharegpt = _load_to_sharegpt()
    dataset = Dataset.from_dict({"instruction": ["Sum"], "input": ["2+2"], "output": ["4"]})
    converted = to_sharegpt(dataset, merged_prompt = "{instruction}\n{input}")
    assert converted[0]["conversations"][0]["value"] == "Sum\n2+2"


def test_missing_input_column_says_which_column_is_missing():
    from datasets import Dataset

    to_sharegpt = _load_to_sharegpt()
    dataset = Dataset.from_dict({"prompt": ["hi"], "output": ["yo"]})
    try:
        to_sharegpt(dataset)
    except KeyError as error:
        assert "instruction" in str(error)
        assert "prompt" in str(error)
    else:
        raise AssertionError("expected a KeyError naming the missing input column")


def test_conversation_extension_keeps_the_real_prompts():
    to_sharegpt = _load_to_sharegpt()
    converted = to_sharegpt(_alpaca(), conversation_extension = 2)
    values = [turn["value"] for turn in converted[0]["conversations"]]
    assert "" not in values
    assert len(converted[0]["conversations"]) == 4


def test_null_cells_do_not_render_as_the_word_none():
    from datasets import Dataset

    to_sharegpt = _load_to_sharegpt()
    dataset = Dataset.from_dict({"instruction": ["ok", None], "output": [None, "fine"]})
    converted = to_sharegpt(dataset)
    values = [turn["value"] for row in converted for turn in row["conversations"]]

    assert "None" not in values
    assert values == ["ok", "", "", "fine"]


def test_null_cells_match_the_merged_prompt_path():
    from datasets import Dataset

    to_sharegpt = _load_to_sharegpt()
    rows = {"instruction": ["ok", None], "output": ["a", "b"]}

    merged = to_sharegpt(Dataset.from_dict(rows), merged_prompt = "{instruction}")
    plain = to_sharegpt(Dataset.from_dict(rows))

    assert [r["conversations"] for r in merged] == [r["conversations"] for r in plain]
