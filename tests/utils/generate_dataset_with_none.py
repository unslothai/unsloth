"""Synthetic chatml/sharegpt/alpaca datasets with intentional None/empty turns for dataset_none_detect.py."""

from datasets import Dataset

# ChatML (messages, role/content). pyarrow needs uniform column types, so
# messages=None / non-list (P1) rows live in a SEPARATE dataset.

_CHATML_ROWS = [
    # clean rows
    {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Name a colour."},
            {"role": "assistant", "content": "Blue."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Tell me a joke."},
            {"role": "assistant", "content": "Why did the chicken cross the road?"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Capital of France?"},
            {"role": "assistant", "content": "Paris."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Count to 3."},
            {"role": "assistant", "content": "1, 2, 3."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "A programming language."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Translate 'hello' to Spanish."},
            {"role": "assistant", "content": "Hola."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is gravity?"},
            {"role": "assistant", "content": "A fundamental force."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Who wrote Hamlet?"},
            {"role": "assistant", "content": "Shakespeare."},
        ]
    },
    # bad rows: None/empty turn content (all values are lists, so pyarrow is happy)
    {
        "messages": [
            {"role": "user", "content": None},
            {"role": "assistant", "content": "Sure!"},
        ]
    },  # None content
    {
        "messages": [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "OK."},
        ]
    },  # empty string
    {
        "messages": [
            {"role": "user", "content": "   "},
            {"role": "assistant", "content": "Got it."},
        ]
    },  # whitespace only
    {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": None},
        ]
    },  # None assistant
    {
        "messages": [
            {"role": "user", "content": None},
            {"role": "assistant", "content": None},
        ]
    },  # both None
    {
        "messages": [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
        ]
    },  # both empty
    {
        "messages": [
            {"role": "user", "content": "Anything?"},
            {"role": "assistant", "content": "  \t  "},
        ]
    },  # tab whitespace
    {
        "messages": [None, {"role": "assistant", "content": "Reply"}]
    },  # None turn element
]

# P1 rows: messages is None or non-list. Plain dicts (not an HF Dataset) since
# pyarrow can't mix list/non-list in one column; the runner mocks find_none_chatml.
_CHATML_P1_ROWS = [
    {"messages": None},  # whole column None
    {"messages": "not a list"},  # wrong type
]


def make_chatml_p1_rows() -> list:
    """Raw P1 rows (not an HF Dataset) for direct mock testing."""
    return list(_CHATML_P1_ROWS)


# ShareGPT (conversations, from/value)

_SHAREGPT_ROWS = [
    # clean
    {
        "conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there!"},
        ]
    },
    {
        "conversations": [
            {"from": "human", "value": "What time is it?"},
            {"from": "gpt", "value": "I don't know."},
        ]
    },
    {
        "conversations": [
            {"from": "human", "value": "Good morning"},
            {"from": "gpt", "value": "Good morning!"},
        ]
    },
    {
        "conversations": [
            {"from": "human", "value": "Tell me about AI."},
            {"from": "gpt", "value": "AI stands for Artificial Intelligence."},
        ]
    },
    {
        "conversations": [
            {"from": "human", "value": "Bye"},
            {"from": "gpt", "value": "Goodbye!"},
        ]
    },
    # bad
    {
        "conversations": [
            {"from": "human", "value": None},
            {"from": "gpt", "value": "Sure!"},
        ]
    },
    {
        "conversations": [
            {"from": "human", "value": ""},
            {"from": "gpt", "value": "OK."},
        ]
    },
    {
        "conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": None},
        ]
    },
    {"conversations": None},  # P1: whole column is None
    {"conversations": [None, {"from": "gpt", "value": "Hi"}]},
]

# Alpaca (instruction / output columns)

_ALPACA_ROWS = [
    # clean
    {
        "instruction": "Summarise this text.",
        "input": "The sky is blue.",
        "output": "The sky is blue.",
    },
    {"instruction": "Translate to French.", "input": "Hello", "output": "Bonjour"},
    {"instruction": "What is 10*10?", "input": "", "output": "100"},
    {
        "instruction": "Name the planets.",
        "input": "",
        "output": "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune",
    },
    {
        "instruction": "Write a haiku.",
        "input": "",
        "output": "Old pond — / frog jumps in / water's sound",
    },
    # bad
    {"instruction": None, "input": "", "output": "Some output"},
    {"instruction": "", "input": "", "output": "Some output"},
    {"instruction": "Valid instruction", "input": "", "output": None},
    {"instruction": None, "input": "", "output": None},
    {"instruction": "  ", "input": "", "output": ""},
]


def make_chatml_dataset() -> Dataset:
    return Dataset.from_list(_CHATML_ROWS)


def make_sharegpt_dataset() -> Dataset:
    return Dataset.from_list(_SHAREGPT_ROWS)


def make_alpaca_dataset() -> Dataset:
    return Dataset.from_list(_ALPACA_ROWS)


if __name__ == "__main__":
    print("Synthetic dataset sizes:")
    print(
        f"  chatml:   {len(_CHATML_ROWS)} rows (+ {len(_CHATML_P1_ROWS)} P1 mock rows)"
    )
    print(f"  sharegpt: {len(_SHAREGPT_ROWS)} rows")
    print(f"  alpaca:   {len(_ALPACA_ROWS)} rows")
    print(
        "\nImport make_chatml_dataset, make_sharegpt_dataset, make_alpaca_dataset, make_chatml_p1_rows."
    )
