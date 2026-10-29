#!/usr/bin/env python3
"""Minimal test for raw text training, without heavy dependencies."""

import sys
import os
import tempfile
import unicodedata
from pathlib import Path
import importlib.util


class MockDataset:
    def __init__(self, data_dict):
        self.data = data_dict
        self.column_names = list(data_dict.keys())

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.data[idx]
        elif isinstance(idx, int):
            return {key: values[idx] for key, values in self.data.items()}
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    @classmethod
    def from_dict(cls, data_dict):
        return cls(data_dict)


# __spec__ must be set or find_spec raises ValueError when transformers probes datasets.
datasets_mock = type(sys)("datasets")
datasets_mock.__spec__ = importlib.util.spec_from_loader("datasets", loader = None)
datasets_mock.Dataset = MockDataset

current_dir = os.path.dirname(__file__)
raw_text_path = os.path.join(os.path.dirname(current_dir), "unsloth", "dataprep", "raw_text.py")

spec = importlib.util.spec_from_file_location("raw_text", raw_text_path)
raw_text_module = importlib.util.module_from_spec(spec)

# Restore sys.modules after import; a leaked mock breaks later test modules' datasets imports.
_real_datasets = sys.modules.get("datasets")
sys.modules["datasets"] = datasets_mock
try:
    spec.loader.exec_module(raw_text_module)
finally:
    if _real_datasets is None:
        del sys.modules["datasets"]
    else:
        sys.modules["datasets"] = _real_datasets

RawTextDataLoader = raw_text_module.RawTextDataLoader
TextPreprocessor = raw_text_module.TextPreprocessor


def test_raw_text_loader():
    """Test basic RawTextDataLoader functionality."""

    class MockTokenizer:
        def __init__(self):
            self.eos_token = "</s>"
            self.eos_token_id = 2

        def __call__(
            self,
            text,
            return_tensors = None,
            add_special_tokens = False,
        ):
            words = text.split()
            token_ids = list(range(len(words)))

            if return_tensors == "pt":

                class MockTensor:
                    def __init__(self, data):
                        self.data = data

                    def __getitem__(self, idx):
                        return self.data

                    def __len__(self):
                        return len(self.data)

                    def tolist(self):
                        return self.data

                return {"input_ids": [MockTensor(token_ids)]}
            return {"input_ids": token_ids}

        def decode(
            self,
            token_ids,
            skip_special_tokens = False,
        ):
            return " ".join([f"word_{i}" for i in token_ids])

    test_content = "This is a test file for raw text training. " * 10
    with tempfile.NamedTemporaryFile(mode = "w", suffix = ".txt", delete = False) as f:
        f.write(test_content)
        test_file = f.name

    try:
        tokenizer = MockTokenizer()
        loader = RawTextDataLoader(tokenizer, chunk_size = 5, stride = 2)

        text_dataset = loader.load_from_file(test_file, return_tokenized = False)
        assert len(text_dataset) > 0, "Should create at least one chunk"
        assert "text" in text_dataset.column_names, "Dataset should have 'text' column"

        tokenized_dataset = loader.load_from_file(test_file, return_tokenized = True)
        assert len(tokenized_dataset) > 0, "Should create at least one tokenized chunk"
        assert (
            "input_ids" in tokenized_dataset.column_names
        ), "Dataset should have 'input_ids' column"
        assert (
            "attention_mask" in tokenized_dataset.column_names
        ), "Dataset should have 'attention_mask' column"

        first_sample = tokenized_dataset[0]
        assert isinstance(first_sample["input_ids"], list), "input_ids should be a list"
        assert isinstance(first_sample["attention_mask"], list), "attention_mask should be a list"
        assert len(first_sample["input_ids"]) == len(
            first_sample["attention_mask"]
        ), "input_ids and attention_mask should have same length"

        assert "labels" in tokenized_dataset.column_names, "Dataset should have 'labels' column"
        assert first_sample["labels"] == first_sample["input_ids"], "labels should match input_ids"

        try:
            bad_loader = RawTextDataLoader(tokenizer, chunk_size = 0, stride = 2)
            assert False, "Should raise ValueError for chunk_size=0"
        except ValueError as e:
            assert "chunk_size must be positive" in str(e)

        try:
            bad_loader = RawTextDataLoader(tokenizer, chunk_size = 5, stride = 10)
            assert False, "Should raise ValueError for stride >= chunk_size"
        except ValueError as e:
            assert "stride" in str(e) and "chunk_size" in str(e)

        # smart_chunk_text bypasses the constructor guard, so a bad stride would loop forever.
        long_text = "This is a test file for raw text training. " * 10
        valid_chunks = loader.smart_chunk_text(long_text, chunk_size = 5, stride = 2)
        assert len(valid_chunks) > 0, "Valid stride should produce chunks"

        try:
            loader.smart_chunk_text(long_text, chunk_size = 5, stride = 5)
            assert False, "Should raise ValueError for stride == chunk_size"
        except ValueError as e:
            assert "stride" in str(e) and "chunk_size" in str(e)

        try:
            loader.smart_chunk_text(long_text, chunk_size = 5, stride = 10)
            assert False, "Should raise ValueError for stride > chunk_size"
        except ValueError as e:
            assert "stride" in str(e) and "chunk_size" in str(e)

        preprocessor = TextPreprocessor()
        clean_text = preprocessor.clean_text("  messy   text  \n\n\n  ")
        assert "messy text" in clean_text, "Should clean text properly"
        paragraph_text = preprocessor.clean_text("Line 1\r\n\r\n\r\nLine 2")
        assert (
            paragraph_text == "Line 1\n\nLine 2"
        ), "Should preserve paragraph breaks while normalizing newlines"

        # Non-ASCII whitespace must become one space, not be deleted, or adjacent words fuse.
        unicode_whitespace_cases = [
            ("hello\u00a0world", "hello world"),
            ("hello\u202fworld", "hello world"),
            ("hello\u2009world", "hello world"),
            ("hello\u3000world", "hello world"),
            ("hello\u2002world", "hello world"),
            ("hello\x0bworld", "hello world"),
            ("hello\x0cworld", "hello world"),
        ]
        for raw, expected in unicode_whitespace_cases:
            assert preprocessor.clean_text(raw) == expected, (
                f"Should normalize Unicode/control whitespace to a single space " f"for {raw!r}"
            )

        mixed = preprocessor.clean_text("Section\u00a01\r\n\r\nBody\ftext\u202fhere")
        assert (
            mixed == "Section 1\n\nBody text here"
        ), "Should preserve paragraph breaks and normalize Unicode whitespace simultaneously"

        assert preprocessor.clean_text("a\tb") == "a b"
        assert preprocessor.clean_text("a\t\tb") == "a b"

        assert preprocessor.clean_text("foo \n\n bar") == "foo\n\nbar"

        assert preprocessor.clean_text("word1 \u200b word2") == "word1 word2"
        assert preprocessor.clean_text("a \ue000 b") == "a b"
        assert preprocessor.clean_text("prefix \ufffd suffix") == "prefix suffix"

        assert preprocessor.clean_text("foo \u200b\nbar") == "foo\nbar"
        assert preprocessor.clean_text("foo\n\ue000 bar") == "foo\nbar"
        assert preprocessor.clean_text("a \u200b\n\nb") == "a\n\nb"

        idempotent_inputs = [
            "  messy   text  \n\n\n  ",
            "Line 1\r\n\r\n\r\nLine 2",
            "hello\u00a0world",
            "Section\u00a01\r\n\r\nBody\ftext\u202fhere",
            "word1 \u00a9 word2",
            "a \u00e9 b",
        ]
        for raw in idempotent_inputs:
            once = preprocessor.clean_text(raw)
            twice = preprocessor.clean_text(once)
            assert once == twice, f"clean_text should be idempotent for {raw!r}"

        stats = preprocessor.validate_dataset(text_dataset)
        assert stats["total_samples"] > 0, "Should count samples"
        assert "warnings" in stats, "Should include warnings"

        # Plain ASCII: a cp1252 Windows console cannot encode a check mark.
        print("All tests passed!")
        return True

    # No broad except here: swallowing failures made every assertion above unable to fail.
    finally:
        os.unlink(test_file)


def test_clean_text_keeps_text_in_any_script():
    """Top level on purpose: an assertion inside test_raw_text_loader used to be swallowed."""
    preprocessor = TextPreprocessor()
    for script_text in [
        "Le caf\u00e9 \u00e9tait tr\u00e8s bon.",
        "\u00bfD\u00f3nde est\u00e1 la ni\u00f1a?",
        "Gr\u00f6\u00dfe und Stra\u00dfe",
        "\u673a\u5668\u5b66\u4e60\u5f88\u6709\u8da3\u3002",
        "\u3053\u3093\u306b\u3061\u306f\u3001\u4e16\u754c\u3002",
        "\u0645\u0631\u062d\u0628\u0627\u060c \u0628\u0627\u0644\u0639\u0627\u0644\u0645",
        "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440!",
        "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e\u0964",
        "\uc548\ub155\ud558\uc138\uc694.",
        "\u0393\u03b5\u03b9\u03ac \u03c3\u03bf\u03c5",
        "\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645",
        "\u0dc1\u0dca\u200d\u0dbb\u0dd3",
        "Price: \u20ac100, x \u2264 4, \u00a9 2026 Acme\u2122",
        "I \u2764\ufe0f you 1\ufe0f\u20e3 \U0001f468\u200d\U0001f469\u200d\U0001f467",
        "\U0001f3f4\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f",
        "\u06dd\u0661\u0662 \u0600\u0663",
        "\U00013000\U00013430\U00013001",
        "f\u2061(x) = a\u2062b",
        # Garay (Unicode 16) is unassigned in older Unicode databases and must survive anyway.
        "\U00010d50\U00010d51",
    ]:
        assert (
            preprocessor.clean_text(script_text) == script_text
        ), f"clean_text must keep text in any script: {script_text!r}"


def test_clean_text_drops_invisible_characters():
    """The control for the test above: invisible characters are still removed."""
    preprocessor = TextPreprocessor()
    for raw, expected in [
        ("a\x00b", "ab"),
        ("a\x1bb", "ab"),
        ("\ufeffhello", "hello"),
        ("co\u00adop", "coop"),
        ("a\u200bb", "ab"),
        ("\u202eabc", "abc"),
        ("a\u200eb\u2060c", "abc"),
        ("\u2066abc\u2069", "abc"),
        ("a\ue000b", "ab"),
        ("a\ufffdb", "ab"),
        ("a\uffffb", "ab"),
    ]:
        assert preprocessor.clean_text(raw) == expected, raw


def test_clean_text_decision_cannot_drift_between_interpreters():
    """Only Cc, Co and Cs are frozen by Unicode; other categories differ across Python versions."""
    immutable = {"Cc", "Co", "Cs"}
    assert set(raw_text_module._TextCharTable._DROP_CATEGORIES) <= immutable, (
        "clean_text may only key on the General_Category values Unicode has frozen "
        f"({sorted(immutable)}); the rest differ between Python versions."
    )

    preprocessor = TextPreprocessor()
    for codepoint in (0x00AD, 0x200B, 0x200E, 0x2060, 0x2066, 0xFEFF, 0xFFFD, 0xE0001):
        assert preprocessor.clean_text(f"a{chr(codepoint)}b") == "ab", hex(codepoint)

    # Unassigned (Cn) means newer than this interpreter's database, so it must survive.
    unassigned = [
        cp
        for cp in range(0x10D40, 0x10D90)
        if unicodedata.category(chr(cp)) == "Cn" and not (cp & 0xFFFE) == 0xFFFE
    ]
    for codepoint in unassigned[:8]:
        text = f"a{chr(codepoint)}b"
        assert preprocessor.clean_text(text) == text, hex(codepoint)


def test_smart_chunk_text_single_chunk_no_eos_returns_plain_list():
    """smart_chunk_text's single-chunk branch must return a plain list for
    input_ids even when the tokenizer has no eos_token_id, matching the
    multi-chunk branch's unconditional tolist()/list() conversion."""

    class MockTensor:
        def __init__(self, data):
            self.data = data

        def __getitem__(self, idx):
            return self.data

        def __len__(self):
            return len(self.data)

        def tolist(self):
            return self.data

    class MockTokenizerNoEos:
        def __init__(self):
            self.eos_token = None
            self.eos_token_id = None

        def __call__(
            self,
            text,
            return_tensors = None,
            add_special_tokens = False,
        ):
            token_ids = list(range(len(text.split())))
            if return_tensors == "pt":
                return {"input_ids": [MockTensor(token_ids)]}
            return {"input_ids": token_ids}

        def decode(
            self,
            token_ids,
            skip_special_tokens = False,
        ):
            return " ".join(f"word_{i}" for i in token_ids)

    loader = RawTextDataLoader(MockTokenizerNoEos(), chunk_size = 2048, stride = 512)
    result = loader.smart_chunk_text(
        "hello world short text", chunk_size = 2048, stride = 512, return_tokenized = True
    )
    input_ids = result[0]["input_ids"]
    assert isinstance(
        input_ids, list
    ), f"input_ids should be a plain list even without an eos_token_id, got {type(input_ids)}"
    assert input_ids == [0, 1, 2, 3], f"unexpected input_ids: {input_ids}"
    print("test_smart_chunk_text_single_chunk_no_eos_returns_plain_list passed")
    return True


def test_smart_chunk_text_no_eos_on_intermediate_full_chunks():
    """Only the final chunk gets EOS; mid-stride chunks stay exactly chunk_size long."""

    class WordTokenizer:
        def __init__(self):
            self.eos_token = "</s>"
            self.eos_token_id = -1

        def __call__(
            self,
            text,
            return_tensors = None,
            add_special_tokens = False,
        ):
            token_ids = list(range(len(text.split())))
            if return_tensors == "pt":
                return {"input_ids": [token_ids]}
            return {"input_ids": token_ids}

        def decode(
            self,
            token_ids,
            skip_special_tokens = False,
        ):
            return " ".join(f"word_{i}" for i in token_ids)

    text = " ".join(f"w{i}" for i in range(37))  # several full chunks plus a short tail
    loader = RawTextDataLoader(WordTokenizer(), chunk_size = 10, stride = 3)

    tokenized_chunks = loader.chunk_text(text, return_tokenized = True)
    assert len(tokenized_chunks) > 2, "test needs several chunks to cover the intermediate case"
    for i, chunk in enumerate(tokenized_chunks):
        ids = chunk["input_ids"]
        is_last = i == len(tokenized_chunks) - 1
        if is_last:
            assert ids[-1] == -1, f"last chunk should end with eos_token_id, got {ids}"
        else:
            assert (
                len(ids) == 10
            ), f"chunk {i} should stay exactly chunk_size (10), got {len(ids)}: {ids}"
            assert (
                ids[-1] != -1
            ), f"chunk {i} is not the last chunk but ends with eos_token_id: {ids}"

    text_chunks = loader.chunk_text(text, return_tokenized = False)
    assert len(text_chunks) > 2
    for i, chunk in enumerate(text_chunks):
        is_last = i == len(text_chunks) - 1
        assert (
            chunk.endswith("</s>") == is_last
        ), f"chunk {i} (last={is_last}) eos suffix mismatch: {chunk!r}"

    print("test_smart_chunk_text_no_eos_on_intermediate_full_chunks passed")
    return True


def test_load_from_file_skips_non_object_json_lines():
    """Non-object .jsonl lines (valid JSON, not dicts) are skipped, not fatal."""
    # Each case would hit data[field] and raise TypeError without the isinstance guard.
    with tempfile.NamedTemporaryFile("w", suffix = ".jsonl", delete = False) as f:
        f.write('"context"\n["text", "x"]\n42\n{"text": "keep this"}\n')
        path = f.name
    try:
        text = RawTextDataLoader(None)._read_file_by_format(path, "json_lines")
        assert text == "keep this", text
    finally:
        os.unlink(path)

    print("test_load_from_file_skips_non_object_json_lines passed")
    return True


def test_smart_chunk_text_empty_input_returns_no_chunks():
    """Empty/whitespace text must yield no chunks. This tokenizer keeps one token
    per char (like BPE/SentencePiece keeping spaces), so a len(tokens)==0 check
    would miss whitespace; the fix guards on text.strip() before tokenizing."""

    class WhitespacePreservingTokenizer:
        def __init__(self, eos_token_id):
            self.eos_token = "</s>" if eos_token_id is not None else None
            self.eos_token_id = eos_token_id

        def __call__(
            self,
            text,
            return_tensors = None,
            add_special_tokens = False,
        ):
            token_ids = [ord(c) % 100 for c in text]
            if return_tensors == "pt":
                return {"input_ids": [token_ids]}
            return {"input_ids": token_ids}

        def decode(
            self,
            token_ids,
            skip_special_tokens = False,
        ):
            return "".join(chr(32 + (t % 90)) for t in token_ids)

    for eos_token_id in (2, None):
        loader = RawTextDataLoader(
            WhitespacePreservingTokenizer(eos_token_id), chunk_size = 2048, stride = 512
        )
        # Whitespace tokenizes to >0 tokens, so [] proves the pre-tokenize guard.
        assert len(loader.tokenizer("   \n\t  ")["input_ids"]) > 0
        for text in ("", "   \n\t  "):
            for return_tokenized in (True, False):
                assert (
                    loader.smart_chunk_text(
                        text, chunk_size = 2048, stride = 512, return_tokenized = return_tokenized
                    )
                    == []
                ), f"no chunks for empty input (eos={eos_token_id}, text={text!r}, tokenized={return_tokenized})"
                assert loader.chunk_text(text, return_tokenized = return_tokenized) == [], (
                    f"chunk_text: no chunks for empty input "
                    f"(eos={eos_token_id}, text={text!r}, tokenized={return_tokenized})"
                )
    print("test_smart_chunk_text_empty_input_returns_no_chunks passed")
    return True


def test_negative_stride_is_rejected():
    """A negative stride passes both checks yet silently drops tokens between chunks; reject it."""

    class CharTokenizer:
        def __init__(self):
            self.eos_token = "</s>"
            self.eos_token_id = 2

        def __call__(
            self,
            text,
            return_tensors = None,
            add_special_tokens = False,
        ):
            token_ids = [ord(c) % 100 for c in text]
            if return_tensors == "pt":
                return {"input_ids": [token_ids]}
            return {"input_ids": token_ids}

        def decode(
            self,
            token_ids,
            skip_special_tokens = False,
        ):
            return "".join(chr(32 + (t % 90)) for t in token_ids)

    tokenizer = CharTokenizer()
    text = "x" * 100

    try:
        RawTextDataLoader(tokenizer, chunk_size = 10, stride = -5)
        assert False, "the constructor should reject a negative stride"
    except ValueError as e:
        assert "stride" in str(e) and "non-negative" in str(e), str(e)

    loader = RawTextDataLoader(tokenizer, chunk_size = 10, stride = 0)
    try:
        loader.smart_chunk_text(text, chunk_size = 10, stride = -5)
        assert False, "smart_chunk_text should reject a negative stride"
    except ValueError as e:
        assert "stride" in str(e) and "non-negative" in str(e), str(e)

    chunks = loader.smart_chunk_text(text, chunk_size = 10, stride = 0)
    assert len(chunks) > 0, "stride = 0 should still produce chunks"

    print("test_negative_stride_is_rejected passed")
    return True


def test_load_from_files_all_empty_raises():
    """All-empty file list must raise (like load_from_file) instead of returning
    a 0-row text-column dataset in return_tokenized mode."""

    class WhitespacePreservingTokenizer:
        eos_token = "</s>"
        eos_token_id = 2

        def __call__(
            self,
            text,
            return_tensors = None,
            add_special_tokens = False,
        ):
            token_ids = [ord(c) % 100 for c in text]
            if return_tensors == "pt":
                return {"input_ids": [token_ids]}
            return {"input_ids": token_ids}

    loader = RawTextDataLoader(WhitespacePreservingTokenizer(), chunk_size = 2048, stride = 512)
    paths = []
    try:
        for content in ("", "   \n\t  "):
            with tempfile.NamedTemporaryFile("w", suffix = ".txt", delete = False) as f:
                f.write(content)
                paths.append(f.name)
        raised = False
        try:
            loader.load_from_files(paths, return_tokenized = True)
        except ValueError as e:
            raised = True
            assert "empty" in str(e).lower() or "whitespace" in str(e).lower(), str(e)
        assert raised, "load_from_files must raise when all files are empty/whitespace"
    finally:
        for p in paths:
            os.unlink(p)
    print("test_load_from_files_all_empty_raises passed")
    return True


def test_validate_dataset_handles_tokenized_and_text_columns():
    """input_ids datasets need a tokenizer to decode for validation; without one it raises ValueError."""

    class MockTokenizer:
        def __init__(self):
            self.eos_token = "</s>"
            self.eos_token_id = 2

        def __call__(
            self,
            text,
            return_tensors = None,
            add_special_tokens = False,
        ):
            words = text.split()
            token_ids = list(range(len(words)))

            if return_tensors == "pt":

                class MockTensor:
                    def __init__(self, data):
                        self.data = data

                    def __getitem__(self, idx):
                        return self.data

                    def __len__(self):
                        return len(self.data)

                    def tolist(self):
                        return self.data

                return {"input_ids": [MockTensor(token_ids)]}
            return {"input_ids": token_ids}

        def decode(
            self,
            token_ids,
            skip_special_tokens = False,
        ):
            return " ".join(f"word_{i}" for i in token_ids)

    tokenizer = MockTokenizer()
    loader = RawTextDataLoader(tokenizer, chunk_size = 5, stride = 2)
    preprocessor = TextPreprocessor()

    test_content = "This is a test file for raw text training. " * 10
    with tempfile.NamedTemporaryFile(mode = "w", suffix = ".txt", delete = False) as f:
        f.write(test_content)
        test_file = f.name

    try:
        text_dataset = loader.load_from_file(test_file, return_tokenized = False)
        stats = preprocessor.validate_dataset(text_dataset)
        assert stats["total_samples"] > 0, "Should count samples from text column"
        assert "warnings" in stats

        tokenized_dataset = loader.load_from_file(test_file, return_tokenized = True)
        stats = preprocessor.validate_dataset(tokenized_dataset, tokenizer = tokenizer)
        assert stats["total_samples"] > 0, "Should count samples decoded from input_ids"
        assert "warnings" in stats
        assert stats["max_length"] > 0

        try:
            preprocessor.validate_dataset(tokenized_dataset)
            assert False, "Should raise ValueError when input_ids present but no tokenizer given"
        except ValueError as e:
            assert "tokenizer" in str(e).lower(), str(e)

        class FakeEmptyDataset:
            column_names = ["some_other_column"]

            def __len__(self):
                return 0

        try:
            preprocessor.validate_dataset(FakeEmptyDataset())
            assert False, "Should raise ValueError when neither text nor input_ids column exists"
        except ValueError as e:
            assert "text" in str(e).lower() and "input_ids" in str(e).lower(), str(e)

        print("test_validate_dataset_handles_tokenized_and_text_columns passed")
        return True

    finally:
        os.unlink(test_file)


def test_validate_dataset_accepts_objects_without_column_names():
    """Do not dispatch on column_names: it would reject mapping-like inputs that dataset['text'] accepts."""

    preprocessor = TextPreprocessor()
    texts = ["first sample with enough characters", "second sample with enough characters"]
    longest = max(len(t) for t in texts)

    class DuckTypedDataset:
        # Only __len__ + __getitem__, the pre-existing implicit contract.
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(next(iter(self.data.values())))

        def __getitem__(self, key):
            return self.data[key]

    stats = preprocessor.validate_dataset(DuckTypedDataset({"text": texts}))
    assert stats["total_samples"] == 2, stats
    assert stats["empty_samples"] == 0, stats
    assert stats["max_length"] == longest, stats

    stats = preprocessor.validate_dataset({"text": texts})
    assert stats["max_length"] == longest, stats

    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None:
        stats = preprocessor.validate_dataset(pd.DataFrame({"text": texts}))
        assert stats["total_samples"] == 2, stats
        assert stats["max_length"] == longest, stats

    print("test_validate_dataset_accepts_objects_without_column_names passed")
    return True


def test_validate_dataset_streams_instead_of_materialising_columns():
    """Stream via Dataset.iter(); dataset[column] materialises every row, which dominates peak memory."""

    class BatchedDataset:
        column_names = ["input_ids"]

        def __init__(self, rows):
            self.rows = rows
            self.materialised = 0

        def __len__(self):
            return len(self.rows)

        def iter(self, batch_size):
            for start in range(0, len(self.rows), batch_size):
                yield {"input_ids": self.rows[start : start + batch_size]}

        def __getitem__(self, key):
            self.materialised += 1
            return self.rows

    class Tokenizer:
        def decode(
            self,
            token_ids,
            skip_special_tokens = False,
        ):
            return " ".join(f"word_{i}" for i in token_ids)

    dataset = BatchedDataset([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    stats = TextPreprocessor().validate_dataset(dataset, tokenizer = Tokenizer())

    assert stats["total_samples"] == 3, stats
    assert stats["empty_samples"] == 0, stats
    assert dataset.materialised == 0, "column was materialised instead of streamed"

    print("test_validate_dataset_streams_instead_of_materialising_columns passed")
    return True


def test_validate_dataset_reports_zero_min_length_when_nothing_has_content():
    """min_length must be 0, not inf, when no sample has content, so normalise it outside the guard."""

    preprocessor = TextPreprocessor()

    for label, texts in (("all blank", ["", "   ", "\n"]), ("no rows", [])):
        stats = preprocessor.validate_dataset({"text": texts})
        assert stats["min_length"] == 0, (label, stats)
        assert stats["max_length"] == 0, (label, stats)
        assert not any("very short" in w for w in stats["warnings"]), (label, stats)

    stats = preprocessor.validate_dataset({"text": ["hi", "a much longer sample of text"]})
    assert stats["min_length"] == 2, stats
    assert any("very short" in w for w in stats["warnings"]), stats

    print("test_validate_dataset_reports_zero_min_length_when_nothing_has_content passed")
    return True


def _eos_tokenizer():
    class Tokenizer:
        eos_token_id = 99
        eos_token = "</s>"

        def __call__(self, text, **kwargs):
            return {"input_ids": [[int(word) for word in text.split()]]}

        def decode(self, ids, **kwargs):
            return "".join(f" {i}" for i in ids)

    return Tokenizer()


def test_tokenized_chunks_reserve_space_for_the_final_eos():
    for size in (2, 4):
        for stride in range(size):
            loader = RawTextDataLoader(_eos_tokenizer(), chunk_size = size, stride = stride)
            for count in (size - 1, size, 2 * size - stride, 2 * size, 3 * size):
                if count == 0:
                    continue
                chunks = loader.chunk_text(" ".join(map(str, range(count))))
                assert all(len(chunk["input_ids"]) <= size for chunk in chunks)
                assert all(chunk["input_ids"] != [99] for chunk in chunks)
                assert all(
                    len(chunk["attention_mask"]) == len(chunk["input_ids"]) for chunk in chunks
                )
                assert chunks[-1]["input_ids"][-1] == 99
                restored = list(chunks[0]["input_ids"])
                for chunk in chunks[1:]:
                    restored.extend(t for t in chunk["input_ids"] if t not in restored)
                assert restored == list(range(count)) + [99]
                # A chunk's first token is never a label, so the last content token must still be one.
                labelled = {t for chunk in chunks for t in chunk["input_ids"][1:]}
                assert count < 2 or count - 1 in labelled
                if stride == 0 and count % size == 0:
                    assert len(chunks[-1]["input_ids"]) == size
                dataset = loader.create_causal_dataset(chunks)
                assert dataset["labels"] == dataset["input_ids"]


def test_text_chunks_reserve_space_for_the_final_eos():
    # MLX truncates chunks at max_seq_length, which would drop an overflowing EOS.
    for size in (2, 4):
        for stride in range(size):
            loader = RawTextDataLoader(_eos_tokenizer(), chunk_size = size, stride = stride)
            for count in (size - 1, size, 2 * size - stride, 2 * size, 3 * size):
                if count == 0:
                    continue
                chunks = loader.chunk_text(" ".join(map(str, range(count))), return_tokenized = False)
                pieces = [chunk.replace("</s>", " </s>").split() for chunk in chunks]
                assert all(len(piece) <= size for piece in pieces)
                assert all(piece != ["</s>"] for piece in pieces)
                assert pieces[-1][-1] == "</s>"
                restored = list(pieces[0])
                for piece in pieces[1:]:
                    restored.extend(t for t in piece if t not in restored)
                assert restored == [str(i) for i in range(count)] + ["</s>"]
                labelled = {t for piece in pieces for t in piece[1:]}
                assert count < 2 or str(count - 1) in labelled


def test_chunk_size_one_keeps_the_final_eos_overflow():
    loader = RawTextDataLoader(_eos_tokenizer(), chunk_size = 1, stride = 0)
    for count in (1, 2, 3):
        chunks = loader.chunk_text(" ".join(map(str, range(count))))
        assert all(chunk["input_ids"] != [99] for chunk in chunks)
        assert all(len(chunk["attention_mask"]) == len(chunk["input_ids"]) for chunk in chunks)
        assert chunks[-1]["input_ids"] == [count - 1, 99]
        restored = [token for chunk in chunks for token in chunk["input_ids"]]
        assert restored == list(range(count)) + [99]


if __name__ == "__main__":
    success = test_raw_text_loader()
    test_clean_text_keeps_text_in_any_script()
    test_clean_text_drops_invisible_characters()
    test_clean_text_decision_cannot_drift_between_interpreters()
    success = test_smart_chunk_text_single_chunk_no_eos_returns_plain_list() and success
    success = test_smart_chunk_text_no_eos_on_intermediate_full_chunks() and success
    success = test_load_from_file_skips_non_object_json_lines() and success
    success = test_smart_chunk_text_empty_input_returns_no_chunks() and success
    success = test_load_from_files_all_empty_raises() and success
    success = test_negative_stride_is_rejected() and success
    success = test_validate_dataset_handles_tokenized_and_text_columns() and success
    success = test_validate_dataset_accepts_objects_without_column_names() and success
    success = test_validate_dataset_streams_instead_of_materialising_columns() and success
    success = test_validate_dataset_reports_zero_min_length_when_nothing_has_content() and success
    test_tokenized_chunks_reserve_space_for_the_final_eos()
    test_text_chunks_reserve_space_for_the_final_eos()
    test_chunk_size_one_keeps_the_final_eos_overflow()
    sys.exit(0 if success else 1)
