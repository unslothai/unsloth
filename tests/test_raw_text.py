#!/usr/bin/env python3
"""Minimal test for raw text training, without heavy dependencies."""

import sys
import os
import tempfile
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


# __spec__ must be set so importlib.util.find_spec doesn't raise ValueError when transformers' import_utils later probes
# for the real `datasets` package.
datasets_mock = type(sys)("datasets")
datasets_mock.__spec__ = importlib.util.spec_from_loader("datasets", loader = None)
datasets_mock.Dataset = MockDataset

current_dir = os.path.dirname(__file__)
raw_text_path = os.path.join(os.path.dirname(current_dir), "unsloth", "dataprep", "raw_text.py")

spec = importlib.util.spec_from_file_location("raw_text", raw_text_path)
raw_text_module = importlib.util.module_from_spec(spec)

# The mock is only in place while raw_text executes its `from datasets import Dataset`.
# Leaving it in sys.modules poisoned every later test module in the same session: `from datasets import IterableDataset`
# then raised ImportError and tests/utils/test_packing.py failed to collect.
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

        # smart_chunk_text validation: called directly, chunk_size/stride are its own arguments and bypass the
        # constructor guard, so it must guard itself or an invalid stride makes `start_idx += chunk_size - stride`
        # non-positive and the chunking loop never terminates (hangs).
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

        # Non-ASCII horizontal whitespace (NBSP, thin/em/ideographic space, VT, FF) must normalize to one ASCII space,
        # not be deleted, or adjacent words fuse on HTML/PDF/OCR input.
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

        # Spaces around newlines trimmed on both sides, even across multiple newlines.
        assert preprocessor.clean_text("foo \n\n bar") == "foo\n\nbar"

        # Stripping a non-ASCII char between spaces must not leave a double space
        assert preprocessor.clean_text("word1 \u00a9 word2") == "word1 word2"
        assert preprocessor.clean_text("a \u00e9 b") == "a b"
        assert preprocessor.clean_text("prefix \U0001f600 suffix") == "prefix suffix"

        # Stripping a non-ASCII char adjacent to a newline must not leave a stray space.
        assert preprocessor.clean_text("foo \u00e9\nbar") == "foo\nbar"
        assert preprocessor.clean_text("foo\n\u00e9 bar") == "foo\nbar"
        # The double-space collapse must not swallow a paragraph break near a non-ASCII char.
        assert preprocessor.clean_text("a \u00a9\n\nb") == "a\n\nb"

        # Idempotence: clean_text twice == once.
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

        print("✅ All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    finally:
        os.unlink(test_file)


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
    print("✅ test_smart_chunk_text_single_chunk_no_eos_returns_plain_list passed!")
    return True


def test_load_from_file_skips_non_object_json_lines():
    """Non-object .jsonl lines (valid JSON, not dicts) are skipped, not fatal."""
    # "context" contains "text", ["text"] holds it, 42 isn't iterable -- each
    # would reach data[field] and raise TypeError without the isinstance guard.
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
    """chunk_size > 0 and stride < chunk_size both pass for a negative stride, but
    `start_idx += chunk_size - stride` then advances by MORE than chunk_size, so the
    tokens between one chunk's end and the next chunk's start are never emitted.
    Nothing raises and nothing is logged, so the caller trains on a corpus with holes
    in it: chunk_size = 10 with stride = -5 emits 70 of a 100 token document."""

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

    # Both entry points validate stride, so both need the lower bound.
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

    # stride = 0 stays valid: it just means the chunks do not overlap.
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
    """validate_dataset() must work for both dataset shapes:
    - text-column datasets (return_tokenized=False), no tokenizer needed
    - input_ids-column datasets (return_tokenized=True, the default), which
      require a tokenizer to decode back to text for validation
    Also asserts the clear ValueError when input_ids is present but no
    tokenizer was passed, and when neither column exists.
    """

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
    """Dispatching on `column_names` must not narrow the accepted input types.

    validate_dataset() read dataset["text"] directly, so it worked for any
    mapping-like object: DataFrames, plain dicts, custom __getitem__ wrappers.
    """

    preprocessor = TextPreprocessor()
    texts = ["first sample with enough characters", "second sample with enough characters"]
    longest = max(len(t) for t in texts)

    class DuckTypedDataset:
        # Only __len__ + __getitem__, i.e. the pre-existing implicit contract.
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
    """Columns must be streamed via Dataset.iter(), not copied whole.

    dataset[column] pulls every row into Python objects at once, which for token
    ids is the bulk of peak memory and grows with the dataset.
    """

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
    """`min_length` must not come back as infinity.

    It is seeded with float("inf") and only ever lowered inside the loop, on exactly
    the iterations that also append to `text_lengths`. The inf->0 normalisation sat
    inside `if text_lengths:`, so within that guard it could never see inf: the branch
    was dead, and the case it existed for, a dataset where no sample has content,
    skipped the line entirely and returned min_length = inf to the caller.

    The warning guard has to move with it. With the normalisation hoisted, min_length
    becomes 0 for an empty dataset, and `0 < 10` would newly claim "some samples are
    very short" about zero measured samples.
    """

    preprocessor = TextPreprocessor()

    for label, texts in (("all blank", ["", "   ", "\n"]), ("no rows", [])):
        stats = preprocessor.validate_dataset({"text": texts})
        assert stats["min_length"] == 0, (label, stats)
        assert stats["max_length"] == 0, (label, stats)
        assert not any("very short" in w for w in stats["warnings"]), (label, stats)

    # a genuinely short sample must still be reported
    stats = preprocessor.validate_dataset({"text": ["hi", "a much longer sample of text"]})
    assert stats["min_length"] == 2, stats
    assert any("very short" in w for w in stats["warnings"]), stats

    print("test_validate_dataset_reports_zero_min_length_when_nothing_has_content passed")
    return True


if __name__ == "__main__":
    success = test_raw_text_loader()
    success = test_smart_chunk_text_single_chunk_no_eos_returns_plain_list() and success
    success = test_load_from_file_skips_non_object_json_lines() and success
    success = test_smart_chunk_text_empty_input_returns_no_chunks() and success
    success = test_load_from_files_all_empty_raises() and success
    success = test_negative_stride_is_rejected() and success
    success = test_validate_dataset_handles_tokenized_and_text_columns() and success
    success = test_validate_dataset_accepts_objects_without_column_names() and success
    success = test_validate_dataset_streams_instead_of_materialising_columns() and success
    success = test_validate_dataset_reports_zero_min_length_when_nothing_has_content() and success
    sys.exit(0 if success else 1)
