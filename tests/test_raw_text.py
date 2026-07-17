#!/usr/bin/env python3
"""Minimal test for raw text training, without heavy dependencies."""

import sys
import os
import tempfile
from pathlib import Path
import importlib.util


# Mock the datasets module (not installed).
class MockDataset:
    def __init__(self, data_dict):
        self.data = data_dict
        self.column_names = list(data_dict.keys())

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx):
        if isinstance(idx, str):
            # Column access, e.g. dataset['text'].
            return self.data[idx]
        elif isinstance(idx, int):
            # Row access by index.
            return {key: values[idx] for key, values in self.data.items()}
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    @classmethod
    def from_dict(cls, data_dict):
        return cls(data_dict)


# __spec__ must be set so importlib.util.find_spec doesn't raise ValueError when
# transformers' import_utils later probes for the real `datasets` package.
datasets_mock = type(sys)("datasets")
datasets_mock.__spec__ = importlib.util.spec_from_loader("datasets", loader = None)
datasets_mock.Dataset = MockDataset
sys.modules["datasets"] = datasets_mock

# Import raw_text directly to avoid unsloth/__init__.py dependencies.
current_dir = os.path.dirname(__file__)
raw_text_path = os.path.join(os.path.dirname(current_dir), "unsloth", "dataprep", "raw_text.py")

spec = importlib.util.spec_from_file_location("raw_text", raw_text_path)
raw_text_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw_text_module)

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

        # Text output (legacy mode).
        text_dataset = loader.load_from_file(test_file, return_tokenized = False)
        assert len(text_dataset) > 0, "Should create at least one chunk"
        assert "text" in text_dataset.column_names, "Dataset should have 'text' column"

        # Tokenized output (new efficient mode).
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

        # labels field (for causal LM training).
        assert "labels" in tokenized_dataset.column_names, "Dataset should have 'labels' column"
        assert first_sample["labels"] == first_sample["input_ids"], "labels should match input_ids"

        # Constructor validation.
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

        # smart_chunk_text validation: called directly, chunk_size/stride are its own
        # arguments and bypass the constructor guard, so it must guard itself or an
        # invalid stride makes `start_idx += chunk_size - stride` non-positive and the
        # chunking loop never terminates (hangs).
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

        # Preprocessor.
        preprocessor = TextPreprocessor()
        clean_text = preprocessor.clean_text("  messy   text  \n\n\n  ")
        assert "messy text" in clean_text, "Should clean text properly"
        paragraph_text = preprocessor.clean_text("Line 1\r\n\r\n\r\nLine 2")
        assert (
            paragraph_text == "Line 1\n\nLine 2"
        ), "Should preserve paragraph breaks while normalizing newlines"

        # Non-ASCII horizontal whitespace (NBSP, thin/em/ideographic space, VT, FF) must
        # normalize to one ASCII space, not be deleted, or adjacent words fuse on HTML/PDF/OCR input.
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

        # Mixed paragraph + Unicode whitespace.
        mixed = preprocessor.clean_text("Section\u00a01\r\n\r\nBody\ftext\u202fhere")
        assert (
            mixed == "Section 1\n\nBody text here"
        ), "Should preserve paragraph breaks and normalize Unicode whitespace simultaneously"

        # Tabs collapse to a single space.
        assert preprocessor.clean_text("a\tb") == "a b"
        assert preprocessor.clean_text("a\t\tb") == "a b"

        # Spaces around newlines trimmed on both sides, even across multiple newlines.
        assert preprocessor.clean_text("foo \n\n bar") == "foo\n\nbar"

        # Stripping a non-ASCII char between spaces must not leave a double space
        # (also guards idempotence: otherwise "word1 (c) word2" needs a second pass).
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

        # Validation.
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


if __name__ == "__main__":
    success = test_raw_text_loader()
    success = test_smart_chunk_text_single_chunk_no_eos_returns_plain_list() and success
    sys.exit(0 if success else 1)
