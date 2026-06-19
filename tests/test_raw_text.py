#!/usr/bin/env python3
"""
Minimal test for raw text training implementation.
Tests basic functionality without heavy dependencies.
"""

import sys
import os
import tempfile
from pathlib import Path
import importlib.util


# Mock the datasets module since it's not installed
class MockDataset:
    def __init__(self, data_dict):
        self.data = data_dict
        self.column_names = list(data_dict.keys())

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx):
        if isinstance(idx, str):
            # Allow accessing columns by name like dataset['text']
            return self.data[idx]
        elif isinstance(idx, int):
            # Allow accessing individual rows by index
            return {key: values[idx] for key, values in self.data.items()}
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    @classmethod
    def from_dict(cls, data_dict):
        return cls(data_dict)


# Mock datasets module. __spec__ must be set so importlib.util.find_spec
# does not raise ValueError when transformers' import_utils probes for
# the real `datasets` package later in the test session.
datasets_mock = type(sys)("datasets")
datasets_mock.__spec__ = importlib.util.spec_from_loader("datasets", loader = None)
datasets_mock.Dataset = MockDataset
sys.modules["datasets"] = datasets_mock

# Import the raw_text module directly to avoid unsloth/__init__.py dependencies
current_dir = os.path.dirname(__file__)
raw_text_path = os.path.join(
    os.path.dirname(current_dir), "unsloth", "dataprep", "raw_text.py"
)

spec = importlib.util.spec_from_file_location("raw_text", raw_text_path)
raw_text_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw_text_module)

RawTextDataLoader = raw_text_module.RawTextDataLoader
TextPreprocessor = raw_text_module.TextPreprocessor


def test_raw_text_loader():
    """Test basic RawTextDataLoader functionality."""

    # Mock tokenizer for testing
    class MockTokenizer:
        def __init__(self):
            self.eos_token = "</s>"
            self.eos_token_id = 2  # Mock EOS token ID

        def __call__(self, text, return_tensors = None, add_special_tokens = False):
            words = text.split()
            token_ids = list(range(len(words)))

            if return_tensors == "pt":
                # Mock tensor-like object
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

        def decode(self, token_ids, skip_special_tokens = False):
            return " ".join([f"word_{i}" for i in token_ids])

    # Create test file
    test_content = "This is a test file for raw text training. " * 10
    with tempfile.NamedTemporaryFile(mode = "w", suffix = ".txt", delete = False) as f:
        f.write(test_content)
        test_file = f.name

    try:
        # Test loader
        tokenizer = MockTokenizer()
        loader = RawTextDataLoader(tokenizer, chunk_size = 5, stride = 2)

        # Test loading with text output (legacy mode)
        text_dataset = loader.load_from_file(test_file, return_tokenized = False)
        assert len(text_dataset) > 0, "Should create at least one chunk"
        assert "text" in text_dataset.column_names, "Dataset should have 'text' column"

        # Test loading with tokenized output (new efficient mode)
        tokenized_dataset = loader.load_from_file(test_file, return_tokenized = True)
        assert len(tokenized_dataset) > 0, "Should create at least one tokenized chunk"
        assert (
            "input_ids" in tokenized_dataset.column_names
        ), "Dataset should have 'input_ids' column"
        assert (
            "attention_mask" in tokenized_dataset.column_names
        ), "Dataset should have 'attention_mask' column"

        # Verify tokenized data structure
        first_sample = tokenized_dataset[0]
        assert isinstance(first_sample["input_ids"], list), "input_ids should be a list"
        assert isinstance(
            first_sample["attention_mask"], list
        ), "attention_mask should be a list"
        assert len(first_sample["input_ids"]) == len(
            first_sample["attention_mask"]
        ), "input_ids and attention_mask should have same length"

        # Verify labels field exists (for causal LM training)
        assert (
            "labels" in tokenized_dataset.column_names
        ), "Dataset should have 'labels' column"
        assert (
            first_sample["labels"] == first_sample["input_ids"]
        ), "labels should match input_ids"

        # Test constructor validation
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

        # Test preprocessor
        preprocessor = TextPreprocessor()
        clean_text = preprocessor.clean_text("  messy   text  \n\n\n  ")
        assert "messy text" in clean_text, "Should clean text properly"
        paragraph_text = preprocessor.clean_text("Line 1\r\n\r\n\r\nLine 2")
        assert (
            paragraph_text == "Line 1\n\nLine 2"
        ), "Should preserve paragraph breaks while normalizing newlines"

        # Non-ASCII horizontal whitespace separators (NBSP, thin space,
        # ideographic space, narrow NBSP, em space, vertical tab, form feed)
        # should be normalized to a single ASCII space, not deleted, otherwise
        # adjacent words get silently fused together on HTML/PDF/OCR inputs.
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
                f"Should normalize Unicode/control whitespace to a single space "
                f"for {raw!r}"
            )

        # Mixed paragraph + Unicode whitespace realistic input
        mixed = preprocessor.clean_text("Section\u00a01\r\n\r\nBody\ftext\u202fhere")
        assert mixed == "Section 1\n\nBody text here", (
            "Should preserve paragraph breaks and normalize Unicode "
            "whitespace simultaneously"
        )

        # Tabs should collapse to a single space
        assert preprocessor.clean_text("a\tb") == "a b"
        assert preprocessor.clean_text("a\t\tb") == "a b"

        # Spaces around newlines should be trimmed on both sides, even with
        # multiple consecutive newlines
        assert preprocessor.clean_text("foo \n\n bar") == "foo\n\nbar"

        # Non-whitespace non-ASCII characters sitting between spaces should
        # not leave an interior double space after being stripped. This
        # guards the idempotence invariant too: without the extra collapse
        # pass, "word1 (c) word2" first reduces to "word1  word2" and only
        # becomes "word1 word2" on a second call.
        assert preprocessor.clean_text("word1 \u00a9 word2") == "word1 word2"
        assert preprocessor.clean_text("a \u00e9 b") == "a b"
        assert preprocessor.clean_text("prefix \U0001f600 suffix") == "prefix suffix"

        # Stripping a non-ASCII character adjacent to a newline must not
        # leave a stray leading/trailing space on the neighbouring line.
        assert preprocessor.clean_text("foo \u00e9\nbar") == "foo\nbar"
        assert preprocessor.clean_text("foo\n\u00e9 bar") == "foo\nbar"
        # The double-space collapse pass must not swallow a legitimate
        # paragraph break when a non-ASCII char sits near it.
        assert preprocessor.clean_text("a \u00a9\n\nb") == "a\n\nb"

        # Idempotence: running clean_text twice should give the same result
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

        # Test validation
        stats = preprocessor.validate_dataset(text_dataset)
        assert stats["total_samples"] > 0, "Should count samples"
        assert "warnings" in stats, "Should include warnings"

        print("✅ All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    finally:
        # Cleanup
        os.unlink(test_file)


if __name__ == "__main__":
    success = test_raw_text_loader()
    sys.exit(0 if success else 1)
