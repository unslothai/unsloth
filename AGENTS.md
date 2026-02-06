# Agent Instructions for Unsloth

This guide helps agentic coding assistants work effectively with the Unsloth codebase.

## Build/Lint/Test Commands

### Installation
```bash
# Install in editable mode with HuggingFace dependencies
pip install -e ".[huggingface]"

# For CUDA support (choose one)
pip install -e ".[cu118]"  # CUDA 11.8
pip install -e ".[cu121]"  # CUDA 12.1
pip install -e ".[cu124]"  # CUDA 12.4

# For Apple Silicon (MPS)
pip install -e ".[apple]"
```

### Linting and Formatting
```bash
# Run Ruff linter (auto-fix enabled)
ruff check --fix --exit-non-zero-on-fix .

# Run Ruff formatter with custom kwarg spacing
python scripts/run_ruff_format.py <file1> <file2>

# Run all pre-commit hooks
pre-commit run --all-files
```

### Testing
```bash
# Run all tests with unittest
python -m unittest discover -v -s tests -p "*_test.py"

# Run a single test file
python -m unittest tests.mps.test_sanity -v

# Run a single test class
python -m unittest tests.mps.test_sanity.TestClassName -v

# Run a single test method
python -m unittest tests.mps.test_sanity.TestClassName.test_method -v

# Run tests matching a pattern
python -m unittest discover -s tests -p "test_mps_*.py"
```

### Build
```bash
# Build package
python -m build

# Build wheel only
python -m build --wheel
```

## Code Style Guidelines

### General
- Python 3.9+ compatibility required
- Apache 2.0 license headers on all files
- 4 spaces for indentation
- 100 character line limit (enforced by Ruff)

### Imports
Order imports as follows:
1. Standard library (`import os`, `from typing import ...`)
2. Third-party packages (`import torch`, `from transformers import ...`)
3. First-party Unsloth imports (`from unsloth.models import ...`)

Example:
```python
import torch
import gc
from typing import Optional, Tuple
from transformers import AutoModel
from unsloth.models import FastLanguageModel
```

### Naming Conventions
- `snake_case` for functions, variables, modules
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants
- Private functions prefixed with underscore (`_private_func`)

### Type Hints
- Use type hints for function signatures
- Use `from __future__ import annotations` for forward references
- Use `Optional[Type]` or `Type | None` for nullable types

### Error Handling
- Use specific exceptions, not bare `except:`
- Provide meaningful error messages
- Use warnings for non-fatal issues

### Documentation
- Use docstrings for modules, classes, and public functions
- Follow Google-style docstrings
- Include type information in docstrings if not using type hints

### Pre-commit Hooks
All commits must pass:
- Ruff linter with auto-fix
- Ruff formatter with custom kwarg spacing script

Run `pre-commit install` to set up hooks locally.

### GPU/Platform Considerations
- Code must handle CUDA, MPS (Apple Silicon), and XPU (Intel) backends
- Use `DEVICE_TYPE` from `unsloth.device_type` for platform checks
- Never assume CUDA is available
- MPS does not support bitsandbytes or Triton

### Testing Guidelines
- Use Python's built-in `unittest` (not pytest)
- Test files should be named `test_*.py` or `*_test.py`
- Place tests in `tests/` directory
- Use descriptive test method names: `test_<what_is_being_tested>`
- Mock external services and heavy computations when possible

## Project Structure

```
unsloth/
├── models/          # Model implementations (llama.py, gemma.py, etc.)
├── kernels/         # CUDA kernels and optimized operations
├── utils/           # Utility functions
├── dataprep/        # Data preparation utilities
├── registry/        # Model registry
└── tests/           # Test suite
```

## Important Notes

- Unsloth must be imported BEFORE `trl`, `transformers`, or `peft` for optimizations to apply
- The library patches third-party libraries at import time
- GPU compatibility is critical - test on multiple backends when possible
- VRAM optimization is a core concern - always consider memory efficiency
