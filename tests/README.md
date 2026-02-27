# Unsloth Test Organization

## Directory Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration
│
├── mps/                     # MPS (Metal Performance Shaders) tests
│   ├── test_sanity.py
│   ├── test_loading.py
│   ├── test_device.py
│   ├── test_integration.py
│   ├── test_e2e_metal.py
│   ├── test_fallbacks.py
│   └── test_loading_profile.py
│
├── mlx/                     # MLX-specific tests
│   ├── test_bridge.py       # PyTorch↔MLX bridge tests
│   ├── test_models.py       # MLX model tests
│   └── test_utils.py        # MLX utility tests
│
├── metal/                   # Metal kernel tests
│   └── test_rms_layernorm.py
│
├── utils/                   # Utility tests
│   ├── test_qat.py
│   ├── test_packing.py
│   ├── test_attention_masks.py
│   └── ...
│
├── saving/                  # Model saving tests
│   ├── language_models/
│   ├── vision_models/
│   └── ...
│
├── qlora/                   # QLoRA-specific tests
│   └── ...
│
├── manual/                  # Manual test scripts (not unittest)
│   ├── test_gradient_diagnostic.py
│   ├── test_mini_lora.py
│   ├── test_mlx_merge.py
│   ├── test_mlx_training_infrastructure.py
│   ├── test_sft_training_mac.py
│   └── test_torch_func_mps.py
│
└── verification/            # Feature verification scripts
    ├── verify_4bit_logic.py
    ├── verify_apple.py
    ├── verify_apple_hardening.py
    ├── verify_apple_memory_save.py
    ├── verify_mlx.py
    ├── verify_unsloth_mac.py
    └── verify_vision_support.py
```

## Running Tests

### Unit Tests (unittest)
```bash
# Run all tests
python -m unittest discover -v -s tests -p "*_test.py"

# Run MPS tests
python -m unittest discover -v -s tests/mps -p "*_test.py"

# Run specific test
python -m unittest tests.mlx.test_bridge -v
```

### Manual Tests
```bash
# Run manual verification scripts
python tests/verification/verify_mlx.py
python tests/manual/test_mini_lora.py
```

## Test Categories

| Category | Location | Purpose |
|----------|----------|---------|
| Unit Tests | `tests/mps/`, `tests/mlx/`, etc. | Automated unittest, CI-ready |
| Manual Tests | `tests/manual/` | Interactive debugging, exploratory |
| Verification | `tests/verification/` | Feature validation scripts |
| Benchmarks | `benchmarks/` | Performance profiling |
