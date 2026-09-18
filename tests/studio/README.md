# Studio tests

Pytest suite for the Studio backend's MLX dispatch surface and CLI behaviours.
Every test in this directory runs on a Linux+CPU box; no Apple Silicon, NVIDIA
GPU, AMD ROCm runtime, or Intel XPU build is required.

## MLX dispatch coverage

Three files cover the CUDA / ROCm / XPU / MLX / CPU dispatch logic by spoofing
hardware probes from a single test host:

### `test_hardware_dispatch_matrix.py`

Comprehensive hardware dispatch matrix. Each row in the `PROFILES` list is a
parametrized `HardwareProfile` dataclass that pins:

- `platform.system()` and `platform.machine()`
- `torch.cuda.is_available()`
- `torch.version.hip` (None on NVIDIA, e.g. `"6.1"` on ROCm)
- `torch.xpu.is_available()` and `torch.xpu.get_device_name()`
- `torch.backends.mps.is_available()`
- whether a fake `mlx` package is registered in `sys.modules`

For every profile the suite asserts:

1. `unsloth._IS_MLX` (re-evaluated under the spoof) flips correctly.
2. `utils.hardware.detect_hardware()` returns the right `DeviceType`.
3. `utils.hardware.IS_ROCM` matches expectation.
4. `utils.hardware.is_apple_silicon()` agrees with the platform spoof.

Bundled profiles:

| Profile                | platform        | cuda | hip   | xpu | mlx | mps  | _IS_MLX | DEVICE | IS_ROCM |
|------------------------|-----------------|------|-------|-----|-----|------|---------|--------|---------|
| `nvidia_cuda`          | Linux x86_64    | True | None  | F   | -   | F    | F       | CUDA   | F       |
| `amd_rocm`             | Linux x86_64    | True | "6.1" | F   | -   | F    | F       | CUDA   | T       |
| `intel_xpu`            | Linux x86_64    | F    | None  | T   | -   | F    | F       | XPU    | F       |
| `apple_silicon_mlx`    | Darwin arm64    | F    | None  | F   | T   | T    | T       | MLX    | F       |
| `apple_silicon_no_mlx` | Darwin arm64    | F    | None  | F   | -   | T    | F       | CPU    | F       |
| `linux_arm64_with_mlx` | Linux arm64     | F    | None  | F   | T   | F    | F       | CPU    | F       |
| `cpu_only`             | Linux x86_64    | F    | None  | F   | -   | F    | F       | CPU    | F       |

Plus two negative-space canaries protecting the dispatch priority order:

- `test_cuda_takes_priority_over_mlx_when_both_available`
- `test_xpu_takes_priority_over_mlx_when_both_available`

To extend coverage, add a row to `PROFILES`. Pytest's parametrize picks up new
entries automatically.

### `test_is_mlx_dispatch_gate.py`

Targeted regression for the `unsloth._IS_MLX` source-level structure. Walks
the AST of `unsloth/__init__.py` and asserts that the `_IS_MLX` assignment
is a `BoolOp(And)` of `platform.system() == "Darwin"`,
`platform.machine() == "arm64"`, and `find_spec("mlx") is not None`. Catches
accidental rewrites that drop a predicate.

### `test_mlx_training_worker_behaviors.py`

AST-level checks on `studio/backend/core/training/worker.py` for the MLX
training worker contract (token forwarding, secret stripping, dataset path,
etc.). Pure-torch unit tests using monkeypatch fakes for `mlx`, `mlx.core`,
and `unsloth_zoo.mlx_loader`.

## Running the dispatch suite

```bash
# All MLX dispatch coverage in one go (~5 seconds)
pytest tests/studio/test_hardware_dispatch_matrix.py \
       tests/studio/test_is_mlx_dispatch_gate.py \
       tests/studio/test_mlx_training_worker_behaviors.py -v

# Just the parametrized matrix (23 tests, ~2 seconds)
pytest tests/studio/test_hardware_dispatch_matrix.py -v

# Just the AST guard on _IS_MLX
pytest tests/studio/test_is_mlx_dispatch_gate.py -v
```

Add `-k <profile_name>` to filter to a single hardware profile, e.g.
`pytest tests/studio/test_hardware_dispatch_matrix.py -k apple_silicon_mlx`.

## Other Studio tests in this directory

The remaining files (`test_cancel_*`, `test_cli_*`, `test_chat_preset_*`,
`test_export_*`, `test_llama_cpp_wall_clock_cap`,
`test_stream_cancel_registration_timing`, `test_studio_gguf_export_script_pin`,
`test_studio_text_descender_clipping`) are conventional unit tests that do
not depend on the dispatch matrix. They run on the same Linux+CPU CI matrix
without any hardware spoofing.

The `install/` subdirectory contains tests for the Studio installer
(`./install.sh`) Python stack selection logic.
