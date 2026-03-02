# fix: update GGUF save paths to use ~/.unsloth/llama.cpp with Windows support

## Description

Aligns `save_pretrained_gguf` and `push_to_hub_gguf` in `unsloth/save.py` with the updated `unsloth_zoo/llama_cpp.py`, which now builds and installs llama.cpp components into `~/.unsloth/llama.cpp` instead of the current working directory.

### Changes

- **Import `LLAMA_CPP_DEFAULT_DIR` and `IS_WINDOWS`** from `unsloth_zoo.llama_cpp` to reference the correct llama.cpp install path
- **Update example usage paths** in `save_pretrained_gguf` to use `os.path.join` with platform-correct binary locations (`build/bin/Release/` on Windows, root dir on Linux) and `.exe` suffix on Windows
- **Update error message** in `save_to_gguf` to provide platform-appropriate manual build instructions (`cmake` on Windows, `make` on Linux)
- **Update README template** in `push_to_hub_gguf` to use platform-neutral binary names without hardcoded path prefixes

### What did NOT change

The **core GGUF conversion logic** (`save_to_gguf` → `check_llama_cpp` / `install_llama_cpp` / `convert_to_gguf` / `quantize_gguf`) already delegates to `unsloth_zoo.llama_cpp` functions, which transparently pick up the new `~/.unsloth/llama.cpp` path. No logic changes were needed — only user-facing strings.

### Related

- Depends on `unsloth-zoo` branch: `feature/llama-cpp-windows-support`
