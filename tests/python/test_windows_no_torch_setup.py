from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"


def _powershell_block(source: str, marker: str) -> str:
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"Unclosed PowerShell block after {marker!r}")


def test_windows_direct_torch_installs_are_skipped_in_no_torch_mode():
    source = SETUP_PS1.read_text(encoding = "utf-8")
    guarded = _powershell_block(source, "if (-not $NoTorchMode) {")

    for install_path in (
        "installing PyTorch (AMD ROCm",
        "installing PyTorch (CPU-only)",
        "installing PyTorch with CUDA support",
        "installing Triton for Windows",
    ):
        assert install_path in guarded

    # The shared dependency pass installs the dedicated no-torch runtime and
    # therefore must remain outside the direct torch/Triton guard.
    assert 'python "$PSScriptRoot\\install_python_stack.py"' not in guarded
