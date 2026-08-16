# Run Unsloth Studio on a Modal GPU with the LOCAL thinking-passthrough fix applied,
# so you can point Claude Code at it and check whether thinking actually works.
#
#   modal run modal_studio_thinking.py
#   UNSLOTH_MODEL=unsloth/Qwen3.5-9B-GGUF modal run modal_studio_thinking.py
#   UNSLOTH_FORCE_REBUILD=1 modal run modal_studio_thinking.py   # re-copy after more edits
#
# Unlike modal_studio_pr.py this does NOT fetch a PR. The fix lives on a local branch,
# so the overlay copies the changed backend files straight off this working tree onto
# the image. They are plain Python and the frontend is untouched, so no rebuild is
# needed -- install.sh does not re-run.
#
# The base layer is byte-identical to modal_studio.py / modal_studio_pr.py, so it is a
# cache hit if you have already built either of those.
#
# WHAT THIS IS FOR
#   Before the fix: /v1/messages parsed `thinking` / `enable_thinking` and dropped them,
#   so the model stayed in its load-time default and Claude Code could never turn
#   thinking on. After the fix they are forwarded as chat_template_kwargs.
#   Verify by hand -- see the checklist the container prints on startup.

import os
from pathlib import Path

import modal

PORT = 8000
GPU = "H100"
FORCE_REBUILD = os.environ.get("UNSLOTH_FORCE_REBUILD") == "1"

# A thinking-capable model, since that is the whole point of the fix. Must be one
# whose template takes enable_thinking / reasoning_effort -- an always-on reasoning
# model ignores the kwarg by design and proves nothing.
MODEL = os.environ.get("UNSLOTH_MODEL", "unsloth/Qwen3.5-9B-GGUF")

# The commit the fix branch was cut from. The base layer's clone is cached and can be
# months older than this, and the patched files reference symbols that only exist at
# this revision -- copying them onto a stale checkout makes the backend fail to import.
# Pinning here keeps the overlay reproducible; bump it if you rebase the fix.
BASE_COMMIT = os.environ.get(
    "UNSLOTH_BASE_COMMIT", "1af3000fcadd320d6568888fb668449b21ea6fa2"
)

# The fix. Backend-only, no frontend, no dependency changes.
PATCHED_FILES = [
    "studio/backend/models/inference.py",
    "studio/backend/routes/inference.py",
    "studio/backend/tests/test_anthropic_messages.py",  # lets you run pytest in-container
]

image = (
    # ── base: byte-identical to modal_studio.py so the layer is reused ──
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install(
        "git", "curl", "wget", "rsync", "ca-certificates", "build-essential",
        "cmake", "libcurl4-openssl-dev", "xz-utils", "unzip",
        "ffmpeg", "libgl1", "libglib2.0-0",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/unslothai/unsloth.git /root/unsloth",
        "cd /root/unsloth && bash install.sh --local",
        # MUST build with a GPU attached: install.sh probes nvidia-smi to pick the
        # torch wheel -- a GPU-less build silently installs CPU-only torch.
        gpu="H100",
    )
)

# ── pin the checkout to the fix's base commit ──
image = image.run_commands(
    f"cd /root/unsloth && git fetch --depth 1 origin {BASE_COMMIT}",
    f"cd /root/unsloth && git checkout --detach {BASE_COMMIT}",
)

# ── local-fix overlay ──
# Modal re-imports this module inside the container, where __file__ lives at /root
# and the worktree does not exist. Anything touching the local filesystem has to be
# local-only; the container gets the already-built image, so it needs no layers here.
# These layers hash on file content, so editing the fix re-copies without FORCE_REBUILD.
if modal.is_local():
    REPO_ROOT = Path(__file__).parent
    for _f in PATCHED_FILES:
        if not (REPO_ROOT / _f).is_file():
            raise SystemExit(
                f"Missing {_f} under {REPO_ROOT}.\n"
                "Run this from the worktree that has the thinking fix checked out."
            )
        image = image.add_local_file(REPO_ROOT / _f, f"/root/unsloth/{_f}", copy=True)

image = image.run_commands(
    # Fail the build rather than serve a stale image that silently still drops thinking.
    "grep -q '_anthropic_reasoning_args' /root/unsloth/studio/backend/routes/inference.py",
    "grep -q 'AnthropicThinkingConfig' /root/unsloth/studio/backend/models/inference.py",
    # install.sh at the pinned commit needs bubblewrap to build the GGUF engine and
    # cannot sudo. Installed here rather than in the base apt_install so the base
    # layer stays byte-identical to modal_studio.py and keeps its cache.
    "apt-get update -y && apt-get install -y bubblewrap",
    # The checkout moved the frontend sources past the baked dist, so rebuild it.
    # Everything else reports up to date. Needs a GPU for the same reason the base
    # layer does: install.sh probes nvidia-smi and a GPU-less run can downgrade torch.
    "cd /root/unsloth && bash install.sh --local",
    # install.sh writes into the HF cache, and Modal refuses to mount a volume onto a
    # non-empty path. Clear it so the hf_cache volume can attach at runtime; nothing
    # here is worth keeping, the volume is for model downloads.
    "rm -rf /root/.cache/huggingface",
    gpu="H100",
    force_build=FORCE_REBUILD,
).env({"UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK": "1"})

app = modal.App("unsloth-studio-thinking-fix")
hf_cache = modal.Volume.from_name("unsloth-studio-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=12 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    # secrets=[modal.Secret.from_name("huggingface")],  # uncomment for gated models
)
def studio():
    import subprocess

    venv_python = Path.home() / ".unsloth/studio/unsloth_studio/bin/python"

    # The baked llama.cpp bundle was selected on the build machine and aborts on other
    # GPU archs; re-resolve against the runtime GPU. No-op when it already matches.
    heal = subprocess.run(
        [str(venv_python), "/root/unsloth/studio/install_llama_prebuilt.py",
         "--install-dir", "/root/.unsloth/llama.cpp"],
    )
    if heal.returncode != 0:
        print("WARNING: llama.cpp re-resolve failed; inference may crash on this GPU", flush=True)

    # Proves the patched code is live in the venv Studio actually runs, not just on disk.
    unit = subprocess.run(
        [str(venv_python), "-m", "pytest", "-q",
         "tests/test_anthropic_messages.py", "-k", "reasoning or thinking"],
        cwd="/root/unsloth/studio/backend",
        capture_output=True,
        text=True,
    )
    print("─" * 60)
    print("thinking-passthrough unit tests:", unit.stdout.strip().splitlines()[-1]
          if unit.stdout.strip() else "(no output)")
    if unit.returncode != 0:
        print("WARNING: unit tests FAILED -- the fix may not be applied correctly")
        print(unit.stdout[-1500:])
    print("─" * 60, flush=True)

    cmd = [
        str(venv_python), "/root/unsloth/studio/backend/run.py",
        "--host", "0.0.0.0", "--port", str(PORT),
    ]

    # run.py at this commit has no --password flag and no password env var; the
    # first visit to the UI sets up the account. Passing one exits with code 2.
    fresh = not list((Path.home() / ".unsloth").rglob("auth.db"))

    with modal.forward(PORT) as tunnel:
        print("=" * 72)
        print(f"  Unsloth Studio: {tunnel.url}")
        print("  Login -> " + (
            "first visit: set up the account in the browser"
            if fresh else "existing account (auth.db survived the restart)"
        ))
        print()
        print("  Testing: LOCAL branch worktree-anthropic-thinking-passthrough")
        print("           (thinking / reasoning controls forwarded on /v1/messages)")
        print()
        print("  HOW TO VERIFY")
        print(f"   1. Open the URL, load {MODEL}, wait for it to finish loading.")
        print("   2. Settings -> API keys -> create one. Copy it.")
        print("   3. On your laptop:")
        print(f"        UNSLOTH_STUDIO_URL={tunnel.url} \\")
        print("          unsloth start claude --api-key <the key>")
        print("   4. Press Ctrl+O to turn on verbose mode.")
        print("   5. Ask something that needs reasoning, e.g.")
        print("        A bat and a ball cost $1.10. The bat costs $1.00 more")
        print("        than the ball. How much is the ball?")
        print()
        print("   PASS -> a reasoning trace appears in verbose mode.")
        print("   FAIL -> still no trace; the request is still being dropped.")
        print()
        print("   Control: the same prompt in Studio's own chat should think too.")
        print("   If neither thinks, the model is not in a thinking-capable config")
        print("   and the test proves nothing -- try a different GGUF.")
        print()
        print("  (wait for the 'Unsloth Studio running' line below)")
        print("=" * 72, flush=True)
        subprocess.run(cmd, cwd="/root/unsloth", check=True)


@app.local_entrypoint()
def main():
    studio.remote()
