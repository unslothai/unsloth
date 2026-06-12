"""Bake a pinned llama.cpp prebuilt into the Docker image, deterministically.

Why not studio/install_llama_prebuilt.py: that resolver selects a bundle for
the CURRENT host (nvidia-smi, /proc/driver/nvidia, installed CUDA runtime),
which is exactly what an image build must not do -- a B200 build host, a
GPU-less CI runner and a laptop must all produce byte-identical layers. This
script instead pins release + asset by build target only:

    amd64 -> app-<tag>-linux-x64-cuda12-portable.tar.gz   (sm_70..sm_120)
    arm64 -> app-<tag>-linux-arm64-cuda13-portable.tar.gz (sm_90..sm_121)

The portable bundles carry their own CUDA runtime libs and dynamically load
the CUDA backend at runtime, so the same binaries also run CPU-only.

Every download is sha256-verified against the release's own
llama-prebuilt-sha256.json. The converter (convert_hf_to_gguf.py) and its
gguf-py library are hydrated from the SAME release's source tarball so the
tensor mappings match the binaries -- the layout unsloth_zoo's
check_llama_cpp() expects: binaries, converter and gguf-py/ at the install
dir root.

Usage (in the Dockerfile):
    python fetch_llama_prebuilt.py <tag> <targetarch> <install_dir>
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

RELEASE_REPO = "unslothai/llama.cpp"


def fetch(url: str, dest: str) -> None:
    request = urllib.request.Request(url, headers = {"User-Agent": "unsloth-docker-build"})
    with urllib.request.urlopen(request, timeout = 600) as response, open(dest, "wb") as f:
        shutil.copyfileobj(response, f, length = 1 << 20)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_verified(base_url: str, name: str, sums: dict, work: str) -> str:
    path = os.path.join(work, name)
    fetch(f"{base_url}/{name}", path)
    expected = sums.get(name, {}).get("sha256")
    if not expected:
        raise SystemExit(f"FAIL: {name} not listed in llama-prebuilt-sha256.json")
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"FAIL: sha256 mismatch for {name}: expected {expected}, got {actual}")
    print(f"verified {name} sha256={actual[:16]}...")
    return path


def extracted_root(extract_dir: str) -> str:
    children = os.listdir(extract_dir)
    if len(children) == 1 and os.path.isdir(os.path.join(extract_dir, children[0])):
        return os.path.join(extract_dir, children[0])
    return extract_dir


def main() -> None:
    tag, target_arch, install_dir = sys.argv[1], sys.argv[2] or "amd64", sys.argv[3]
    base_url = f"https://github.com/{RELEASE_REPO}/releases/download/{tag}"
    assets = {
        "amd64": f"app-{tag}-linux-x64-cuda12-portable.tar.gz",
        "arm64": f"app-{tag}-linux-arm64-cuda13-portable.tar.gz",
    }
    if target_arch not in assets:
        raise SystemExit(f"FAIL: unsupported TARGETARCH={target_arch}")
    bundle_name = assets[target_arch]
    source_name = f"llama.cpp-source-{tag}.tar.gz"

    with tempfile.TemporaryDirectory() as work:
        sha_path = os.path.join(work, "llama-prebuilt-sha256.json")
        fetch(f"{base_url}/llama-prebuilt-sha256.json", sha_path)
        sums = json.load(open(sha_path))["artifacts"]

        # Binaries: flat tarball, llama-quantize / llama-server / lib*.so at root.
        bundle_path = fetch_verified(base_url, bundle_name, sums, work)
        bundle_dir = os.path.join(work, "bundle")
        os.makedirs(bundle_dir)
        with tarfile.open(bundle_path) as tf:
            tf.extractall(bundle_dir, filter = "tar")
        os.makedirs(install_dir, exist_ok = True)
        root = extracted_root(bundle_dir)
        for entry in os.listdir(root):
            target = os.path.join(install_dir, entry)
            shutil.move(os.path.join(root, entry), target)
            if os.path.isfile(target) and not entry.startswith("lib") and ".so" not in entry:
                os.chmod(target, 0o755)

        # Converter + gguf-py from the same-tag source tarball, so the python
        # side's tensor mappings match the binaries (mirrors unsloth_zoo's
        # _hydrate_converter_sources).
        source_path = fetch_verified(base_url, source_name, sums, work)
        source_dir = os.path.join(work, "source")
        os.makedirs(source_dir)
        with tarfile.open(source_path) as tf:
            tf.extractall(source_dir, filter = "tar")
        src_root = extracted_root(source_dir)
        converter = os.path.join(src_root, "convert_hf_to_gguf.py")
        gguf_py = os.path.join(src_root, "gguf-py")
        if not (os.path.isfile(converter) and os.path.isdir(gguf_py)):
            raise SystemExit(f"FAIL: source tarball for {tag} is missing converter files")
        for script in os.listdir(src_root):
            if script.startswith("convert_") and script.endswith(".py"):
                shutil.copy2(os.path.join(src_root, script), os.path.join(install_dir, script))
        shutil.copytree(gguf_py, os.path.join(install_dir, "gguf-py"), dirs_exist_ok = True)
        conversion = os.path.join(src_root, "conversion")
        if os.path.isdir(conversion):
            shutil.copytree(conversion, os.path.join(install_dir, "conversion"), dirs_exist_ok = True)

    # Mirror the install into build/bin/ via hardlinks (zero extra bytes).
    # Studio's setup.sh treats an executable build/bin/llama-server +
    # build/bin/llama-quantize as a complete local build and skips its
    # source-build fallback -- which would otherwise fire inside the image
    # build, where the host-probing prebuilt updater cannot succeed, and
    # compile a CPU-only llama.cpp over the baked CUDA bundle. Hardlinks
    # (not symlinks) keep $ORIGIN rpath resolution working from build/bin
    # and avoid a cycle when setup.sh later relinks the root quantizer to
    # build/bin/llama-quantize.
    build_bin = os.path.join(install_dir, "build", "bin")
    os.makedirs(build_bin, exist_ok = True)
    for entry in os.listdir(install_dir):
        source = os.path.join(install_dir, entry)
        if os.path.isfile(source) and not os.path.islink(source):
            try:
                os.link(source, os.path.join(build_bin, entry))
            except OSError:
                shutil.copy2(source, os.path.join(build_bin, entry))

    # Sanity: the server binary must execute on a GPU-less host (the CUDA
    # backend is a dlopen'd plugin, so --version works anywhere).
    out = subprocess.run(
        [os.path.join(install_dir, "llama-server"), "--version"],
        capture_output = True,
        text = True,
        timeout = 120,
    )
    banner = (out.stdout + out.stderr).strip()
    print(banner.splitlines()[0] if banner else "(no version banner)")
    if "version" not in banner:
        raise SystemExit(
            f"FAIL: llama-server --version did not report a version: rc={out.returncode}"
        )
    for required in (
        "llama-quantize",
        "convert_hf_to_gguf.py",
        "gguf-py",
        "UNSLOTH_PREBUILT_INFO.json",
    ):
        if not os.path.exists(os.path.join(install_dir, required)):
            raise SystemExit(f"FAIL: {required} missing from {install_dir}")
    print(f"OK: llama.cpp {tag} ({bundle_name}) installed at {install_dir}")


if __name__ == "__main__":
    main()
