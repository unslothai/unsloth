# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Bake a pinned llama.cpp prebuilt into the Docker image, deterministically.

Not studio/install_llama_prebuilt.py: that resolver selects a bundle for the CURRENT
host, where an image build must produce byte-identical layers everywhere. Pins release
and asset by build target only:

    amd64 -> app-<tag>-linux-x64-cuda12-portable.tar.gz   (sm_70..sm_120)
    arm64 -> app-<tag>-linux-arm64-cuda13-portable.tar.gz (sm_90..sm_121)

The portable bundles dlopen the CUDA backend, so the same binaries also run CPU-only.
Every download is sha256-verified, and the converter + gguf-py come from the SAME
release's source tarball so the tensor mappings match the binaries. A "latest" tag is
resolved at build time; pass a concrete tag for a reproducible build.

Usage (in the Dockerfile):
    python fetch_llama_prebuilt.py <tag|latest> <targetarch> <install_dir>
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

RELEASE_REPO = "unslothai/llama.cpp"


def base_build_tag(tag: str) -> str:
    """Normalized upstream build out of a release tag: b10715 from b10715-mix-86bd2d3.
    Mirrors llama_cpp_freshness.parse_base_build; a tag that is not bNNNN is kept."""
    match = re.match(r"b(\d+)", tag.strip())
    return f"b{match.group(1)}" if match else tag.strip()


def resolve_latest_tag(repo: str) -> str:
    url = f"https://github.com/{repo}/releases/latest"
    request = urllib.request.Request(url, headers = {"User-Agent": "unsloth-docker-build"})
    with urllib.request.urlopen(request, timeout = 60) as response:
        final_url = response.geturl()
    marker = "/releases/tag/"
    if marker not in final_url:
        raise SystemExit(
            f"FAIL: could not resolve latest release of {repo} (landed on {final_url})"
        )
    return final_url.rsplit(marker, 1)[1].strip("/")


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


def sanity_check_binaries(install_dir, build_bin):
    checks = (
        # exit 0 too: a dynamic-loader failure prints "version `GLIBC_...' not found",
        # which the substring alone accepts even though it never reached main
        (os.path.join(install_dir, "llama-server"), "version", True),
        # no --version, and an unknown flag need not exit 0, so only the banner counts
        (os.path.join(install_dir, "llama-quantize"), "usage", False),
        (os.path.join(build_bin, "llama-quantize"), "usage", False),
    )
    for binary, expect, require_zero_exit in checks:
        out = subprocess.run(
            [binary, "--version"],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        banner = (out.stdout + out.stderr).strip()
        print(
            os.path.relpath(binary, install_dir),
            "->",
            banner.splitlines()[0] if banner else "(no output)",
        )
        if expect not in banner:
            raise SystemExit(
                f"FAIL: {binary} did not print '{expect}': rc={out.returncode}\n{banner[:400]}"
            )
        if require_zero_exit and out.returncode != 0:
            raise SystemExit(
                f"FAIL: {binary} --version exited {out.returncode}; the banner text is "
                f"not enough on its own because a dynamic-loader failure prints "
                f'"version `GLIBC_...\' not found" too\n{banner[:400]}'
            )


def main() -> None:
    tag, target_arch, install_dir = sys.argv[1], sys.argv[2] or "amd64", sys.argv[3]
    if tag in ("", "latest"):
        tag = resolve_latest_tag(RELEASE_REPO)
        print(f"resolved latest {RELEASE_REPO} release: {tag}")
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

    # Studio's freshness check wants the install_llama_prebuilt.py schema: "tag" is the
    # NORMALIZED BASE build, "release_tag" the full release. No timestamp, so layers
    # stay identical.
    marker_path = os.path.join(install_dir, "UNSLOTH_PREBUILT_INFO.json")
    try:
        with open(marker_path) as f:
            marker = json.load(f)
    except (OSError, ValueError):
        marker = {}
    llama_tag = base_build_tag(marker.get("upstream_tag") or tag)
    marker.setdefault("tag", llama_tag)
    marker.setdefault("release_tag", tag)
    marker.setdefault("published_repo", RELEASE_REPO)
    with open(marker_path, "w") as f:
        json.dump(marker, f, indent = 2)
        f.write("\n")
    print(
        f"marker augmented for freshness: tag={marker['tag']} "
        f"release_tag={marker['release_tag']} published_repo={RELEASE_REPO}"
    )

    # Mirror into build/bin/ so Studio's setup.sh sees a complete local build and skips
    # its source-build fallback, which would compile CPU-only llama.cpp over the baked
    # CUDA bundle. Hardlinks keep the $ORIGIN rpath.
    build_bin = os.path.join(install_dir, "build", "bin")
    os.makedirs(build_bin, exist_ok = True)
    for entry in os.listdir(install_dir):
        source = os.path.join(install_dir, entry)
        if os.path.isfile(source) and not os.path.islink(source):
            try:
                os.link(source, os.path.join(build_bin, entry))
            except OSError:
                shutil.copy2(source, os.path.join(build_bin, entry))
        elif os.path.islink(source):
            target = os.readlink(source)
            dest = os.path.join(build_bin, entry)
            if "/" not in target and not os.path.lexists(dest):
                os.symlink(target, dest)

    sanity_check_binaries(install_dir, build_bin)
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
