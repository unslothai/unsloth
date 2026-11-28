#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Convert/prepare GGUF models for Qualcomm Hexagon NPU (GenieX) and register them for Unsloth Studio.

Unlike AMD's proprietary `.q4nx` format, Qualcomm GenieX (`ggml-hexagon` HTP0) accelerates
standard `Q4_0`, `Q8_0`, and selected tiled/K-quant GGUFs natively without kernel shape
allowlists. This tool:
  1. Downloads a GGUF from Hugging Face (with resumable HTTP Range transfers), or takes any
     local GGUF (F16/BF16/Q8_0/Q4_K_M/Unsloth export).
  2. Converts non-native or source GGUFs to Hexagon-native `Q4_0` (with `Q8_0` output &
     embedding tensors by default, both HTX/HVX-accelerated on HTP0) via `llama-quantize.exe`.
  3. Registers the result in GenieX's model cache (`geniex.json`) so it appears immediately
     in `geniex list` and Unsloth Studio's NPU Model Hub / Chat picker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Optional

import httpx

HEXAGON_NATIVE_QUANTS = (
    "Q4_0",
    "Q4_1",
    "Q8_0",
    "Q4_K_M",
    "Q4_K_S",
    "Q5_K_M",
    "Q5_K_S",
    "Q6_K",
    "Q3_K_M",
    "Q3_K_S",
    "Q2_K",
    "IQ4_NL",
    "MXFP4",
)
_NO_REQUANTIZE_SOURCES = frozenset({"F32", "F16", "BF16"})
_QUANT_TAG_RE = re.compile(
    r"(?:^|[-_.])(BF16|F16|F32|Q8_0|Q6_K|Q5_K_[SML]|Q4_K_[SML]|Q4_1|Q4_0|Q3_K_[SML]|Q2_K_S|Q2_K|IQ4_XS|IQ4_NL|IQ3_XXS|IQ3_XS|IQ3_[SM]|IQ2_XXS|IQ2_XS|IQ2_[SM]|IQ1_XXS|IQ1_XS|IQ1_[SM]|MXFP4)(?:[-_.]|$)",
    re.IGNORECASE,
)


def resolve_geniex_cache(override: Optional[Path] = None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    env_dir = os.environ.get("GENIEX_DATADIR", "").strip()
    if env_dir:
        return Path(os.path.expanduser(env_dir)).resolve()
    return (Path.home() / ".cache" / "geniex").resolve()


def resolve_studio_exports() -> Path:
    for env_key in ("UNSLOTH_STUDIO_HOME", "STUDIO_HOME"):
        val = os.environ.get(env_key, "").strip()
        if val:
            return Path(os.path.expanduser(val)) / "exports"
    unsloth_home = os.environ.get("UNSLOTH_HOME", "").strip()
    if unsloth_home:
        return Path(os.path.expanduser(unsloth_home)) / "studio" / "exports"
    return Path.home() / ".unsloth" / "studio" / "exports"


def find_llama_quantize() -> Optional[Path]:
    env_bin = os.environ.get("UNSLOTH_LLAMA_QUANTIZE_BIN", "").strip()
    if env_bin and Path(env_bin).is_file():
        return Path(env_bin)
    which_bin = shutil.which("llama-quantize")
    if which_bin and Path(which_bin).is_file():
        return Path(which_bin)
    candidates = (
        Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "Release" / "llama-quantize.exe",
        Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "llama-quantize.exe",
        Path.home() / ".unsloth" / "studio" / "bin" / "llama-quantize.exe",
    )
    for c in candidates:
        if c.is_file():
            return c
    return None


def detect_quant_from_filename(path: Path) -> Optional[str]:
    m = _QUANT_TAG_RE.search(path.stem)
    return m.group(1).upper() if m else None


def detect_quant_from_gguf(path: Path) -> Optional[str]:
    tag = detect_quant_from_filename(path)
    if tag:
        return tag
    try:
        import gguf  # type: ignore

        reader = gguf.GGUFReader(str(path))
        field = reader.get_field("general.file_type")
        if field is not None:
            ftype = int(field.parts[field.data[0]])
            name = gguf.LlamaFileType(ftype).name
            return name.replace("MOSTLY_", "").upper()
    except Exception:
        pass
    return None


def derive_repo_name(source_path: Path, explicit_name: Optional[str]) -> str:
    if explicit_name and explicit_name.strip():
        name = explicit_name.strip().replace("\\", "/").strip("/")
        return name if "/" in name else f"local/{name}"
    parent_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", source_path.parent.name).strip("-") or "model"
    stem_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", source_path.stem).strip("-")
    slug = parent_slug if parent_slug.lower() not in ("exports", "downloads", "models", ".") else stem_slug
    digest = hashlib.sha1(os.path.normcase(str(source_path.parent)).encode("utf-8")).hexdigest()[:6]
    return f"local/{slug[:48]}-{digest}"


def download_hf_file(
    repo_id: str,
    filename: str,
    dest_dir: Path,
    revision: str = "main",
    max_retries: int = 12,
) -> Path:
    """Download a GGUF from Hugging Face using resumable HTTP Range transfers."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_path = dest_dir / Path(filename).name
    part_path = dest_dir / f"{final_path.name}.part"
    if final_path.is_file() and final_path.stat().st_size > 0 and not part_path.exists():
        print(f"[HF] Using existing file: {final_path} ({final_path.stat().st_size / 1e9:.2f} GB)")
        return final_path

    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
    headers: dict[str, str] = {"User-Agent": "unsloth-qualcomm-npu-bridge/1.0"}
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for attempt in range(1, max_retries + 1):
        downloaded = part_path.stat().st_size if part_path.is_file() else 0
        req_headers = dict(headers)
        if downloaded > 0:
            req_headers["Range"] = f"bytes={downloaded}-"
        try:
            with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, connect=20.0)) as client:
                with client.stream("GET", url, headers=req_headers) as resp:
                    if resp.status_code == 416 and part_path.is_file() and downloaded > 0:
                        os.replace(part_path, final_path)
                        print(f"[HF] Completed: {final_path}")
                        return final_path
                    if resp.status_code == 200:
                        downloaded = 0
                        mode = "wb"
                        total = int(resp.headers.get("Content-Length") or 0)
                    elif resp.status_code == 206:
                        mode = "ab"
                        cr = resp.headers.get("Content-Range", "")
                        total = int(cr.rsplit("/", 1)[-1]) if "/" in cr and cr.rsplit("/", 1)[-1].isdigit() else (
                            downloaded + int(resp.headers.get("Content-Length") or 0)
                        )
                    else:
                        resp.raise_for_status()
                        return final_path

                    last_pct = -5
                    t0 = time.monotonic()
                    with open(part_path, mode) as fh:
                        for chunk in resp.iter_bytes(chunk_size=1 << 20):
                            if not chunk:
                                continue
                            fh.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                pct = int(downloaded * 100 / total)
                                if pct >= last_pct + 5 or pct == 100:
                                    last_pct = pct
                                    elapsed = max(0.1, time.monotonic() - t0)
                                    mb_s = (downloaded / 1e6) / elapsed
                                    print(
                                        f"[HF] {final_path.name}: {pct}% "
                                        f"({downloaded / 1e9:.2f}/{total / 1e9:.2f} GB, {mb_s:.1f} MB/s)",
                                        flush=True,
                                    )
                    if total == 0 or downloaded >= total:
                        os.replace(part_path, final_path)
                        print(f"[HF] Download complete: {final_path} ({final_path.stat().st_size / 1e9:.2f} GB)")
                        return final_path
        except Exception as exc:  # noqa: BLE001
            wait_s = min(30, 2 * attempt)
            print(f"[HF] Attempt {attempt}/{max_retries} interrupted ({exc}); resuming in {wait_s}s...", flush=True)
            time.sleep(wait_s)

    raise RuntimeError(f"Failed to download {repo_id}/{filename} after {max_retries} attempts.")


def quantize_gguf(
    src_gguf: Path,
    dst_gguf: Path,
    target_quant: str = "Q4_0",
    *,
    pure: bool = False,
    threads: int = 0,
) -> Path:
    """Convert/quantize `src_gguf` to `target_quant` for Qualcomm Hexagon HTP0."""
    quantizer = find_llama_quantize()
    if quantizer is None:
        raise FileNotFoundError(
            "llama-quantize.exe was not found. Install or build llama.cpp in Unsloth Studio first."
        )
    dst_gguf.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst_gguf.with_name(f".quantizing-{dst_gguf.name}")

    src_quant = detect_quant_from_gguf(src_gguf)
    base_cmd = [str(quantizer)]
    if (src_quant and src_quant not in _NO_REQUANTIZE_SOURCES) or not src_quant:
        base_cmd.append("--allow-requantize")

    attempts: list[list[str]] = []
    if pure:
        attempts.append([*base_cmd, "--pure", str(src_gguf), str(tmp_dst), target_quant.upper()])
    elif target_quant.upper() == "Q4_0":
        # Both Q4_0 and Q8_0 have native 32x2/tiled HVX kernels in libggml-htp-v73.so;
        # keeping output.weight & token_embd at Q8_0 preserves accuracy while staying on HTP0.
        attempts.append([
            *base_cmd,
            "--output-tensor-type",
            "Q8_0",
            "--token-embedding-type",
            "Q8_0",
            str(src_gguf),
            str(tmp_dst),
            target_quant.upper(),
        ])
        # Fallback if a model architecture rejects mixed head/embedding overrides.
        attempts.append([*base_cmd, str(src_gguf), str(tmp_dst), target_quant.upper()])
    else:
        attempts.append([*base_cmd, str(src_gguf), str(tmp_dst), target_quant.upper()])

    last_code = 1
    for idx, cmd in enumerate(attempts):
        if threads > 0:
            cmd = [*cmd, str(threads)]
        if tmp_dst.exists():
            tmp_dst.unlink()
        print(f"[QUANTIZE] Running (attempt {idx + 1}/{len(attempts)}): {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, check=False)
        last_code = proc.returncode
        if last_code == 0 and tmp_dst.is_file() and tmp_dst.stat().st_size > 0:
            os.replace(tmp_dst, dst_gguf)
            print(f"[QUANTIZE] Wrote {dst_gguf} ({dst_gguf.stat().st_size / 1e9:.2f} GB)", flush=True)
            return dst_gguf
        if tmp_dst.exists():
            tmp_dst.unlink()

    raise RuntimeError(f"llama-quantize exited with code {last_code}")


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        try:
            if os.path.samefile(src, dst):
                return
        except OSError:
            pass
        if dst.stat().st_size == src.stat().st_size:
            return
        dst.unlink()
    try:
        os.link(src, dst)
        print(f"[LINK] Hardlinked {src} -> {dst}")
    except OSError:
        shutil.copy2(src, dst)
        print(f"[COPY] Copied {src} -> {dst}")


def register_in_geniex(
    gguf_path: Path,
    repo_name: str,
    precision: str,
    geniex_cache: Path,
    *,
    display_name: Optional[str] = None,
    mtp_path: Optional[Path] = None,
) -> Path:
    """Place/hardlink `gguf_path` (and optional MTP draft head) into GenieX cache and `geniex.json`."""
    parts = [p for p in repo_name.strip().replace("\\", "/").split("/") if p]
    if len(parts) == 1:
        parts = ["local", parts[0]]
    model_dir = geniex_cache / "models" / Path(*parts)
    model_dir.mkdir(parents=True, exist_ok=True)

    dest_gguf = model_dir / gguf_path.name
    _link_or_copy(gguf_path, dest_gguf)
    size_bytes = dest_gguf.stat().st_size
    prec_key = precision.strip().upper()
    manifest_path = model_dir / "geniex.json"

    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
        except (OSError, ValueError):
            manifest = {}

    model_file = manifest.get("ModelFile")
    if not isinstance(model_file, dict):
        model_file = {}
    model_file[prec_key] = {
        "Name": dest_gguf.name,
        "Downloaded": True,
        "Size": size_bytes,
    }

    manifest["Name"] = "/".join(parts)
    manifest["ModelName"] = display_name or manifest.get("ModelName") or parts[-1].removesuffix("-GGUF")
    manifest["ModelType"] = manifest.get("ModelType") or "llm"
    manifest["PluginId"] = "llama_cpp"
    manifest["ModelFile"] = model_file
    manifest.setdefault("MMProjFile", {"Name": "", "Downloaded": False, "Size": 0})
    manifest.setdefault("TokenizerFile", {"Name": "", "Downloaded": False, "Size": 0})
    extra_files = manifest.get("ExtraFiles")
    if not isinstance(extra_files, list):
        extra_files = []

    if mtp_path is not None and mtp_path.is_file():
        dest_mtp = model_dir / mtp_path.name
        _link_or_copy(mtp_path, dest_mtp)
        mtp_size = dest_mtp.stat().st_size
        mtp_entry = {"Name": dest_mtp.name, "Downloaded": True, "Size": mtp_size}
        manifest["MtpFile"] = mtp_entry
        extra_files = [e for e in extra_files if not (isinstance(e, dict) and e.get("Name") == dest_mtp.name)]
        extra_files.append(mtp_entry)
        print(f"[MTP] Registered draft head: {dest_mtp} ({mtp_size / 1e9:.2f} GB)")

    manifest["ExtraFiles"] = extra_files

    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest), encoding="utf-8")
    os.replace(tmp, manifest_path)
    print(f"[REGISTER] {manifest['Name']}:{prec_key} -> {manifest_path}")
    return manifest_path


def scan_studio_exports(
    geniex_cache: Path,
    *,
    auto_quantize: bool = False,
    target_quant: str = "Q4_0",
    pure: bool = False,
) -> int:
    """Discover local GGUFs under Unsloth Studio's `exports/` and register them for Hexagon NPU."""
    exports_dir = resolve_studio_exports()
    if not exports_dir.is_dir():
        print(f"[SCAN] Exports directory does not exist: {exports_dir}")
        return 0
    count = 0
    for root, dirs, files in os.walk(exports_dir):
        rel_depth = len(Path(root).relative_to(exports_dir).parts)
        if rel_depth >= 4:
            dirs[:] = []
        for fname in sorted(files):
            if not fname.lower().endswith(".gguf"):
                continue
            low = fname.lower()
            if low.startswith(".") or "mmproj" in low or low.endswith(".part"):
                continue
            src = Path(root) / fname
            quant = detect_quant_from_gguf(src)
            repo_name = derive_repo_name(src, None)
            if quant in HEXAGON_NATIVE_QUANTS:
                register_in_geniex(src, repo_name, quant, geniex_cache)
                count += 1
            elif auto_quantize:
                base_stem = re.sub(r"[-_.]?(BF16|F16|F32)$", "", src.stem, flags=re.IGNORECASE)
                out_name = f"{base_stem}-{target_quant.upper()}.gguf"
                parts = repo_name.split("/")
                out_path = geniex_cache / "models" / Path(*parts) / out_name
                if not out_path.is_file():
                    quantize_gguf(src, out_path, target_quant, pure=pure)
                register_in_geniex(out_path, repo_name, target_quant, geniex_cache)
                count += 1
            else:
                print(
                    f"[SKIP] {src} (quant={quant or 'unknown'}) is not natively Hexagon-accelerated; "
                    "pass --auto-quantize to convert to Q4_0."
                )
    print(f"[SCAN] Registered {count} model(s) from {exports_dir}")
    return 0


def _quant_output_name(src_path: Path, src_quant: Optional[str], target_quant: str) -> str:
    stem = src_path.stem
    tq = target_quant.upper()
    if src_quant and re.search(rf"[-_.]{re.escape(src_quant)}$", stem, re.IGNORECASE):
        new_stem = re.sub(rf"([-_.]){re.escape(src_quant)}$", rf"\1{tq}", stem, flags=re.IGNORECASE)
    else:
        new_stem = f"{stem}-{tq}"
    return f"{new_stem}.gguf"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gguf", nargs="?", type=Path, help="Local GGUF file path")
    parser.add_argument("--hf-repo", help="Hugging Face repo ID (e.g. tablinum/Moon-Beam-8B-A1B-GGUF)")
    parser.add_argument("--hf-file", help="GGUF filename in --hf-repo (e.g. Moon-Beam-8B-A1B-Q4_K_M.gguf)")
    parser.add_argument("--hf-mtp-file", help="Optional MTP draft head GGUF path in --hf-repo")
    parser.add_argument("--hf-rev", default="main", help="Hugging Face revision (default: main)")
    parser.add_argument("--mtp-gguf", type=Path, help="Optional local MTP draft head GGUF path")
    parser.add_argument("--name", help="GenieX repo name (<org>/<repo>, default derived)")
    parser.add_argument("--quant", default="Q4_0", help="Target Hexagon quantization (default: Q4_0)")
    parser.add_argument("--pure", action="store_true", help="Use pure --pure quantization without Q8_0 head/embed")
    parser.add_argument("--no-convert", "--skip-quantize", action="store_true", dest="no_convert", help="Register input GGUF without re-quantizing")
    parser.add_argument("--also-register-source", action="store_true", help="Also register the downloaded/source GGUF alongside converted target")
    parser.add_argument(
        "--delete-source-after-quantize",
        action="store_true",
        help="Remove temporary source GGUF after conversion succeeds (useful on tight NVMe drives)",
    )
    parser.add_argument("--scan-exports", action="store_true", help="Scan Unsloth Studio exports/ and register Hexagon models")
    parser.add_argument("--auto-quantize", action="store_true", help="With --scan-exports, quantize F16/BF16/non-Hexagon GGUFs to --quant")
    parser.add_argument("--geniex-cache", type=Path, help="Override GenieX cache root (default: GENIEX_DATADIR or ~/.cache/geniex)")
    parser.add_argument("--threads", type=int, default=0, help="Worker thread count for llama-quantize")
    args = parser.parse_args(argv)

    geniex_cache = resolve_geniex_cache(args.geniex_cache)

    if args.scan_exports:
        return scan_studio_exports(
            geniex_cache,
            auto_quantize=args.auto_quantize,
            target_quant=args.quant.upper(),
            pure=args.pure,
        )

    src_gguf: Optional[Path] = args.gguf
    mtp_gguf: Optional[Path] = args.mtp_gguf
    repo_name = args.name
    if args.hf_repo:
        if not args.hf_file:
            parser.error("--hf-file is required when --hf-repo is specified.")
        repo_name = repo_name or args.hf_repo
        parts = [p for p in repo_name.split("/") if p]
        if len(parts) == 1:
            parts = ["hf", parts[0]]
        dl_dir = geniex_cache / "models" / Path(*parts)
        src_gguf = download_hf_file(args.hf_repo, args.hf_file, dl_dir, revision=args.hf_rev)
        if args.hf_mtp_file:
            mtp_gguf = download_hf_file(args.hf_repo, args.hf_mtp_file, dl_dir, revision=args.hf_rev)
    elif src_gguf is None:
        parser.error("Provide a local GGUF path, --hf-repo + --hf-file, or --scan-exports.")

    if not src_gguf.is_file():
        raise FileNotFoundError(f"Source GGUF not found: {src_gguf}")
    if mtp_gguf is not None and not mtp_gguf.is_file():
        raise FileNotFoundError(f"MTP draft GGUF not found: {mtp_gguf}")

    repo_name = derive_repo_name(src_gguf, repo_name)
    src_quant = detect_quant_from_gguf(src_gguf)
    target_quant = args.quant.upper()

    if args.no_convert or (src_quant and src_quant == target_quant):
        register_in_geniex(src_gguf, repo_name, src_quant or target_quant, geniex_cache, mtp_path=mtp_gguf)
        return 0

    if args.also_register_source and src_quant in HEXAGON_NATIVE_QUANTS:
        register_in_geniex(src_gguf, repo_name, src_quant, geniex_cache, mtp_path=mtp_gguf)

    parts = [p for p in repo_name.split("/") if p]
    if len(parts) == 1:
        parts = ["local", parts[0]]
    out_name = _quant_output_name(src_gguf, src_quant, target_quant)
    out_path = geniex_cache / "models" / Path(*parts) / out_name
    if not out_path.is_file() or out_path.stat().st_size == 0:
        quantize_gguf(src_gguf, out_path, target_quant, pure=args.pure, threads=args.threads)
    else:
        print(f"[QUANTIZE] Using existing converted file: {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")

    register_in_geniex(out_path, repo_name, target_quant, geniex_cache, mtp_path=mtp_gguf)

    if args.delete_source_after_quantize and not args.also_register_source and src_gguf.resolve() != out_path.resolve():
        try:
            src_size_gb = src_gguf.stat().st_size / 1e9
            src_gguf.unlink()
            print(f"[CLEANUP] Deleted source GGUF {src_gguf.name} ({src_size_gb:.2f} GB)")
        except OSError as exc:
            print(f"[CLEANUP] Warning: could not delete {src_gguf}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
