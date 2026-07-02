# Plan: Publish unslothai/stable-diffusion.cpp mirror + our own CPU/Apple prebuilts

## Context

The Unsloth Studio native diffusion engine downloads a prebuilt `sd-cli` / `sd-server`
(stable-diffusion.cpp) via `studio/install_sd_cpp_prebuilt.py`. Today it pulls from
**leejet/stable-diffusion.cpp** upstream releases. We want to own this like we own
**unslothai/llama.cpp**: a fork that builds and publishes OUR OWN prebuilt binaries on a
schedule, so we control reproducibility, integrity, and the macOS load floor.

**Why native is CPU/Apple-only.** On a GPU host, diffusers + our optimizations (regional
`torch.compile` ~2.2x, cuDNN/flash attention, FP8/INT8/NVFP4 quant, first-block-cache) is
faster than sd.cpp's CUDA path, which has none of those levers — so GPU hosts route to
diffusers. Native sd.cpp only wins where diffusers is weak: **CPU and Apple**. Therefore we
build native binaries ONLY for the platforms where native is actually the faster engine, and
skip CUDA/ROCm/Vulkan entirely (GPU = diffusers/torch). This also makes the CI far cheaper.

The Studio side is already prepared: `install_sd_cpp_prebuilt.py` reads `UNSLOTH_SD_CPP_REPO`
(repo override) + `UNSLOTH_SD_CPP_TAG` (pinned tag) and verifies the GitHub asset `digest`
(`_verify_sha256`). So the bulk of the work is the mirror repo + release CI; the Studio change
is a small default flip + resolver tweak.

## Coverage (user-confirmed): CPU / Apple ONLY

| Platform | Arch | Build | Runner | Notes |
|---|---|---|---|---|
| macOS | arm64 | Metal (`-DSD_METAL=ON`) | macos-26, `OSX_DEPLOYMENT_TARGET=14.0` | Apple fast path (diffusers/MPS weak) |
| macOS | x86_64 | CPU | macos-15-intel, `OSX_DEPLOYMENT_TARGET=13.3` | Intel Macs |
| Linux | x86_64 | CPU | ubuntu-22.04 (glibc 2.35) | **also covers WSL** (WSL = Linux x64) |
| Linux | aarch64 | CPU | ubuntu-24.04-arm | ARM servers |
| Windows | x86_64 | CPU | windows-2022 (MSVC+Ninja) | |

**Explicitly out of scope:** CUDA, ROCm, Vulkan native builds; GPU runners; cudart bundling;
per-gfx matrices. GPU stays on diffusers/torch.

## Reference pattern (verified this session)

`unslothai/llama.cpp` builds via `.github/workflows/unsloth-prebuilt.yml` (orchestrator) + six
per-accel children + `scripts/unsloth/` helpers (`assemble_metadata.py`, `package_bundle.py`,
`assert_macho_minos.sh`). Mechanisms to mirror: `resolve` (supply-chain aging — only build a
release public >=6h; stamp build-info + Unsloth fingerprint; upload ONE source artifact all
children extract) -> per-platform children (build from the source artifact, load-gate, package,
upload) -> `assemble` (fingerprint gate + manifest/sha256 index + coverage gate + **atomic
draft->publish**, no partial releases). Template files fetched to `workspace_81/temp/llamacpp_workflows/`.

## Key facts (verified)

- leejet builds both `sd-cli` and `sd-server` (`examples/cli`, `examples/server`) — the mirror
  ships both (sd-server is used by PR #6768's persistent server).
- leejet naming: `sd-<tag>-bin-<Darwin-macOS-…-arm64 | Linux-Ubuntu-…-x86_64 | win-cpu-x64>.zip`.
  leejet already ships macOS arm64, Linux x64 CPU, Windows CPU — we ADD macOS x86_64 and Linux
  aarch64 (the gaps in our target set), and rebuild the rest under our own fingerprint/integrity.
- Studio resolver (`resolve_release_asset`, `install_sd_cpp_prebuilt.py:88`): filters to `.zip`;
  macOS = darwin/macos + arch token; Linux = `linux` + arch + (no accel marker for auto/cpu);
  Windows = `bin-win` + `avx2` else any. For a CPU-only mirror the resolver needs essentially NO
  change — macOS x86_64 and Linux aarch64 already match by arch token; just confirm the Windows
  CPU asset resolves (contains `bin-win`, falls back to the plain build).

## Design

### A. Mirror repo (fork of leejet/stable-diffusion.cpp)

Fork so upstream C++ stays intact; add only `.github/workflows/` + `scripts/unsloth/`. Adapt the
llama.cpp orchestrator, heavily simplified (no CUDA/ROCm/Vulkan, no PR-mix):

- **`resolve`**: pick the upstream leejet tag with the >=6h aging window; reuse leejet's
  `master-<count>-<sha>` as the mirror tag (keeps `UNSLOTH_SD_CPP_TAG` comparable to upstream);
  stamp a source tarball with build-info + the "Compiled by the Unsloth team" fingerprint; upload
  the source artifact. Skip-if-already-published like llama.cpp.
- **Build children** (reusable `workflow_call`), each `cmake -DSD_BUILD_EXAMPLES=ON` (cli+server):
  - `macos` (arm64 Metal + x64 CPU): pinned `CMAKE_OSX_DEPLOYMENT_TARGET`, `@loader_path` rpath,
    load-gate via `assert_macho_minos.sh` (adapted for `sd-cli`/`sd-server`).
  - `cpu-linux` (x64 + arm64) and `cpu-windows` (x64, MSVC+Ninja).
- **Asset naming = leejet-compatible**, all `.zip`:
  `sd-<tag>-bin-Darwin-macOS-arm64.zip`, `sd-<tag>-bin-Darwin-macOS-x86_64.zip`,
  `sd-<tag>-bin-Linux-Ubuntu-24.04-x86_64.zip`, `sd-<tag>-bin-Linux-Ubuntu-24.04-aarch64.zip`,
  `sd-<tag>-bin-win-cpu-x64.zip`.
- **`assemble`**: fingerprint gate (verify the mark in every archive), generate
  `sd-prebuilt-manifest.json` + `sd-prebuilt-sha256.json`, coverage gate (all 5 assets present),
  atomic draft->publish. GitHub sets each asset `digest`, which the Studio already verifies.
- **Signing/notarization:** none. The Studio downloads via `urllib` (not a browser), so no macOS
  quarantine xattr is set and Gatekeeper does not block CLI-run binaries (matches llama.cpp).

### B. Studio-side switch (PR on the diffusion stack, after the mirror's first green release)

Small, in `studio/install_sd_cpp_prebuilt.py` + its test:
1. `DEFAULT_REPO = "unslothai/stable-diffusion.cpp"`; `DEFAULT_TAG` = the mirror's first tag.
2. Confirm `resolve_release_asset` picks correctly for all 5 CPU/Apple hosts (add a Windows CPU
   token only if the plain-`bin-win` fallback proves insufficient; likely no change needed).
   Keep the leejet fallback (env override still points back upstream).
3. Extend `test_sd_cpp_install.py` `_ASSETS` to the mirror's 5-asset set; assert host->pick for
   macOS arm64/x64, Linux x64/arm64, Windows x64; assert GPU hosts are unaffected (still diffusers).

## Critical files

- New (mirror repo): `.github/workflows/unsloth-sd-prebuilt.yml` (+ `-macos.yml`, `-cpu-linux.yml`,
  `-cpu-windows.yml`), `scripts/unsloth/{assemble_metadata.py,package_bundle.py,assert_macho_minos.sh}`.
- Studio: `studio/install_sd_cpp_prebuilt.py`, `studio/backend/tests/test_sd_cpp_install.py`.
- Local templates to adapt: `workspace_81/temp/llamacpp_workflows/{unsloth-prebuilt.yml,unsloth-prebuilt-macos.yml,unsloth-prebuilt-cpu.yml}`.

## Sequencing (chicken-and-egg)

1. Build the mirror repo + CI; `publish=false` dry run to validate the 5-way matrix (~10-20 min,
   no GPU runners so cheap).
2. First green **published** release with all 5 assets + manifest/sha256.
3. THEN the Studio PR flips `DEFAULT_REPO`/`DEFAULT_TAG` + resolver test (on the diffusion stack).

## Verification

- **Resolver unit tests** (hermetic): feed the mirror's 5 asset names to `resolve_release_asset`
  for macOS arm64/x64, Linux x64/arm64, Windows x64 -> correct pick; and CUDA/GPU host -> still
  routes to diffusers (native not selected).
- **CI dry run**: `publish=false` artifact-only run; inspect the 5 archives each contain `sd-cli`
  (+ `sd-server`) and carry the fingerprint.
- **Live install smoke** (this Linux box): `UNSLOTH_SD_CPP_REPO=unslothai/stable-diffusion.cpp
  python studio/install_sd_cpp_prebuilt.py --print-asset` then real `install()`, confirm
  `sd-cli --version` + `sd-server` launch, and drive one native CPU generation via the Studio.
- **Integrity**: each published archive matches its manifest sha256 and the GitHub asset digest.

## Staging (user-confirmed): fork + push CI now

Execution order:
1. **Preflight permissions**: `gh auth status`; confirm the token can create/fork under the
   `unslothai` org and enable Actions. If it CANNOT, stop and report (fall back to scaffold-only,
   or a private fork under danielhanchen), rather than pushing somewhere unintended.
2. **Fork** leejet/stable-diffusion.cpp -> `unslothai/stable-diffusion.cpp` (clone into the
   workspace to add files). Keep upstream C++ intact.
3. **Add CI + scripts** on a branch: `.github/workflows/unsloth-sd-prebuilt.yml` +
   `-macos.yml`/`-cpu-linux.yml`/`-cpu-windows.yml`, `scripts/unsloth/*`. Commit as Daniel Han
   (no AI/bot mentions, no emojis, no em dashes). `unset GH_TOKEN`/use `gh` creds for pushes that
   touch `.github/workflows/*` (needs `workflow` scope).
4. **Dry run**: trigger the orchestrator with `publish=false` (artifact-only), confirm all 5
   archives build + carry `sd-cli`/`sd-server` + the fingerprint. Iterate until green.
5. **First publish**: `publish=true` (or let the schedule run) -> a real release with the 5
   assets + manifest/sha256.
6. **Studio PR** (section B) on the diffusion stack once the release tag exists.

## Follow-ups (not this task)

- Nightly schedule + auto-bump of the Studio `DEFAULT_TAG` (PR bot), like llama.cpp.
- Add GPU native builds later ONLY if a real need appears (today: GPU = diffusers/torch).
