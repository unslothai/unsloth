# llama.cpp prebuilt resolution (download-host fast path)

`install_llama_prebuilt.py` installs prebuilt llama.cpp binaries published under
`unslothai/llama.cpp`. Historically it discovered the latest release by
enumerating the GitHub REST API (`GET /repos/unslothai/llama.cpp/releases?per_page=100`).
That endpoint is capped at 60 requests/hour for anonymous callers, per IP, which
shared or NAT'd IPs, VPNs, CI fleets, or a few repeated installs exhaust easily.
Once exhausted, every attempt returns `HTTP 403 rate limit exceeded` and the
installer falls back to a slow source build.

## The fast path

For the fork repo (`unslothai/llama.cpp`) with an unpinned or `latest` request,
resolution happens against the release-assets CDN, which is not subject to the
API rate limit:

1. `GET https://github.com/unslothai/llama.cpp/releases/latest/download/llama-prebuilt-sha256.json`
   (302-redirects to `release-assets.githubusercontent.com`). Gives the
   `release_tag` and every artifact's `sha256` plus source provenance.
2. Using that `release_tag`, fetch the coverage manifest at the tag-pinned URL
   `.../releases/download/<release_tag>/llama-prebuilt-manifest.json` (which
   asset serves which GPU/arch).
3. Each asset and the source archive is downloaded by name via
   `.../releases/download/<release_tag>/<asset>` (same CDN) and
   `codeload.github.com`. No API call.

Net result for a normal install: zero `api.github.com` calls. Checksum and
source-provenance validation are byte-for-byte identical to the API path; the
fast path reuses `parse_approved_release_checksums`, `parse_published_release_bundle`,
and `_validate_checksums_against_bundle`. It is an optimization, not a weaker
trust path.

### Code

- `iter_resolved_published_releases()` tries `_download_host_resolved_release(repo)`
  first, gated on the fork repo, a `latest` request, the escape hatch, and the
  caller's `allow_download_host_fast_path`.
- `_download_host_resolved_release()` builds a synthetic release payload with
  tag-pinned CDN URLs and hands it to the existing parsers.
- Any failure (older release lacking the JSON assets, a rejected checksum, a
  transient network error) falls through to the unchanged GitHub API enumeration,
  so there is no regression on availability.

### Single-latest tradeoff and the macOS walk-back

The CDN `releases/latest/download/` path can only surface the single latest
release; older tags are not enumerable without the API. So the fast path yields
only the latest and returns:

- On Windows and Linux this is the intended behavior. If the latest release's
  asset is broken, the installer drops to a source build rather than an older
  release (the general 2-deep release fallback is reduced to 1 here). This is an
  accepted tradeoff for avoiding the rate-limited API.
- On macOS the caller passes `allow_download_host_fast_path = False`, so macOS
  keeps the full API enumeration and its multi-release walk-back, which exists to
  skip a run of prebuilts built for a newer macOS than the host.

### Escape hatch

```
UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE=1
```

forces the legacy API-enumeration path (debugging / bisecting only).

## No new CI publishing is required

The fast path consumes assets the release pipeline already publishes. The only
invariants CI must keep, all already true today, are:

1. Every release uploads both `llama-prebuilt-manifest.json` and
   `llama-prebuilt-sha256.json` as release assets.
2. The newest usable release is the one GitHub marks Latest (not `draft`, not
   `prerelease`); `releases/latest/download/` resolves to it.
3. Asset filenames match what the manifest and checksum JSON name (they are
   downloaded by name via `releases/download/<tag>/<asset>`).

If a future release omits the JSON assets or is only a pre-release, the client
silently falls back to the API path for that install.

## Published asset schemas

### `llama-prebuilt-manifest.json` (coverage)

```jsonc
{
  "schema_version": 1,
  "component": "llama.cpp",
  "source_repo": "unslothai/llama.cpp",
  "source_repo_url": "https://github.com/unslothai/llama.cpp",
  "source_ref_kind": "mix",
  "requested_source_ref": "b9964-mix-53618c5",
  "resolved_source_ref": "b9964-mix-53618c5",
  "source_commit": "5c1660c...",
  "source_commit_short": "5c1660c",
  "upstream_repo": "ggml-org/llama.cpp",
  "upstream_tag": "b9964",
  "merged_prs": [ { "repo": "...", "number": 24423, "sha": "...", "title": "..." } ],
  "generated_at_utc": "2026-07-11T22:40:44Z",
  "artifacts": [
    {
      "asset_name": "app-b9964-mix-53618c5-windows-x64-rocm-gfx1151.zip",
      "install_kind": "windows-rocm",
      "gfx_target": "gfx1151",
      "mapped_targets": ["gfx1151"]
    },
    {
      "asset_name": "app-b9964-mix-53618c5-windows-x64-cuda12-newer.zip",
      "install_kind": "windows-cuda",
      "bundle_profile": "cuda12-newer",
      "runtime_line": "cuda12",
      "coverage_class": "newer",
      "supported_sms": ["86", "89", "90", "100", "120"],
      "min_sm": 86,
      "max_sm": 120,
      "rank": 20,
      "toolkit_version": "12.9"
    }
    // ... one entry per published bundle (CPU, x64/arm64 CUDA, Windows CUDA,
    //     per-gfx ROCm, macOS, Vulkan)
  ]
}
```

Client selection by `install_kind`:

- CUDA bundles: match the host SMs against `supported_sms` plus the
  `[min_sm, max_sm]` range and the driver-compatible `runtime_line`.
  `toolkit_version` is published for humans and is not read by the client.
- ROCm bundles: match the host `rocm_gfx_target` against the artifact's
  `mapped_targets` list, or against the `gfx_target` family label
  (`published_rocm_choice_for_host`). Not parsed from the asset filename.
- CPU / macOS bundles: matched by `install_kind` alone (macOS also gates on a
  minimum-OS check).

### `llama-prebuilt-sha256.json` (checksums + provenance)

```jsonc
{
  "schema_version": 1,
  "component": "llama.cpp",
  "release_tag": "b9964-mix-53618c5",   // the fast path reads this to pin URLs
  "upstream_tag": "b9964",
  "source_commit": "5c1660c...",
  "source_repo": "unslothai/llama.cpp",
  "artifacts": {
    "app-b9964-mix-53618c5-windows-x64-rocm-gfx1151.zip": {
      "kind": "windows-rocm-app",
      "repo": "unslothai/llama.cpp",
      "sha256": "b974d444...e2dd4",
      "source_commit": "5c1660c...",
      "source_commit_short": "5c1660c",
      "upstream_tag": "b9964"
    }
    // ... one entry per artifact, including the source archive and (optionally)
    //     the manifest asset itself, whose hash is cross-checked against the
    //     manifest actually downloaded.
  }
}
```

Every installed binary is verified against its `sha256` here at download time
(`download_file_verified`), and any asset without an approved hash is dropped
(`apply_approved_hashes` fails closed), so the fast path cannot install an
unverified binary.

## Host-key derivation (reference)

`detect_host()` produces a `HostInfo`; selection maps it to an artifact by:

| OS / accelerator            | selects `install_kind`      | discriminator                          |
|-----------------------------|-----------------------------|----------------------------------------|
| Windows x64 + NVIDIA        | `windows-cuda`              | `supported_sms` + driver runtime       |
| Windows x64 + AMD ROCm      | `windows-rocm`              | manifest `gfx_target`/`mapped_targets` |
| Windows x64 CPU             | `windows-cpu`               | -                                      |
| Windows arm64               | `windows-arm64`             | -                                      |
| Linux x64 + NVIDIA          | `linux-cuda`                | `supported_sms` + runtime line         |
| Linux arm64 + NVIDIA        | `linux-arm64-cuda`          | `supported_sms` + runtime line         |
| Linux x64/arm64 + AMD ROCm  | `linux-rocm`                | manifest `gfx_target`/`mapped_targets` |
| Linux CPU (x64/arm64)       | `linux-cpu` / `linux-arm64` | -                                      |
| macOS arm64 / x64           | `macos-arm64` / `macos-x64` | minimum-OS gate                        |
