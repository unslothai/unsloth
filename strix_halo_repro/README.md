# Strix Halo memory reporting: a repro kit

AMD reported that llama.cpp reads the Variable Graphics Memory carve-out as
shared iGPU memory on Ryzen AI Max (Strix Halo), so large models will not load,
and that Unsloth Studio then uses very little of the GPU. They said the effect
is visible with the carve-out set to 96 GiB.

We ran the same measurements on a devlab Strix Halo box and **could not
reproduce the over-report there**, because that machine carves out 64 GiB of its
128 GB. At 64 GiB the numbers stay inside the machine and nothing looks wrong:

| reading | devlab box, 64 GiB carve-out |
|---|---|
| carve-out, from the registry | 64.00 GiB |
| RAM Windows can see | 63.65 GiB |
| so the machine holds | 127.65 GiB |
| Vulkan heaps, raw | 37.22 GiB + 74.43 GiB (device-local) |
| **what ggml reports as GPU memory** | **111.65 GiB** = the two heaps added together |
| what HIP reports | 99.74 GiB |

The summation is confirmed: 37.22 + 74.43 is ggml's 111.65 to the byte, because
`ggml_backend_vk_get_device_memory` adds every heap on an integrated device. But
111.65 is still under the 127.65 GiB the machine has, so nothing is claimed that
cannot exist, and a 77 GiB model loaded and served on that box.

Whether the sum goes past the machine depends on the carve-out, which is the one
thing a CI runner cannot change: setting it needs the firmware or the Adrenalin
panel plus a reboot, and the runner has no administrator. Hence this kit, for
someone with a box they can configure.

## What you need

- A Windows Ryzen AI Max / Max+ machine (gfx1150 or gfx1151).
- Python 3.9 or newer from python.org. The Microsoft Store stub does not work;
  the script says so rather than failing obscurely.
- About 1 GB of free disk for two llama.cpp release zips, plus whatever a model
  needs if you use the optional model step.
- No administrator rights, no pip install, no Docker.

## Setting the carve-out

The interesting configuration is a large one. On most of these machines it is
set in one of two places, and both need a reboot:

- **Firmware**: look for UMA Frame Buffer Size, or Dedicated Graphics Memory,
  under an Advanced or NBIO menu.
- **AMD Software: Adrenalin Edition**: Performance, then Tuning, then Variable
  Graphics Memory.

Set it as high as the machine allows (96 GiB is what AMD reported against),
reboot, and run the kit. Running it at your current setting first is also
useful: two carve-outs on one machine is a stronger result than one.

## Running it

```powershell
cd strix_halo_repro
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

Takes a few minutes and about 200 MB of downloads. To also ask what llama.cpp
would place for a model you already have:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 -ModelPath D:\gguf\model-00001-of-00004.gguf
```

To measure AMD's patched HIP runtime beside the shipped one:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 `
  -PatchedDllUrl "https://.../hiprtc-builtins0716.zip" `
  -PatchedDllSha256 "72cb13857ef4da1bbcaa0a5a415c2db1e138898b430e36758f040e038f416131"
```

The script prints an output directory at the end. **Send that whole directory
back.** It holds `REPORT.md` and the raw JSON behind every number, so a reading
can be re-checked rather than taken on trust.

## What it measures, and what a bad answer looks like

1. **The host.** Carve-out from the registry, RAM Windows can see, GPU name and
   driver version. WMI's `AdapterRAM` is recorded too, and it will say 4 GB on
   any large card because the field is a 32-bit integer; it is there as the trap
   it is, not as a reading.

2. **llama.cpp.** Two release builds, ROCm gfx1151 and Vulkan, at a pinned tag.
   Both are hashed so the report names exactly which binaries answered.

3. **What every layer reports.** Per-heap Vulkan sizes and flags, the number
   `ggml_backend_vk_get_device_memory` returns, the number `hipMemGetInfo`
   returns, and what `llama-server --list-devices` prints. The three statements
   at the end of `REPORT.md` are the result:
   - `over_report`: the number llama.cpp places against is larger than the
     machine's physical memory. **If this holds, the claim reproduces.**
   - `sums_heaps`: that number equals the heaps added together rather than the
     largest one, which is the mechanism.
   - `hip_is_vgm_only`: HIP reports the carve-out alone, so the two backends
     disagree because they answer different questions.

4. **The largest single allocation** the HIP runtime allows, found by bisection,
   writing and reading back every candidate so a pointer that is not backed by
   memory does not count. The search stops at 92% of the machine on purpose:
   AMD's own note warns that committing near all of RAM can hang the host or
   bugcheck it, and the question is whether an allocation the shipped runtime
   refuses now succeeds, not what the new ceiling is.

5. **Unsloth Studio's own Vulkan probe**, fetched verbatim at a pinned commit
   and run against the same build. A re-implementation would answer a different
   question: the claim is about what Studio sees.

6. **What `--fit` would place** for a model you name, via `llama-fit-params`.
   This is where an over-report turns into a model that will not load: fit plans
   against the number from step 3, so if that number is larger than the machine,
   it will offload more than can exist.

## What we already know from the 64 GiB box

Worth having beside your numbers, so a difference is attributable:

- The largest single `hipMalloc` was **110.2 GiB**, not the 64 GiB AMD described.
  The vendor's own note says the stock rule is
  `max(dedicated_VRAM_heap, 0.75 x shared_GART_heap)`, so on a box with a small
  carve-out and a large aperture the GART branch wins and there is no 64 GiB
  constant to hit. On a 96 GiB carve-out the dedicated branch may dominate
  instead, in which case their patch changes nothing there and the interesting
  arm is the memory report rather than the cap.
- The Vulkan build was faster than the ROCm build on the same model:
  **+22.9% prefill, +8.3% decode**, each gap wider than the spread within an arm.
- Prefill time grows about 6.35 ms per token, so a 20 minute first-token
  deadline is crossed near 197k tokens and a 250k prompt projects to about 26
  minutes. Studio renews that deadline on prompt progress as of 2026-09-03, and
  the server does emit the progress it renews on.
