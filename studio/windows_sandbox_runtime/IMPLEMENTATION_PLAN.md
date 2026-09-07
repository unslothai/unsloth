# Native Windows Python bootstrap implementation

## Product contract

Use Studio's selected release CPython and its packages. Run each Python tool in
one LPAC process with zero payload capabilities; support threads and explain
worker/subprocess denial. Never grant a package initializer temporary startup
authority. Terminal remains a separate compatibility problem. Do not replace
the current PR's Windows launchers or change their profile selection as a side
effect of bringing in this experimental runtime.

## Delivery stages

1. **Port the reusable Python infrastructure.** Bring the protected runtime
   generations, dependency admission, native ABI hosts, bounded startup protocol,
   per-invocation identity, Job ownership, process policy and focused tests onto
   the current PR. Keep worker imports independent of Studio logging and retain
   ownership through failed preparation/cleanup. A failed native startup must
   terminate without CRT/DLL detach callbacks running under temporary authority.
2. **Replace the Pillow diagnostic adapter with an ordinary component.** Inspect
   exact PE bytes without executable loading. Retain resource identity, language,
   manifest bytes and content hashes. Admit only one strictly empty assembly
   resource; reject unknown semantics. Pin the source while preparing its
   activation context. Match the actual mapped image for HMODULE_VALID requests,
   transfer a context reference per load, and retain the hook and owner references
   through finalization. Use a pinned Detours source build, never fixed ntdll
   offsets or edits to installed Pillow binaries.
3. **Integrate and qualify the expanded bootstrap.** Add a versioned context-plan
   channel bound to the protected final snapshot. Integrate delayed startup-token
   assignment and a reviewed private Winsock catalog instead of relying on the
   earlier host-registry warmup. Prove retained authority, real DNS/network denial,
   initialization timing, cancellation and parent-death cleanup. Install the
   adapter only at a demonstrated thread-quiescent point. This stage is required
   before enabling the expanded Python backend; the preceding components do not
   implement it implicitly.
4. **Deliver and select per execution kind.** Build all admitted ABI artifacts,
   verify installed-package delivery, and key qualification by the complete
   runtime/helper/OS/policy/manifest-plan identity. Python and Terminal must not
   share an availability claim for a Python-only profile. Required execution must
   refuse an unavailable/unqualified combination. Keep the existing explicit
   Limited behavior and its disclosures.

## Version-specific maintenance

- Discover the selected interpreter and package files; never choose a nearby
  interpreter, hardcode a Pillow ABI filename, edit the selected environment, or
  replace CUDA packages. ABI registry membership is not qualification.
- Build each CPython minor against matching release headers. Unknown versions,
  architectures, debug and free-threaded builds receive an explicit unsupported
  result. Add adapters only with their native matrix results.
- The private asyncio Proactor wakeup adapter uses CPython implementation details.
  Its support remains limited to the tested minor registry. Exercise loop creation,
  cross-thread wakeups, full-buffer behavior and shutdown on each supported minor;
  do not treat a public asyncio import as validation of these private fields.
- Inspect the final protected image bytes, not a source path that can be replaced
  between scanning and copying. Hash full images and preserve resource ID/name,
  language, codepage and exact XML bytes. A changed image or manifest invalidates
  the plan even when a package version string is unchanged.
- Treat XML semantics as an allowlist. A new dependency, directive, resource
  layout or language ambiguity requires review; do not delete metadata to make
  loading succeed. An image without a manifest requires no compatibility context.
- Pin compiler, SDK and Detours revision and record source/header/binary hashes.
  The current loader's use of KernelBase!CreateActCtxW remains an observed Windows
  behavior, not a supported Windows sandbox extension contract. After an OS update,
  repeat the loader reachability and wrong-module controls before qualification.
- Keep compatibility and enforcement tests separate. An import pass cannot
  replace a host-secret, DNS, IPC or lifecycle denial test with a positive control.

## Acceptance evidence

Component delivery requires strict manifest negative cases, real selected-Pillow
resource inspection, a native build with warnings as errors, wrong-HMODULE and
unsupported-request refusal, repeated actual DLL unload/reload, and startup-failure
controls. Expanded-runtime activation additionally requires every stage 3 gate
plus real tool streaming and per-kind profile selection from stage 4.

As implemented in this change, the native activation adapter is a separately
built component. It is not compiled into the packaged Python host. The expanded
private-catalog startup remains a diagnostic composition; importing the earlier
bootstrap infrastructure does not convert that composition into a production
launch. The production backend selector is unchanged.

## API references

- [ACTCTXW and authoritative hModule semantics](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-actctxw)
- [Activation-context reference ownership](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-addrefactctx)
- [Detours thread enlistment](https://github.com/microsoft/Detours/wiki/DetourUpdateThread)
- [Detours transaction commit](https://github.com/microsoft/Detours/wiki/DetourTransactionCommit)
