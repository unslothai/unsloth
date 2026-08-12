# ADR 0006 — The native-path lease survives as a desktop *read* optimisation for the Studio backend; platform uploads always carry real bytes, and a lease is never sent to the platform

* Status: Accepted
* Date: 2026-08-12
* Scope: Faz 0B decides; Faz 5 (documents) and Faz 7 (chat attachments) execute. `features/native-intents`, `features/rag`, `features/chat`
* Supersedes: nothing. Superseded by: nothing.

## Context

Plan line 424 requires a decision record for native path upload. ADR 0001 defers
to this file by name (`:68`, "See the native-path decision record") and states the
consequence it is built on (`:155`, "Desktop-only capabilities (native path
leases) cannot migrate at all").

The native-path lease is what lets the desktop build accept a file the user
dropped or picked **without the webview reading its bytes**. It is easy to mistake
for a transport optimisation when it is actually a permission mechanism, so the
flow is stated before the decision.

### How the lease works today

The mechanism is entirely Tauri IPC — no HTTP — which is why
`contract-matrix.md:121` records that `features/native-intents/api.ts` has no row
in the matrix at all:

1. The user drops or picks a path. `registerNativeAttachmentPath(path)`,
   `registerNativeModelPath(path)` or `registerNativeDatasetPath(path)`
   (`features/native-intents/api.ts`) hands the path to Rust, which returns a
   `NativeIntent` carrying an opaque `token`.
2. When the app is ready to act, `consumeNativePathToken(token, operation)`
   exchanges the token for a `NativePathLeaseResponse`
   (`features/native-intents/types.ts:37-40`):

   ```ts
   export interface NativePathLeaseResponse {
     nativePathLease: string;
     displayLabel: string;
     expiresAtMs: number;
   }
   ```

3. The lease string goes to the **Studio backend**, a local process on the same
   machine, which therefore *can* open the path:

   ```ts
   // features/training/api/datasets-api.ts:112-118
   export function uploadNativeTrainingDataset(
     nativePathLease: string,
   ): Promise<UploadDatasetResponse> {
     const form = new FormData();
     form.append("nativePathLease", nativePathLease);
     return uploadTrainingDatasetForm(form);
   }
   ```

   Same shape in `features/rag/api/rag-api.ts:58-67`:

   ```ts
   /** A desktop drop the webview can only name through a Rust-signed grant. */
   export interface NativeUploadRef { nativePathLease: string; }
   export type UploadSource = File | NativeUploadRef;

   async function ragUpload(path, source, ocr?, caption?) {
     const form = new FormData();
     if (source instanceof File) form.append("file", source);
     else form.append("nativePathLease", source.nativePathLease);
   ```

So `UploadSource` is a two-branch union: browser bytes, or a signed reference the
local server dereferences itself. It is consumed by
`uploadKnowledgeBaseDocument:135`, `uploadThreadDocument:158` and
`uploadProjectDocument:181`.

Availability is a runtime probe, not a build flag
(`features/native-intents/use-native-readiness.ts`):
`useNativePathLeasesSupported()` returns `false` immediately when `!isTauri`, and
in Tauri polls `GET /api/health` for `native_path_leases_supported === true` up to
`MAX_READINESS_POLLS = 720`. `chat-page.tsx:2035` passes the result down as
`nativeReadsDisabled={!nativePathLeasesSupported}` (`:3395`).

The codebase also already has the *other* half — a byte-reading path with no
lease. `nativeAttachmentIntentToFile` (`features/native-intents/native-attachment-file.ts`)
calls `readNativeAttachmentFile(token)`, base64-decodes the payload in JS and
builds a real `File`. Both halves exist because both are needed: reference-passing
for large files a local server will read, byte-reading for files the webview needs
in hand.

### What the platform accepts

`POST /api/v1/datasets/<dataset_id>/documents` dispatches on a query parameter
(`api/apps/restful_apis/document_api.py:519,530-541`):

```python
upload_type = (request.args.get("type") or "local").lower()
…
if upload_type == "web":   return await _upload_web_document(dataset_id, kb, tenant_id)
if upload_type == "empty": return await _upload_empty_document(dataset_id, kb, tenant_id)
if upload_type != "local":
    return get_error_data_result(message='`type` must be one of "local", "web", or "empty".', code=RetCode.ARGUMENT_ERROR)
return await _upload_local_documents(kb, tenant_id)
```

and `local` means multipart bytes, nothing else (`:658-670`):

```python
async def _upload_local_documents(kb, tenant_id):
    form = await request.form
    files = await request.files
    if "file" not in files:
        return get_error_data_result(message="No file part!", code=RetCode.ARGUMENT_ERROR)
    file_objs = files.getlist("file")
```

`type=web` takes a URL; `type=empty` creates a placeholder row. There is **no
path-based ingest** — no `type=path`, no `local_path` form field, and no route
that opens a filesystem path supplied by a client. `POST /documents/ingest`
(`:1469`) is not an upload: it takes `doc_ids` plus a `run` value and re-parses
documents that already exist.

Two size limits read the same environment variable with different defaults:
`DOC_MAXIMUM_SIZE = int(os.environ.get("MAX_CONTENT_LENGTH", 128 * 1024 * 1024))`
(`common/settings.py:406`) and
`app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH", 1024 * 1024 * 1024))`
(`api/apps/__init__.py:81`). The effective per-document ceiling is the smaller of
the two.

### Why a lease could not be sent to the platform even if we wanted to

Three independent reasons, any one sufficient:

1. **The platform is not on the user's machine.** A lease references a local path.
   A remote server dereferencing it either fails (no such path) or — worse —
   succeeds against *its own* filesystem at that path. That is a client-controlled
   server-side file read, which is the bug class, not the feature.
2. **The lease is minted by our Tauri process for our Studio backend.** The
   platform has no verifier for it and no reason to acquire one.
3. **There is no field to put it in.** `_upload_local_documents` reads
   `files["file"]` plus `parent_path` and `parser_config`, and ignores the rest, so
   a `nativePathLease` field is silently dropped. The failure mode is
   `"No file part!"`, not a diagnosable rejection of the lease.

## Decision

**1. The lease stays, and its scope is narrowed to "local Studio backend only".**
`features/native-intents` is not removed. Model loading, GGUF inventory and
training-dataset upload remain Studio-local per ADR 0001 decision 2, and the lease
is how the desktop build avoids copying multi-gigabyte model files through the
webview. Those paths do not change.

**2. A lease is never placed in a platform request, in any field, under any name.**
Hard rule, not a default. Platform uploads are multipart bytes via `type=local`,
or a URL via `type=web`.

**3. For platform-bound uploads the desktop reads the bytes and posts a real
`File`.** `nativeAttachmentIntentToFile` already does exactly this, so the desktop
drop experience is preserved: the user still drags a file from Finder and it still
uploads. The change is invisible to them — bytes travel through the webview
instead of being read by a local server.

**4. `UploadSource`'s `NativeUploadRef` branch becomes Studio-only, and the
platform-facing upload signature takes `File` alone.** The union is not kept "just
in case" on the platform side: a function that accepts a variant it must reject at
runtime is a control that appears to work, which ADR 0001 decision 5 forbids.
Concretely — Faz 5's platform document upload accepts `File`;
`uploadNativeTrainingDataset` keeps its lease because its server is local.

**5. Base64 in the IPC payload is accepted as-is and not optimised.**
`readNativeAttachmentFile` returns `{name, mimeType, base64}` and the JS side
decodes it through `charCodeAt` into a `Uint8Array`. That costs roughly 2× the
file size in transient memory and is fine for the attachment-sized files it was
written for. It is not fine for a 128 MiB document, which is why decision 6 bounds
it rather than leaving the ceiling implicit.

**6. Platform uploads are size-checked client-side before the read, against the
platform's own limit.** The limit comes from configuration (default 128 MiB per
`common/settings.py:406`), never hardcoded, and the check uses the size the lease
already reports — so an oversized file is rejected **before** any bytes enter the
webview. A user dropping a 2 GiB file gets an immediate, specific message instead
of an out-of-memory crash followed by a 413.

**7. `useNativePathLeasesSupported()` keeps gating the desktop drop affordance,
and its meaning is documented as "this build can name a dropped OS path", not
"this build can upload".** Browser deployments have no lease and no path
registration, but they still have the ordinary file input, so upload works
everywhere. The two capabilities are decoupled in the UI so a browser user is
never shown a disabled control for a desktop-only mechanism.

**8. No `type=path` is requested from the backend.** Adding one would mean the
platform opening a filesystem path a client named; even restricted to a configured
directory it is a traversal surface for a capability multipart already covers. It
is also a backend source change, which the standing instruction excludes from
Faz 0 absent a prior ADR. Recorded as rejected, not deferred, so a later phase
does not rediscover it as an optimisation.

## Alternatives rejected

* **Send `nativePathLease` to the platform as-is** — no verifier, no such form
  field, so it is ignored and the upload arrives empty with `"No file part!"`. And
  a backend that *did* dereference it would be opening a path of the client's
  choosing on the server's disk.
* **Add a path-ingest route to the platform** (`type=path` or a `local_path`
  field) — multipart already transfers the file. A path buys skipping one copy on
  the single deployment where client and server share a disk, in exchange for a
  client-controlled server-side read on every deployment where they do not.
* **Keep `UploadSource` as a union on platform functions and throw on the
  `NativeUploadRef` branch** — a signature advertising a capability it always
  refuses. ADR 0001 decision 5.
* **Remove `features/native-intents` entirely** — load-bearing for Studio-local
  model loading and training-dataset upload, neither of which migrates. Removing
  it breaks working desktop features to tidy a boundary.
* **Stream the file from Rust straight to the platform, bypassing the webview** —
  the Tauri side would need the platform base URL and the platform bearer token,
  putting a second copy of the credential in a second process. ADR 0002 decision 6
  keeps the token where it must be and nowhere else, and one HTTP client per
  backend (ADR 0001 decision 4) is the point of the client boundary.
* **Read the bytes with no size check and let the platform answer 413** — the
  out-of-memory happens in the webview during decode, before any request is sent.
  The user gets a crash instead of an error message.
* **Hardcode 128 MiB as the client-side limit** — `MAX_CONTENT_LENGTH` is an
  environment variable, and the two backend defaults already disagree (128 MiB vs
  1 GiB). A deployment that raises it would find the client rejecting files the
  server would have accepted.

## Consequences

* Desktop uploads to the platform move bytes through the webview. For a file at
  the 128 MiB ceiling that is a real memory spike (base64 string plus decoded
  array), bounded by decision 6 but not eliminated. The upload progress UI must
  show a read phase before the request phase, or a large drop looks frozen.
* `features/native-intents` splits conceptually without splitting on disk:
  lease-passing (Studio-local, unchanged) and byte-reading (every platform
  upload). The two look interchangeable at the call site and are not, so the
  distinction is stated at both entry points.
* A browser-only deployment loses drag-and-drop **from the OS** but keeps file
  picking and in-browser drag-and-drop. This is the desktop-vs-browser product
  difference ADR 0001 already records, now with the upload path spelled out.
* The client-side size limit must be configured alongside the platform. A mismatch
  becomes a client misconfiguration showing as a spurious rejection — easier to
  diagnose than a 413 mid-upload, but it needs wiring to the same source of truth.
* Because leases never leave the local machine, no phase reasons about
  `expiresAtMs` against platform latency. Lease lifetime stays a Studio concern.
