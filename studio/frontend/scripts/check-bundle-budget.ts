// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Budget for the JavaScript Unsloth must download, parse and execute before the first screen exists.
 *
 * Two PRs (#8623, #8624) moved 1.5 MB of decoded resources off this path by hand, each measuring it with a
 * throwaway Chromium harness. Nothing then stopped the next static import from putting it back, and nothing
 * noticed when one did: a single `import` of a dialog that is closed on load was carrying 2.4 MB.
 *
 * The eager set is not inferred here. Vite already computes it and writes it into index.html: the entry
 * `<script type="module">` plus one `<link rel="modulepreload">` per chunk in the entry's STATIC import closure.
 * That is exactly what the browser fetches before the app boots. Chunks reachable only through `import()` carry
 * no preload link and are correctly not counted. A parser-blocking classic `<script src>`
 * (public/theme-boot.js) is added to that: it is not Vite's, but it runs before the module graph and so is part
 * of the same wait.
 *
 * Raising a budget is a normal thing to do. Doing it in the same diff as the import that needed it is the point.
 */

import { readFileSync, realpathSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";

const HERE = dirname(fileURLToPath(import.meta.url));
const DIST = resolve(HERE, "..", "dist");

/**
 * Measured on the build these were set from, plus a little headroom. Transfer is what crosses the wire. Raw is
 * what the main thread has to parse and execute, which is the part that shows up as a slow launch on a weak
 * machine.
 */
export const BUDGET = {
  // Re-measured together on one machine and build: 6,023.4 KB raw / 1,802.1 KB transfer at 5b78bafd04,
  // plus the margin the previous raises chose (210.8 KB raw, 65.1 KB transfer). The growth since is
  // the vLLM / SGLang engine options inside the model config page and Resources tab (#11491), which
  // live inline in the chat and settings chunks: lazy-loading the separable inference-engines
  // module recovers only 0.6 KB transfer, so what ran out is headroom, not laziness.
  // The browser panel's bookmarks, tab dragging and page zoom then added 2.8 KB raw / 1.0 KB
  // transfer (6,231.7 -> 6,234.5 KB raw, 1,865.0 -> 1,866.0 KB transfer, one machine and build),
  // inside the already-eager chat chunk; raw is raised by that so the margin stays where it was.
  // Re-measured again at cc84c7dc7a: 6,278.6 KB raw / 1,879.0 KB transfer, 11.8 KB transfer over.
  // Measured commit by commit from the browser panel (#12347), no single import did it: the panel
  // itself left 2.9 KB transfer over, the Sandbox settings tab (#12215) added 16.4 KB raw / 4.5 KB
  // transfer (its strings in every eager locale and the sandbox pick in the chat adapter), the
  // message action bar rework (#12735) 2.9 KB raw, and the other merges in that range the rest,
  // all in the eager chat, index and settings chunks. Headroom ran out, so the same margin is put
  // back.
  // The long-chat render work (#13255: deferred fences and reasoning rows) adds 33.3 KB raw / 13.1 KB
  // transfer to the eager chat chunk (6,477.1 -> 6,510.4 KB raw, 1,942.6 -> 1,955.7 KB transfer at
  // 160cb41fe, one machine and build); raised by that so main's margin stays where it was.
  // Project sources follow-ups (#13224: queued drops, retry on failed sources, folder drops) add
  // 6.6 KB raw / 2.0 KB transfer, mostly in the eager chat chunk (6,513.0 -> 6,519.6 KB raw,
  // 1,956.4 -> 1,958.4 KB transfer against d4ea5a51d, one machine and build). Lazy-loading the
  // folder walker recovered 0.5 KB raw and no transfer, so the budget is raised by that growth.
  // Then 19 small Studio merges took main over again with no single import doing it: 6,518.8 ->
  // 6,533.6 KB raw, 1,958.0 -> 1,963.7 KB transfer from 5f38afbc2d to 33176a75e9 (one machine and
  // build). The largest step is Regenerate title in the Library menu (#13491, +5.2 KB raw / +2.2 KB
  // transfer, code that shares the eager chat row menu module); the Library card and composer tweaks
  // and the edit_file download link add the rest, 0.2 to 1.7 KB transfer each. Raised by that growth
  // so main's margin stays where it was.
  // Twelve more Studio merges added 7.1 KB raw / 2.3 KB transfer (6,534.6 -> 6,541.7 KB raw,
  // 1,963.9 -> 1,966.2 KB transfer from f68585e2e9 to a7dc8a0640, one machine and build), again with
  // no single import doing it. The largest step is per-chat file history and /rewind (#13522, +4.3 KB
  // raw / +1.3 KB transfer in the eager chat chunk); the rest are 0.1 to 0.6 KB each. Raised by that
  // growth so main's margin stays where it was.
  // Sidebar server paging adds 17.4 KB raw / 5.6 KB transfer to the first screen (6,544.5 -> 6,561.9
  // KB raw, 1,967.1 -> 1,972.7 KB transfer against b73110c6f, one machine and build), raised by that.
  // /compact and the in-loop compaction divider add 3.6 KB raw / 1.4 KB transfer to the eager startup
  // JS (6,563.4 -> 6,567.0 KB raw, 1,973.0 -> 1,974.4 KB transfer against 280c45695, one machine
  // and build), raised by that.
  // Five merges since then add 2.1 KB raw / 0.8 KB transfer (6,567.3 -> 6,569.4 KB raw, 1,974.5 ->
  // 1,975.3 KB transfer from a062c04d51 to c35052e42f, one machine and build). The step that crossed
  // is the token activity Chart view (#13558, +0.7 KB raw / +0.3 KB transfer): its persisted store
  // pulls zustand/middleware into the startup set as a 99th chunk. Raised by that growth so main's
  // margin stays where it was.
  // Six merges since add 2.9 KB raw / 1.4 KB transfer (6,570.7 -> 6,573.6 KB raw, 1,975.5 -> 1,976.9 KB
  // transfer from b8d76e1696 to 8af12324b0, one machine and build), with the chunk count unchanged and
  // no single import doing it. The largest step is the audio model selector fix (#13517, +2.9 KB raw /
  // +0.9 KB transfer); thinking traces (#13545), MLX precision folders (#13565) and vLLM thinking
  // effort (#13571) add the rest. Raised by that growth so main's margin stays where it was.
  // The KV cache, GPU memory limit and seed engine options (#13596) add 5.7 KB raw / 1.4 KB transfer
  // (6,573.5 -> 6,579.2 KB raw, 1,976.9 -> 1,978.3 KB transfer against abfe3142c, one machine and
  // build), inline in the eager chat and settings chunks like the other engine options; raised by that.
  // Read aloud's token cap and length warning add 0.2 KB raw / 0.1 KB transfer to the eager chat chunk
  // (6,739,816 -> 6,740,052 B raw, 2,021,725 -> 2,021,835 B transfer against af9d97837e, one machine
  // and build), which took main over; raised by that.
  // Seven merges since add 2.1 KB raw / 0.7 KB transfer (6,741,756 -> 6,743,873 B raw, 2,025,885 ->
  // 2,026,588 B transfer from b44035a64 to cbf0f593d, one machine and build, Node 22), with the chunk count
  // unchanged and no new dependency. The step that crossed is eval-loss early stopping patience (#13626,
  // +806 B raw / +311 B transfer: its training field and strings); the maximized window layout fix in the
  // app provider (#13647) 539 B raw, the + menu reorder (#13679, measured together with #13659) 711 B raw,
  // and #13703 / #13700 the last 61 B. Raised by that growth so main's margin stays where it was.
  // The chat menu's Compact conversation item adds 1.6 KB raw / 0.3 KB transfer to the eager chat chunk
  // (6,741,807 -> 6,743,381 B raw, 2,021,316 -> 2,021,655 B transfer against 2a03c22812), raised by that.
  // Eleven frontend merges since add 4.6 KB raw / 1.2 KB transfer (6,745,911 -> 6,750,599 B raw, 2,027,285 ->
  // 2,028,544 B transfer from 85b6414c68 to 8611f1846c, one machine and build, Node 24), with the chunk count
  // unchanged at 95 and no new dependency. No single step crossed: the composer draft kept across chats
  // (#13722) 2.1 KB raw / 0.5 KB transfer, the collapsed sidebar rail's Images and Audio entries (#13731)
  // 2.1 KB raw / 0.4 KB transfer, and #13665, #13614, #13732, #13661, #13538 the rest (#13744's revert
  // gave back 0.4 KB raw). Raised by that growth so main's margin stays where it was.
  // Then the Audio API card in Settings (#12821) 2,019 B raw / 557 B transfer and #13630 132 B raw
  // (6,750,599 -> 6,752,750 B raw, 2,028,544 -> 2,029,099 B transfer to dd56fb3572, same machine and build).
  // Numbers wrapped tightly in dollars render as math (#13649): the shell-context checks that keep
  // `$1$?`-style shell text literal add 6,984 B raw / 2,836 B transfer to the eager chat chunk
  // (6,753,387 -> 6,760,371 B raw, 2,029,398 -> 2,032,234 B transfer against eba7641835, one machine
  // and build, chunk count unchanged at 95), raised by that.
  // Re-embedding documents after an embedding model switch (the save toast's action and the Settings row)
  // adds 3,018 B raw / 905 B transfer (6,760,371 -> 6,763,389 B raw, 2,032,234 -> 2,033,139 B transfer
  // against 525f3ceae, one machine and build, Node 24, chunk count unchanged at 95). Lazy-loading its store
  // from both save sites cost 0.3 KB transfer more in chunk overhead, so the budget is raised by that growth.
  // Keeping the tool pills in step with the loaded model's tool support (#13621) adds 1.2 KB raw / 0.4 KB
  // transfer to the eager chat chunk (6,753,387 -> 6,754,540 B raw, 2,029,398 -> 2,029,795 B transfer
  // against 97ed9415f7, before #13649, one machine and build), raised by that.
  // Offering the newest audio.cpp release in updates (#13589) adds 385 B raw / 80 B transfer to the eager set
  // (6,764,579 -> 6,764,964 B raw, 2,033,464 -> 2,033,544 B transfer from 91f8457ab3 to 7b6b2a8e48, one machine
  // and build, Node 24, chunk count unchanged at 95), which put main 43 B raw over. Raised by that growth so
  // main's margin stays where it was.
  // Naming the Windows update that adds MXC's built-in container (#13798) adds 358 B raw / 93 B transfer, its
  // string in the eager English locale (6,765,003 -> 6,765,361 B raw against 3c0061e16, one machine and build,
  // Node 24, chunk count unchanged at 95), which put main 55 B raw over. Raised by that growth.
  // Each chat keeping its own browser tabs (#13800: the store's per-chat state, background pages in the
  // native view, and the off-screen question strings) adds 5,278 B raw / 1,928 B transfer (6,765,003 ->
  // 6,770,281 B raw, 2,033,563 -> 2,035,491 B transfer against a8585676d4, one machine and build, Node 24,
  // chunk count unchanged at 95), raised by that.
  // Always allow on browser agent prompts (#13777) adds 91 B raw / 42 B transfer to the eager chat chunk
  // (6,770,922 -> 6,771,013 B raw, 2,035,640 -> 2,035,682 B transfer against cd947468f1, one machine and build,
  // Node 24, chunk count unchanged at 95), which put main 71 B raw over. Raised by that growth.
  // Word list numbering through the default style and basedOn (#13755) adds 228 B raw / 39 B transfer to the
  // eager attachment reader (6,770,962 -> 6,771,190 B raw, 2,035,676 -> 2,035,715 B transfer against 7ae1a5386,
  // one machine and build, Node 24, chunk count unchanged at 95). Raised by that growth.
  // The "Only show models that fit" Settings row adds 289 B raw / 77 B transfer (6,771,372 -> 6,771,661 B raw,
  // 2,035,798 -> 2,035,875 B transfer against 52a3fbefe, 95 chunks, identical under Node 22.22.2 and 24). Main
  // already measured 111 B raw / 54 B transfer over the previous limits, so they are set to the measured build.
  // Main was 48 B raw / 29 B transfer over at ffa8fc3b9 (6,771,709 B raw, 2,035,904 B transfer). The browser
  // panel's notice for pages that hit a sandbox limit (#13761) adds 143 B raw / 44 B transfer (-> 6,771,852 B raw,
  // 2,035,948 B transfer, one machine and build, Node 24, chunk count unchanged at 95). Raised by both.
  // Keeping site storage across browser panel pages (#13762) adds 1,644 B raw / 466 B transfer, the page-storage
  // module the eager browser store imports for tab close and duplicate (6,771,852 -> 6,773,496 B raw, 2,035,948 ->
  // 2,036,414 B transfer against 82f16ca55, one machine and build, Node 24, chunk count unchanged at 95). Raised by
  // that growth.
  // Hub Run refusing audio models neither Audio nor chat can run (#13812) adds 77 B raw / 65 B transfer
  // (6,773,445 -> 6,773,522 B raw, 2,036,392 -> 2,036,457 B transfer against 59ab069b5, one machine and build,
  // Node 24, chunk count unchanged at 95). Raised by that growth.
  // The Windows Desktop install location row on the first-run screen (#13446) adds 1,324 B raw / 342 B
  // transfer (6,773,496 -> 6,774,820 B raw, 2,036,386 -> 2,036,728 B transfer against 59ab069b5, one machine
  // and build, Node 24, chunk count unchanged at 95). Lazy-loading it cost more transfer and a 96th startup
  // chunk (Rollup split react-dom out), so it is imported statically and the budget raised by that growth.
  // Opening the OS folder dialog for model folders on Desktop (#13843) adds 235 B raw / 336 B transfer
  // (6,777,794 -> 6,778,029 B raw, 2,037,725 -> 2,038,061 B transfer against 199987c63, one machine and build,
  // Node 24, chunk count unchanged at 95). Raised by that growth.
  // The Windows Long Paths explainer on the first-run startup screen (#13523) adds 1,816 B raw / 519 B
  // transfer (6,778,788 -> 6,780,604 B raw, 2,038,337 -> 2,038,856 B transfer against 62e063618, one machine
  // and build, Node 24, chunk count unchanged at 95). Main already measured 3,656 B raw / 1,180 B transfer over
  // the previous limits there, so they are set to the measured build.
  // The downloads panel's "Starting…" row (#13738) adds 1,404 B raw / 353 B transfer (6,778,788 -> 6,780,192 B raw,
  // 2,038,337 -> 2,038,690 B transfer against 7c235de6f, one machine and build, Node 24, 95 chunks). Raised by that growth.
  // The Image downloads > Info stamp row in Settings > Library, its search entry and strings (#13444) add 348 B
  // raw / 83 B transfer to the eager settings and English locale chunks (6,780,553 -> 6,780,901 B raw,
  // 2,038,848 -> 2,038,931 B transfer against 9f5b133254, one machine and build, Node 24, chunk count unchanged
  // at 95). Raised by that growth.
  // Drafter training in the Train tab adds about 4.9 KB raw / 1.4 KB transfer (6,628.3 KB raw, 1,992.9 KB transfer
  // measured against cd424654a, one machine and build, 95 chunks): its persisted training-store fields, their
  // migration and the en strings hydrate at startup; the section itself is not in the startup set.
  // The Run settings Load in 4-bit switch (#13841: the per-model field threaded through every chat load path and
  // the resident-config check) adds 1,448 B raw / 433 B transfer, inline in the eager chat chunk (6,787,367 ->
  // 6,788,815 B raw, 2,040,805 -> 2,041,238 B transfer against 4494fa58c, one machine and build, Node 24, chunk
  // count unchanged at 95). Raised by that growth.
  // The Train tab's Preview tokens button, its dialog and strings (#13850) add 2,233 B raw / 858 B transfer
  // (6,788,764 -> 6,790,997 B raw, 2,041,243 -> 2,042,101 B transfer against 5e5511027, one machine and build,
  // Node 24, chunk count unchanged at 95). Lazy-loading the dialog cost more (2.7 KB transfer) and a 96th
  // startup chunk, so it is imported statically and the budget raised by that growth.
  // Three small Studio merges since add 747 B raw / 302 B transfer (6,791,021 -> 6,791,768 B raw, 2,042,102 ->
  // 2,042,404 B transfer from 587079569a to 4a8b727dfd, one machine and build, Node 24, chunk count unchanged at
  // 95), with no single import doing it: Deep research under Tools in the + menu (#13959) took main over, and
  // the connection name and provider menu order (#13964) and the attachment card polish (#13970) add the rest.
  // Raised by that growth so main's margin stays where it was.
  // The Hub fit filter sizing safetensors rows the way transformers loads them adds 497 B raw / 184 B transfer
  // (6,791,768 -> 6,792,265 B raw, 2,037,833 -> 2,038,017 B transfer against 23bbee09b1, one machine and build,
  // Node 26, chunk count unchanged at 95). A synchronous predicate in a module already in the startup set, so
  // nothing to lazy-load; raised by that growth.
  // Seven Studio merges after 17901061d1 add 5,031 B raw / 1,798 B transfer (6,791,768 -> 6,796,799 B raw,
  // 2,042,404 -> 2,044,202 B transfer from 17901061d1 to 6106de938c, one machine and build, Node 24, chunk count
  // unchanged at 95), measured commit by commit. The two largest steps are inline chat markdown tables with
  // their hover controls and expand dialog (#13973, +2,561 B raw / +1,063 B transfer, in the eager chat chunk)
  // and Latest Unsloth Models ranking (#13969, +1,290 B raw / +445 B transfer); the composer alignment
  // (#13965), the resizable hub split view (#13966) and the fit filter (#13945, already raised for above) add the
  // rest. Set to the measured build plus the margin main had at 17901061d1 (160 B raw, 189 B transfer).
  // Chat with files closing on its X, the app tooltips and the higher-contrast traces (#13989) add 1,339 B raw /
  // 377 B transfer (6,796,799 -> 6,798,138 B raw, 2,044,202 -> 2,044,579 B transfer against 1574001bc1, one
  // machine and build, Node 24, chunk count unchanged at 95). Raised by that growth so main's margin stays where
  // it was.
  // Three Studio merges since add 3,117 B raw / 1,058 B transfer (6,798,138 -> 6,801,255 B raw, 2,044,534 ->
  // 2,045,592 B transfer from 4ebc263832~1 to d4e005e73e, one machine and build, Node 24, chunk count unchanged at
  // 95). Nearly all of it is searching a model maker in the Model hub (#13998, +3,038 B raw / +1,000 B transfer):
  // its owner-to-uploads search lives in use-hub-model-search, a hook the startup screen already loads, so there
  // is no separate module to lazy-load. #13999 and #13993 add 79 B raw / 58 B transfer. Raised by that growth.
  // Sliding the selected pill on every segmented toggle (#14030) adds 4,432 B raw / 1,696 B transfer
  // (6,800,980 -> 6,805,412 B raw, 2,045,645 -> 2,047,341 B transfer against 15e4bc8fde, one machine and build,
  // Node 24, chunk count unchanged at 95): the shared slide module and SegmentedPill, imported by toggles that
  // already live in startup chunks, so there is no separate module to lazy-load. Raised by that growth.
  // QwenCloud as a connection (#14033) adds 167 B raw / 43 B transfer (6,804,234 -> 6,804,401 B raw, 2,047,526 ->
  // 2,047,569 B transfer against ca3d1d0ce4, one machine and build, Node 24, chunk count unchanged at 95): its
  // connection row, logo mapping and capability entries live in tables the startup chunks already load. Raised
  // by that growth.
  // Z.ai as a connection (#14034) adds 73 B raw / 29 B transfer (6,804,389 -> 6,804,462 B raw, 2,047,551 ->
  // 2,047,580 B transfer against f172c8d7c4, one machine and build, Node 24, chunk count unchanged at 95): its
  // logo mapping, capability row and vision gate entries live in tables the startup chunks already load. Raised
  // by that growth.
  // The audio.cpp v0.9.2 picker rows (#14073) add 253 B raw / 59 B transfer to the eager audio catalog
  // table (one machine and build, chunk count unchanged at 95). Raised by that.
  // The audio "More models" group and catalog refresh (#14072) add 2,559 B raw / 911 B transfer (6,810,132 ->
  // 6,812,691 B raw, 2,049,298 -> 2,050,209 B transfer against f2ed00705f, one machine and build, Node 24, chunk
  // count unchanged at 95): the picker rows and catalog hook live in the audio picker the startup chunks load.
  // Raised by that growth.
  // Audio failure actions, seeds and the single voice picker (#14075) add 492 B raw / 134 B transfer (6,812,691 ->
  // 6,813,183 B raw, 2,050,209 -> 2,050,343 B transfer against 4bed2fa246, one machine and build, Node 24, chunk count
  // unchanged at 95): the shared audio failure and seed helpers are imported by startup chunks. Raised by that growth.
  // The Images pill making images and videos with the loaded local models (#14110) adds 5,425 B raw / 1,442 B
  // transfer (6,814,532 -> 6,819,957 B raw, 2,050,886 -> 2,052,328 B transfer against 82523339bd, one machine and build,
  // Node 24, chunk count unchanged at 95): the local image and video cards in the eager tool UI registry and the
  // composer and adapter gates in the eager chat chunk. Raised by that growth.
  // Model memory modes, the refusal toasts and the chat-context save phase (#14108) add 4,315 B raw / 1,495 B
  // transfer (6,819,958 -> 6,824,273 B raw, 2,052,278 -> 2,053,773 B transfer against e2a214bc08, one machine and
  // build, Node 24, chunk count unchanged at 95): the refusal and evicted-model toast helpers are imported by the
  // eager chat, image and video chunks. Raised by that growth.
  // Matching letter shortcuts by the letter the layout types (#14113) adds 328 B raw / 137 B transfer
  // (6,809,493 -> 6,809,821 B raw, 2,049,157 -> 2,049,294 B transfer against a6da154b3, one machine and build,
  // Node 24, chunk count unchanged at 95): the matcher runs on every keydown from the first screen, so there is
  // no separate module to lazy-load. Raised by that growth.
  // Main was still over after those raises (2fee36ffa3: 6,825,378 B raw / 2,053,999 B transfer against 6,818,878 /
  // 2,051,590): each was sized from its PR's own delta on top of a limit that already sat below the build, so the
  // shortfall carried forward (it starts at #14073, whose limit was 92 B of transfer under its own build). Moving the
  // FLUX.3 prompt upsampler client (#14137) out of images/api.ts, which startup chunks import, saves 4,372 B raw /
  // 1,291 B transfer. With #14113 on top that leaves 6,821,334 B raw / 2,052,857 B transfer (78e5eefa4d plus this
  // change). Three more PRs landed while main was over, without raising the limit: the darker picker toggle (#14183,
  // +76 B raw / -243 B transfer), one tooltip design everywhere (#14172, +3,897 B raw / +1,376 B transfer: the shared
  // tooltip is in the eager shell) and pinning FLUX.3 Klein in the Images picker (#14184, +418 B raw / +82 B
  // transfer). That gives 6,825,725 B raw / 2,054,072 B transfer at b227654927 plus this change (one machine and
  // build, Node 24, chunk count unchanged at 95; CI has matched this machine to the byte). Both limits are reset to
  // that build plus a 160 B raw / 189 B transfer margin, measured absolutely rather than by delta.
  // Opening //host links as web links (#14190) adds 177 B raw / 52 B transfer (6,825,725 -> 6,825,902 B raw,
  // 2,054,072 -> 2,054,124 B transfer against 4dc8bf0bd2 and 551dc28123, one machine and build, Node 24, chunk
  // count unchanged at 95): open-link.ts is the shared link opener the eager shell imports. Raised by that growth,
  // again to the measured build plus the same 160 B raw / 189 B transfer margin.
  // Keeping the FLUX.3 Image Klein GGUF row under its Fast FP8 row (#14197) adds 188 B raw / 66 B transfer
  // (6,825,902 -> 6,826,090 B raw, 2,054,124 -> 2,054,190 B transfer against 94464babd6 and 1de12ab309, one machine
  // and build, Node 24, chunk count unchanged at 95): the picker catalog is imported by startup code (chat-adapter.ts).
  // Raised by that growth, to the measured build plus the same 160 B raw / 189 B transfer margin.
  // Keeping the session when auth.db is busy (#14151) adds 829 B raw / 393 B transfer (6,826,090 -> 6,826,919 B
  // raw, 2,054,190 -> 2,054,583 B transfer against a33463fb16, one machine and build, Node 24, chunk count
  // unchanged at 96): the refresh retry and outcome live in features/auth/api.ts, which every authFetch imports.
  // Raised by that growth, to the measured build plus the same 160 B raw / 189 B transfer margin.
  // Chunked fence lines, hoisted line colour, the progressive highlight queue and pointer state
  // (#13919) add 11,012 B raw / 3,984 B transfer (6,826,518 ->
  // 6,837,530 B raw, 2,054,604 -> 2,058,588 B transfer against ffd6c3033, one machine and build, Node 24,
  // chunk count unchanged at 96), all in the eager chat chunk the fences render from. Raised to that
  // build plus the same 160 B raw / 189 B transfer margin.
  // Choosing a working folder when creating a project (#13873) adds 1,756 B raw / 719 B transfer (6,837,530 ->
  // 6,839,286 B raw, 2,058,588 -> 2,059,307 B transfer against 6c3a9e9b3, main's build being its limit less the margin;
  // one machine and build, Node 24, chunk count unchanged at 96): the picker and the folder reconciliation live in the
  // create dialog, which the eager sidebar imports. Raised to the measured build plus the same 160 B raw / 189 B
  // transfer margin.
  // Keeping the Read aloud speech model loaded beside the chat model (#14191) adds 1,442 B raw / 723 B transfer
  // (6,826,504 -> 6,827,946 B raw, 2,054,599 -> 2,055,322 B transfer against 9faa995217, one machine and build,
  // Node 24, chunk count unchanged at 96): the speech slot routing, evicted-model reports and Voice tab rows sit in
  // the audio hook, chat adapter and settings chunks the startup set already loads. Raised by that growth.
  // Developer mode opt-in for experimental CPU video generation (#14247) adds 1,777 B raw / 654 B transfer
  // (6,840,406 -> 6,842,183 B raw, 2,059,667 -> 2,060,321 B transfer against 88afa84b87 and 26e246d407, one machine
  // and build, Node 24, 96 chunks): its strings join the eager English catalog and the developer-mode
  // flag is read by the eager shell (__root.tsx, app-sidebar.tsx, env.ts, hardware-verdict.ts). Raised by that
  // growth, to the measured build plus a 160 B raw / 189 B transfer margin.
  // Naming a chat export after its model, title and day (#12586) adds 615 B raw / 65 B transfer (6,842,183 ->
  // 6,842,798 B raw, 2,060,321 -> 2,060,386 B transfer against 46e4f50bea, one machine and build, Node 24,
  // 96 chunks): the name builder sits beside the chat exporters in the eager chat chunk. Raised by that growth.
  transferBytes: 2_060_575,
  rawBytes: 6_842_958,
};

// The chunk count is reported but not budgeted. Splitting a page out of the entry raises it while lowering the
// bytes, which is the behaviour this is trying to reward; capping it would penalise the fix.

/** A start tag: lowercased name, and its attributes by lowercased name. A valueless attribute like `defer` is
 * present with an empty-string value, as HTML defines it. */
type StartTag = { name: string; attrs: Map<string, string> };

/**
 * ASCII whitespace, which is what separates one attribute from the next. CR is in here for safety only: the
 * parser's input preprocessor turns every CR into an LF before the tokenizer sees it, so a CRLF file cannot
 * behave differently. https://html.spec.whatwg.org/multipage/parsing.html#preprocessing-the-input-stream
 */
const WHITESPACE = new Set(["\t", "\n", "\f", "\r", " "]);

/** First index at or after `from` that is not ASCII whitespace. */
function skipWhitespace(html: string, from: number): number {
  let i = from;
  while (i < html.length && WHITESPACE.has(html[i] as string)) {
    i += 1;
  }
  return i;
}

/** First index at or after `from` that ends a name or an unquoted value. */
function scanUntil(html: string, from: number, stop: string): number {
  let i = from;
  while (
    i < html.length &&
    !WHITESPACE.has(html[i] as string) &&
    !stop.includes(html[i] as string)
  ) {
    i += 1;
  }
  return i;
}

/** One attribute value, from the `=`. Quoted values run to their closing quote. */
function readValue(
  html: string,
  from: number,
): { value: string; next: number } {
  const i = skipWhitespace(html, from + 1);
  const quote = html[i];
  if (quote !== '"' && quote !== "'") {
    const end = scanUntil(html, i, ">"); // Unquoted: ends at whitespace or `>`.
    return { value: html.slice(i, end), next: end };
  }
  const close = html.indexOf(quote, i + 1);
  return close < 0
    ? { value: html.slice(i + 1), next: html.length }
    : { value: html.slice(i + 1, close), next: close + 1 };
}

/**
 * Reads the attributes of one start tag, beginning just past its name, and returns where the tag ends.
 *
 * This is the tokenizer's attribute states, narrowed to what a build artefact can contain. The two rules that
 * matter, and that no regex over the tag text can express:
 *
 *   - `>` ends the tag only OUTSIDE a quoted value. Inside one it is ordinary text ("anything else: append the
 *     current input character to the current attribute's value"), so `data-note="a > b"` is one attribute and
 *     the tag does not end there.
 *     https://html.spec.whatwg.org/multipage/parsing.html#attribute-value-(double-quoted)-state
 *   - An attribute begins only after whitespace, `/` or a previous value, so a NAME is the only place `async`
 *     or `type` can be read from. A value is arbitrary text: `data-mode="load async later"` contains no `async`
 *     attribute. https://html.spec.whatwg.org/multipage/parsing.html#before-attribute-name-state
 *
 * Both were live bugs while this searched the tag text instead: the first dropped the entry chunk, the second
 * dropped the parser-blocking script, and in each case enough of the build survived to satisfy the shape guard
 * below, so the gate reported a comfortable pass on a startup path it had not measured.
 */
function readAttributes(
  html: string,
  start: number,
): { attrs: Map<string, string>; end: number; closed: boolean } {
  const attrs = new Map<string, string>();
  let i = start;
  while (i < html.length) {
    // Before attribute name: whitespace is ignored, and `/` (self-closing, or a
    // stray solidus) is not part of a name.
    if (WHITESPACE.has(html[i] as string) || html[i] === "/") {
      i += 1;
      continue;
    }
    if (html[i] === ">") {
      return { attrs, end: i + 1, closed: true };
    }
    const nameEnd = scanUntil(html, i, "/>=");
    const name = html.slice(i, nameEnd).toLowerCase();
    // After attribute name: whitespace may separate the name from its `=`.
    i = skipWhitespace(html, nameEnd);
    let value = "";
    if (html[i] === "=") {
      ({ value, next: i } = readValue(html, i));
    }
    // A repeated attribute is a parse error and the browser keeps the first, so a
    // later `src=` cannot overwrite the one the browser actually fetches.
    if (name && !attrs.has(name)) {
      attrs.set(name, value);
    }
  }
  return { attrs, end: i, closed: false };
}

/**
 * End of a comment, from just past its `<!--`. `<!-->` and `<!--->` close there rather than running on: the
 * comment start and comment start dash states both end the comment on `>`. Read as unterminated, the rest of the
 * file disappears along with the tags in it.
 * https://html.spec.whatwg.org/multipage/parsing.html#comment-start-state
 */
const COMMENT_END = /^-?>|--!?>/;
function endOfComment(html: string, start: number): number {
  const m = COMMENT_END.exec(html.slice(start));
  return m ? start + m.index + m[0].length : html.length;
}

/** End of a script element's raw text, from just past its start tag. */
const SCRIPT_END = /<\/script[\t\n\f\r >/]/i;
function endOfScriptBody(html: string, start: number): number {
  const m = SCRIPT_END.exec(html.slice(start));
  return m ? start + m.index : html.length;
}

/** Tag open state: only an ASCII letter after `<` starts a tag name. */
const TAG_NAME_START = /[a-z]/i;

/**
 * Every start tag in the document, in order.
 *
 * Comments and script bodies are skipped rather than scanned, for the same reason the attribute parser exists: a
 * tag is only a tag where the browser sees one. A `<script src>` commented out during debugging is not
 * downloaded and must not be charged, and a tag written inside a string in an inline script is text.
 * https://html.spec.whatwg.org/multipage/parsing.html#script-data-state
 *
 * A script body ends at the first `</script`, which is the tokenizer's answer unless the body itself contains
 * `<!-- <script`: those escaped states let a `</script>` be text. Erring there reads a bit of script body as
 * markup, so the mistake is to charge a chunk that is not there rather than to miss one. Checked against parse5
 * over 40,000 generated documents, that is the only case left.
 */
function* startTags(html: string): Generator<StartTag> {
  let i = 0;
  while (i < html.length) {
    const lt = html.indexOf("<", i);
    if (lt < 0) {
      return;
    }
    if (html.startsWith("<!--", lt)) {
      i = endOfComment(html, lt + 4);
      continue;
    }
    if (!TAG_NAME_START.test(html[lt + 1] ?? "")) {
      i = lt + 1;
      continue;
    }
    // Tag name state: the name ends at whitespace, `/` or `>`.
    const nameEnd = scanUntil(html, lt + 1, "/>");
    const name = html.slice(lt + 1, nameEnd).toLowerCase();
    const { attrs, end, closed } = readAttributes(html, nameEnd);
    if (!closed) {
      return; // A tag the file ends in the middle of is never emitted, or fetched.
    }
    yield { name, attrs };
    i = name === "script" ? endOfScriptBody(html, end) : end;
  }
}

/** Reads an attribute off a tag. Absent is undefined; valueless is `""`. */
function attr(tag: StartTag, name: string): string | undefined {
  return tag.attrs.get(name);
}

/** True for a valueless attribute like `defer`, which carries no value to read. */
function hasAttr(tag: StartTag, name: string): boolean {
  return tag.attrs.has(name);
}

/** `rel` is a space-separated token list, and its tokens are case-insensitive. */
function relTokens(tag: StartTag): string[] {
  return (attr(tag, "rel") ?? "").toLowerCase().split(/\s+/).filter(Boolean);
}

/**
 * True when the tag carries `blocking="render"`, which holds the FIRST RENDER back until the resource has been
 * fetched and evaluated.
 *
 * The spec's own reading: "Let value be the value of el's blocking attribute... converted to ASCII lowercase...
 * split on ASCII whitespace", then "An element is potentially render-blocking if its blocking tokens set
 * contains 'render'", and in prepare the script element, "If el is potentially render-blocking, then block
 * rendering on el", which is reached for any external script, `async` or not. The async/defer carve-out applies
 * only to what is IMPLICITLY render-blocking.
 * https://html.spec.whatwg.org/multipage/urls-and-fetching.html#blocking-attributes
 * https://html.spec.whatwg.org/multipage/scripting.html#prepare-the-script-element
 *
 * Measured, not assumed: a `<script async blocking="render" src>` held back for two seconds moved first
 * contentful paint from 20 ms to 2,020 ms in Chromium 151 and from 11 ms to 2,009 ms in WebKit 26.5, with
 * Chromium reporting the request's renderBlockingStatus as "blocking". Firefox has not shipped it
 * (bugzil.la/1751383) and simply treats the script as async.
 */
function blocksRender(tag: StartTag): boolean {
  return (attr(tag, "blocking") ?? "")
    .toLowerCase()
    .split(/\s+/)
    .includes("render");
}

/**
 * The eager set, in three buckets, as paths relative to `dist/`. `entry` and `preloads` are Vite's output and
 * stay apart so the caller can tell "Vite stopped emitting preload links" from "this page really does load one
 * chunk": flattened, a build with no preloads reads as a very small app. `blocking` is the classic `<script
 * src>` in `<head>`, which is not Vite's and carries no preload link, but is parser-blocking: it is downloaded
 * and run before the module graph starts. `public/theme-boot.js` is one, and left out it could grow without
 * limit inside a gate whose whole subject is startup JavaScript.
 */
export type EagerSet = {
  entry: string[];
  preloads: string[];
  blocking: string[];
};

/**
 * A `type` the browser still runs as a classic script. An absent type is the same thing; anything else
 * (`importmap`, `application/json`, a template) is not code that runs, and `module` is handled on its own.
 *
 * Exact strings, deliberately. The spec matches this attribute on JavaScript MIME type ESSENCE, so a parameter
 * makes it match nothing: `text/javascript; charset=utf-8` is not evaluated, and Chromium, Firefox and WebKit do
 * not even fetch such a script. Bytes the browser never requests are not startup cost.
 *
 * The whole essence list, not the four anyone would write today. The legacy spellings are not historical
 * trivia: every one of them still executes. Measured in Chromium 151, a script tagged
 * `application/x-javascript`, `text/jscript`, `text/javascript1.5`, `text/livescript`, `application/x-ecmascript`
 * or `text/x-javascript` runs, while `text/javascript; charset=utf-8` and `application/json` do not. The list is
 * frozen, so this does not grow. https://mimesniff.spec.whatwg.org/#javascript-mime-type
 */
const CLASSIC_TYPES = new Set([
  "application/ecmascript",
  "application/javascript",
  "application/x-ecmascript",
  "application/x-javascript",
  "text/ecmascript",
  "text/javascript",
  "text/javascript1.0",
  "text/javascript1.1",
  "text/javascript1.2",
  "text/javascript1.3",
  "text/javascript1.4",
  "text/javascript1.5",
  "text/jscript",
  "text/livescript",
  "text/x-ecmascript",
  "text/x-javascript",
]);

/** Same-origin, build-relative, and not a traversal out of `dist/`. */
function distRelative(url: string | undefined): string | undefined {
  if (!url?.startsWith("/") || url.startsWith("//")) {
    return undefined; // Absent, external, or protocol-relative: not ours to budget.
  }
  const path = url.slice(1).split(/[?#]/)[0];
  return path && !path.split("/").includes("..") ? path : undefined;
}

/** What the browser fetches and runs before the app boots, in document order. */
export function eagerSetFromHtml(html: string): EagerSet {
  const set: EagerSet = { entry: [], preloads: [], blocking: [] };
  const seen = new Set<string>();
  const add = (into: string[], url: string | undefined, prefix = "") => {
    const path = distRelative(url);
    if (!path?.startsWith(prefix) || seen.has(path)) {
      return;
    }
    seen.add(path);
    into.push(path);
  };

  const tags = [...startTags(html)];
  for (const tag of tags.filter((t) => t.name === "script")) {
    const type = attr(tag, "type")?.toLowerCase();
    if (type === "module") {
      // Vite's entry, always one of its own hashed assets.
      add(set.entry, attr(tag, "src"), "assets/");
    } else if (!type || CLASSIC_TYPES.has(type)) {
      // Counts from anywhere in the build, not just assets/. `defer` is included: a deferred script runs after
      // parsing but BEFORE DOMContentLoaded, in document order with the module entry, which is itself deferred, so
      // it is on exactly the timeline this budgets. Only `async` is out, having no ordering relationship to the
      // first screen at all. `async` also wins when both are present, which is why it is the one tested. Unless it
      // is asked to block rendering, which restores the relationship the exclusion assumes is absent:
      // `blocking="render"` is the documented way to keep a boot script off the parser without letting the
      // unthemed page paint, and it delays the first screen by the whole fetch and evaluation. Not counting it
      // would leave the one thing this gate exists to bound, bytes between the navigation and the first screen,
      // unbounded.
      if (!hasAttr(tag, "async") || blocksRender(tag)) {
        add(set.blocking, attr(tag, "src"));
      }
    }
  }
  for (const tag of tags.filter((t) => t.name === "link")) {
    if (relTokens(tag).includes("modulepreload")) {
      add(set.preloads, attr(tag, "href"), "assets/");
    }
  }
  return set;
}

/** Flattened eager set, in the order the browser gets to it. */
export function eagerChunksFromHtml(html: string): string[] {
  const { entry, preloads, blocking } = eagerSetFromHtml(html);
  return [...blocking, ...entry, ...preloads];
}

/**
 * Text that marks React-preview runtime code. The preview libraries and the bootstrap that imports
 * them are only ever fetched when a preview opens, so either string in a startup chunk means that
 * code was pulled onto the startup path: a static import where a lazy one was meant.
 */
export const LAZY_ONLY_NEEDLES = ["__unslothModules", "__vite_ssr_import__"];

/** Every (chunk, needle) pair where an eager chunk contains a needle. A chunk that cannot be read is skipped. */
export function eagerChunksContaining(
  distDir: string,
  chunks: string[],
  needles: string[],
): { chunk: string; needle: string }[] {
  const found: { chunk: string; needle: string }[] = [];
  for (const chunk of chunks) {
    let text: string;
    try {
      text = readFileSync(join(distDir, chunk), "utf8");
    } catch {
      continue; // measure() reports a missing chunk; this only looks inside the ones that exist.
    }
    for (const needle of needles) {
      if (text.includes(needle)) {
        found.push({ chunk, needle });
      }
    }
  }
  return found;
}

type Measured = { name: string; raw: number; transfer: number };

/**
 * What the browser actually downloads. The backend gzips the `/assets` mount only (studio/backend/main.py mounts
 * `_AssetGZipMiddleware` there); everything else goes out through a plain FileResponse, so its raw size IS its
 * transfer size.
 */
function transferBytes(name: string, bytes: Buffer): number {
  return name.startsWith("assets/")
    ? gzipSync(bytes, { level: 6 }).byteLength
    : bytes.byteLength;
}

function measure(names: string[]): Measured[] | string {
  const out: Measured[] = [];
  for (const name of names) {
    let bytes: Buffer;
    try {
      bytes = readFileSync(join(DIST, name));
    } catch {
      // index.html names a file the build did not emit. Reporting it rather than a stack trace, because the
      // alternative reading, that the budget is fine, is the one that must never be reachable.
      return `dist/index.html references ${name}, which is not in the build`;
    }
    out.push({
      name,
      raw: bytes.byteLength,
      transfer: transferBytes(name, bytes),
    });
  }
  return out;
}

function kb(bytes: number): string {
  return `${(bytes / 1024).toFixed(1)} KB`;
}

function main(): number {
  let html: string;
  try {
    html = readFileSync(join(DIST, "index.html"), "utf8");
  } catch {
    console.error("no dist/index.html -- run `npm run build` first");
    return 2;
  }

  const { entry, preloads, blocking } = eagerSetFromHtml(html);
  const names = [...blocking, ...entry, ...preloads];

  // Counting Vite's own output only: a parser-blocking classic script is not evidence that the
  // module graph was read correctly, so it cannot stand in for the entry when deciding whether this
  // still understands the build. A code-split build of this app is dozens of chunks. One or none
  // means the shape this reads has changed and the number below would be fiction: with
  // `build.modulePreload: false` the links disappear and the entry alone measured 424 KB of a 5,207
  // KB startup path, reporting 4.8 MB to spare. A comfortable pass is the one answer this must
  // never give by accident. Counting scripts and links together rather than requiring both: when
  // the entry module is nothing but imports, Vite inlines it into one `<script>` per imported chunk
  // and emits no preload links at all, which is a complete measurement. So the total is what
  // decides whether this is a code-split build. But at least one entry is required on top of that
  // total, because preloads without one is not a build shape Vite emits: a modulepreload link
  // exists to announce the entry's static import closure, so links surviving while the entry does
  // not means the entry was read wrong, not that it is absent. Left to the total alone, the app's
  // 48 links carry the guard while the largest single chunk in the startup path silently leaves the
  // measurement, which is exactly how two mis-parses of this file's own making stayed invisible.
  const fromVite = entry.length + preloads.length;
  if (entry.length === 0 || fromVite < 2) {
    console.error(
      entry.length === 0
        ? `dist/index.html yielded no module entry (and ${preloads.length} preload link(s)), so there is nothing trustworthy to measure here.`
        : `dist/index.html yielded ${fromVite} eager chunk(s) from Vite, so there is nothing trustworthy to measure here.`,
    );
    console.error(
      'A code-split build served from the site root gives a `<script type="module" src="/assets/...">` plus one `<link rel="modulepreload" href="/assets/...">` per statically imported chunk. If the shape changed on purpose -- `build.modulePreload` turned off, a non-root or relative `base`, a different `build.assetsDir`, `renderBuiltUrl` pointing at a CDN -- teach scripts/check-bundle-budget.ts the new shape rather than leaving a gate that measures nothing.',
    );
    return 2;
  }

  const sized = measure(names);
  if (typeof sized === "string") {
    console.error(sized);
    return 2;
  }
  const measured = sized.sort((a, b) => b.raw - a.raw);
  const raw = measured.reduce((sum, c) => sum + c.raw, 0);
  const transfer = measured.reduce((sum, c) => sum + c.transfer, 0);

  console.log(
    `eager startup JS: ${kb(raw)} raw, ${kb(transfer)} transfer, ${measured.length} chunks`,
  );
  console.log("largest:");
  for (const c of measured.slice(0, 8)) {
    console.log(
      `  ${kb(c.raw).padStart(10)} raw  ${kb(c.transfer).padStart(9)} transfer  ${c.name}`,
    );
  }

  const lazyOnly = eagerChunksContaining(DIST, names, LAZY_ONLY_NEEDLES);
  if (lazyOnly.length > 0) {
    console.error("\nReact-preview runtime code is on the startup path:");
    for (const { chunk, needle } of lazyOnly) {
      console.error(`  ${chunk} contains ${needle}`);
    }
    console.error(
      "It belongs only in chunks reached through import(). Find the static import that pulled it in.",
    );
    return 1;
  }

  const over: string[] = [];
  if (transfer > BUDGET.transferBytes) {
    over.push(`transfer ${kb(transfer)} > ${kb(BUDGET.transferBytes)}`);
  }
  if (raw > BUDGET.rawBytes) {
    over.push(`raw ${kb(raw)} > ${kb(BUDGET.rawBytes)}`);
  }

  if (over.length > 0) {
    console.error(`\nover the startup budget: ${over.join(", ")}`);
    console.error(
      "Something is now imported statically that the first screen does not need. " +
        "Either load it on use (React.lazy, lazyRouteComponent, or a dynamic import " +
        "at the point of use), or raise BUDGET in this file in the same PR, with the " +
        "measurement that justifies it. If the chunk count above went up, check the " +
        "opposite first: a dynamic import of a module the startup set already carries " +
        "loads nothing later, and splits that module's graph into extra startup chunks.",
    );
    return 1;
  }
  console.log(
    `\nwithin budget (${kb(BUDGET.transferBytes - transfer)} transfer, ${kb(BUDGET.rawBytes - raw)} raw to spare)`,
  );
  return 0;
}

/**
 * True when this file was run, rather than imported by the tests. Compared through realpath on both sides.
 * `import.meta.url` is already the real path (node resolves modules through symlinks), while `process.argv[1]`
 * is the path as typed, so a checkout reached through a symlinked directory made the two disagree and the whole
 * check became a silent no-op that exited 0.
 */
function invokedDirectly(): boolean {
  const argv = process.argv[1];
  if (!argv) {
    return false;
  }
  const here = fileURLToPath(import.meta.url);
  try {
    return realpathSync(argv) === realpathSync(here);
  } catch {
    return resolve(argv) === resolve(here);
  }
}

if (invokedDirectly()) {
  process.exit(main());
}
