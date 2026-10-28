// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Budget for JavaScript fetched and run before the first screen: Vite's entry plus modulepreload
 * links (static import closure) plus parser-blocking classic scripts such as public/theme-boot.js.
 */

import { readFileSync, realpathSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";

const HERE = dirname(fileURLToPath(import.meta.url));
const DIST = resolve(HERE, "..", "dist");

/** Measured plus headroom. Transfer = wire bytes; raw = bytes the main thread parses. */
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
  // A setting to hide the macOS menu bar icon (#11937) adds 349 B raw / 78 B transfer (6,842,183 -> 6,842,532 B raw,
  // 2,060,321 -> 2,060,399 B transfer against 46e4f50bea and 2b20770c91, one machine and build, Node 24, 96 chunks):
  // its strings join the eager English catalog and settings search index. #14223 and #14232 add no raw bytes.
  // Raised by that growth, to the measured build at 6ccf0e4e74 (6,842,532 B raw / 2,060,400 B transfer) plus a
  // 160 B raw / 189 B transfer margin.
  // The Loaded models card's eject confirmation (#14161, eager shell) adds 356 B raw / 158 B transfer over 28a222f20
  // (6,842,874 B raw, 2,060,540 B transfer); limits = that build plus the same margin.
  // Naming a chat export after its model, title and day (#12586) adds 615 B raw / 65 B transfer (6,842,183 ->
  // 6,842,798 B raw, 2,060,321 -> 2,060,386 B transfer against 46e4f50bea, one machine and build, Node 24,
  // 96 chunks): the name builder sits beside the chat exporters in the eager chat chunk. Raised by that growth.
  // Opening Parallel Slots from the API page's Slots readout (#14225) adds 546 B raw / 212 B transfer (6,842,518 ->
  // 6,843,064 B raw, 2,060,427 -> 2,060,639 B transfer against bff0a25b4, one machine and build, Node 24, 96 chunks):
  // the field focus and per-visit Advanced reveal live in the Run settings page and the picker, both in the eager chat
  // chunk. Raised to the measured build plus the same 160 B raw / 189 B transfer margin.
  // Reconciled with main at 87c6408455: the combined build measures 6,844,101 B raw / 2,060,934 B transfer
  // (Node 24, 96 chunks). The API Settings button stays in its lazy page. Keep the same 160 B / 189 B margin.
  // FLUX.3 workflows, controls and LoRAs (#14312) add 426 B raw / 138 B transfer (6,844,101 -> 6,844,527 B raw,
  // 2,060,887 -> 2,061,025 B transfer against 6a19cb20b9, one machine and build, Node 24, 96 chunks), and #14228
  // adds 95 B raw / 15 B transfer on top, measured at d3a82576a4 as 6,844,622 B raw / 2,061,040 B transfer. Raised
  // to that build plus the same 160 B raw / 189 B transfer margin.
  // The modern toggles (#14322) add 449 B raw / 86 B transfer (6,844,459 -> 6,844,908 B raw, 2,061,006 ->
  // 2,061,092 B transfer against ad2fc0659c, one machine and build, Node 24, 96 chunks): switch.tsx is a shared UI
  // primitive the startup set already carries. #14329 removed 163 B raw just before. Raised to that build plus the
  // same 160 B raw / 189 B transfer margin.
  // The Desktop chat-only install choice (#14245) adds 679 B raw / 205 B transfer (6,842,169 -> 6,842,848 B raw,
  // 2,060,319 -> 2,060,524 B transfer against 26e246d407, one machine and build, Node 24, 96 chunks): the
  // first-run checkbox lives in the eager startup screen and the Train hint in the eager sidebar. Raised by that.
  // Explicit browser download names retain the prior startup margin.
  // The About tab's iGPU memory row (#14317): merged with main at ca0252968 the build measures 6,848,044 B raw /
  // 2,062,332 B transfer (Node 24, 96 chunks). Raised to that plus the same 160 B raw / 189 B transfer margin.
  // Token usage tracking (#14076) keeps the saved-count matching, recount scheduling and their review guards in the
  // eager chat runtime: merged with main at 4e0fe5b99 the build measures 6,857,322 B raw / 2,064,700 B transfer
  // (Node 24, 96 chunks). Raised to that plus the same 160 B raw / 189 B transfer margin.
  // The Allow oversized model loads row in Settings (#14379) adds its switch and strings to the eager settings
  // chunk and locale: 1,886 B raw / 628 B transfer (6,857,308 -> 6,859,194 B raw, 2,064,700 -> 2,065,328 B
  // transfer against f1f283a24e, one machine and build, Node 24, 96 chunks). Raised by that, same margin.
  // The Rosetta chat-only messages (#14374) in the eager sidebar and hardware verdict add 298 B raw / 127 B
  // transfer (6,859,194 -> 6,859,492 B raw, 2,065,328 -> 2,065,455 B transfer against 34c5203ebb, one machine and
  // build, Node 24, 96 chunks). Raised to that build plus the same 160 B raw / 189 B transfer margin.
  // The retrieved passage viewer (#13987) in the eager chat UI adds 3,132 B raw / 939 B transfer (6,859,492 ->
  // 6,862,624 B raw, 2,065,455 -> 2,066,394 B transfer against 98010f6eee, one machine and build, 96 chunks).
  // Raised to that build plus the same 160 B raw / 189 B transfer margin.
  // Chat safety for diagrams, redirect links and approval cards adds 617 B raw / 238 B transfer (6,862,624 ->
  // 6,863,241 B raw, 2,066,394 -> 2,066,632 B transfer against 731cb16c2, one machine and build, the diagram
  // source check itself loaded with the first diagram). Raised to that build plus the same margin.
  // The sidebar Projects menu's Projects home item and its strings add 240 B raw / no transfer (6,862,692 ->
  // 6,862,932 B raw, 2,066,441 -> 2,066,440 B transfer against 2aeeceb07e, one machine and build, 96 chunks).
  // Merged with main at 2552ee3895 the build measures 6,863,532 B raw / 2,066,692 B transfer (Node 26, 97 chunks).
  // Raised to that build plus the same 160 B raw / 189 B transfer margin.
  // Slider tracks round to whole screen pixels at any zoom: theme-boot.js publishes --device-px before first
  // paint, 1,501 -> 1,847 B (served uncompressed, so 346 B raw and transfer). Raised by that, same margin.
  // Main at dcf5c4009 measures 6,870,312 B raw / 2,068,954 B transfer, already over the line above. Browser panel
  // page downloads add 1,468 B raw / 354 B transfer on top (6,871,780 B raw, 2,069,308 B transfer, one machine and
  // build, Node 24, 97 chunks). Raised to that build plus the same 160 B raw / 189 B transfer margin.
  // Browser store sync across windows (#14277) adds 545 B raw / 288 B transfer on top (6,872,325 B raw, 2,069,596 B
  // transfer merged with main at 4962983b0, Node 24, 97 chunks). Raised to that plus the same margin.
  // The runtime update card's skip and Undo (#14436) add 1,706 B raw / 424 B transfer against 444f63275c.
  // Finished chat replies render lone markdown markers as written, cut ones keep the repair (#14391): +179 B raw /
  // +52 B transfer measured against main (6,870,298 -> 6,870,477 B raw, one machine and build). Raised by that.
  // Picking a preset forgetting a model's remembered sampling (#14128) keeps its preset marks in the eager chat
  // runtime store, about 1,860 B raw / 440 B transfer: merged with main at acb31d909b the build measures
  // 6,876,062 B raw / 2,070,545 B transfer (Node 24, 97 chunks). Raised to that plus the same margin.
  // Sharing a connection with every account adds the shared flags to the eager provider mapping: 180 B raw /
  // 20 B transfer (6,862,624 -> 6,862,804 B raw, 2,066,394 -> 2,066,414 B transfer against 723a43b02, one
  // machine and build, Node 22, 96 chunks); the share switch itself is in the lazy Connections tab. Raised by that.
  // The browser panel's page history (#14283) on top measures 6,876,588 B raw / 2,070,700 B transfer (merged with
  // main at 04d2beec9, Node 24, 97 chunks). Raised to that plus the same margin.
  // The MCP app tool arguments on the approval card (#14458) in the eager chat UI add 341 B raw / 110 B transfer
  // (6,862,706 -> 6,863,047 B raw, 2,066,422 -> 2,066,532 B transfer against 2aeeceb07e, one machine and build,
  // 96 chunks). Raised by that.
  // Floating panels remembering their dropped spot (#14355) add 701 B raw / 211 B transfer. Raised by that.
  // The Settings > Chat memory editor (#14357) keeps its strings in the eager English locale and the AGENTS.md
  // cache counter in the chat adapter (the section itself is lazy): +989 B raw / +364 B transfer over main
  // (6,863,532 -> 6,864,521 B raw, 2,066,692 -> 2,067,056 B transfer at e08466e82, 97 chunks). Raised by that.
  // REAP on the Export page (#14130) passes its options through the eager export runtime store (the card is
  // in the lazy export page): +371 B raw / +119 B transfer (6,878,631 -> 6,879,002 B raw, 2,071,507 ->
  // 2,071,626 B transfer against 4df9ce74f, one machine and build, Node 24, 97 chunks). Raised by that.
  // The NPU setup card (#14524): +439 B raw / +122 B transfer against fc91656ff (Node 24, 97 chunks), where main
  // was already 1 B transfer over; transfer set to the measured build, raw raised by the growth.
  // #14524 set transfer to its own measured build with no headroom; merged next to #14508 the startup set
  // compresses 7 B larger (2_071_823 B at 5ab80ffdb4, Node 24), so main is over. Measured + 189 B, raw unchanged
  // (already 244 B above the measured 6_879_345).
  // Chat fork affordances (the sidebar fork mark, the "Forked from" divider and the shared fork toasts) in the
  // eager chat chunk: +1_460 B raw / +361 B transfer against 77e742c8e2 (6_879_345 -> 6_880_805 raw,
  // 2_071_823 -> 2_072_184 transfer, 97 chunks). Raised by that growth.
  // Projects recency and the polling cuts behind projects and sources (#14586) live in the eager chat chunk:
  // +505 B raw / +309 B transfer (6,879,331 -> 6,879,836 B raw, 2,071,848 -> 2,072,157 B transfer against
  // 2749eeedaa, one machine and build, Node 26, 97 chunks). Raised by that, on top of the fork affordances above.
  // The shared custom folder store and its Undo / already-added toasts in the eager model picker: +557 B raw /
  // +627 B transfer against 2749eeedaa (6_879_345 -> 6_879_902 raw, 2_071_823 -> 2_072_450 transfer, 97
  // chunks); the folders dialog itself stays lazy. Raised by that, on top of the above.
  // Shortcut fixes (#14597): stop action, isModEnter, zoom guard and strings in the eager locales:
  // +3,339 B raw / +1,727 B transfer (6,881,936 -> 6,885,275 B raw,
  // 2,073,188 -> 2,074,915 B transfer against b2ba86a514, Node 26, 98 chunks). Raised by that.
  // Run settings save state and Advanced groups (#14589) merged with main at 8bcbf837bd: 6,725.4 KB raw /
  // 2,026.7 KB transfer (Node 24, 98 chunks). Raised to that plus about 200 B.
  // #14590 (eager locale strings, help entry, API buttons, update card close) merged with main at 356721b31e:
  // 6,729.6 KB raw / 2,027.3 KB transfer (Node 24, 98 chunks). Raised to that plus about 200 B.
  // Passive Settings update check: +24 B raw / +43 B transfer against e24fa71e1b
  // (6,891,238 -> 6,891,262 raw, 2,076,088 -> 2,076,131 transfer, Node 26). Raised by that.
  // Local addresses in the desktop browser panel (#14290): the grant, per-tab bookkeeping and agent check live in
  // the eager browser store, executor and panel: +1,808 B raw / +551 B transfer (6,891,248 -> 6,893,056 B raw,
  // 2,076,155 -> 2,076,706 B transfer against c2f75c486, one machine and build). Raised to that plus the same margin.
  // Opt-in reply times (#14500: the Settings > Chat row, its eager English strings and the reply footer gate):
  // +1,514 B raw / +418 B transfer over main (6,891,262 -> 6,892,776 B raw, 2,076,131 -> 2,076,549 B transfer
  // merged at c2f75c4, Node 24, 98 chunks). Raised by that, on top of the above.
  // #14592 + #14601 (system polling, floating panels) merged with main at 0c001fceb: 6,907,598 B raw /
  // 2,080,708 B transfer (Node 26, 98 chunks). Raised to that plus about 200 B.
  // Settings API / Remote / LAN strings (#14596) merged with main at 870f37741: 6,907,886 B raw / 2,080,818 B
  // transfer (Node 26, 98 chunks). Raised to that plus about 200 B.
  // Apple unified-memory fit hint (#14598) merged with main at a29487c52: 6,909,320 B raw / 2,081,284 B
  // transfer (Node 26, 98 chunks). Raised to that plus about 200 B.
  // Desktop Chat only upgrade (#14593), incl. the Train row opening the full install dialog, merged with main
  // at c5a9fe021: 6,912,687 B raw / 2,082,221 B transfer (Node 24, 97 chunks). Raised to that plus about 200 B.
  // The MLX PLE and expert offload controls (#14326): +7,406 B raw / +2,073 B transfer, plus about 500 B raw
  // (a CI build measured about 100 B over a local one). Raised by that.
  // Install Training button under the greyed Train row's hint: +424 B raw / +141 B transfer against ee55c3c9a
  // (6,920,512 -> 6,920,936 B raw, 2,084,382 -> 2,084,523 B transfer, Node 24, 97 chunks). Raised by that.
  // MOSS-Image (#14552): its two image catalog groups, family label and generation defaults sit in the eager
  // catalog the chat chunk imports: +1,298 B raw / +181 B transfer (6,919,946 -> 6,921,244 B raw, 2,084,333 ->
  // 2,084,514 B transfer against 4e2838ec5f, 2026-10-22, one machine and build, Node 24, chunk count unchanged
  // at 97). Raised by that growth so main's margin stays where it was.
  // A chat going back to its model when it is still loaded (#14568): the eager chat model notice and the
  // adopt-only pick in the chat runtime, +1,970 B raw / +627 B transfer (6,922,858 -> 6,924,828 B raw,
  // 2,085,257 -> 2,085,884 B transfer against 1f5576747, Node 24, 98 chunks), where main was already 315 B raw /
  // 239 B transfer over. Set to the measured build plus about 300 B raw / 100 B transfer (CI builds measure
  // about 100 B over a local one).
  // Hub metadata caching and hidden-row filtering (#14265): +548 B raw / +146 B transfer (6,925,639 ->
  // 6,926,187 B raw, 2,086,290 -> 2,086,436 B transfer against 0ee5cfc58, Node 24, 98 chunks), where main
  // was already over the line above. Set to the merged build plus about 200 B.
  // macOS Desktop stray key text guard (#14815), eager in main.tsx: +278 B raw / +114 B transfer
  // (6,925,639 -> 6,925,917 B raw, 2,086,290 -> 2,086,404 B transfer against 14e0178e0, Node 24, chunk count
  // unchanged at 98). Raised by +278 B raw / +129 B transfer so main's margin stays where it was.
  // Browser use step names, docs link and card fence strip (#14681): +1,089 B raw / +298 B transfer (6,925,028 ->
  // 6,926,117 B raw, 2,086,327 -> 2,086,625 B transfer against 394aff8545, Node 26, 99 chunks). Raised by that.
  // The waiting-for-approval mark, Desktop attention request and tool_approved handling (#14877): measured
  // 6,929,942 B raw / 2,087,788 B transfer merged with main at b3004af53 (Node 24, 99 chunks). Set to
  // that plus about 300 B raw / 100 B transfer, as for #14568.
  // Skill ZIP import/export (#14181) adds 2,721 B raw / 758 B transfer to the eager Skills dialog
  // and locale strings (6,929,934 -> 6,932,655 B raw, 2,087,782 -> 2,088,540 B transfer against
  // fb104d2b57, one machine and build, Node 22.23.2, unchanged at 99 chunks). Raised by that growth.
  // Language and platform tabs on the API page examples (#15084) add 138 B raw / 53 B transfer: Vite's preload map
  // in the entry and settings chunks naming the example box, now a lazy chunk shared by the API page and Settings
  // (6,932,829 -> 6,932,967 B raw, 2,088,616 -> 2,088,669 B transfer against cfc593f0bd, Node 26, 99 chunks).
  // Per-account chat preferences (#15025): +338 B raw / +93 B transfer over the line above (Node 22.23.2, 99 chunks).
  // Live prefill progress, the Generation details setting and the loaded placement readout (#14582): the
  // stream frame reader, live store slice, reply footer line and timing tooltip rows in the eager chat
  // chunk, +8,585 B raw / +2,268 B transfer over main (6,932,829 -> 6,941,414 B raw, 2,088,586 ->
  // 2,090,854 B transfer against 1492952a09, Node 22, 99 chunks). Raised by that on top of #15084 and #15025.
  // The pinned Run settings footer adds 25 B raw / 49 B transfer to the eager model picker (6,932,829 -> 6,932,854 B
  // raw, 2,088,616 -> 2,088,665 B transfer against 2c2e04474b, Node 26, unchanged at 99 chunks). Raised by that.
  // Storage overview in Library and System settings (#15083) adds 360 B raw / 56 B transfer, its strings in the
  // eager en locale plus 46 B in the settings chunk; the overview itself is a lazy chunk (6,932,829 -> 6,933,189 B
  // raw, 2,088,616 -> 2,088,672 B transfer against cfc593f0bd, Node 26, unchanged at 99 chunks). Raised by that.
  // Its "System and Engine" page title adds 11 B raw / 34 B transfer to the eager en locale (6,933,189 -> 6,933,200 B
  // raw, 2,088,699 -> 2,088,733 B transfer merged with f49cbdf535, Node 26, 99 chunks). Raised by that.
  // Scheduled tasks (#13871) add 15.9 KB raw / 5.0 KB transfer (results poll, unseen dots, strings; page and
  // dialog lazy): 6,766.5 -> 6,782.4 KB raw, 2,041.5 -> 2,046.5 KB transfer against main 90164297d, one machine
  // and build, Node 24, 105 chunks. Raised by that growth over main's budget.
  // The control heights and API page round (#15232) adds 2,639 B raw / 717 B transfer, mostly its token table,
  // regenerate and tooltip strings in the eager locales (6,945,181 -> 6,947,820 B raw, 2,095,616 -> 2,096,333 B
  // transfer against 6eeb690f8d, one machine and build, 105 chunks). Raised by that growth.
  // The run settings dropdown per model (#15247: saved settings store, dropdown, pencil and Reset to Default)
  // adds 9.9 KB raw / 3.2 KB transfer inside the eager model config page: 6,787.4 -> 6,797.3 KB raw, 2,047.9 ->
  // 2,051.1 KB transfer against d4be247222, one machine and build, Node 26, 105 chunks. Raised by that growth.
  // Settings and the row menu on the loaded model in Recommended (#15249) add 82 B raw / 176 B transfer to the eager
  // model picker (6,960,469 -> 6,960,551 B raw, 2,100,277 -> 2,100,453 B transfer against c9ddc5ab5e, one machine and
  // build, Node 26, 105 chunks). Raised by that growth.
  // Cosmos3 video families (#14554): +466 B raw / +218 B transfer. Raised by that.
  // Lookup-table pricing (#14465): +1,517 B raw / +470 B transfer on top of the lines above (measured merged with
  // 4db3eaeb55, Node 24, 99 chunks).
  // Chat pins synced through the server (#15293): the sync runtime is a lazy chunk, but the modules it shares
  // with startup code (chat-settings-api, sidebar-organization-keys) split into their own eager chunks, plus the
  // mount hook and store wiring: +3,074 B raw / +2,025 B transfer (6,958,762 -> 6,961,836 B raw, 2,099,260 ->
  // 2,101,285 B transfer against 9c397197e0, Node 24, 106 -> 109 chunks). Raised by that growth.
  // The per-tool MCP switches at the end of Settings > Agents (#14348) add 3,593 B raw / 1,086 B transfer
  // (6,972,239 -> 6,975,832 B raw, 2,102,226 -> 2,103,312 B transfer against a5956882f6, one machine and build,
  // Node 24, 103 chunks both): the Agents panel and its tools dialog are lazy, so the growth is their strings in the
  // eager English catalog plus the agents-mcp scroll target. Raised by that.
  // The chat's Unsloth tools preset, its Desktop media links and the tools dialog's chat note (#14349) add
  // 1,363 B raw / 366 B transfer on top (6,975,832 -> 6,977,195 B raw, 2,103,312 -> 2,103,678 B transfer against
  // 8eeb3df91b, one machine and build, Node 24, 103 chunks both), in the eager chat chunk and English catalog.
  // Raised by that.
  transferBytes: 2_104_662,
  rawBytes: 6_981_585,
};

// Chunk count is not budgeted: splitting a page out raises it while lowering bytes.

/** Lowercased tag name and attributes; a valueless attribute has value "". */
type StartTag = { name: string; attrs: Map<string, string> };

/** ASCII whitespace; CR is defensive since the HTML preprocessor turns CR into LF. */
const WHITESPACE = new Set(["\t", "\n", "\f", "\r", " "]);

function skipWhitespace(html: string, from: number): number {
  let i = from;
  while (i < html.length && WHITESPACE.has(html[i] as string)) {
    i += 1;
  }
  return i;
}

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

function readValue(
  html: string,
  from: number,
): { value: string; next: number } {
  const i = skipWhitespace(html, from + 1);
  const quote = html[i];
  if (quote !== '"' && quote !== "'") {
    const end = scanUntil(html, i, ">");
    return { value: html.slice(i, end), next: end };
  }
  const close = html.indexOf(quote, i + 1);
  return close < 0
    ? { value: html.slice(i + 1), next: html.length }
    : { value: html.slice(i + 1, close), next: close + 1 };
}

/**
 * Tokenizer attribute states: `>` inside a quoted value does not end the tag, and only an
 * attribute NAME can be `async`/`type`. Regex over tag text got both wrong.
 */
function readAttributes(
  html: string,
  start: number,
): { attrs: Map<string, string>; end: number; closed: boolean } {
  const attrs = new Map<string, string>();
  let i = start;
  while (i < html.length) {
    if (WHITESPACE.has(html[i] as string) || html[i] === "/") {
      i += 1;
      continue;
    }
    if (html[i] === ">") {
      return { attrs, end: i + 1, closed: true };
    }
    const nameEnd = scanUntil(html, i, "/>=");
    const name = html.slice(i, nameEnd).toLowerCase();
    i = skipWhitespace(html, nameEnd);
    let value = "";
    if (html[i] === "=") {
      ({ value, next: i } = readValue(html, i));
    }
    // Duplicate attributes: the browser keeps the first.
    if (name && !attrs.has(name)) {
      attrs.set(name, value);
    }
  }
  return { attrs, end: i, closed: false };
}

/** Comment end; `<!-->` and `<!--->` close immediately per the comment start states. */
const COMMENT_END = /^-?>|--!?>/;
function endOfComment(html: string, start: number): number {
  const m = COMMENT_END.exec(html.slice(start));
  return m ? start + m.index + m[0].length : html.length;
}

const SCRIPT_END = /<\/script[\t\n\f\r >/]/i;
function endOfScriptBody(html: string, start: number): number {
  const m = SCRIPT_END.exec(html.slice(start));
  return m ? start + m.index : html.length;
}

const TAG_NAME_START = /[a-z]/i;

/**
 * Every start tag in order, skipping comments and script bodies as the browser does.
 * Escaped `<!-- <script` bodies may over-count a chunk, never miss one (checked vs parse5).
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
    const nameEnd = scanUntil(html, lt + 1, "/>");
    const name = html.slice(lt + 1, nameEnd).toLowerCase();
    const { attrs, end, closed } = readAttributes(html, nameEnd);
    if (!closed) {
      return;
    }
    yield { name, attrs };
    i = name === "script" ? endOfScriptBody(html, end) : end;
  }
}

function attr(tag: StartTag, name: string): string | undefined {
  return tag.attrs.get(name);
}

function hasAttr(tag: StartTag, name: string): boolean {
  return tag.attrs.has(name);
}

function relTokens(tag: StartTag): string[] {
  return (attr(tag, "rel") ?? "").toLowerCase().split(/\s+/).filter(Boolean);
}

/**
 * blocking="render" delays first render for any external script, async included (HTML spec).
 */
function blocksRender(tag: StartTag): boolean {
  return (attr(tag, "blocking") ?? "")
    .toLowerCase()
    .split(/\s+/)
    .includes("render");
}

/**
 * Eager set relative to dist/. entry and preloads stay separate so missing preload links are
 * detectable; `blocking` holds parser-blocking classic scripts like public/theme-boot.js.
 */
export type EagerSet = {
  entry: string[];
  preloads: string[];
  blocking: string[];
};

/**
 * Types the browser runs as classic script: the full frozen JS MIME essence list, exact strings
 * (a parameter like `; charset=utf-8` makes it not run). mimesniff.spec.whatwg.org
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

function distRelative(url: string | undefined): string | undefined {
  if (!url?.startsWith("/") || url.startsWith("//")) {
    return undefined;
  }
  const path = url.slice(1).split(/[?#]/)[0];
  return path && !path.split("/").includes("..") ? path : undefined;
}

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
      add(set.entry, attr(tag, "src"), "assets/");
    } else if (!type || CLASSIC_TYPES.has(type)) {
      // `defer` counts (runs before DOMContentLoaded); `async` does not, unless blocking="render".
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

/** Only the `/assets` mount is gzipped by the backend; elsewhere raw size is transfer size. */
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
      // Report it: concluding the budget is fine must never be reachable here.
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

  // Shape guard: one or no Vite chunks means the build shape changed and the number is fiction.
  // Require an entry too: preloads without an entry means the entry was mis-parsed.
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

/** Run directly vs imported. Compare realpaths: argv[1] keeps symlinks, import.meta.url does not. */
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
