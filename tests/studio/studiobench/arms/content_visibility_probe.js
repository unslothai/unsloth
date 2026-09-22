// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/* POTENCY PROBE for `content-visibility: auto` on Unsloth's chat message roots. Installed through
 * `SBENCH_EXTRA_INIT_SCRIPT`, reporting through `SBENCH_PAGE_CONSOLE`; needs `--password` (and
 * `--password-b`, which defaults to `--password` and is wrong when two Unsloth instances are
 * attached, since each mints its own bootstrap password). A run carrying this is a PROBE RUN and
 * its payload is never scored: the probe forces layout on every sample, including the rendering it
 * asks about.
 * Without it the run dies on an HTTP 401 only after the browser has started. The invocation is
 * SBENCH_EXTRA_INIT_SCRIPT=tests/studio/studiobench/arms/content_visibility_probe.js
 * SBENCH_PAGE_CONSOLE="CVPOT " python -m tests.studio.studiobench --tier fast --ab probe
 * --attach URL --attach-b URL --password ... --password-b ...; see the studiobench README.
 *
 * WHY A PROBE AT ALL. A null from an arm that never fired measures the cascade, or a selector, or
 * the fact that nothing was off screen -- a null already produced twice here. So nothing times
 * anything; it answers only whether the browser genuinely skipped rendering an off-screen subtree.
 * Three independent answers, weakest first: `cvAuto` (computed `auto`, proving only that the
 * declaration won the cascade), `skipEvents` (`contentvisibilityautostatechange` with `skipped`,
 * fired by the engine alone, so a non-zero count is proof), and `offUnrendered` (descendants with
 * no layout boxes -- the route everybody reaches for first, which DOES NOT WORK; kept only so its
 * zero is on the record. See `descendantBoxes`).
 *
 * IT ALSO WATCHES THE SIZING TRAP. Size containment while skipping means a skipped root's height is
 * its `contain-intrinsic-size`; with `auto` that is the LAST REMEMBERED SIZE (css-sizing-4 5.2),
 * and a root never rendered without containment falls back to the <length>. Both wreck scroll
 * geometry and they differ, so scrollHeight, the roots on the fallback and the roots on their
 * PADDING ALONE (a remembered size of zero) are reported per sample, read against the unarmed side.
 *
 * Reporting goes out via the console: Unsloth ships `connect-src \'self\'`, so a beacon to another
 * port is blocked by CSP before it is sent.
 */
(function () {
	"use strict";

	var PREFIX = "CVPOT ";
	var MESSAGE_SELECTOR = "[data-message-id]";
	var VIEWPORT_SELECTOR = ".aui-thread-viewport";
	var SAMPLE_MS = 2000;
	var MAX_ROOTS_SCANNED = 60;
	var MAX_DESCENDANTS_PER_ROOT = 40;
	/* Per role, the two heights a skipped root can land on, meaning OPPOSITE things: `fallback` is the
	 * declared `contain-intrinsic-size`, used because no last remembered size exists yet (working as
	 * designed); `padding` is the root's own padding alone, a remembered size of ZERO recorded while
	 * the root was mounted and empty, which is the trap. On this app they collide -- the user root's
	 * fallback is 60px against 40px of padding -- so each is matched against its OWN target height,
	 * fallback first so it wins ties, and a root that is neither is counted as neither. A single
	 * `height <= 64` bucket put a user root sitting exactly on its fallback into both counters. */
	var ROLE_PX = {
		assistant: { fallback: 300, padding: 18 },
		user: { fallback: 60, padding: 40 }
	};
	/* `getBoundingClientRect().height` is the BORDER BOX, so both targets carry the root's padding: an
	 * assistant root on its 300px fallback measures 318. Comparing against the bare declared length
	 * pinned `fallbackBite` at zero, reading as "the fallback is never used". */
	function targetHeight(px, which) {
		return (which === "fallback" ? px.fallback : 0) + px.padding;
	}
	/* Half the gap between the two smallest interesting heights, so neither test can reach the other's
	 * target; sub-pixel layout means an exact equality test would miss. */
	var PX_EPS = 2;

	if (typeof window === "undefined" || !window.document) {
		return;
	}
	if (window.__cvPotInstalled) {
		return;
	}
	window.__cvPotInstalled = true;

	var doc = window.document;
	var watched = [];
	var watchedSet = typeof WeakSet === "function" ? new WeakSet() : null;
	/* element -> its record, so a sample can ask whether THIS root is skipped. Weak so a discarded root
	 * is not held alive by the probe. */
	var recOf = typeof WeakMap === "function" ? new WeakMap() : null;
	var ev = { stateChange: 0, skip: 0, unskip: 0, watchers: 0, listenerErrors: 0 };
	var seq = 0;

	function computed(el, prop) {
		try {
			var s = window.getComputedStyle(el);
			return s ? String(s.getPropertyValue(prop) || "").trim() : "";
		} catch (e) {
			return "";
		}
	}

	function all(selector, scope) {
		try {
			var root = scope || doc;
			var list = root.querySelectorAll(selector);
			var out = [];
			for (var i = 0; i < list.length; i++) {
				out.push(list[i]);
			}
			return out;
		} catch (e) {
			return [];
		}
	}

	function rect(el) {
		try {
			return el.getBoundingClientRect();
		} catch (e) {
			return null;
		}
	}

	function intersects(a, b) {
		if (!a || !b) {
			return false;
		}
		return a.bottom > b.top && a.top < b.bottom && a.right > b.left && a.left < b.right;
	}

	/* Attached BEFORE anything is read and exactly once per element: the one signal here that no author
	 * CSS can fake. */
	function watch(el) {
		try {
			if (watchedSet) {
				if (watchedSet.has(el)) {
					return;
				}
				watchedSet.add(el);
			} else if (el.__cvPotWatched) {
				return;
			} else {
				el.__cvPotWatched = true;
			}
			var rec = { el: el, skipped: null, events: 0 };
			el.addEventListener("contentvisibilityautostatechange", function (e) {
				ev.stateChange += 1;
				rec.events += 1;
				if (e && e.skipped) {
					ev.skip += 1;
					rec.skipped = true;
				} else {
					ev.unskip += 1;
					rec.skipped = false;
				}
			});
			watched.push(rec);
			if (recOf) {
				recOf.set(el, rec);
			} else {
				el.__cvPotRec = rec;
			}
			ev.watchers += 1;
		} catch (e) {
			ev.listenerErrors += 1;
		}
	}

	/* How many of this element's first `cap` element descendants generate a layout box.
	 *
	 * A KNOWN FALSE NEGATIVE, KEPT ON PURPOSE. A skipped subtree is not laid out, so its descendants
	 * have no boxes -- but ASKING breaks it: `getClientRects()` inside a locked subtree makes Chromium
	 * render it to answer. On a 100K thread this returned 0 off-screen unrendered roots while the event
	 * counter recorded 22 roots simultaneously skipped. Use `ev_skip`. */
	function descendantBoxes(el, cap) {
		var kids;
		try {
			kids = el.querySelectorAll("*");
		} catch (e) {
			return -1;
		}
		var limit = Math.min(kids.length, cap);
		var n = 0;
		for (var i = 0; i < limit; i++) {
			try {
				if (kids[i].getClientRects().length > 0) {
					n += 1;
				}
			} catch (e2) {
				/* an element that cannot be measured is counted as having no box */
			}
		}
		return n;
	}

	/* Whether the browser last told us THIS root is skipping its contents. `null` means no transition
	 * has been seen, which is not "rendered"; callers that care check for `true` explicitly. */
	function skippedState(el) {
		var rec = null;
		try {
			rec = recOf ? recOf.get(el) : el.__cvPotRec;
		} catch (e) {
			rec = null;
		}
		return rec ? rec.skipped : null;
	}

	function roleOf(el) {
		try {
			return String(el.getAttribute("data-role") || "");
		} catch (e) {
			return "";
		}
	}

	function percentile(sorted, q) {
		if (sorted.length === 0) {
			return 0;
		}
		var i = Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * q)));
		return sorted[i];
	}

	function sample() {
		seq += 1;
		var roots = all(MESSAGE_SELECTOR);
		var i;
		for (i = 0; i < roots.length; i++) {
			watch(roots[i]);
		}

		var vp = null;
		try {
			vp = doc.querySelector(VIEWPORT_SELECTOR);
		} catch (e) {
			vp = null;
		}
		var vpRect = vp ? rect(vp) : null;

		var out = {
			seq: seq,
			origin: String(window.location.origin || ""),
			href_thread: String(window.location.hash || window.location.pathname || ""),
			messages: roots.length,
			cvAuto: 0,
			cvVisible: 0,
			armedOffscreen: 0,
			offUnrendered: 0,
			offRendered: 0,
			armedOnscreen: 0,
			onRendered: 0,
			onUnrendered: 0,
			skippedNow: 0,
			fallbackBite: 0,
			paddingOnly: 0,
			droppedDetached: 0,
			scanned: 0,
			codeBlocks: 0,
			codeBlocksAuto: 0
		};

		var heights = [];
		var scanned = 0;
		for (i = 0; i < roots.length; i++) {
			var el = roots[i];
			var cv = computed(el, "content-visibility");
			if (cv === "auto") {
				out.cvAuto += 1;
			} else if (cv === "visible") {
				out.cvVisible += 1;
			}
			if (out.cis === undefined) {
				out.cis = computed(el, "contain-intrinsic-size");
				out.cis_role = roleOf(el);
			}
			var r = rect(el);
			if (r) {
				heights.push(Math.round(r.height));
				/* Two MUTUALLY EXCLUSIVE buckets, in priority order: a skipped root's height is its padding plus
				 * its intrinsic size, so landing on the declared <length> means no remembered size exists yet
				 * and landing on the padding alone means one exists and is ZERO. The fallback is tested first
				 * and wins ties, since such a root behaves exactly as declared and must not be charged to the
				 * trap. Anything else is counted as neither. */
				/* ONLY WHILE SKIPPED: `content-visibility: auto` computes to `auto` whether or not the element
				 * is skipping, and size containment applies only while it is. An on-screen armed root has its
				 * ordinary rendered height, which must not be able to land in a role target and masquerade as
				 * the finding. */
				var px = ROLE_PX[roleOf(el)];
				if (cv === "auto" && px && skippedState(el) === true) {
					if (Math.abs(r.height - targetHeight(px, "fallback")) <= PX_EPS) {
						out.fallbackBite += 1;
					} else if (Math.abs(r.height - targetHeight(px, "padding")) <= PX_EPS) {
						out.paddingOnly += 1;
					}
				}
			}
			/* Asked only of armed roots, and only of as many as can be asked cheaply: this forces layout, and
			 * the probe must not become the load. */
			if (cv === "auto" && r && vpRect && scanned < MAX_ROOTS_SCANNED) {
				scanned += 1;
				var hasOwnBox = r.width > 0 || r.height > 0;
				var boxes = descendantBoxes(el, MAX_DESCENDANTS_PER_ROOT);
				var off = !intersects(r, vpRect);
				if (off) {
					out.armedOffscreen += 1;
					if (hasOwnBox && boxes === 0) {
						out.offUnrendered += 1;
					} else if (boxes > 0) {
						out.offRendered += 1;
					}
				} else {
					out.armedOnscreen += 1;
					if (boxes > 0) {
						out.onRendered += 1;
					} else {
						out.onUnrendered += 1;
					}
				}
			}
		}
		out.scanned = scanned;

		var blocks = all('[data-streamdown="code-block"]');
		out.codeBlocks = blocks.length;
		for (i = 0; i < blocks.length; i++) {
			if (computed(blocks[i], "content-visibility") === "auto") {
				out.codeBlocksAuto += 1;
			}
		}

		/* CONNECTED roots only, dropping the detached ones as we go. `thread_reopen` rebuilds the thread,
		 * so old roots leave the document and receive no further transitions; one whose last event said
		 * `skipped` would be counted in `skippedNow` for the rest of the session. Pruning also stops
		 * `watched` growing a strong reference per root. */
		var live = [];
		for (i = 0; i < watched.length; i++) {
			var wel = watched[i].el;
			var connected = false;
			try {
				connected = wel.isConnected !== false && doc.contains(wel);
			} catch (e) {
				connected = false;
			}
			if (!connected) {
				out.droppedDetached += 1;
				continue;
			}
			live.push(watched[i]);
			if (watched[i].skipped === true) {
				out.skippedNow += 1;
			}
		}
		watched = live;

		heights.sort(function (a, b) {
			return a - b;
		});
		out.h_min = heights.length ? heights[0] : 0;
		out.h_p50 = percentile(heights, 0.5);
		out.h_max = heights.length ? heights[heights.length - 1] : 0;
		out.h_list = heights.slice(0, 24);
		out.h_sum = 0;
		for (i = 0; i < heights.length; i++) {
			out.h_sum += heights[i];
		}

		if (vp) {
			try {
				out.vp_scrollHeight = vp.scrollHeight;
				out.vp_clientHeight = vp.clientHeight;
				out.vp_scrollTop = Math.round(vp.scrollTop);
			} catch (e) {
				/* leave the keys absent rather than reporting a zero that looks like a collapse */
			}
		}

		out.ev_stateChange = ev.stateChange;
		out.ev_skip = ev.skip;
		out.ev_unskip = ev.unskip;
		out.ev_watchers = ev.watchers;
		out.ev_listenerErrors = ev.listenerErrors;

		try {
			window.console.log(PREFIX + JSON.stringify(out));
		} catch (e) {
			/* nothing to do: the console is the only way out */
		}
	}

	window.__cvPotSample = sample;

	/* THE LISTENER HAS TO EXIST BEFORE THE ELEMENT'S FIRST TRANSITION, and polling cannot guarantee
	 * that. `contentvisibilityautostatechange` fires on a CHANGE: a root inserted off screen becomes
	 * skipped once and never changes again while the user stays put, so on a two-second tick every
	 * root mounted inside the first tick emits its only event into a void and `ev_skip` stays zero --
	 * the false NOT RUN this file exists to prevent. A whole thread can mount inside two seconds.
	 *
	 * So roots are adopted at INSERTION, from a MutationObserver installed before the app boots; its
	 * callback runs in a microtask before the next lifecycle update. The interval is kept because a
	 * root can acquire the property later without being re-inserted, and `adoptAll` is idempotent.
	 *
	 * OBSERVE THE DOCUMENT, NOT `documentElement`: an init script runs after the Document exists but
	 * possibly before the parser creates the root element, and `observe(null, ...)` would throw into
	 * the catch below, silently falling back to the tick -- the same false zero, reintroduced by the
	 * fix for it. */
	function adoptAll() {
		var roots = all(MESSAGE_SELECTOR);
		for (var i = 0; i < roots.length; i++) {
			watch(roots[i]);
		}
	}

	/* Only the ADDED NODES, never a document-wide re-scan: streaming produces thousands of mutations a
	 * second and a `querySelectorAll` per mutation would make this probe the load. `watch` is
	 * idempotent, so a node reached twice costs a WeakSet lookup. */
	function adoptAdded(records) {
		for (var i = 0; i < records.length; i++) {
			var added = records[i].addedNodes;
			for (var j = 0; j < (added ? added.length : 0); j++) {
				var node = added[j];
				if (!node || node.nodeType !== 1) {
					continue;
				}
				try {
					if (typeof node.matches === "function" && node.matches(MESSAGE_SELECTOR)) {
						watch(node);
					}
				} catch (e) {
					ev.listenerErrors += 1;
				}
				var inner = all(MESSAGE_SELECTOR, node);
				for (var k = 0; k < inner.length; k++) {
					watch(inner[k]);
				}
			}
		}
	}

	try {
		if (typeof window.MutationObserver === "function") {
			new window.MutationObserver(adoptAdded).observe(doc, {
				childList: true,
				subtree: true
			});
		} else {
			ev.listenerErrors += 1;
		}
	} catch (e) {
		ev.listenerErrors += 1;
	}

	adoptAll();
	try {
		window.setInterval(sample, SAMPLE_MS);
	} catch (e) {
		/* a probe that cannot schedule itself reports nothing, which is the honest failure */
	}
})();
