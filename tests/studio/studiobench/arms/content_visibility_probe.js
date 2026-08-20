// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/* POTENCY PROBE for `content-visibility: auto` on Studio's chat message roots.
 *
 * Installed through `SBENCH_EXTRA_INIT_SCRIPT`, reporting through `SBENCH_PAGE_CONSOLE`:
 *
 *     SBENCH_EXTRA_INIT_SCRIPT=tests/studio/studiobench/arms/content_visibility_probe.js \
 *     SBENCH_PAGE_CONSOLE="CVPOT " \
 *     python -m tests.studio.studiobench --tier fast --ab probe --out outputs/probe ...
 *
 * A run carrying this is a PROBE RUN and its payload is never scored: the probe forces layout on
 * every sample, and one of the things it forces is the very rendering it is asking about. See the
 * warning on `descendantBoxes`.
 *
 * WHY A PROBE AT ALL, AND WHY IT RUNS BEFORE ANY TIMING IS COLLECTED.
 *
 * A null from an arm that never fired is not a measurement of the mechanism; it is a measurement
 * of the cascade, or of a selector, or of the fact that nothing was ever off screen. This exact
 * null has already been produced twice in this codebase for `content-visibility`. So nothing here
 * times anything. It answers one question: did the browser genuinely skip rendering an off-screen
 * subtree, on the elements this arm targets, in the running app.
 *
 * THREE INDEPENDENT ANSWERS, WEAKEST FIRST.
 *
 *   1. cvAuto            elements whose COMPUTED content-visibility is `auto`. This proves the
 *                        declaration won the cascade and nothing more. An element can compute to
 *                        `auto` and be painted on every single frame because it never stopped
 *                        being relevant to the user.
 *   2. skipEvents        `contentvisibilityautostatechange` events with `skipped === true`. This
 *                        event is fired by the engine's own relevance machinery and by nothing
 *                        else: no stylesheet, no selector and no author code can produce one. A
 *                        non-zero count is proof that a subtree was locked and its rendering
 *                        skipped.
 *   3. offUnrendered     off-screen armed roots whose element descendants generate NO layout
 *                        boxes. This is the route everybody reaches for first and it DOES NOT
 *                        WORK for `content-visibility: auto`. It is kept, and reported, only so
 *                        that its zero is on the record next to a non-zero skip count rather than
 *                        being rediscovered. See `descendantBoxes`.
 *
 * IT ALSO WATCHES THE SIZING TRAP. `content-visibility: auto` imposes size containment while
 * skipping, so a skipped root's height is its `contain-intrinsic-size`. With the `auto` keyword
 * that is the LAST REMEMBERED SIZE (css-sizing-4 5.2, 5.2.1), and an element that has never been
 * rendered without size containment has none and falls back to the <length>. Both failure modes
 * wreck scroll geometry on a long thread and they are different, so the viewport's scrollHeight,
 * the count of roots sitting on the declared fallback, and the count sitting on their PADDING
 * ALONE (a remembered size of zero) are all reported every sample, to be read against the
 * unarmed side of the same session.
 *
 * OUT OF THE PAGE VIA THE CONSOLE. Studio ships `connect-src 'self'`, so a beacon to a collector
 * on another port is blocked by CSP before it is sent. The console is the one channel that costs
 * nothing and cannot be silently dropped.
 */
(function () {
	"use strict";

	var PREFIX = "CVPOT ";
	var MESSAGE_SELECTOR = "[data-message-id]";
	var VIEWPORT_SELECTOR = ".aui-thread-viewport";
	var SAMPLE_MS = 2000;
	var MAX_ROOTS_SCANNED = 60;
	var MAX_DESCENDANTS_PER_ROOT = 40;
	/* The fallback lengths this arm ships. A skipped root whose height lands on one of these to
	 * the pixel has NO last remembered size, which is the trap, not the design. */
	var FALLBACK_PX = { assistant: 300, user: 60 };

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

	/* Attached BEFORE anything is read, and exactly once per element. The event is the only
	 * signal here that no amount of author CSS can fake. */
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
			ev.watchers += 1;
		} catch (e) {
			ev.listenerErrors += 1;
		}
	}

	/* How many of this element's first `cap` element descendants generate a layout box.
	 *
	 * A KNOWN FALSE NEGATIVE, KEPT ON PURPOSE. The reasoning is that a skipped subtree is not laid
	 * out, so its descendants have no boxes. The reasoning is correct and the measurement is still
	 * useless, because ASKING is what breaks it: `getClientRects()` on content inside a locked
	 * subtree makes Chromium render that subtree in order to answer, so the probe unlocks exactly
	 * what it came to observe. Measured on a 100K thread this returned 0 off-screen unrendered
	 * roots while the event counter recorded 22 roots simultaneously in the skipped state. Read
	 * alone it is a clean, confident, wrong "the arm did not fire". Use `ev_skip`. */
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
				/* The padding-only signature. `content-visibility: auto` imposes size containment
				 * while skipping, so a skipped root's height is its padding plus its intrinsic
				 * size. A root that lands on its padding alone is reporting an intrinsic size of
				 * ZERO, which means the `auto` keyword found a last remembered size of zero --
				 * recorded while the message root existed but had not yet been filled -- rather
				 * than falling back to the declared <length>. That is a different failure from
				 * "the fallback is too small" and it has to be counted separately. */
				if (cv === "auto" && r.height > 0 && r.height <= 64) {
					out.paddingOnly = (out.paddingOnly || 0) + 1;
				}
				var fb = FALLBACK_PX[roleOf(el)];
				if (cv === "auto" && fb && Math.abs(r.height - fb) < 1) {
					out.fallbackBite += 1;
				}
			}
			/* The geometry question is asked only of armed roots, and only of as many as can be
			 * asked cheaply: this forces layout, and the probe must not become the load. */
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

		for (i = 0; i < watched.length; i++) {
			if (watched[i].skipped === true) {
				out.skippedNow += 1;
			}
		}

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
	try {
		window.setInterval(sample, SAMPLE_MS);
	} catch (e) {
		/* a probe that cannot schedule itself reports nothing, which is the honest failure */
	}
})();
