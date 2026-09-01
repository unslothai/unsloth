// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/* POTENCY PROBE for `content-visibility: auto` on Unsloth's chat message roots.
 *
 * Installed through `SBENCH_EXTRA_INIT_SCRIPT`, reporting through `SBENCH_PAGE_CONSOLE`:
 *
 *     SBENCH_EXTRA_INIT_SCRIPT=tests/studio/studiobench/arms/content_visibility_probe.js \
 *     SBENCH_PAGE_CONSOLE="CVPOT " \
 *     python -m tests.studio.studiobench --tier fast --ab probe --out outputs/probe \
 *         --attach http://127.0.0.1:PORT --attach-b http://127.0.0.1:OTHER \
 *         --password "$(cat "$STUDIO_HOME_A/auth/.bootstrap_password")" \
 *         --password-b "$(cat "$STUDIO_HOME_B/auth/.bootstrap_password")"
 *
 * The credential flags are spelled out rather than trailed off with an ellipsis. A probe run
 * drives a real Unsloth like every other command in the loop, and without `--password` it dies on
 * an HTTP 401 only after the browser has already started. See "You need an Unsloth, and you need
 * its password" in the studiobench README.
 *
 * ONE PASSWORD PER ARM when both arms are attached. Two separately booted Unsloth instances mint two
 * different bootstrap passwords, so reusing the first for the treatment is a 401 on the second
 * arm only, after the browser is up. `--password-b` defaults to `--password`, which is right
 * for the single-Unsloth case and wrong for this one.
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
 * OUT OF THE PAGE VIA THE CONSOLE. Unsloth ships `connect-src 'self'`, so a beacon to a collector
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
	/* Per role, the two heights a skipped root can land on, and they mean OPPOSITE things:
	 *
	 *   fallback  the declared `contain-intrinsic-size` length, used because no last remembered
	 *             size exists yet. The arm is working as designed and the length is a guess.
	 *   padding   the root's own padding and nothing else, which is a remembered size of ZERO,
	 *             recorded while the root was mounted and still empty. That is the trap.
	 *
	 * They have to be told apart or the probe attributes one to the other, and on this app they
	 * are close enough to collide: the user root's fallback is 60px against 40px of padding. So
	 * each is matched against its OWN target height, the fallback first so that it wins ties, and
	 * a root that is neither is counted as neither. A single `height <= 64` bucket put a user root
	 * sitting exactly on its fallback into both counters at once. */
	var ROLE_PX = {
		assistant: { fallback: 300, padding: 18 },
		user: { fallback: 60, padding: 40 }
	};
	/* `getBoundingClientRect().height` is the BORDER BOX, so both targets carry the root's own
	 * padding: an assistant root on its 300px fallback measures 318, and one whose remembered
	 * size is zero measures 18. Comparing the rect against the bare declared length instead left
	 * `fallbackBite` structurally pinned at zero, which reads as "the fallback is never used" no
	 * matter what the browser did. */
	function targetHeight(px, which) {
		return (which === "fallback" ? px.fallback : 0) + px.padding;
	}
	/* Half the gap between the two smallest interesting heights, so neither test can reach the
	 * other's target. Sub-pixel layout means an exact equality test would miss. */
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
	/* element -> its record, so a sample can ask whether THIS root is currently skipped. Weak so
	 * that a root the app has thrown away is not held alive by the probe. */
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

	/* Whether the browser last told us THIS root is skipping its contents.
	 *
	 * `null` means no transition has been seen, which is not the same as "rendered" and must not
	 * be read as either. Callers that care about the difference check for `true` explicitly. */
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
				/* Two MUTUALLY EXCLUSIVE buckets, in priority order. `content-visibility: auto`
				 * imposes size containment while skipping, so a skipped root's height is its
				 * padding plus its intrinsic size. Landing on the declared <length> means no
				 * remembered size exists yet; landing on the padding alone means one exists and
				 * it is ZERO, recorded while the root was mounted and still empty. The fallback
				 * is tested first and wins ties, because a root sitting on its fallback is
				 * behaving exactly as the declaration asks and must never be charged to the
				 * trap. Anything else is counted as neither. */
				/* ONLY WHILE SKIPPED. `content-visibility: auto` computes to `auto` whether or
				 * not the element is currently skipping, and size containment applies only while
				 * it is. An armed root that is on screen has its ordinary rendered height, and if
				 * that height happens to land within the tolerance of a role target it would be
				 * charged to the remembered-size trap. Ordinary geometry must not be able to
				 * masquerade as the finding. */
				var px = ROLE_PX[roleOf(el)];
				if (cv === "auto" && px && skippedState(el) === true) {
					if (Math.abs(r.height - targetHeight(px, "fallback")) <= PX_EPS) {
						out.fallbackBite += 1;
					} else if (Math.abs(r.height - targetHeight(px, "padding")) <= PX_EPS) {
						out.paddingOnly += 1;
					}
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

		/* CONNECTED roots only, and the detached ones are dropped as we go.
		 *
		 * `thread_reopen` is in the film: it tears the thread down and rebuilds it, so the old
		 * roots leave the document. A detached root receives no further transitions, so one whose
		 * last event said `skipped` would go on being counted in `skippedNow` for the rest of the
		 * session, reporting roots as skipped that are not in the page at all. Pruning here also
		 * stops `watched` growing a strong reference per root for the life of a long run. */
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

	/* THE LISTENER HAS TO EXIST BEFORE THE ELEMENT'S FIRST TRANSITION, and polling cannot
	 * guarantee that.
	 *
	 * `contentvisibilityautostatechange` fires on a CHANGE of state. A root that is inserted
	 * off screen becomes skipped once, at its first lifecycle update, and then never changes
	 * again while the user stays where they are. If the listener is attached on a two-second
	 * tick, every root mounted and skipped inside that first tick emits its only event into a
	 * void, `ev_skip` stays at zero, and the probe reports precisely the false NOT RUN it was
	 * written to prevent. A whole thread can mount inside two seconds; a seeded one does.
	 *
	 * So roots are adopted at INSERTION, from a MutationObserver installed before the app boots.
	 * The observer callback runs in a microtask after the mutation and before the next lifecycle
	 * update, which is early enough. The interval is kept, because a root can also acquire the
	 * property later without being re-inserted, and `adoptAll` is idempotent.
	 *
	 * OBSERVE THE DOCUMENT, NOT `documentElement`. An init script is evaluated after the Document
	 * exists but before the page's own scripts run, and at that instant the parser may not have
	 * created the root element yet. `observe(null, ...)` throws, the catch below would swallow it,
	 * and adoption would silently fall back to the two-second tick: the exact false zero this
	 * whole arrangement exists to prevent, reintroduced by the fix for it. A Document node is a
	 * valid target and it always exists here. */
	function adoptAll() {
		var roots = all(MESSAGE_SELECTOR);
		for (var i = 0; i < roots.length; i++) {
			watch(roots[i]);
		}
	}

	/* Only the ADDED NODES, never a document-wide re-scan. Streaming produces thousands of
	 * mutations a second, and a `querySelectorAll` per mutation would make this probe the load
	 * it is trying to observe. `watch` is idempotent, so a node reached twice costs a WeakSet
	 * lookup. */
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
