// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * knobs.js -- runtime ablation knobs for the SHIPPED Unsloth Studio build.
 *
 * Injected with Playwright's context.add_init_script BEFORE the app boots, so an external tester
 * can run every ablation against the same production bundle the user gets, with the ablated and
 * control runs sharing the same bytes of application code.
 *
 * Each arm removes ONE candidate mechanism and nothing else, so a cheaper thread names the fix:
 *
 *   A  visibility:hidden on completed messages -- removes paint/raster only. If A wins, stop
 *      painting the prefix rather than building it.
 *   B  content-visibility:auto + contain-intrinsic-size, undoing the index.css:2537 override --
 *      removes off-screen style, layout AND paint. B beating A says the pixels are not the cost,
 *      and re-tests that override's claim that containment on message roots "was no help".
 *   C  display:none -- also removes layout geometry and the sibling count. C winning where B does
 *      not points at the thread container's layout mode, not at per-element style.
 *   D  detach the autoscroll subtree MutationObserver (use-intent-aware-autoscroll.tsx:502), whose
 *      callback reads el.scrollHeight and so forces synchronous layout of the whole thread per
 *      streamed character. If D wins, rAF-coalesce that read rather than touching the messages.
 * Observed with subtree+characterData, so it fires per streamed character.
 *   E  neutralise the --aui-scroll-stabilizer write: a custom property is inherited, so writing it
 *      on the scroll container invalidates inherited style for every message. If E wins, register
 *      it with inherits:false or move the write off the thread's ancestor.
 * If E wins, `CSS.registerProperty` with inherits:false is the fix.
 *   F  freeze React's scheduler (MessageChannel port1.onmessage = performWorkUntilDeadline),
 *      leaving the DOM untouched. If F wins the fix is memoisation / virtualisation, not CSS.
 *   G  CONTROL: identical DOM, no knob, prior turns actually IN the viewport. A G that is not
 *      slower than the baseline means the baseline was never skipping anything, which would
 *      invalidate the whole comparison.
 *
 * THE TWO FAILURE MODES THIS FILE EXISTS TO PREVENT (see arms/manifest.py); both are silent.
 *
 *   1. THE ARM CHANGED THE OUTPUT. An earlier stub fixture rendered 552 highlighted spans where
 *      the real page renders 2,561 and read as a clean 4x win. digest() and counts() catch that:
 *      the canonical form carries code-block and span counts per message, so a highlighting
 *      difference is caught even when every character of text matches. Two declared gaps: the
 *      digest does not serialise descendants' attributes (so arm B's inline style on a code block
 *      is invisible, which is why B is EQUIVALENT and not EXACT) and it knows nothing about
 *      geometry. Arms A and C therefore mutate NOTHING, targeting the prefix with
 *      `:nth-child(-n+K)` so EXACT is a claim they can make; when that is impossible they fall
 *      back to a marker attribute and say in the returned reason that they dropped to EQUIVALENT.
 *   2. THE ARM DID NOT FIRE, and "no effect" got written down as evidence. So every arm has a
 *      potency counter the arm CAUSES ("N elements compute to visibility:hidden", not "the
 *      stylesheet was appended"), and a patch that could not be installed lands in unavailable[]
 *      so UNAVAILABLE can never be misread as "had no effect".
 *
 * CASCADE FACTS, verified against CSS Cascading and Inheritance Level 5. The index.css override
 * sits inside `@layer utilities`, and for IMPORTANT declarations layer order is REVERSED, so an
 * unlayered `!important` rule LOSES at any specificity -- the obvious approach silently does
 * nothing, which is failure mode 2. Inline declarations sort BEFORE layers, so an inline
 * `!important` beats every author declaration. Each CSS arm therefore emits its sheet twice (once
 * unlayered, once inside `@layer utilities`), verifies with getComputedStyle, and only escalates
 * to inline !important when the verification says the rule did not take, counting the escalation.
 *
 * ARMS ARE APPLIED ONCE, over the prefix that exists at that moment. Messages created during the
 * window are NOT ablated, deliberately: the hypothesis is about the completed off-screen prefix,
 * and a live MutationObserver re-applying the knob would add per-mutation work to the hot path --
 * an ablation that installs an observer to remove an observer measures itself. For the same reason
 * arm B marks only code blocks inside COMPLETED messages.
 *
 * PREBOOT VERSUS RUNTIME. D, E and F patch APIs the app captures during boot and are read from
 * window.__sbArmConfig.preboot. An arm not listed is NOT INSTALLED AT ALL, because an inactive
 * wrapper still costs a call frame on every observe or setProperty and a control run carrying that
 * overhead is not a control. D and E are active from injection (apply("D") reports the arm already
 * active); F captures the port at preboot and toggles the freeze in apply("F").
 *
 * COUNTERS. potency() returns integers in two kinds. EVENT COUNTERS are monotonic for the life of
 * the page (suppressedViewportObserves, observeCallsTotal, suppressedStabilizerSets,
 * suppressedSchedulerCallbacks, capturedSchedulerPorts, the bookkeeping counters, ...). GAUGES are
 * point-in-time reads of the live DOM, re-measured on every call unless {live:false}
 * (visibilityHiddenConfirmed, cvAutoMessages, displayNoneConfirmed, controlVisibleMessages, ...).
 * A stored 0 and a measured 0 are different facts: measuring live keeps the "before" read a real
 * observation, and checks arm G at the END of the window after the app's autoscroll has had every
 * chance to undo the scroll position. Re-measuring calls getComputedStyle over every message root,
 * forcing style recalc, so potency() must be called OUTSIDE the timed window.
 *
 * HASHING: FNV-1a, 32 bit, hex. Not cryptographic, and small enough that a collision on a
 * multi-megabyte canonical form is not impossible, so digest() also returns canonicalLength and
 * rawLength. Compare the pair, not the hash alone.
 */

(function () {
	"use strict";

	if (typeof window === "undefined" || !window) {
		return;
	}

	var W = window;

	// Idempotence: add_init_script runs for every document including iframes, and double
	// wrapping would double every count and make the freeze flag ambiguous.
	if (W.__sbArmsInstalled) {
		return;
	}
	W.__sbArmsInstalled = true;

	var VERSION = "1.0.0";
	var CANONICAL_FORMAT_VERSION = 1;

	// Selectors. Every one of these is a verified fact about the shipped build, not a guess.
	var VIEWPORT_SELECTOR = ".aui-thread-viewport.aui-stream-viewport";
	var VIEWPORT_CLASS = "aui-stream-viewport";
	var MESSAGE_SELECTOR = "[data-message-id]";
	var CODE_BLOCK_SELECTOR = '[data-streamdown="code-block"]';
	var RUNNING_SELECTOR = '[data-status="running"]';
	var STABILIZER_PROP = "--aui-scroll-stabilizer";
	var MARKER_ATTR = "data-sb-arm";

	var ARM_IDS = ["A", "B", "C", "D", "E", "F", "G"];
	var PREBOOT_ARMS = { D: true, E: true, F: true };

	var MAX_RETAINED_SCHEDULER_EVENTS = 256;
	var MAX_CAPTURED_CHANNELS = 64;
	var DEFAULT_MAX_CANONICAL_CHARS = 4000000;
	var DEFAULT_CONTROL_VISIBLE_TARGET = 3;
	var MAX_STYLESHEET_DEPTH = 8;

	// A missing or malformed __sbArmConfig means no preboot arms: absent config is not
	// permission to patch a control run.

	var cfg = {};
	try {
		if (W.__sbArmConfig && typeof W.__sbArmConfig === "object") {
			cfg = W.__sbArmConfig;
		}
	} catch (e) {
		cfg = {};
	}

	var debugOn = false;
	try {
		debugOn = cfg.debug === true;
	} catch (e) {
		debugOn = false;
	}

	function debug() {
		if (!debugOn) {
			return;
		}
		try {
			var args = ["[sbArms]"];
			for (var i = 0; i < arguments.length; i++) {
				args.push(arguments[i]);
			}
			if (W.console && typeof W.console.debug === "function") {
				W.console.debug.apply(W.console, args);
			}
		} catch (e) {
			/* logging must never be the reason a run fails */
		}
	}

	var armedPreboot = [];
	try {
		var rawPreboot = cfg.preboot;
		if (rawPreboot && typeof rawPreboot.length === "number") {
			for (var pi = 0; pi < rawPreboot.length; pi++) {
				var pid = String(rawPreboot[pi]).toUpperCase();
				if (PREBOOT_ARMS[pid] === true && armedPreboot.indexOf(pid) === -1) {
					armedPreboot.push(pid);
				}
			}
		}
	} catch (e) {
		armedPreboot = [];
	}

	var controlVisibleTarget = DEFAULT_CONTROL_VISIBLE_TARGET;
	try {
		var rawTarget = cfg.controlVisibleTarget;
		if (typeof rawTarget === "number" && isFinite(rawTarget) && rawTarget > 0) {
			controlVisibleTarget = Math.floor(rawTarget);
		}
	} catch (e) {
		controlVisibleTarget = DEFAULT_CONTROL_VISIBLE_TARGET;
	}

	var redeliverOnUnfreeze = true;
	try {
		if (cfg.redeliverOnUnfreeze === false) {
			redeliverOnUnfreeze = false;
		}
	} catch (e) {
		redeliverOnUnfreeze = true;
	}

	var doc = null;
	try {
		doc = W.document || null;
	} catch (e) {
		doc = null;
	}


	function freshEventCounters() {
		return {
			observeCallsTotal: 0,
			observeCallsPassedThrough: 0,
			suppressedViewportObserves: 0,
			setPropertyCallsTotal: 0,
			customPropSets: 0,
			suppressedStabilizerSets: 0,
			capturedSchedulerPorts: 0,
			schedulerCallbacksDelivered: 0,
			suppressedSchedulerCallbacks: 0,
			schedulerCallbacksRedelivered: 0,
			schedulerResumeKicks: 0,
			retainedSchedulerEvents: 0,
			droppedSchedulerEvents: 0,
			schedulerListenerAdds: 0,
			schedulerCallbacksWithoutHandler: 0,
			armsApplied: 0,
			armsReverted: 0,
			styleSheetsInstalled: 0,
			markedElements: 0,
			inlineFallbacks: 0,
			inlinePropsSet: 0,
			digestsTaken: 0
		};
	}

	function freshGauges() {
		return {
			visibilityHiddenConfirmed: 0,
			cvAutoMessages: 0,
			cvAutoCodeBlocks: 0,
			contentVisibilityAutoConfirmed: 0,
			displayNoneConfirmed: 0,
			controlVisibleMessages: 0,
			controlVisiblePriorMessages: 0
		};
	}

	var counters = freshEventCounters();
	var gauges = freshGauges();

	var state = {
		applied: {},
		frozen: false,
		markers: [],
		inline: [],
		sheets: {},
		channels: [],
		scrollRestore: null,
		originals: {
			observe: null,
			observeDescriptor: null,
			setProperty: null,
			setPropertyDescriptor: null,
			MessageChannel: null
		},
		installed: { D: false, E: false, F: false }
	};

	var unavailable = {};

	function markUnavailable(armId, reason) {
		try {
			if (!Object.prototype.hasOwnProperty.call(unavailable, armId)) {
				unavailable[armId] = String(reason);
			}
		} catch (e) {
			/* nothing sensible to do, and throwing here would take the app down */
		}
	}

	function nowStamp() {
		try {
			return new Date().toISOString();
		} catch (e) {
			return null;
		}
	}

	function result(applied, reason, affected) {
		return {
			applied: !!applied,
			reason: String(reason || ""),
			affected: typeof affected === "number" && isFinite(affected) ? Math.floor(affected) : 0
		};
	}

	function revertResult(reverted, reason) {
		return { reverted: !!reverted, reason: String(reason || "") };
	}

	// None of these throw on a missing document, detached node or empty match; a failed
	// query returns [] and the caller turns that into a reason, not a zero.

	function toArray(nodeList) {
		var out = [];
		try {
			if (!nodeList) {
				return out;
			}
			for (var i = 0; i < nodeList.length; i++) {
				out.push(nodeList[i]);
			}
		} catch (e) {
			/* partial list is better than none */
		}
		return out;
	}

	function queryAll(selector, root) {
		try {
			var scope = root || doc;
			if (!scope || typeof scope.querySelectorAll !== "function") {
				return [];
			}
			return toArray(scope.querySelectorAll(selector));
		} catch (e) {
			return [];
		}
	}

	function queryOne(selector, root) {
		try {
			var scope = root || doc;
			if (!scope || typeof scope.querySelector !== "function") {
				return null;
			}
			return scope.querySelector(selector);
		} catch (e) {
			return null;
		}
	}

	function messageRoots() {
		return queryAll(MESSAGE_SELECTOR);
	}

	function isRunning(el) {
		// The only DOM signal of an in-flight message is [data-status="running"] on the root or a
		// descendant (assistant-ui status: running|complete|incomplete|requires-action).
		try {
			if (!el) {
				return false;
			}
			if (typeof el.getAttribute === "function" && el.getAttribute("data-status") === "running") {
				return true;
			}
			return queryOne(RUNNING_SELECTOR, el) !== null;
		} catch (e) {
			// Unknown status counts as running, i.e. not ablatable: the other way would ablate the
			// streaming message and contaminate the measurement.
			return true;
		}
	}

	// Filtering in JS rather than CSS `:has()`: an unsupported compound selector invalidates the
	// whole selector list, so the rule would be dropped at parse time and the arm would report a
	// clean zero difference.
	// One querySelector per root.
	function completedMessageRoots() {
		var roots = messageRoots();
		var out = [];
		for (var i = 0; i < roots.length; i++) {
			if (!isRunning(roots[i])) {
				out.push(roots[i]);
			}
		}
		return out;
	}

	function runningMessageRoots() {
		var roots = messageRoots();
		var out = [];
		for (var i = 0; i < roots.length; i++) {
			if (isRunning(roots[i])) {
				out.push(roots[i]);
			}
		}
		return out;
	}

	function codeBlocksIn(root) {
		return queryAll(CODE_BLOCK_SELECTOR, root);
	}

	function allCodeBlocks() {
		return queryAll(CODE_BLOCK_SELECTOR);
	}

	function viewportEl() {
		return queryOne(VIEWPORT_SELECTOR);
	}

	function rectOf(el) {
		try {
			if (!el || typeof el.getBoundingClientRect !== "function") {
				return null;
			}
			return el.getBoundingClientRect();
		} catch (e) {
			return null;
		}
	}

	function rectsIntersect(a, b) {
		if (!a || !b) {
			return false;
		}
		return a.bottom > b.top && a.top < b.bottom && a.right > b.left && a.left < b.right;
	}

	function computedValue(el, prop) {
		try {
			if (!el || typeof W.getComputedStyle !== "function") {
				return "";
			}
			var cs = W.getComputedStyle(el);
			if (!cs) {
				return "";
			}
			if (typeof cs.getPropertyValue === "function") {
				return String(cs.getPropertyValue(prop) || "");
			}
			return "";
		} catch (e) {
			return "";
		}
	}

	function countComputed(els, prop, expected) {
		var n = 0;
		for (var i = 0; i < els.length; i++) {
			var v = computedValue(els[i], prop);
			if (v.trim() === expected) {
				n += 1;
			}
		}
		return n;
	}

	function supportsProp(prop, value) {
		try {
			if (W.CSS && typeof W.CSS.supports === "function") {
				return W.CSS.supports(prop, value) === true;
			}
		} catch (e) {
			/* fall through */
		}
		// No CSS.supports means support cannot be proven; assume supported and let the computed
		// check decide, since a false UNAVAILABLE hides an arm that works.
		return true;
	}

	// One space-separated attribute serves arms A, B and C ([data-sb-arm~="A"]). It IS a DOM
	// change and appears in digest().raw deliberately: such an arm is EQUIVALENT, not EXACT,
	// and Python declares `attr:data-sb-arm` (plus `attr:style`) as its allowed diff.

	function markElement(el, armId) {
		try {
			if (!el || typeof el.getAttribute !== "function") {
				return false;
			}
			var prev = el.getAttribute(MARKER_ATTR);
			var tokens = prev ? String(prev).split(/\s+/) : [];
			var seen = false;
			var clean = [];
			for (var i = 0; i < tokens.length; i++) {
				if (!tokens[i]) {
					continue;
				}
				clean.push(tokens[i]);
				if (tokens[i] === armId) {
					seen = true;
				}
			}
			if (!seen) {
				clean.push(armId);
			}
			state.markers.push({ el: el, prev: prev });
			el.setAttribute(MARKER_ATTR, clean.join(" "));
			counters.markedElements += 1;
			return true;
		} catch (e) {
			return false;
		}
	}

	function markAll(els, armId) {
		var n = 0;
		for (var i = 0; i < els.length; i++) {
			if (markElement(els[i], armId)) {
				n += 1;
			}
		}
		return n;
	}

	function unmarkAll() {
		for (var i = state.markers.length - 1; i >= 0; i--) {
			var rec = state.markers[i];
			try {
				if (rec.prev === null || rec.prev === undefined) {
					rec.el.removeAttribute(MARKER_ATTR);
				} else {
					rec.el.setAttribute(MARKER_ATTR, rec.prev);
				}
			} catch (e) {
				/* best effort: a detached node cannot be restored and does not matter */
			}
		}
		state.markers = [];
	}

	function setInline(el, prop, value, priority) {
		try {
			if (!el || !el.style || typeof el.style.setProperty !== "function") {
				return false;
			}
			var prevValue = "";
			var prevPriority = "";
			try {
				prevValue = el.style.getPropertyValue(prop) || "";
				prevPriority = el.style.getPropertyPriority(prop) || "";
			} catch (e) {
				prevValue = "";
				prevPriority = "";
			}
			var hadStyleAttr = false;
			try {
				hadStyleAttr =
					typeof el.hasAttribute === "function" ? el.hasAttribute("style") : true;
			} catch (e) {
				hadStyleAttr = true;
			}
			state.inline.push({
				el: el,
				prop: prop,
				prevValue: prevValue,
				prevPriority: prevPriority,
				hadStyleAttr: hadStyleAttr
			});
			el.style.setProperty(prop, value, priority || "");
			counters.inlinePropsSet += 1;
			return true;
		} catch (e) {
			return false;
		}
	}

	function restoreInline() {
		for (var i = state.inline.length - 1; i >= 0; i--) {
			var rec = state.inline[i];
			try {
				if (rec.prevValue === "") {
					rec.el.style.removeProperty(rec.prop);
				} else {
					rec.el.style.setProperty(rec.prop, rec.prevValue, rec.prevPriority);
				}
				if (!rec.hadStyleAttr) {
					var left = "";
					try {
						left = String(rec.el.getAttribute("style") || "").trim();
					} catch (e2) {
						left = "x";
					}
					if (left === "") {
						// Leaving style="" behind would be a permanent, invisible difference in every future digest of this page.
						rec.el.removeAttribute("style");
					}
				}
			} catch (e) {
				/* best effort */
			}
		}
		state.inline = [];
	}

	// Every arm sheet is emitted twice, unlayered and inside `@layer utilities`. The rule to beat
	// (index.css:2536-2538) is layered, and for important declarations layered beats unlayered at
	// any specificity; the unlayered copy covers a build with no layers. Nothing is trusted:
	// applyX() rechecks getComputedStyle afterwards.

	function ensureStyleSheet(armId, cssBody) {
		try {
			if (state.sheets[armId]) {
				return true;
			}
			if (!doc || typeof doc.createElement !== "function") {
				return false;
			}
			var parent = doc.head || doc.documentElement || doc.body;
			if (!parent || typeof parent.appendChild !== "function") {
				return false;
			}
			var el = doc.createElement("style");
			el.setAttribute("data-sb-arm-sheet", armId);
			el.textContent = cssBody + "\n@layer utilities {\n" + cssBody + "\n}\n";
			parent.appendChild(el);
			state.sheets[armId] = el;
			counters.styleSheetsInstalled += 1;
			return true;
		} catch (e) {
			return false;
		}
	}

	function removeStyleSheet(armId) {
		try {
			var el = state.sheets[armId];
			if (!el) {
				return false;
			}
			if (el.parentNode && typeof el.parentNode.removeChild === "function") {
				el.parentNode.removeChild(el);
			}
			delete state.sheets[armId];
			return true;
		} catch (e) {
			return false;
		}
	}

	// PREFIX PATH (preferred): messages are direct children of the viewport and the streamed one
	// is last, so a `> [data-message-id]:nth-child(-n+K)` selector picks exactly the completed
	// run. It mutates nothing, so the raw digest is byte-identical and arms A and C can claim
	// EXACT. `[data-message-id]` is needed: nth-child alone also matches spacers and buttons.
	// MARKER PATH (fallback): when the completed set is not a leading prefix, or no viewport is
	// found, each completed root is marked with `data-sb-arm`; the arm drops to EQUIVALENT and
	// says so in its reason.

	function cssPrefixA(k) {
		return (
			VIEWPORT_SELECTOR + " > [data-message-id]:nth-child(-n+" + k + ") {\n" +
			"\tvisibility: hidden !important;\n" +
			"}\n"
		);
	}

	function cssPrefixB(k) {
		return (
			"html body " + VIEWPORT_SELECTOR + " > [data-message-id]:nth-child(-n+" + k + ") " +
			'[data-streamdown="code-block"] {\n' +
			"\tcontent-visibility: auto !important;\n" +
			"}\n"
		);
	}

	function cssPrefixC(k) {
		return (
			VIEWPORT_SELECTOR + " > [data-message-id]:nth-child(-n+" + k + ") {\n" +
			"\tdisplay: none !important;\n" +
			"}\n"
		);
	}

	var CSS_A =
		"body [data-message-id][" + MARKER_ATTR + '~="A"],\n' +
		"[data-message-id][" + MARKER_ATTR + '~="A"] {\n' +
		"\tvisibility: hidden !important;\n" +
		"}\n";

	var CSS_B =
		"html body .aui-thread-root [data-streamdown=\"code-block\"][" + MARKER_ATTR + '~="B"],\n' +
		"html body [data-streamdown=\"code-block\"][" + MARKER_ATTR + '~="B"] {\n' +
		"\tcontent-visibility: auto !important;\n" +
		"}\n";

	var CSS_C =
		"body [data-message-id][" + MARKER_ATTR + '~="C"],\n' +
		"[data-message-id][" + MARKER_ATTR + '~="C"] {\n' +
		"\tdisplay: none !important;\n" +
		"}\n";

	// Returns the leading run of viewport children with no running message, else null. `k` is a
	// 1-based nth-child bound; `roots` are the completed roots inside it.
	function prefixInfo() {
		try {
			var vp = viewportEl();
			if (!vp) {
				return null;
			}
			var kids = [];
			try {
				kids = toArray(vp.children);
			} catch (e) {
				kids = [];
			}
			if (kids.length === 0) {
				return null;
			}
			var firstRunning = -1;
			var i;
			for (i = 0; i < kids.length; i++) {
				var kid = kids[i];
				var isMessage = false;
				try {
					isMessage =
						typeof kid.getAttribute === "function" &&
						kid.getAttribute("data-message-id") !== null;
				} catch (e2) {
					isMessage = false;
				}
				if (isMessage && isRunning(kid)) {
					firstRunning = i;
					break;
				}
			}
			var k = firstRunning === -1 ? kids.length : firstRunning;
			if (k <= 0) {
				return null;
			}
			var roots = [];
			for (i = 0; i < k; i++) {
				try {
					if (
						typeof kids[i].getAttribute === "function" &&
						kids[i].getAttribute("data-message-id") !== null
					) {
						roots.push(kids[i]);
					}
				} catch (e3) {
					/* a child we cannot read is a child we do not ablate */
				}
			}
			if (roots.length === 0) {
				return null;
			}
			var skipped = completedMessageRoots().length - roots.length;
			return { vp: vp, k: k, roots: roots, skipped: skipped > 0 ? skipped : 0 };
		} catch (e) {
			return null;
		}
	}


	var imul =
		typeof Math.imul === "function"
			? Math.imul
			: function (a, b) {
					var ah = (a >>> 16) & 0xffff;
					var al = a & 0xffff;
					var bh = (b >>> 16) & 0xffff;
					var bl = b & 0xffff;
					return ((al * bl + (((ah * bl + al * bh) << 16) >>> 0)) | 0);
			  };

	// FNV-1a 32-bit over UTF-16 code units, low byte first. Non-cryptographic: compare it
	// together with canonicalLength.
	function fnv1a32(str) {
		var h = 0x811c9dc5;
		for (var i = 0; i < str.length; i++) {
			var c = str.charCodeAt(i);
			h ^= c & 0xff;
			h = imul(h, 16777619);
			h ^= (c >>> 8) & 0xff;
			h = imul(h, 16777619);
		}
		return ("0000000" + (h >>> 0).toString(16)).slice(-8);
	}

	// JSON escapes quotes, backslashes, newlines and controls; the pipe is escaped on top, so no
	// field can contain the separator and diffKeys() can split() the canonical form.
	function enc(s) {
		try {
			return JSON.stringify(String(s === null || s === undefined ? "" : s))
				.slice(1, -1)
				.replace(/\|/g, "\\u007c");
		} catch (e) {
			return "";
		}
	}

	function dec(f) {
		try {
			return JSON.parse('"' + f + '"');
		} catch (e) {
			return "";
		}
	}

	function normaliseStyleValue(value, skipStyleProps) {
		if (!skipStyleProps || !skipStyleProps.length) {
			return value;
		}
		// Parsed by the browser's CSS parser on a detached element: splitting on ";" and ":" breaks
		// on url(data:...) and any value containing a semicolon.
		try {
			if (doc && typeof doc.createElement === "function") {
				var scratch = doc.createElement("div");
				scratch.style.cssText = String(value);
				for (var i = 0; i < skipStyleProps.length; i++) {
					scratch.style.removeProperty(String(skipStyleProps[i]));
				}
				return String(scratch.style.cssText || "");
			}
		} catch (e) {
			/* fall through to the textual fallback */
		}
		var parts = String(value).split(";");
		var kept = [];
		for (var j = 0; j < parts.length; j++) {
			var seg = parts[j];
			var colon = seg.indexOf(":");
			if (colon < 0) {
				continue;
			}
			var name = seg.slice(0, colon).trim().toLowerCase();
			var drop = false;
			for (var k = 0; k < skipStyleProps.length; k++) {
				if (String(skipStyleProps[k]).trim().toLowerCase() === name) {
					drop = true;
					break;
				}
			}
			if (!drop) {
				kept.push(seg.trim());
			}
		}
		return kept.join("; ");
	}

	// One DOM walk feeds both serialisations: walking twice lets the page change between the raw
	// and normalised passes, faking a raw-only difference.
	function collectRecords() {
		var roots = messageRoots();
		var records = [];
		var totalCodeBlocks = 0;
		var totalCodeSpans = 0;
		for (var i = 0; i < roots.length; i++) {
			var el = roots[i];
			var rec = {
				tag: "",
				id: "",
				attrs: [],
				text: "",
				codeBlocks: 0,
				codeSpans: 0
			};
			try {
				rec.tag = String(el.tagName || el.nodeName || "?").toLowerCase();
			} catch (e) {
				rec.tag = "?";
			}
			try {
				rec.id = String(el.getAttribute("data-message-id") || "");
			} catch (e) {
				rec.id = "";
			}
			try {
				var attrs = el.attributes;
				var list = [];
				for (var a = 0; a < attrs.length; a++) {
					list.push({
						name: String(attrs[a].name),
						value: String(attrs[a].value === undefined ? "" : attrs[a].value)
					});
				}
				list.sort(function (x, y) {
					return x.name < y.name ? -1 : x.name > y.name ? 1 : 0;
				});
				rec.attrs = list;
			} catch (e) {
				rec.attrs = [];
			}
			try {
				rec.text = String(el.textContent === null || el.textContent === undefined ? "" : el.textContent);
			} catch (e) {
				rec.text = "";
			}
			var blocks = codeBlocksIn(el);
			rec.codeBlocks = blocks.length;
			var spans = 0;
			for (var b = 0; b < blocks.length; b++) {
				spans += queryAll("span", blocks[b]).length;
			}
			rec.codeSpans = spans;
			totalCodeBlocks += rec.codeBlocks;
			totalCodeSpans += rec.codeSpans;
			records.push(rec);
		}
		return {
			records: records,
			messages: roots.length,
			codeBlocks: totalCodeBlocks,
			codeSpans: totalCodeSpans
		};
	}

	function serialise(collected, skipAttributes, skipStyleProps, observedKeys) {
		var lines = [];
		lines.push("V|" + CANONICAL_FORMAT_VERSION);
		lines.push("S|messages|" + collected.messages);
		lines.push("S|codeBlocks|" + collected.codeBlocks);
		lines.push("S|codeSpans|" + collected.codeSpans);
		observedKeys["structure:messages"] = true;
		observedKeys["structure:codeBlocks"] = true;
		observedKeys["structure:codeSpans"] = true;
		observedKeys["structure:order"] = true;
		observedKeys.tag = true;
		observedKeys.text = true;

		var skip = {};
		for (var s = 0; s < skipAttributes.length; s++) {
			skip[String(skipAttributes[s]).toLowerCase()] = true;
		}

		for (var i = 0; i < collected.records.length; i++) {
			var rec = collected.records[i];
			lines.push("M|" + i + "|" + enc(rec.tag) + "|" + enc(rec.id));
			for (var a = 0; a < rec.attrs.length; a++) {
				var name = rec.attrs[a].name;
				var lower = String(name).toLowerCase();
				if (skip[lower] === true) {
					continue;
				}
				var value = rec.attrs[a].value;
				if (lower === "style") {
					value = normaliseStyleValue(value, skipStyleProps);
				}
				observedKeys["attr:" + lower] = true;
				lines.push("A|" + i + "|" + enc(name) + "|" + enc(value));
			}
			lines.push("T|" + i + "|" + enc(rec.text));
			lines.push("B|" + i + "|" + rec.codeBlocks + "|" + rec.codeSpans);
		}
		return lines.join("\n");
	}

	function digest(options) {
		var opts = options && typeof options === "object" ? options : {};
		var skipAttributes = ["style"];
		try {
			if (opts.skipAttributes && typeof opts.skipAttributes.length === "number") {
				skipAttributes = [];
				for (var i = 0; i < opts.skipAttributes.length; i++) {
					skipAttributes.push(String(opts.skipAttributes[i]));
				}
			}
		} catch (e) {
			skipAttributes = ["style"];
		}
		var skipStyleProps = [];
		try {
			if (opts.skipStyleProps && typeof opts.skipStyleProps.length === "number") {
				for (var j = 0; j < opts.skipStyleProps.length; j++) {
					skipStyleProps.push(String(opts.skipStyleProps[j]));
				}
			}
		} catch (e) {
			skipStyleProps = [];
		}
		var keepCanonical = opts.keepCanonical === true;
		var maxCanonicalChars = DEFAULT_MAX_CANONICAL_CHARS;
		try {
			if (typeof opts.maxCanonicalChars === "number" && isFinite(opts.maxCanonicalChars) && opts.maxCanonicalChars >= 0) {
				maxCanonicalChars = Math.floor(opts.maxCanonicalChars);
			}
		} catch (e) {
			maxCanonicalChars = DEFAULT_MAX_CANONICAL_CHARS;
		}

		var out = {
			raw: null,
			normalised: null,
			canonicalLength: 0,
			rawLength: 0,
			observedKeys: [],
			canonical: null,
			canonicalRaw: null,
			truncated: false,
			skipAttributes: skipAttributes.slice(0),
			skipStyleProps: skipStyleProps.slice(0),
			formatVersion: CANONICAL_FORMAT_VERSION,
			hash: "fnv1a32",
			messages: 0,
			codeBlocks: 0,
			codeSpans: 0,
			error: null
		};

		try {
			var collected = collectRecords();
			out.messages = collected.messages;
			out.codeBlocks = collected.codeBlocks;
			out.codeSpans = collected.codeSpans;

			var rawKeys = {};
			// `raw` skips nothing, including the arm's own marker attribute: an EXACT arm that touches an
			// attribute is not EXACT, and this is where that is caught.
			var rawCanonical = serialise(collected, [], [], rawKeys);

			var normKeys = {};
			var normCanonical = serialise(collected, skipAttributes, skipStyleProps, normKeys);

			out.raw = fnv1a32(rawCanonical);
			out.normalised = fnv1a32(normCanonical);
			out.rawLength = rawCanonical.length;
			out.canonicalLength = normCanonical.length;

			var keys = [];
			for (var k in rawKeys) {
				if (Object.prototype.hasOwnProperty.call(rawKeys, k)) {
					keys.push(k);
				}
			}
			keys.sort();
			out.observedKeys = keys;

			if (keepCanonical) {
				if (rawCanonical.length > maxCanonicalChars || normCanonical.length > maxCanonicalChars) {
					// A prefix is kept for debugging but `truncated` is set and diffKeys() refuses a truncated
					// pair: a quiet cut would turn "not looked at" into "identical".
					out.truncated = true;
				}
				out.canonicalRaw = rawCanonical.slice(0, maxCanonicalChars);
				out.canonical = normCanonical.slice(0, maxCanonicalChars);
			}
			counters.digestsTaken += 1;
		} catch (e) {
			out.error = String((e && e.message) || e);
		}
		return out;
	}

	// Compares the RAW canonical forms: the normalised hashes already answer "equivalent?", while
	// manifest.py needs what actually differs to check it against the DECLARED diff. Normalised
	// forms would hide a second difference, since the normaliser removes the declared one.
	// Keys collapse to a stable vocabulary (`attr:<name>`, `text`, `tag`, `structure:<what>`): a
	// declared diff must be writable in advance, and `#msg-8fc2.attr.style` is not.
	// Fails closed: an unretained, truncated or unparseable canonical form yields a
	// `__unavailable:` key, which can never appear in a declared diff, so the arm voids.

	function parseCanonical(text) {
		var entries = {};
		var indexToId = {};
		var lines = String(text).split("\n");
		for (var i = 0; i < lines.length; i++) {
			var line = lines[i];
			if (!line) {
				continue;
			}
			var f = line.split("|");
			var kind = f[0];
			if (kind === "V") {
				continue;
			}
			if (kind === "S") {
				if (!entries["#global"]) {
					entries["#global"] = {};
				}
				entries["#global"]["structure:" + f[1]] = f[2];
				continue;
			}
			var idx = f[1];
			if (kind === "M") {
				var id = dec(f[3] === undefined ? "" : f[3]);
				var entryId = id ? "id:" + id : "idx:" + idx;
				indexToId[idx] = entryId;
				if (!entries[entryId]) {
					entries[entryId] = {};
				}
				entries[entryId].tag = dec(f[2] === undefined ? "" : f[2]);
				entries[entryId]["structure:order"] = String(idx);
				continue;
			}
			var owner = indexToId[idx];
			if (!owner || !entries[owner]) {
				continue;
			}
			if (kind === "A") {
				entries[owner]["attr:" + dec(f[2] === undefined ? "" : f[2]).toLowerCase()] =
					dec(f.slice(3).join("|"));
			} else if (kind === "T") {
				entries[owner].text = dec(f.slice(2).join("|"));
			} else if (kind === "B") {
				entries[owner]["structure:codeBlocks"] = f[2];
				entries[owner]["structure:codeSpans"] = f[3];
			}
		}
		return entries;
	}

	function diffKeys(a, b) {
		try {
			if (!a || !b || typeof a !== "object" || typeof b !== "object") {
				return ["__unavailable:digest-missing__"];
			}
			if (a.truncated === true || b.truncated === true) {
				return ["__unavailable:canonical-truncated__"];
			}
			var ca = a.canonicalRaw;
			var cb = b.canonicalRaw;
			if (typeof ca !== "string" || typeof cb !== "string") {
				return ["__unavailable:canonical-not-retained__"];
			}
			if (ca === cb) {
				return [];
			}
			var ea = parseCanonical(ca);
			var eb = parseCanonical(cb);
			var found = {};
			var ids = {};
			var id;
			for (id in ea) {
				if (Object.prototype.hasOwnProperty.call(ea, id)) {
					ids[id] = true;
				}
			}
			for (id in eb) {
				if (Object.prototype.hasOwnProperty.call(eb, id)) {
					ids[id] = true;
				}
			}
			for (id in ids) {
				if (!Object.prototype.hasOwnProperty.call(ids, id)) {
					continue;
				}
				var ra = ea[id];
				var rb = eb[id];
				if (!ra || !rb) {
					found["structure:messagePresence"] = true;
					continue;
				}
				var keys = {};
				var key;
				for (key in ra) {
					if (Object.prototype.hasOwnProperty.call(ra, key)) {
						keys[key] = true;
					}
				}
				for (key in rb) {
					if (Object.prototype.hasOwnProperty.call(rb, key)) {
						keys[key] = true;
					}
				}
				for (key in keys) {
					if (!Object.prototype.hasOwnProperty.call(keys, key)) {
						continue;
					}
					if (ra[key] !== rb[key]) {
						found[key] = true;
					}
				}
			}
			var out = [];
			for (var k in found) {
				if (Object.prototype.hasOwnProperty.call(found, k)) {
					out.push(k);
				}
			}
			out.sort();
			return out;
		} catch (e) {
			return ["__unavailable:diff-failed__"];
		}
	}

	// Beyond the required API: the same comparison with examples, for a human reading a VOIDED
	// verdict who needs to know which message drifted and by how much.
	function diffDetail(a, b, limit) {
		var cap = typeof limit === "number" && limit > 0 ? Math.floor(limit) : 20;
		var out = [];
		try {
			if (!a || !b || typeof a.canonicalRaw !== "string" || typeof b.canonicalRaw !== "string") {
				return out;
			}
			var ea = parseCanonical(a.canonicalRaw);
			var eb = parseCanonical(b.canonicalRaw);
			var ids = {};
			var id;
			for (id in ea) {
				if (Object.prototype.hasOwnProperty.call(ea, id)) {
					ids[id] = true;
				}
			}
			for (id in eb) {
				if (Object.prototype.hasOwnProperty.call(eb, id)) {
					ids[id] = true;
				}
			}
			for (id in ids) {
				if (!Object.prototype.hasOwnProperty.call(ids, id) || out.length >= cap) {
					continue;
				}
				var ra = ea[id] || {};
				var rb = eb[id] || {};
				var keys = {};
				var key;
				for (key in ra) {
					if (Object.prototype.hasOwnProperty.call(ra, key)) {
						keys[key] = true;
					}
				}
				for (key in rb) {
					if (Object.prototype.hasOwnProperty.call(rb, key)) {
						keys[key] = true;
					}
				}
				for (key in keys) {
					if (!Object.prototype.hasOwnProperty.call(keys, key) || out.length >= cap) {
						continue;
					}
					if (ra[key] !== rb[key]) {
						out.push({
							entry: id,
							key: key,
							before: String(ra[key] === undefined ? "" : ra[key]).slice(0, 200),
							after: String(rb[key] === undefined ? "" : rb[key]).slice(0, 200)
						});
					}
				}
			}
		} catch (e) {
			/* a partial detail list is still useful and must not throw */
		}
		return out;
	}


	function counts() {
		var out = {
			messages: 0,
			assistantMessages: 0,
			userMessages: 0,
			completedMessages: 0,
			runningMessages: 0,
			codeBlocks: 0,
			codeSpans: 0,
			highlightedTokens: 0,
			chars: 0,
			domNodes: 0,
			visibleMessages: 0,
			// Zero discipline: visibleMessages == 0 from a missing thread viewport is a different fact
			// from nothing being on screen, and must not print the same.
			viewportFound: false
		};
		try {
			var roots = messageRoots();
			out.messages = roots.length;
			var vp = viewportEl();
			out.viewportFound = !!vp;
			var vpRect = vp ? rectOf(vp) : null;
			for (var i = 0; i < roots.length; i++) {
				var el = roots[i];
				var role = "";
				try {
					role = String(el.getAttribute("data-role") || "");
				} catch (e) {
					role = "";
				}
				if (role === "assistant") {
					out.assistantMessages += 1;
				} else if (role === "user") {
					out.userMessages += 1;
				}
				if (isRunning(el)) {
					out.runningMessages += 1;
				} else {
					out.completedMessages += 1;
				}
				try {
					out.chars += String(el.textContent || "").length;
				} catch (e2) {
					/* a node that cannot be read contributes nothing */
				}
				if (vpRect) {
					var r = rectOf(el);
					if (rectsIntersect(r, vpRect)) {
						out.visibleMessages += 1;
					}
				}
			}
			var blocks = allCodeBlocks();
			out.codeBlocks = blocks.length;
			for (var b = 0; b < blocks.length; b++) {
				var spans = queryAll("span", blocks[b]);
				out.codeSpans += spans.length;
				for (var s = 0; s < spans.length; s++) {
					// A shiki token is a leaf span with text. Both numbers are reported: partial highlighting
					// moves both, a change in span NESTING moves only codeSpans.
					var sp = spans[s];
					var hasElementChild = false;
					try {
						hasElementChild = !!(sp.children && sp.children.length > 0);
					} catch (e3) {
						hasElementChild = false;
					}
					if (hasElementChild) {
						continue;
					}
					var txt = "";
					try {
						txt = String(sp.textContent || "");
					} catch (e4) {
						txt = "";
					}
					if (txt.length > 0) {
						out.highlightedTokens += 1;
					}
				}
			}
			try {
				out.domNodes = doc && typeof doc.getElementsByTagName === "function"
					? doc.getElementsByTagName("*").length
					: 0;
			} catch (e5) {
				out.domNodes = 0;
			}
		} catch (e) {
			/* whatever was counted before the failure is still returned */
		}
		return out;
	}


	function measureGauges() {
		try {
			var roots = messageRoots();
			gauges.visibilityHiddenConfirmed = countComputed(roots, "visibility", "hidden");
			gauges.displayNoneConfirmed = countComputed(roots, "display", "none");
			gauges.cvAutoMessages = countComputed(roots, "content-visibility", "auto");
			gauges.cvAutoCodeBlocks = countComputed(allCodeBlocks(), "content-visibility", "auto");
			gauges.contentVisibilityAutoConfirmed = gauges.cvAutoMessages + gauges.cvAutoCodeBlocks;

			var vp = viewportEl();
			var vpRect = vp ? rectOf(vp) : null;
			var visible = 0;
			var visiblePrior = 0;
			if (vpRect) {
				for (var i = 0; i < roots.length; i++) {
					if (rectsIntersect(rectOf(roots[i]), vpRect)) {
						visible += 1;
						if (i < roots.length - 1) {
							visiblePrior += 1;
						}
					}
				}
			}
			gauges.controlVisibleMessages = visible;
			gauges.controlVisiblePriorMessages = visiblePrior;
		} catch (e) {
			/* keep the last measurement rather than zeroing it, and never throw */
		}
	}

	function potency(options) {
		var opts = options && typeof options === "object" ? options : {};
		try {
			if (opts.live !== false) {
				measureGauges();
			}
		} catch (e) {
			/* fall through and return the stored values */
		}
		var out = {};
		var k;
		for (k in counters) {
			if (Object.prototype.hasOwnProperty.call(counters, k)) {
				out[k] = Math.floor(counters[k]);
			}
		}
		for (k in gauges) {
			if (Object.prototype.hasOwnProperty.call(gauges, k)) {
				out[k] = Math.floor(gauges[k]);
			}
		}
		out.frozen = state.frozen ? 1 : 0;
		out.armsAppliedNow = 0;
		for (var id in state.applied) {
			if (Object.prototype.hasOwnProperty.call(state.applied, id)) {
				out.armsAppliedNow += 1;
			}
		}
		return out;
	}


	function applyA() {
		var info = prefixInfo();
		var roots;
		var via;
		if (info) {
			roots = info.roots;
			ensureStyleSheet("A", cssPrefixA(info.k));
			via =
				":nth-child(-n+" + info.k + ") prefix stylesheet, no DOM mutation (EXACT)" +
				(info.skipped > 0
					? "; " + info.skipped + " completed root(s) after the first running message " +
					  "were left alone"
					: "");
		} else {
			roots = completedMessageRoots();
			if (roots.length === 0) {
				return result(
					false,
					"no completed message roots matched " + MESSAGE_SELECTOR + " (found " +
						messageRoots().length + " message roots, " + runningMessageRoots().length +
						" of them running); nothing was ablated",
					0
				);
			}
			ensureStyleSheet("A", CSS_A);
			markAll(roots, "A");
			via =
				"marker-attribute stylesheet: the completed set is not a leading prefix of the " +
				"viewport's children, so " + MARKER_ATTR + " was added and this arm is " +
				"EQUIVALENT, not EXACT";
		}
		var confirmed = countComputed(roots, "visibility", "hidden");
		if (confirmed === 0) {
			// The stylesheet lost the cascade. Inline important sorts before layers and beats every
			// author rule, so this escalation cannot lose.
			counters.inlineFallbacks += 1;
			for (var i = 0; i < roots.length; i++) {
				setInline(roots[i], "visibility", "hidden", "important");
			}
			confirmed = countComputed(roots, "visibility", "hidden");
			via =
				"inline !important, escalated because the stylesheet lost the cascade (this " +
				"writes the style attribute, so the arm is EQUIVALENT, not EXACT)";
		}
		gauges.visibilityHiddenConfirmed = confirmed;
		if (confirmed === 0) {
			return result(
				false,
				"targeted " + roots.length + " completed message roots but none of them compute " +
					"to visibility:hidden; the arm did not fire and must not be read as no effect",
				0
			);
		}
		return result(
			true,
			"visibility:hidden via " + via + " on " + confirmed + " of " + roots.length +
				" completed message roots",
			confirmed
		);
	}

	// (i) restores content-visibility on code blocks, undoing index.css:2536-2538; (ii) puts
	// content-visibility:auto and contain-intrinsic-size inline on each completed message root.
	// THE ORDERING BUG THIS FUNCTION AVOIDS: content-visibility:auto makes an off-screen element
	// skip its contents, so an offsetHeight read AFTER setting it returns the
	// contain-intrinsic-size placeholder, which then feeds back as the intrinsic size: scroll
	// height collapses, the scrollbar jumps, the autoscroll observer fires and the run measures a
	// page that is not the page. So heights are read in one pass and written in a second.
	// The placeholder claims 200px, or 0.

	function readHeights(els) {
		var heights = [];
		for (var i = 0; i < els.length; i++) {
			var h = 0;
			try {
				h = Number(els[i].offsetHeight) || 0;
			} catch (e) {
				h = 0;
			}
			heights.push(h);
		}
		return heights;
	}

	function applyB() {
		var info = prefixInfo();
		var roots = info ? info.roots : completedMessageRoots();
		if (roots.length === 0) {
			return result(
				false,
				"no completed message roots matched " + MESSAGE_SELECTOR + "; nothing was ablated",
				0
			);
		}

		// Only code blocks in COMPLETED messages: changing containment on the streaming block would
		// change how streamdown and shiki finalise it, altering the output rather than its cost.
		var blocks = [];
		for (var r = 0; r < roots.length; r++) {
			var bs = codeBlocksIn(roots[r]);
			for (var b = 0; b < bs.length; b++) {
				blocks.push(bs[b]);
			}
		}

		// READ PASS. Every height is captured before any property is written.
		// THE ORDERING BUG THIS AVOIDS: content-visibility:auto makes an off-screen element skip its
		// contents, so offsetHeight collapses to the contain-intrinsic-size placeholder the moment it
		// is set. Read afterwards, that placeholder feeds back as the intrinsic size.
		var blockHeights = readHeights(blocks);
		var rootHeights = readHeights(roots);

		// B writes inline styles either way, so it is EQUIVALENT regardless; the prefix selector is
		// still preferred because it keeps a marker attribute off the code blocks, which the canonical
		// form cannot see (it serialises root attributes and descendant COUNTS).
		if (info) {
			ensureStyleSheet("B", cssPrefixB(info.k));
		} else {
			ensureStyleSheet("B", CSS_B);
			markAll(blocks, "B");
		}
		var i;
		for (i = 0; i < blocks.length; i++) {
			// contain-intrinsic-size has to be inline because it is per element; the stylesheet only carries content-visibility.
			setInline(
				blocks[i],
				"contain-intrinsic-size",
				"auto " + Math.max(0, Math.round(blockHeights[i])) + "px",
				"important"
			);
		}

		var cvBlocks = countComputed(blocks, "content-visibility", "auto");
		if (blocks.length > 0 && cvBlocks === 0) {
			// Expected here: index.css:2537 sets content-visibility:visible !important inside @layer
			// utilities, and a layered important rule beats an unlayered one at any specificity, so the
			// layered copy of CSS_B should win; if not, inline important is the guaranteed escalation.
			counters.inlineFallbacks += 1;
			for (i = 0; i < blocks.length; i++) {
				setInline(blocks[i], "content-visibility", "auto", "important");
			}
			cvBlocks = countComputed(blocks, "content-visibility", "auto");
		}

		for (i = 0; i < roots.length; i++) {
			setInline(roots[i], "content-visibility", "auto", "important");
			setInline(
				roots[i],
				"contain-intrinsic-size",
				"auto " + Math.max(0, Math.round(rootHeights[i])) + "px",
				"important"
			);
		}
		var cvRoots = countComputed(roots, "content-visibility", "auto");

		gauges.cvAutoCodeBlocks = cvBlocks;
		gauges.cvAutoMessages = cvRoots;
		gauges.contentVisibilityAutoConfirmed = cvBlocks + cvRoots;

		if (cvBlocks + cvRoots === 0) {
			return result(
				false,
				"touched " + roots.length + " message roots and " + blocks.length + " code blocks " +
					"and none of them compute to content-visibility:auto; the arm did not fire",
				0
			);
		}
		return result(
			true,
			"content-visibility:auto on " + cvRoots + "/" + roots.length + " message roots and " +
				cvBlocks + "/" + blocks.length + " code blocks; contain-intrinsic-size seeded from " +
				"heights measured before the property was set",
			cvBlocks + cvRoots
		);
	}


	function applyC() {
		var info = prefixInfo();
		var roots;
		var via;
		if (info) {
			roots = info.roots;
			ensureStyleSheet("C", cssPrefixC(info.k));
			via =
				":nth-child(-n+" + info.k + ") prefix stylesheet, no DOM mutation (EXACT)" +
				(info.skipped > 0
					? "; " + info.skipped + " completed root(s) after the first running message " +
					  "were left alone"
					: "");
		} else {
			roots = completedMessageRoots();
			if (roots.length === 0) {
				return result(
					false,
					"no completed message roots matched " + MESSAGE_SELECTOR + "; nothing was ablated",
					0
				);
			}
			ensureStyleSheet("C", CSS_C);
			markAll(roots, "C");
			via =
				"marker-attribute stylesheet: the completed set is not a leading prefix of the " +
				"viewport's children, so " + MARKER_ATTR + " was added and this arm is " +
				"EQUIVALENT, not EXACT";
		}
		var confirmed = countComputed(roots, "display", "none");
		if (confirmed === 0) {
			counters.inlineFallbacks += 1;
			for (var i = 0; i < roots.length; i++) {
				setInline(roots[i], "display", "none", "important");
			}
			confirmed = countComputed(roots, "display", "none");
			via =
				"inline !important, escalated because the stylesheet lost the cascade (this " +
				"writes the style attribute, so the arm is EQUIVALENT, not EXACT)";
		}
		gauges.displayNoneConfirmed = confirmed;
		if (confirmed === 0) {
			return result(
				false,
				"targeted " + roots.length + " completed message roots but none compute to " +
					"display:none; the arm did not fire",
				0
			);
		}
		return result(
			true,
			"display:none via " + via + " on " + confirmed + " of " + roots.length +
				" completed message roots",
			confirmed
		);
	}

	// Two discriminators identify the autoscroll observer, both from
	// use-intent-aware-autoscroll.tsx:502: target has .aui-stream-viewport and options carry
	// "aria-expanded" in attributeFilter. Nothing else in the app observes with aria-expanded.
	// EVERY OTHER observe() CALL MUST PASS THROUGH UNCHANGED (reasoning, research panel, theme
	// toggler, tooltip layer, settings dialog, monitor store, composer pill fit): breaking one
	// turns an ablation into a different page.
	// The wrapper forwards `arguments` verbatim, so invalid calls throw the same TypeError from
	// the same function. The matcher cannot throw: an unreadable options.attributeFilter counts as
	// not-matching and passes through, since passing an observe() through is always safe.

	function targetIsStreamViewport(target) {
		try {
			if (!target || target.nodeType !== 1) {
				return false;
			}
			var list = target.classList;
			if (list && typeof list.contains === "function") {
				return list.contains(VIEWPORT_CLASS) === true;
			}
			var cn = String(target.className || "");
			return (" " + cn + " ").indexOf(" " + VIEWPORT_CLASS + " ") >= 0;
		} catch (e) {
			return false;
		}
	}

	function optionsHaveAriaExpanded(options) {
		try {
			if (!options || typeof options !== "object") {
				return false;
			}
			var filter = options.attributeFilter;
			if (!filter || typeof filter.length !== "number") {
				return false;
			}
			for (var i = 0; i < filter.length; i++) {
				if (String(filter[i]) === "aria-expanded") {
					return true;
				}
			}
			return false;
		} catch (e) {
			return false;
		}
	}

	function installD() {
		var proto = null;
		try {
			proto = W.MutationObserver && W.MutationObserver.prototype;
		} catch (e) {
			proto = null;
		}
		if (!proto) {
			markUnavailable("D", "window.MutationObserver is missing in this browser");
			return false;
		}
		var descriptor = null;
		try {
			descriptor = Object.getOwnPropertyDescriptor(proto, "observe");
		} catch (e) {
			descriptor = null;
		}
		if (!descriptor || typeof descriptor.value !== "function") {
			markUnavailable("D", "MutationObserver.prototype.observe is not a data property");
			return false;
		}
		if (descriptor.configurable !== true && descriptor.writable !== true) {
			markUnavailable(
				"D",
				"MutationObserver.prototype.observe is neither configurable nor writable, so the " +
					"patch cannot be installed on this engine"
			);
			return false;
		}
		var original = descriptor.value;

		function observe(target, options) {
			counters.observeCallsTotal += 1;
			var matched = false;
			try {
				matched = targetIsStreamViewport(target) || optionsHaveAriaExpanded(options);
			} catch (e) {
				matched = false;
			}
			if (matched && state.installed.D) {
				counters.suppressedViewportObserves += 1;
				debug("suppressed autoscroll observe()", target);
				return undefined;
			}
			counters.observeCallsPassedThrough += 1;
			return original.apply(this, arguments);
		}

		try {
			Object.defineProperty(observe, "name", { value: "observe", configurable: true });
			Object.defineProperty(observe, "length", { value: original.length, configurable: true });
		} catch (e) {
			/* cosmetic only */
		}

		try {
			Object.defineProperty(proto, "observe", {
				value: observe,
				writable: descriptor.writable !== false,
				enumerable: descriptor.enumerable === true,
				configurable: true
			});
		} catch (e) {
			markUnavailable("D", "defineProperty on MutationObserver.prototype.observe threw: " + e);
			return false;
		}

		state.originals.observe = original;
		state.originals.observeDescriptor = descriptor;
		state.installed.D = true;
		return true;
	}

	// One custom property, by exact name; every other setProperty call forwards its arguments
	// verbatim so coercion and throwing are unchanged. customPropSets counts every
	// custom-property write including suppressed ones, so a reader can see the rate.

	function installE() {
		var proto = null;
		try {
			proto = W.CSSStyleDeclaration && W.CSSStyleDeclaration.prototype;
		} catch (e) {
			proto = null;
		}
		if (!proto) {
			markUnavailable("E", "window.CSSStyleDeclaration is missing in this browser");
			return false;
		}
		var descriptor = null;
		try {
			descriptor = Object.getOwnPropertyDescriptor(proto, "setProperty");
		} catch (e) {
			descriptor = null;
		}
		if (!descriptor || typeof descriptor.value !== "function") {
			markUnavailable("E", "CSSStyleDeclaration.prototype.setProperty is not a data property");
			return false;
		}
		if (descriptor.configurable !== true && descriptor.writable !== true) {
			markUnavailable(
				"E",
				"CSSStyleDeclaration.prototype.setProperty is neither configurable nor writable"
			);
			return false;
		}
		var original = descriptor.value;

		function setProperty(property, value, priority) {
			counters.setPropertyCallsTotal += 1;
			var name = null;
			try {
				// A Symbol argument makes String() throw; name stays null, the call passes through, and the
				// original throws exactly the TypeError it would have.
				name = typeof property === "string" ? property : String(property);
			} catch (e) {
				name = null;
			}
			if (name !== null && name.charCodeAt(0) === 45 && name.charCodeAt(1) === 45) {
				counters.customPropSets += 1;
			}
			if (name === STABILIZER_PROP && state.installed.E) {
				counters.suppressedStabilizerSets += 1;
				return undefined;
			}
			return original.apply(this, arguments);
		}

		try {
			Object.defineProperty(setProperty, "name", { value: "setProperty", configurable: true });
			Object.defineProperty(setProperty, "length", {
				value: original.length,
				configurable: true
			});
		} catch (e) {
			/* cosmetic only */
		}

		try {
			Object.defineProperty(proto, "setProperty", {
				value: setProperty,
				writable: descriptor.writable !== false,
				enumerable: descriptor.enumerable === true,
				configurable: true
			});
		} catch (e) {
			markUnavailable("E", "defineProperty on CSSStyleDeclaration.prototype.setProperty threw: " + e);
			return false;
		}

		state.originals.setProperty = original;
		state.originals.setPropertyDescriptor = descriptor;
		state.installed.E = true;
		return true;
	}

	// THIS ARM IS DOM_CHANGING: while frozen React renders nothing, so it is not rendering the
	// control's page. Its cost is an UPPER BOUND on reconciliation, never a point estimate;
	// manifest.py prints it as `<= x`.
	// React's browser scheduler is the only runtime handle: ReactDOMRoot is not on window, and the
	// scheduler assigns performWorkUntilDeadline to channel.port1.onmessage, so intercepting that
	// assignment puts a dispatcher in front of it.
	// capturedSchedulerPorts separates "React never used MessageChannel on this build" (0 ports,
	// NOT RUN) from "we froze it and nothing changed".
	// SUPPRESSING SCHEDULER MESSAGES CAN WEDGE REACT PERMANENTLY: the loop posts the next message
	// only from inside the handler it just ran, so revert() posts one fresh kick per channel that
	// had suppressions unless config.redeliverOnUnfreeze is false. Suppressed events are not
	// replayed: performWorkUntilDeadline reads its own queue, so one kick resumes the work.
	// Counted as `schedulerResumeKicks`.

	function instrumentSchedulerPort(channel) {
		var port = channel.port1;
		var stored = null;
		var record = { channel: channel, pendingSuppressed: 0 };

		function dispatch(event) {
			var fn = stored;
			if (typeof fn !== "function") {
				counters.schedulerCallbacksWithoutHandler += 1;
				return undefined;
			}
			if (state.frozen) {
				counters.suppressedSchedulerCallbacks += 1;
				record.pendingSuppressed += 1;
				if (counters.retainedSchedulerEvents < MAX_RETAINED_SCHEDULER_EVENTS) {
					counters.retainedSchedulerEvents += 1;
				} else {
					counters.droppedSchedulerEvents += 1;
				}
				return undefined;
			}
			counters.schedulerCallbacksDelivered += 1;
			// Deliberately not wrapped in try/catch: an exception thrown by React must propagate exactly
			// as it would have, or the frozen run differs for reasons unrelated to the ablation.
			return fn.call(port, event);
		}

		// Assigned through the prototype accessor first so the real port calls the dispatcher; this
		// implicitly starts port1, which the scheduler does a moment later anyway.
		port.onmessage = dispatch;

		Object.defineProperty(port, "onmessage", {
			configurable: true,
			enumerable: true,
			get: function () {
				return stored;
			},
			set: function (fn) {
				stored = fn;
			}
		});

		// Some builds attach with addEventListener instead of onmessage; without this, such a build
		// reports captured ports and zero suppressions, reading as "React did not care" when the
		// freeze never reached the handler.
		try {
			var origAdd = port.addEventListener;
			var origRemove = port.removeEventListener;
			var wrappedByOriginal = typeof WeakMap === "function" ? new WeakMap() : null;
			if (typeof origAdd === "function") {
				Object.defineProperty(port, "addEventListener", {
					configurable: true,
					writable: true,
					enumerable: false,
					value: function (type, listener, opts) {
						if (type === "message" && typeof listener === "function") {
							counters.schedulerListenerAdds += 1;
							var wrapped = function (event) {
								if (state.frozen) {
									counters.suppressedSchedulerCallbacks += 1;
									record.pendingSuppressed += 1;
									if (counters.retainedSchedulerEvents < MAX_RETAINED_SCHEDULER_EVENTS) {
										counters.retainedSchedulerEvents += 1;
									} else {
										counters.droppedSchedulerEvents += 1;
									}
									return undefined;
								}
								counters.schedulerCallbacksDelivered += 1;
								return listener.call(this, event);
							};
							if (wrappedByOriginal) {
								wrappedByOriginal.set(listener, wrapped);
							}
							return origAdd.call(this, type, wrapped, opts);
						}
						return origAdd.apply(this, arguments);
					}
				});
			}
			if (typeof origRemove === "function") {
				Object.defineProperty(port, "removeEventListener", {
					configurable: true,
					writable: true,
					enumerable: false,
					value: function (type, listener, opts) {
						if (type === "message" && typeof listener === "function" && wrappedByOriginal) {
							var wrapped = wrappedByOriginal.get(listener);
							if (wrapped) {
								return origRemove.call(this, type, wrapped, opts);
							}
						}
						return origRemove.apply(this, arguments);
					}
				});
			}
		} catch (e) {
			debug("could not wrap port addEventListener", e);
		}

		if (state.channels.length < MAX_CAPTURED_CHANNELS) {
			state.channels.push(record);
		}
		counters.capturedSchedulerPorts += 1;
	}

	function installF() {
		var Original = null;
		try {
			Original = W.MessageChannel;
		} catch (e) {
			Original = null;
		}
		if (typeof Original !== "function") {
			markUnavailable("F", "window.MessageChannel is missing, so React's scheduler cannot be intercepted");
			return false;
		}

		// Probe before committing: a port that refuses a redefined onmessage cannot be intercepted,
		// and the arm must read UNAVAILABLE rather than run and suppress nothing.
		try {
			var probe = new Original();
			Object.defineProperty(probe.port1, "onmessage", {
				configurable: true,
				enumerable: true,
				get: function () {
					return null;
				},
				set: function () {}
			});
			try {
				probe.port1.close();
				probe.port2.close();
			} catch (e2) {
				/* closing is hygiene, not correctness */
			}
		} catch (e) {
			markUnavailable("F", "MessagePort.onmessage cannot be redefined on this engine: " + e);
			return false;
		}

		function WrappedMessageChannel() {
			var channel = new Original();
			try {
				instrumentSchedulerPort(channel);
			} catch (e) {
				// A channel we failed to instrument is still a working channel: the app must not notice, and
				// capturedSchedulerPorts simply does not count it.
				debug("could not instrument a MessageChannel", e);
			}
			return channel;
		}

		try {
			WrappedMessageChannel.prototype = Original.prototype;
			Object.defineProperty(WrappedMessageChannel, "name", {
				value: "MessageChannel",
				configurable: true
			});
		} catch (e) {
			/* cosmetic only */
		}

		try {
			W.MessageChannel = WrappedMessageChannel;
		} catch (e) {
			markUnavailable("F", "window.MessageChannel is not writable: " + e);
			return false;
		}

		state.originals.MessageChannel = Original;
		state.installed.F = true;
		return true;
	}

	function applyF() {
		if (!state.installed.F) {
			return result(false, unavailable.F || "arm F was not armed at injection time", 0);
		}
		if (state.frozen) {
			return result(false, "already frozen", counters.capturedSchedulerPorts);
		}
		state.frozen = true;
		if (counters.capturedSchedulerPorts === 0) {
			// Not an error yet (React may create its channel later); recorded in the reason so a run
			// ending with zero captured ports reads as NOT RUN.
			return result(
				true,
				"freeze flag set, but no MessageChannel has been created yet; if " +
					"capturedSchedulerPorts is still 0 at the end of the window then React never " +
					"used MessageChannel on this build and the arm did NOT run",
				0
			);
		}
		return result(
			true,
			"scheduler frozen across " + counters.capturedSchedulerPorts + " captured port(s); " +
				"this arm is DOM_CHANGING and its cost is an upper bound only",
			counters.capturedSchedulerPorts
		);
	}

	function revertF() {
		if (!state.installed.F) {
			return revertResult(false, "arm F was never installed");
		}
		if (!state.frozen) {
			return revertResult(true, "not frozen");
		}
		state.frozen = false;
		var kicks = 0;
		var suppressed = 0;
		for (var i = 0; i < state.channels.length; i++) {
			var rec = state.channels[i];
			if (rec.pendingSuppressed <= 0) {
				continue;
			}
			suppressed += rec.pendingSuppressed;
			rec.pendingSuppressed = 0;
			if (!redeliverOnUnfreeze) {
				continue;
			}
			try {
				rec.channel.port2.postMessage(null);
				kicks += 1;
				counters.schedulerResumeKicks += 1;
				counters.schedulerCallbacksRedelivered += 1;
			} catch (e) {
				debug("could not kick a frozen scheduler channel", e);
			}
		}
		if (!redeliverOnUnfreeze) {
			return revertResult(
				true,
				"unfrozen; " + suppressed + " suppressed delivery/deliveries were NOT redelivered " +
					"(config.redeliverOnUnfreeze is false), so React's message loop may stay " +
					"stopped until something else schedules work"
			);
		}
		return revertResult(
			true,
			"unfrozen; " + suppressed + " suppressed delivery/deliveries were not replayed, and " +
				kicks + " fresh scheduler message(s) were posted instead to restart the message " +
				"loop. Anything rendered after this point was rendered late"
		);
	}

	// The viewport carries `scroll-smooth` (thread.tsx:1728), so assigning scrollTop would animate
	// and put scroll-animation cost inside the control; scrollBehavior is forced to auto for the
	// assignment and restored afterwards.
	// The app's own autoscroll may pull the thread back at any time, so controlVisibleMessages is
	// a gauge re-measured on every potency() call: the honest question is whether prior turns were
	// visible for the WINDOW, not for an instant.

	function applyG() {
		var vp = viewportEl();
		if (!vp) {
			return result(false, "no element matched " + VIEWPORT_SELECTOR + "; nothing to scroll", 0);
		}
		var roots = messageRoots();
		if (roots.length === 0) {
			return result(false, "no message roots to bring into view", 0);
		}

		var prevBehavior = "";
		var prevBehaviorPriority = "";
		var hadStyleAttr = true;
		try {
			prevBehavior = vp.style.getPropertyValue("scroll-behavior") || "";
			prevBehaviorPriority = vp.style.getPropertyPriority("scroll-behavior") || "";
			hadStyleAttr = typeof vp.hasAttribute === "function" ? vp.hasAttribute("style") : true;
		} catch (e) {
			prevBehavior = "";
			prevBehaviorPriority = "";
		}
		var prevScrollTop = 0;
		try {
			prevScrollTop = Number(vp.scrollTop) || 0;
		} catch (e) {
			prevScrollTop = 0;
		}
		state.scrollRestore = {
			el: vp,
			scrollTop: prevScrollTop,
			behavior: prevBehavior,
			behaviorPriority: prevBehaviorPriority,
			hadStyleAttr: hadStyleAttr
		};

		try {
			vp.style.setProperty("scroll-behavior", "auto", "important");
		} catch (e) {
			debug("could not force scroll-behavior:auto", e);
		}

		var target = Math.min(controlVisibleTarget, Math.max(1, roots.length - 1));
		var anchorIndex = Math.max(0, roots.length - 1 - target);

		function visibleCounts() {
			var vpRect = rectOf(vp);
			var visible = 0;
			var prior = 0;
			if (vpRect) {
				for (var i = 0; i < roots.length; i++) {
					if (rectsIntersect(rectOf(roots[i]), vpRect)) {
						visible += 1;
						if (i < roots.length - 1) {
							prior += 1;
						}
					}
				}
			}
			return { visible: visible, prior: prior };
		}

		var attempts = 0;
		var seen = visibleCounts();
		try {
			var vpRect0 = rectOf(vp);
			var anchorRect = rectOf(roots[anchorIndex]);
			if (vpRect0 && anchorRect) {
				vp.scrollTop = Number(vp.scrollTop) + (anchorRect.top - vpRect0.top);
			}
		} catch (e) {
			debug("initial scroll assignment failed", e);
		}
		seen = visibleCounts();

		// Walk up until enough prior turns are on screen or the top is reached. Bounded iterations: a
		// non-terminating loop would hang the measured window, and "could not reach the target" is a
		// usable result while a hang is not.
		while (seen.prior < target && attempts < 16) {
			attempts += 1;
			var before = 0;
			try {
				before = Number(vp.scrollTop) || 0;
				var step = Math.max(64, Math.floor((Number(vp.clientHeight) || 400) / 2));
				vp.scrollTop = Math.max(0, before - step);
			} catch (e) {
				break;
			}
			var after = 0;
			try {
				after = Number(vp.scrollTop) || 0;
			} catch (e) {
				after = before;
			}
			seen = visibleCounts();
			if (after === before) {
				break; // already at the top
			}
		}

		try {
			if (prevBehavior === "") {
				vp.style.removeProperty("scroll-behavior");
			} else {
				vp.style.setProperty("scroll-behavior", prevBehavior, prevBehaviorPriority);
			}
			if (!hadStyleAttr) {
				var left = String(vp.getAttribute("style") || "").trim();
				if (left === "") {
					vp.removeAttribute("style");
				}
			}
		} catch (e) {
			debug("could not restore scroll-behavior", e);
		}

		gauges.controlVisibleMessages = seen.visible;
		gauges.controlVisiblePriorMessages = seen.prior;

		if (seen.prior < target) {
			return result(
				false,
				"scrolled to the top of the thread and only " + seen.prior + " prior message " +
					"root(s) intersect the viewport, target was " + target + "; the control is not " +
					"in the state it claims",
				seen.visible
			);
		}
		return result(
			true,
			seen.visible + " message root(s) intersect the viewport, " + seen.prior + " of them " +
				"prior turns (target " + target + "); DOM unchanged, scroll position changed",
			seen.visible
		);
	}

	function revertG() {
		var rec = state.scrollRestore;
		if (!rec) {
			return revertResult(true, "arm G was never applied");
		}
		state.scrollRestore = null;
		try {
			rec.el.style.setProperty("scroll-behavior", "auto", "important");
			rec.el.scrollTop = rec.scrollTop;
			if (rec.behavior === "") {
				rec.el.style.removeProperty("scroll-behavior");
			} else {
				rec.el.style.setProperty("scroll-behavior", rec.behavior, rec.behaviorPriority);
			}
			if (!rec.hadStyleAttr) {
				var left = String(rec.el.getAttribute("style") || "").trim();
				if (left === "") {
					rec.el.removeAttribute("style");
				}
			}
			return revertResult(true, "scroll position restored to " + rec.scrollTop + "px");
		} catch (e) {
			return revertResult(false, "could not restore the scroll position: " + e);
		}
	}


	function apply(armId) {
		var id;
		try {
			id = String(armId).toUpperCase();
		} catch (e) {
			return result(false, "arm id is not a string", 0);
		}
		if (ARM_IDS.indexOf(id) === -1) {
			return result(false, "unknown arm id " + JSON.stringify(id), 0);
		}
		if (Object.prototype.hasOwnProperty.call(unavailable, id)) {
			return result(false, unavailable[id], 0);
		}
		if (Object.prototype.hasOwnProperty.call(state.applied, id) && id !== "F") {
			// `applied` means "this call applied it": arms apply once per measured window, so a second
			// call is a no-op instead of re-marking elements and inflating the counters.
			return result(
				false,
				"arm " + id + " was already applied at " + state.applied[id].at +
					"; arms are applied once per measured window by design",
				state.applied[id].affected
			);
		}

		var out;
		try {
			if (id === "A") {
				out = applyA();
			} else if (id === "B") {
				out = applyB();
			} else if (id === "C") {
				out = applyC();
			} else if (id === "D") {
				out = result(
					true,
					"arm D is a preboot arm and has been active since injection; " +
						counters.suppressedViewportObserves + " viewport observe call(s) suppressed " +
						"so far out of " + counters.observeCallsTotal + " total",
					counters.suppressedViewportObserves
				);
			} else if (id === "E") {
				out = result(
					true,
					"arm E is a preboot arm and has been active since injection; " +
						counters.suppressedStabilizerSets + " stabilizer write(s) suppressed so far " +
						"out of " + counters.customPropSets + " custom property writes",
					counters.suppressedStabilizerSets
				);
			} else if (id === "F") {
				out = applyF();
			} else {
				out = applyG();
			}
		} catch (e) {
			// An arm that throws is reported, not propagated: the app is mid-stream and an exception here
			// would end the run with no data at all.
			return result(false, "arm " + id + " threw while applying: " + e, 0);
		}

		if (out.applied) {
			counters.armsApplied += 1;
			state.applied[id] = { at: nowStamp(), affected: out.affected, reason: out.reason };
		}
		debug("apply", id, out);
		return out;
	}

	function revert(armId) {
		var id;
		try {
			id = String(armId).toUpperCase();
		} catch (e) {
			return revertResult(false, "arm id is not a string");
		}
		if (ARM_IDS.indexOf(id) === -1) {
			return revertResult(false, "unknown arm id " + JSON.stringify(id));
		}

		var out;
		try {
			if (id === "A" || id === "B" || id === "C") {
				removeStyleSheet(id);
				// The inline and marker ledgers are global, not per arm: a run applies one arm, and unwinding
				// everything is least likely to leave a stray !important on a page about to be re-measured.
				restoreInline();
				unmarkAll();
				out = revertResult(true, "stylesheet removed, inline properties and markers restored");
			} else if (id === "D") {
				var hadSuppressed = counters.suppressedViewportObserves > 0;
				if (state.originals.observe && W.MutationObserver) {
					try {
						Object.defineProperty(W.MutationObserver.prototype, "observe", {
							value: state.originals.observe,
							writable: true,
							enumerable: false,
							configurable: true
						});
						state.installed.D = false;
					} catch (e2) {
						/* leave it patched rather than throw */
					}
				}
				out = hadSuppressed
					? revertResult(
							false,
							"the original observe() is restored for FUTURE calls, but the " +
								counters.suppressedViewportObserves + " already-suppressed observe " +
								"call(s) cannot be installed retroactively; the autoscroll observer " +
								"for this thread stays detached for the life of the page"
					  )
					: revertResult(true, "original observe() restored; nothing had been suppressed");
			} else if (id === "E") {
				if (state.originals.setProperty && W.CSSStyleDeclaration) {
					try {
						Object.defineProperty(W.CSSStyleDeclaration.prototype, "setProperty", {
							value: state.originals.setProperty,
							writable: true,
							enumerable: false,
							configurable: true
						});
						state.installed.E = false;
					} catch (e3) {
						/* leave it patched rather than throw */
					}
				}
				out = revertResult(
					true,
					"original setProperty() restored; the " + counters.suppressedStabilizerSets +
						" suppressed write(s) are lost, and " + STABILIZER_PROP + " stays unset " +
						"until the app writes it again"
				);
			} else if (id === "F") {
				out = revertF();
			} else {
				out = revertG();
			}
		} catch (e) {
			return revertResult(false, "arm " + id + " threw while reverting: " + e);
		}

		if (out.reverted) {
			counters.armsReverted += 1;
			try {
				delete state.applied[id];
			} catch (e4) {
				/* nothing to do */
			}
		}
		debug("revert", id, out);
		return out;
	}


	if (armedPreboot.indexOf("D") !== -1) {
		installD();
	} else {
		markUnavailable(
			"D",
			"not armed at injection time: D is a preboot arm and must be listed in " +
				"__sbArmConfig.preboot. It is deliberately NOT installed, because an inactive " +
				"wrapper on MutationObserver.prototype.observe still costs a call frame per " +
				"observe and would perturb the control run"
		);
	}

	if (armedPreboot.indexOf("E") !== -1) {
		installE();
	} else {
		markUnavailable(
			"E",
			"not armed at injection time: E is a preboot arm and must be listed in " +
				"__sbArmConfig.preboot. It is deliberately NOT installed, because an inactive " +
				"wrapper on CSSStyleDeclaration.prototype.setProperty would cost a call frame on " +
				"every style write in the control run"
		);
	}

	if (armedPreboot.indexOf("F") !== -1) {
		installF();
	} else {
		markUnavailable(
			"F",
			"not armed at injection time: F is a preboot arm and must be listed in " +
				"__sbArmConfig.preboot. It is deliberately NOT installed, because wrapping " +
				"MessageChannel changes React's scheduler path even when the freeze is off"
		);
	}

	// Feature checks for the runtime arms: capability questions about the browser, not the page,
	// so a false here is UNAVAILABLE and never a zero difference.
	if (!doc) {
		markUnavailable("A", "no document");
		markUnavailable("B", "no document");
		markUnavailable("C", "no document");
		markUnavailable("G", "no document");
	} else {
		if (!supportsProp("content-visibility", "auto")) {
			markUnavailable(
				"B",
				"this browser does not support content-visibility:auto, so the arm cannot be run " +
					"here at all"
			);
		} else if (!supportsProp("contain-intrinsic-size", "auto 200px")) {
			markUnavailable(
				"B",
				"this browser does not support contain-intrinsic-size with the auto keyword; " +
					"running content-visibility:auto without it would collapse the thread height " +
					"and change the page instead of ablating it"
			);
		}
	}

	var available = [];
	for (var ai = 0; ai < ARM_IDS.length; ai++) {
		if (!Object.prototype.hasOwnProperty.call(unavailable, ARM_IDS[ai])) {
			available.push(ARM_IDS[ai]);
		}
	}

	var api = {
		version: VERSION,
		available: available,
		unavailable: unavailable,
		armedPreboot: armedPreboot,
		apply: apply,
		revert: revert,
		potency: potency,
		counts: counts,
		digest: digest,
		diffKeys: diffKeys,
		// Beyond the required API, additive and safe to ignore.
		diffDetail: diffDetail,
		armIds: ARM_IDS.slice(0),
		prebootArmIds: ["D", "E", "F"],
		domChangingArmIds: ["F"],
		markerAttribute: MARKER_ATTR,
		canonicalFormatVersion: CANONICAL_FORMAT_VERSION,
		hashAlgorithm: "fnv1a32 (non-cryptographic change detector, not a security boundary)",
		installedAt: nowStamp(),
		controlVisibleTarget: controlVisibleTarget,
		selectors: {
			viewport: VIEWPORT_SELECTOR,
			message: MESSAGE_SELECTOR,
			codeBlock: CODE_BLOCK_SELECTOR,
			running: RUNNING_SELECTOR,
			stabilizerProperty: STABILIZER_PROP
		},
		appliedArms: function () {
			var out = [];
			for (var id in state.applied) {
				if (Object.prototype.hasOwnProperty.call(state.applied, id)) {
					out.push(id);
				}
			}
			out.sort();
			return out;
		}
	};

	try {
		W.__sbArms = api;
	} catch (e) {
		/* if we cannot publish, the driver will see __sbArms undefined, which is the honest
		   signal that no arm can be run on this page */
	}

	debug("installed", VERSION, "available:", available, "unavailable:", unavailable);
})();
