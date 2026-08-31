// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * knobs.js -- runtime ablation knobs for the SHIPPED Unsloth Studio build.
 *
 * Injected with Playwright's context.add_init_script BEFORE the app boots, so it can patch
 * browser APIs the app is about to use. It needs no compiler, no source edit and no rebuild: the
 * point is that an external tester can run every ablation against the same production bundle the
 * user gets, and that the ablated run and the control run are the same bytes of application code.
 *
 * WHAT AN ARM IS FOR
 *
 * Each arm removes one candidate mechanism and nothing else. If removing it makes the thread
 * cheaper, that mechanism was the cost, and the arm names the fix. The arms:
 *
 *   A  visibility:hidden on completed messages.
 *      Removes paint and raster for the prefix. Keeps layout, keeps the DOM, keeps React.
 *      IF A IS THE WIN: the cost is paint/raster of off-screen content. The fix is to stop
 *      painting the prefix, not to stop building it.
 *
 *   B  content-visibility:auto plus contain-intrinsic-size on completed messages, and the undo
 *      of the index.css override that force-disables content-visibility on code blocks.
 *      Removes style, layout AND paint for anything off screen.
 *      IF B IS THE WIN: the cost is off-screen style plus layout plus paint, and the rule at
 *      studio/frontend/src/index.css:2537 is wrong -- it was kept for a flicker, and the comment
 *      above it asserts containment on message roots "was measured as no help", which is exactly
 *      the claim this arm re-tests. B beating A says the expensive part is not the pixels.
 *
 *   C  display:none on completed messages.
 *      Removes layout geometry and the sibling count as well as everything B removes.
 *      IF C IS THE WIN AND B IS NOT: the cost is layout geometry and the cost of having N
 *      siblings in one flow, not per-element style. That points at the thread container's layout
 *      mode (flex column over N children) rather than at the messages.
 *
 *   D  detach the autoscroll subtree MutationObserver.
 *      use-intent-aware-autoscroll.tsx:502 observes the viewport with subtree+characterData, and
 *      its callback reads el.scrollHeight, which forces synchronous style+layout of the WHOLE
 *      thread on every delivery. During streaming those deliveries arrive per streamed character.
 *      IF D IS THE WIN: the cost is forced synchronous layout inside the autoscroll observer.
 *      The fix is to stop reading scrollHeight per mutation (rAF-coalesce it, or switch to
 *      IntersectionObserver / overflow-anchor), not to touch the messages at all.
 *
 *   E  neutralise the --aui-scroll-stabilizer custom property write.
 *      That same callback writes a custom property on the viewport element. A custom property is
 *      inherited, so writing it on the scroll container invalidates inherited style for the whole
 *      subtree beneath it, which is every message.
 *      IF E IS THE WIN: the cost is inherited custom-property subtree style invalidation. The fix
 *      is to register the property with syntax and inherits:false via CSS.registerProperty, or to
 *      move the write off the ancestor of the thread.
 *
 *   F  freeze React's scheduler while leaving the DOM exactly as it is.
 *      React's browser scheduler drives work through a MessageChannel: port1.onmessage is
 *      performWorkUntilDeadline, and port2.postMessage schedules the next slice. Suppressing the
 *      delivery stops reconciliation, effects and store subscriptions without touching a node.
 *      IF F IS THE WIN: the cost is React subscriptions and reconciliation over the prefix, and
 *      the fix is memoisation / virtualisation / narrower context subscriptions, not CSS.
 *
 *   G  CONTROL. Identical DOM, no knob, thread scrolled so prior turns are actually IN the
 *      viewport. Every other arm removes work that only exists when the prefix is off screen and
 *      cheap; G is the run where the prefix is on screen and expensive. It is the calibration for
 *      "how much does this page cost when nothing is being skipped", and a G that is not slower
 *      than the untouched baseline means the baseline was never skipping anything, which would
 *      invalidate the whole comparison.
 *
 * THE TWO FAILURE MODES THIS FILE EXISTS TO PREVENT
 *
 * See arms/manifest.py. An ablation lies in exactly two ways and both are silent.
 *
 *   1. THE ARM CHANGED THE OUTPUT. The knob also removed some of the work, so the two sides did
 *      not render the same page. The dangerous case is the small invisible difference: an earlier
 *      stub fixture in this codebase produced 552 syntax-highlighted spans where the real page
 *      produces 2,561, and it read as a clean 4x win. It was not fast, it was rendering a fifth
 *      of the content. digest() and counts() exist for that: the canonical serialisation carries
 *      the code-block count and the span count per message, so a highlighting difference is
 *      caught even when every character of text is identical.
 *
 *      TWO THINGS THE DIGEST CANNOT SEE, stated here because they cannot be stated in it. It
 *      walks `[data-message-id]` roots and serialises their attributes and text plus the count of
 *      code blocks and of spans inside them. It does NOT serialise the attributes of descendants,
 *      so an inline style written on a code block is invisible to it (arm B does exactly that,
 *      which is one reason B is EQUIVALENT and not EXACT). And it does not know about geometry,
 *      so a change that only moves pixels is invisible too. Neither gap is patched by making the
 *      digest bigger; they are declared, because a normaliser nobody wrote down is just a bug.
 *
 *      Arms A and C therefore go out of their way to mutate NOTHING: they target the completed
 *      prefix with `:nth-child(-n+K)` against the viewport's direct children rather than by
 *      tagging elements, so the raw digest is byte-identical across the arm and EXACT is a claim
 *      they can actually make. When that is impossible they fall back to a marker attribute and
 *      say in the returned reason that they have dropped to EQUIVALENT.
 *
 *   2. THE ARM DID NOT FIRE. Selector matched nothing, patch failed to install, run completed,
 *      difference was zero, and "no effect" got written down as evidence against the mechanism.
 *      It is evidence of nothing. So every arm has a potency counter that the arm CAUSES: not
 *      "the stylesheet was appended" but "N elements actually compute to visibility:hidden".
 *      A patch that could not be installed lands in unavailable[] and never in available[], so
 *      UNAVAILABLE can never be misread as "had no effect".
 *
 * CASCADE FACTS THIS FILE DEPENDS ON, VERIFIED AGAINST THE SPEC
 *
 * The index.css override lives inside `@layer utilities { ... }` (Tailwind v4 emits native
 * cascade layers). Per CSS Cascading and Inheritance Level 5 and MDN:
 *
 *   - For IMPORTANT declarations the layer order is REVERSED, and important declarations in ANY
 *     layer beat important declarations outside every layer. So appending an unlayered
 *     `!important` rule, at any specificity, LOSES to the layered `!important` rule it is trying
 *     to undo. The obvious approach silently does nothing, which is failure mode 2.
 *   - Element-attached (inline) declarations are sorted BEFORE layers in the cascade, so an
 *     inline `!important` declaration beats every other author declaration whatever its layer.
 *
 * Therefore each CSS arm applies its stylesheet in two copies, one unlayered and one inside
 * `@layer utilities` so it competes in the same layer as the rule it must beat (same layer, same
 * importance, higher specificity, later order -> wins), THEN verifies with getComputedStyle, and
 * only if the verification says the rule did not take effect does it escalate to inline
 * !important, counting the escalation. The verification is the whole point: the stylesheet is a
 * hypothesis about the cascade, the computed style is the observation.
 *
 * ARMS ARE APPLIED ONCE, OVER THE PREFIX THAT EXISTS AT THAT MOMENT
 *
 * apply() runs at the start of a measured window and never again. Messages created during the
 * window are NOT ablated. That is deliberate and not a limitation:
 *
 *   - the hypothesis is about the PREFIX (the N completed turns that are off screen), not about
 *     the turn being streamed, so ablating the in-flight message would change the thing under
 *     measurement rather than the thing under test;
 *   - a live MutationObserver re-applying the knob to each new message would add per-mutation
 *     work to the hot path, which is the exact category of cost the experiment is trying to
 *     remove. An ablation that installs its own observer to remove an observer measures itself.
 *
 * For the same reason arm B marks only code blocks inside COMPLETED messages: touching the code
 * block that is still streaming would change how streamdown and shiki finalise it.
 *
 * PREBOOT VERSUS RUNTIME
 *
 * D, E and F patch APIs the app captures during boot, so they must be installed before the app
 * runs and they are read from window.__sbArmConfig.preboot, set by an earlier init script. An arm
 * that is not listed is NOT INSTALLED AT ALL. Installing a patch and leaving it inactive is not
 * free: the wrapper costs a call frame on every MutationObserver.observe or setProperty in the
 * process, and a control run carrying that overhead is not a control. So the preboot list decides
 * per page load, and a run whose config did not arm D reports D as unavailable with a reason,
 * never as an arm that ran and did nothing.
 *
 * D and E are ACTIVE FROM INJECTION when armed; there is no meaningful apply() for them, because
 * the autoscroll observer is installed once during mount and the suppression has to be in place
 * before that. apply("D") therefore reports the arm as already active and returns the suppression
 * count so far. F is different: the port capture is preboot, the freeze is toggled by apply("F").
 *
 * ARM F IS DOM_CHANGING. Freezing React stops the stream from rendering, so the frozen run is not
 * rendering the same page as the control -- it is rendering less of it. Its cost is therefore an
 * upper bound on what React reconciliation costs and can never be quoted as a point estimate.
 * manifest.py prints it as `<= x`. This is stated here as well because the arm is the one most
 * likely to produce a spectacular number that somebody wants to quote.
 *
 * COUNTERS: EVENT COUNTERS VERSUS GAUGES
 *
 * potency() returns integers only, in two kinds, and the distinction is load-bearing:
 *
 *   EVENT COUNTERS are monotonic and never decrease for the life of the page:
 *     suppressedViewportObserves, observeCallsTotal, observeCallsPassedThrough,
 *     suppressedStabilizerSets, customPropSets, setPropertyCallsTotal,
 *     suppressedSchedulerCallbacks, schedulerCallbacksDelivered, capturedSchedulerPorts, and the
 *     bookkeeping counters (armsApplied, inlineFallbacks, ...).
 *
 *   GAUGES are point-in-time measurements of the live DOM, re-measured on every potency() call
 *   unless {live:false} is passed: visibilityHiddenConfirmed, cvAutoMessages, cvAutoCodeBlocks,
 *   contentVisibilityAutoConfirmed, displayNoneConfirmed, controlVisibleMessages,
 *   controlVisiblePriorMessages.
 *
 * Gauges are deliberately not monotonic. A stored 0 and a measured 0 are different facts, and
 * this file exists to keep them different: measuring live means the "before" read is a real
 * observation of the untouched page rather than a variable that has not been written yet, and it
 * means arm G is checked against what the page looks like at the END of the window, after the
 * app's own autoscroll has had every chance to undo the scroll position G set. Re-measuring calls
 * getComputedStyle over every message root, which forces style recalc, so potency() must be
 * called OUTSIDE the timed window; pass {live:false} to read the last stored values with no DOM
 * access at all.
 *
 * HASHING
 *
 * FNV-1a, 32 bit, hex. NOT CRYPTOGRAPHIC and not a security boundary. It is a change detector
 * against an adversary that does not exist, and 32 bits is small enough that a collision on a
 * multi-megabyte canonical form is not impossible, so digest() also returns canonicalLength and
 * rawLength. Compare the pair, not the hash alone.
 */

(function () {
	"use strict";

	if (typeof window === "undefined" || !window) {
		return;
	}

	var W = window;

	// Idempotence. add_init_script runs for every document including iframes, and a driver that
	// reloads the page must not stack patches: a doubly wrapped observe() would double every
	// count, and a doubly wrapped MessageChannel would make the freeze flag ambiguous.
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

	// ------------------------------------------------------------------------------------
	// Config
	//
	// Read defensively. A missing or malformed __sbArmConfig means no preboot arms, which is the
	// correct default: no config is not permission to install patches into a control run.
	// ------------------------------------------------------------------------------------

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

	// ------------------------------------------------------------------------------------
	// State
	// ------------------------------------------------------------------------------------

	function freshEventCounters() {
		return {
			// D
			observeCallsTotal: 0,
			observeCallsPassedThrough: 0,
			suppressedViewportObserves: 0,
			// E
			setPropertyCallsTotal: 0,
			customPropSets: 0,
			suppressedStabilizerSets: 0,
			// F
			capturedSchedulerPorts: 0,
			schedulerCallbacksDelivered: 0,
			suppressedSchedulerCallbacks: 0,
			schedulerCallbacksRedelivered: 0,
			schedulerResumeKicks: 0,
			retainedSchedulerEvents: 0,
			droppedSchedulerEvents: 0,
			schedulerListenerAdds: 0,
			schedulerCallbacksWithoutHandler: 0,
			// bookkeeping
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
		applied: {},          // armId -> {at, affected, reason}
		frozen: false,
		markers: [],          // {el, prev} for MARKER_ATTR restoration
		inline: [],           // {el, prop, prevValue, prevPriority, hadStyleAttr}
		sheets: {},           // armId -> <style> element
		channels: [],         // captured MessageChannel records
		scrollRestore: null,  // arm G restoration record
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

	// ------------------------------------------------------------------------------------
	// DOM helpers
	//
	// All of them tolerate a missing document, a detached node and a selector that matches
	// nothing, and none of them throw. A query that fails returns an empty array, and the caller
	// turns that into a reason string rather than into a zero.
	// ------------------------------------------------------------------------------------

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
		// The ONLY DOM signal of an in-flight message is a descendant (or the root itself)
		// carrying [data-status="running"]; status.type from assistant-ui is one of
		// running|complete|incomplete|requires-action.
		try {
			if (!el) {
				return false;
			}
			if (typeof el.getAttribute === "function" && el.getAttribute("data-status") === "running") {
				return true;
			}
			return queryOne(RUNNING_SELECTOR, el) !== null;
		} catch (e) {
			// Unknown status is treated as running, i.e. as NOT ablatable. Erring the other way
			// would ablate the message being streamed and contaminate the measurement.
			return true;
		}
	}

	// The completed-message filter is done in JS, one querySelector per root, and NOT with a CSS
	// `:has()` selector. `:has()` is unsupported on older engines and an unsupported compound
	// selector makes the WHOLE selector list invalid, so the rule would be dropped at parse time,
	// match nothing, and the arm would report a clean zero difference. That is precisely the
	// did-not-fire failure this file is built to make impossible, and it is invisible in a
	// screenshot, in a trace and in the console.
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
		// No CSS.supports means we cannot prove support. Treat as supported and let the computed
		// check decide: a false "unavailable" would hide an arm that works.
		return true;
	}

	// ------------------------------------------------------------------------------------
	// Markers and inline styles
	//
	// Arms A, B and C need a CSS hook on specific elements, and the only way to have a stylesheet
	// address "these elements and no others" is an attribute the stylesheet can select. One
	// attribute name is used for all arms, space separated, so [data-sb-arm~="A"] matches. That
	// attribute IS a change to the DOM, so it shows up in digest().raw. It is meant to: an arm
	// that mutates attributes is EQUIVALENT, not EXACT, and the Python side declares
	// `attr:data-sb-arm` (plus `attr:style` when the inline escalation fires) as its allowed diff.
	// Pretending otherwise by hiding the marker from the raw digest would defeat the check.
	// ------------------------------------------------------------------------------------

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
						// Leaving style="" behind would be a permanent, invisible difference in
						// every future digest of this page.
						rec.el.removeAttribute("style");
					}
				}
			} catch (e) {
				/* best effort */
			}
		}
		state.inline = [];
	}

	// ------------------------------------------------------------------------------------
	// Stylesheets
	//
	// Every arm sheet is emitted TWICE: once unlayered, once inside `@layer utilities`.
	//
	// The rule that has to be beaten (index.css:2536-2538) is inside `@layer utilities`, and for
	// important declarations layered beats unlayered no matter the specificity. The layered copy
	// therefore competes in the same layer as its target, where the ordinary rules apply again:
	// same layer, same importance, higher specificity, later in document order, so it wins. The
	// unlayered copy covers the case where the build has no layers at all (then there is nothing
	// layered to lose to, and the unlayered copy is the one that applies). Emitting both is safe
	// because the two copies carry identical declarations, so whichever wins produces the same
	// computed value.
	//
	// Appending to <head> puts the sheet last among same-layer rules, which settles order of
	// appearance. None of this is trusted: applyX() checks getComputedStyle afterwards.
	// ------------------------------------------------------------------------------------

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

	// ------------------------------------------------------------------------------------
	// Two ways to address "the completed messages" from a stylesheet, and why the first one is
	// worth the extra code.
	//
	// PREFIX PATH (preferred). Messages are DIRECT CHILDREN of the viewport; there is no
	// message-list wrapper element. The message being streamed is the last one. So the completed
	// messages are a leading run of the viewport's children and
	//
	//     .aui-thread-viewport.aui-stream-viewport > [data-message-id]:nth-child(-n+K)
	//
	// selects exactly them, with K computed in JS as the child index of the first running message
	// root. This mutates NOTHING. No attribute is added, no inline style is written, and the raw
	// digest of the thread is byte-identical before and after, so arms A and C can honestly claim
	// EXACT invariance. The `[data-message-id]` part is not redundant: nth-child alone would also
	// match spacers, sentinels and buttons that share the viewport, and hiding one of those is a
	// change to the rendered page that the digest cannot see because the digest only walks
	// message roots.
	//
	// MARKER PATH (fallback). If the completed set is not a leading prefix (a message somewhere
	// in the middle is still running, which assistant-ui allows in principle), or the viewport
	// cannot be found, the arm falls back to a `data-sb-arm` attribute on each completed root.
	// That is a real DOM mutation, so the arm drops from EXACT to EQUIVALENT and the returned
	// reason string says so in as many words. It is reported rather than hidden because a
	// silently-EQUIVALENT arm quoted as EXACT is a false number, while a loudly-EQUIVALENT one is
	// just a number with a declared diff.
	// ------------------------------------------------------------------------------------

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

	// Returns the leading run of viewport children that contains no running message, or null when
	// the prefix path is not usable. `k` is a 1-based nth-child bound; `roots` are the message
	// roots inside it, all of them completed by construction.
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

	// ------------------------------------------------------------------------------------
	// Hashing and canonical serialisation
	// ------------------------------------------------------------------------------------

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

	// FNV-1a, 32 bit, over UTF-16 code units taken low byte first. Non-cryptographic: this is a
	// change detector, not a security boundary. Compare it together with canonicalLength.
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

	// Field encoding. JSON escapes quotes, backslashes, newlines and control characters; the pipe
	// is escaped on top of that so no encoded field can ever contain the field separator, which
	// makes the canonical form parseable by a plain split() in diffKeys().
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
		// Parse with the browser's own CSS parser on a detached element rather than by splitting
		// on ";" and ":", which breaks on url(data:...) and on any value containing a semicolon.
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

	// One DOM walk, reused for both serialisations. Walking twice would be slower and, worse,
	// would let the page change between the raw pass and the normalised pass, producing a "raw
	// differs, normalised does not" that is an artefact of the instrument.
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
			// `raw` skips nothing at all: it is the answer to "did anything about the rendered
			// output change", including the arm's own marker attribute. An EXACT arm that touches
			// an attribute is not an EXACT arm, and this is where that gets caught.
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
					// A prefix is retained because it is useful when debugging, but `truncated`
					// is set and diffKeys() refuses to answer from a truncated pair. A quiet cut
					// would turn "we did not look at the rest" into "the rest was identical".
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

	// ------------------------------------------------------------------------------------
	// diffKeys
	//
	// Compares the RAW canonical forms, not the normalised ones. The normalised hashes already
	// answer "are these equivalent"; what manifest.py needs from here is the other question,
	// "what actually differs", so it can check the observed diff against the diff the arm
	// DECLARED it would produce. An arm that declares one difference and produces two is voided,
	// and comparing normalised forms would hide the second one, since the normaliser is exactly
	// the thing that removes the declared difference.
	//
	// Keys are collapsed to a stable vocabulary (`attr:<name>`, `text`, `tag`,
	// `structure:<what>`) rather than per-message paths, because a declared diff has to be
	// writable in advance and `#msg-8fc2.attr.style` is not knowable in advance.
	//
	// Fails closed: if the canonical form was not retained, or was truncated, or cannot be
	// parsed, the returned array contains a key beginning with `__unavailable:` which can never
	// appear in a declared diff, so the arm is voided instead of silently passing.
	// ------------------------------------------------------------------------------------

	function parseCanonical(text) {
		var entries = {};   // entryId -> {key: value}
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
	// verdict who needs to know WHICH message drifted and by how much.
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

	// ------------------------------------------------------------------------------------
	// counts()
	// ------------------------------------------------------------------------------------

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
			// Beyond the required keys, and required by the zero discipline: visibleMessages == 0
			// because the thread viewport was not found is a different fact from
			// visibleMessages == 0 because nothing is on screen, and the two must not print the
			// same.
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
					// A shiki token is a leaf span with text. Both numbers are reported: a
					// fixture that renders a fifth of the highlighting moves both, and a change
					// in span NESTING moves only codeSpans, so keeping them separate says which
					// of the two happened.
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

	// ------------------------------------------------------------------------------------
	// potency()
	// ------------------------------------------------------------------------------------

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

	// ------------------------------------------------------------------------------------
	// Arm A -- visibility:hidden on completed messages
	// ------------------------------------------------------------------------------------

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
			// The stylesheet lost the cascade. Inline important is sorted before layers and beats
			// every author rule, so this is the escalation that cannot lose.
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

	// ------------------------------------------------------------------------------------
	// Arm B -- content-visibility:auto plus contain-intrinsic-size
	//
	// Part (i) restores content-visibility on code blocks, undoing index.css:2536-2538.
	// Part (ii) puts content-visibility:auto and contain-intrinsic-size:auto <measured>px inline
	// on each completed message root.
	//
	// THE ORDERING BUG THIS FUNCTION IS SHAPED TO AVOID: content-visibility:auto makes an
	// off-screen element skip its contents, so the moment it is set, offsetHeight collapses to the
	// contain-intrinsic-size placeholder. Reading offsetHeight after setting the property captures
	// the placeholder, which is then fed back as the intrinsic size, and every element ends up
	// claiming to be 200px (or 0) tall. The thread's scroll height collapses, the scrollbar jumps,
	// the autoscroll observer fires on the reflow, and the run measures a page that is not the
	// page. So the heights are all read in one pass FIRST and only then written in a second pass.
	// Two passes also avoid read/write layout thrash, but that is a bonus, not the reason.
	// ------------------------------------------------------------------------------------

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

		// Only code blocks inside COMPLETED messages. The block that is still streaming must not
		// be touched: changing containment on it would change how streamdown and shiki finalise
		// it, which is a change to the output, not an ablation of cost.
		var blocks = [];
		for (var r = 0; r < roots.length; r++) {
			var bs = codeBlocksIn(roots[r]);
			for (var b = 0; b < bs.length; b++) {
				blocks.push(bs[b]);
			}
		}

		// READ PASS. Every height is captured before any property is written.
		var blockHeights = readHeights(blocks);
		var rootHeights = readHeights(roots);

		// WRITE PASS. B writes inline styles on the roots either way, so it is EQUIVALENT
		// whatever happens here; the prefix selector is still preferred because it keeps the
		// code blocks free of a marker attribute that the digest cannot see (the canonical form
		// serialises message-root attributes and DESCENDANT COUNTS, not descendant attributes).
		if (info) {
			ensureStyleSheet("B", cssPrefixB(info.k));
		} else {
			ensureStyleSheet("B", CSS_B);
			markAll(blocks, "B");
		}
		var i;
		for (i = 0; i < blocks.length; i++) {
			// contain-intrinsic-size has to be inline anyway because it is per element, and the
			// stylesheet only carries content-visibility.
			setInline(
				blocks[i],
				"contain-intrinsic-size",
				"auto " + Math.max(0, Math.round(blockHeights[i])) + "px",
				"important"
			);
		}

		var cvBlocks = countComputed(blocks, "content-visibility", "auto");
		if (blocks.length > 0 && cvBlocks === 0) {
			// Expected on this build: index.css:2537 sets content-visibility:visible !important
			// from inside @layer utilities, and for important declarations a layered rule beats
			// an unlayered one at any specificity. The layered copy of CSS_B should win; if it
			// did not, inline important is the escalation that the cascade guarantees.
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

	// ------------------------------------------------------------------------------------
	// Arm C -- display:none on completed messages
	// ------------------------------------------------------------------------------------

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

	// ------------------------------------------------------------------------------------
	// Arm D -- detach the autoscroll subtree observer (PREBOOT)
	//
	// Two discriminators identify the autoscroll observer, and both come from
	// use-intent-aware-autoscroll.tsx:502: the target carries the aui-stream-viewport class, and
	// the options carry "aria-expanded" in attributeFilter. Nothing else in the app observes with
	// aria-expanded, and .aui-stream-viewport marks only real streaming-thread viewports.
	//
	// EVERY OTHER observe() CALL MUST PASS THROUGH UNCHANGED. reasoning.tsx,
	// research-activity-panel.tsx, animated-theme-toggler.tsx, tooltip-modal-layer.ts (twice),
	// settings-dialog.tsx, monitor-frame-store.ts and use-composer-pill-fit.ts all install
	// observers, and breaking any of them would change what the app renders, which converts an
	// ablation into a different page.
	//
	// The wrapper forwards `arguments` verbatim to the original, so calls with missing or invalid
	// arguments throw the same TypeError from the same function they always did. The matching
	// logic cannot throw: if reading options.attributeFilter fails for any reason, the call is
	// treated as not-matching and passes through, because passing an observe() through is always
	// safe and suppressing one that should not have been suppressed is not.
	// ------------------------------------------------------------------------------------

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

	// ------------------------------------------------------------------------------------
	// Arm E -- neutralise the --aui-scroll-stabilizer write (PREBOOT)
	//
	// One custom property, by exact name. Every other setProperty call, custom property or not,
	// passes through with its arguments forwarded verbatim so that coercion and throwing are
	// unchanged. customPropSets counts every custom property write including the suppressed ones,
	// which is the context that tells a reader whether the stabilizer was one write in ten or one
	// in ten thousand.
	// ------------------------------------------------------------------------------------

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
				// A Symbol argument makes String() throw. In that case name stays null, the call
				// passes through, and the original throws exactly the TypeError it would have.
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

	// ------------------------------------------------------------------------------------
	// Arm F -- freeze the React scheduler, keep the DOM (PREBOOT capture, apply() freezes)
	//
	// THIS ARM IS DOM_CHANGING. While frozen, React renders nothing, so the stream stops
	// appearing on screen and the frozen side is not rendering the same page as the control. Its
	// cost is an UPPER BOUND on what React reconciliation costs and can never be quoted as a point
	// estimate; manifest.py prints it as `<= x`. It is here because a bound on React is still
	// worth having when the alternative is guessing, not because the number is clean.
	//
	// React's browser scheduler is the only runtime handle available. The ReactDOMRoot object is
	// not exposed on window, so root.unmount() is not reachable, and #root is just an element.
	// The scheduler does: const channel = new MessageChannel(); channel.port1.onmessage =
	// performWorkUntilDeadline; ... port2.postMessage(null). Intercepting the assignment to
	// port1.onmessage puts a dispatcher in front of performWorkUntilDeadline, and the freeze flag
	// decides whether the dispatcher forwards.
	//
	// capturedSchedulerPorts is what separates "React never used MessageChannel on this build"
	// (0 ports: NOT RUN, the treatment never happened) from "we froze it and nothing changed"
	// (ports captured, callbacks suppressed).
	//
	// SUPPRESSING SCHEDULER MESSAGES CAN WEDGE REACT PERMANENTLY. The message loop only posts the
	// next message from inside the handler it just ran, so a swallowed delivery ends the loop and
	// unfreezing alone does not restart it. revert() therefore posts one fresh message per
	// channel that had suppressions (schedulerResumeKicks) unless config.redeliverOnUnfreeze is
	// false, and says so in the returned reason. The suppressed events themselves are NOT
	// replayed: React's performWorkUntilDeadline ignores the event object and reads its own
	// queue, so one kick resumes exactly the work the swallowed messages would have done, and
	// replaying N events would run N slices that no longer correspond to anything.
	// ------------------------------------------------------------------------------------

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
			// Called without a try/catch on purpose: an exception thrown by React must propagate
			// exactly as it would have. Swallowing it here would make the frozen run behave
			// differently from the control in a way that has nothing to do with the ablation.
			return fn.call(port, event);
		}

		// Assigning through the prototype accessor first, so the real port ends up calling the
		// dispatcher. This implicitly starts port1, which the scheduler does a moment later
		// anyway when it assigns its own handler.
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

		// Some builds attach with addEventListener instead of onmessage. Without this, such a
		// build would report captured ports and zero suppressions, which reads as "we froze it
		// and React did not care" when the truth is that the freeze never reached the handler.
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

		// Probe before committing: if the port instance refuses a redefined onmessage there is no
		// way to intercept, and the arm must read UNAVAILABLE rather than run and suppress
		// nothing.
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
				// A channel we failed to instrument is still a working channel. The app must not
				// notice, and capturedSchedulerPorts simply does not count it.
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
			// Not an error yet: React may create its channel later. It is recorded in the reason
			// so that a run which ends with zero captured ports is read as NOT RUN.
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

	// ------------------------------------------------------------------------------------
	// Arm G -- control: identical DOM, prior turns in the viewport
	//
	// The viewport carries the `scroll-smooth` class (thread.tsx:1728), so a scrollTop assignment
	// would animate. An animated scroll would still be running when the measured window starts,
	// which would put scroll animation cost inside the control and make the control the most
	// expensive arm. scrollBehavior is forced to auto for the assignment and restored afterwards.
	//
	// The app's own autoscroll may pull the thread back to the bottom at any time, which is why
	// controlVisibleMessages is a gauge that is re-measured on every potency() call rather than a
	// number recorded once at apply time: the honest question is whether the prior turns were
	// visible for the WINDOW, not whether they were visible for an instant.
	// ------------------------------------------------------------------------------------

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

		// Walk upwards until enough prior turns are on screen or the top is reached. Bounded
		// iterations: a loop that cannot terminate would hang the measured window, and reporting
		// "could not reach the target" is a usable result while a hang is not.
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

	// ------------------------------------------------------------------------------------
	// apply / revert
	// ------------------------------------------------------------------------------------

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
			// `applied` means "this call applied it". Arms are applied once per measured window
			// by design, so a second call is a no-op and says so rather than re-marking elements
			// and inflating the counters.
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
			// An arm that throws has to be reported, not propagated: the app is mid-stream and an
			// exception here would end the run with no data at all.
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
				// The inline and marker ledgers are global rather than per arm: a run applies one
				// arm, and unwinding everything is the behaviour least likely to leave a stray
				// !important behind on a page that is about to be measured again.
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

	// ------------------------------------------------------------------------------------
	// Install the preboot arms, then publish
	// ------------------------------------------------------------------------------------

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

	// Feature checks for the runtime arms. These are capability questions about the browser, not
	// about the page, so a false here is UNAVAILABLE and never a zero difference.
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
