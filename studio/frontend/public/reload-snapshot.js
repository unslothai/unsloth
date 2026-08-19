// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

(function () {
  var storageKey = "unsloth.reload-snapshot.v1";
  var maxSnapshotLength = 3 * 1024 * 1024;
  var maxSnapshotAgeMs = 10 * 1000;
  var overlay = null;
  var removalTimer = null;

  function clearStoredSnapshot() {
    try {
      sessionStorage.removeItem(storageKey);
    } catch (error) {}
  }

  function readStoredSnapshot() {
    try {
      var value = sessionStorage.getItem(storageKey);
      sessionStorage.removeItem(storageKey);
      return value ? JSON.parse(value) : null;
    } catch (error) {
      clearStoredSnapshot();
      return null;
    }
  }

  function navigationType() {
    try {
      var entries = performance.getEntriesByType("navigation");
      return entries.length ? entries[0].type : null;
    } catch (error) {
      return null;
    }
  }

  function restoreSnapshot() {
    var snapshot = readStoredSnapshot();
    if (
      navigationType() !== "reload" ||
      !snapshot ||
      typeof snapshot.createdAt !== "number" ||
      Date.now() - snapshot.createdAt > maxSnapshotAgeMs ||
      snapshot.path !== location.pathname + location.search ||
      typeof snapshot.html !== "string" ||
      !snapshot.html
    ) {
      return;
    }

    overlay = document.createElement("div");
    overlay.className = "reload-snapshot";
    overlay.setAttribute("aria-hidden", "true");
    overlay.inert = true;
    overlay.innerHTML = snapshot.html;
    document.documentElement.appendChild(overlay);
    removalTimer = setTimeout(removeOverlay, 5000);
  }

  function saveSnapshot() {
    var root = document.getElementById("root");
    if (!root || !root.firstElementChild) return;
    try {
      var clone = root.cloneNode(true);
      var originalElements = Array.from(root.querySelectorAll("*"));
      var clonedElements = Array.from(clone.querySelectorAll("*"));
      for (var index = originalElements.length - 1; index >= 0; index -= 1) {
        var original = originalElements[index];
        var cloned = clonedElements[index];
        if (original.closest("svg")) continue;
        var style = getComputedStyle(original);
        var bounds = original.getBoundingClientRect();
        if (
          style.display === "none" ||
          style.visibility === "hidden" ||
          bounds.bottom <= 0 ||
          bounds.right <= 0 ||
          bounds.top >= innerHeight ||
          bounds.left >= innerWidth
        ) {
          cloned.remove();
        }
      }
      clone
        .querySelectorAll("iframe, object, embed, script, style, link, base")
        .forEach(function (element) {
          element.remove();
        });
      clone.querySelectorAll("*").forEach(function (element) {
        element.removeAttribute("id");
        element.removeAttribute("autofocus");
        element.removeAttribute("srcdoc");
        Array.from(element.attributes).forEach(function (attribute) {
          if (attribute.name.toLowerCase().startsWith("on")) {
            element.removeAttribute(attribute.name);
          }
        });
      });
      var html = clone.innerHTML;
      if (!html || html.length > maxSnapshotLength) {
        clearStoredSnapshot();
        return;
      }
      sessionStorage.setItem(
        storageKey,
        JSON.stringify({
          createdAt: Date.now(),
          path: location.pathname + location.search,
          html: html,
        }),
      );
    } catch (error) {
      clearStoredSnapshot();
    }
  }

  function removeOverlay() {
    if (removalTimer !== null) {
      clearTimeout(removalTimer);
      removalTimer = null;
    }
    if (!overlay) return;
    overlay.remove();
    overlay = null;
  }

  window.addEventListener("pageswap", function (event) {
    if (event.activation && event.activation.navigationType === "reload") {
      saveSnapshot();
    }
  });
  window.addEventListener("unsloth:app-shell-ready", function () {
    if (!overlay) return;
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        removeOverlay();
      });
    });
  });

  restoreSnapshot();
})();
