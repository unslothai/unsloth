// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

(function () {
  var storageKey = "unsloth.reload-snapshot.v1";
  var maxSnapshotLength = 3 * 1024 * 1024;
  var maxSnapshotAgeMs = 10 * 1000;
  var overlay = null;
  var removalTimer = null;
  // Appearance customization reaches the page as inline custom properties on
  // <html> plus these gate attributes (applyCustomizationToDocument in
  // src/features/settings/stores/appearance-custom-store.ts). theme-boot.js
  // resolves only mode and palette, so without carrying them across the reload
  // the restored shell paints in stock colors and restyles once React runs.
  var appearanceAttributes = [
    "data-chat-font",
    "data-code-font-size",
    "data-contrast-adjust",
    "data-ui-font",
    "data-ui-font-size",
  ];

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

  function readStyleSheets() {
    var hrefs = [];
    document
      .querySelectorAll('link[rel="stylesheet"]')
      .forEach(function (link) {
        if (link.href) hrefs.push(link.href);
      });
    return hrefs;
  }

  function readAppearance() {
    var root = document.documentElement;
    var variables = {};
    for (var index = 0; index < root.style.length; index += 1) {
      var name = root.style[index];
      if (name.slice(0, 2) === "--") {
        variables[name] = root.style.getPropertyValue(name);
      }
    }
    var attributes = {};
    appearanceAttributes.forEach(function (name) {
      var value = root.getAttribute(name);
      if (value !== null) attributes[name] = value;
    });
    return { variables: variables, attributes: attributes };
  }

  // React re-applies every one of these once it mounts, so writing them onto
  // the live <html> only brings that forward; nothing here has to be undone
  // when the overlay goes.
  function applyAppearance(appearance) {
    if (!appearance) return;
    var root = document.documentElement;
    var variables = appearance.variables || {};
    Object.keys(variables).forEach(function (name) {
      if (name.slice(0, 2) === "--" && typeof variables[name] === "string") {
        root.style.setProperty(name, variables[name]);
      }
    });
    var attributes = appearance.attributes || {};
    appearanceAttributes.forEach(function (name) {
      if (typeof attributes[name] === "string") {
        root.setAttribute(name, attributes[name]);
      }
    });
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
      !snapshot.html ||
      !Array.isArray(snapshot.styles) ||
      !snapshot.styles.length
    ) {
      return;
    }

    applyAppearance(snapshot.appearance);
    overlay = document.createElement("div");
    overlay.className = "reload-snapshot";
    overlay.setAttribute("aria-hidden", "true");
    overlay.inert = true;
    // A closed shadow tree keeps the copy out of every document query. The
    // markup is a duplicate of the live shell, so leaving it in the page tree
    // makes `#root textarea` (and the UI tests that wait on one) ambiguous.
    var shell = overlay.attachShadow({ mode: "closed" });
    snapshot.styles.forEach(function (href) {
      var link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = href;
      // A rebuilt bundle renames its hashed CSS, so the shell would come back
      // unstyled. Drop it and let the real document through instead.
      link.onerror = removeOverlay;
      shell.appendChild(link);
    });
    // Selectors do not cross the shadow boundary, so the copy carries both the
    // <html> classes it was styled by (dark mode, cursors, font smoothing) and
    // the marker index.css hangs the shell's own rules off.
    var shellRoot = document.createElement("div");
    shellRoot.className =
      "reload-snapshot-shell " +
      (typeof snapshot.rootClass === "string" ? snapshot.rootClass : "");
    shellRoot.innerHTML = snapshot.html;
    shell.appendChild(shellRoot);
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
        // A `display: contents` wrapper generates no box, so its rectangle is
        // empty however much of the viewport its children fill. Judging it by
        // that rectangle takes the whole visible subtree with it.
        var laidOut = style.display !== "contents";
        var bounds = laidOut ? original.getBoundingClientRect() : null;
        if (
          style.display === "none" ||
          style.visibility === "hidden" ||
          (laidOut &&
            (bounds.bottom <= 0 ||
              bounds.right <= 0 ||
              bounds.top >= innerHeight ||
              bounds.left >= innerWidth))
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
          appearance: readAppearance(),
          styles: readStyleSheets(),
          rootClass: document.documentElement.className,
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
