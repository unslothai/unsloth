// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

(function () {
  var storageKey = "unsloth.reload-snapshot.v1";
  var maxSnapshotLength = 3 * 1024 * 1024;
  var maxSnapshotAgeMs = 10 * 1000;
  var overlay = null;
  var removalTimer = null;
  // Appearance reaches the page as inline custom properties on <html> plus
  // these gate attributes (theme-boot.js for the palette,
  // applyCustomizationToDocument in
  // src/features/settings/stores/appearance-custom-store.ts for the rest).
  // theme-boot.js resolves only mode and palette, so without carrying them
  // across the reload the restored shell paints in stock colors and restyles
  // once React runs.
  var appearanceAttributes = [
    "data-chat-font",
    "data-code-font-size",
    "data-contrast-adjust",
    "data-palette",
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

  // Design tokens the app defines through `:root` do not reach a shadow tree:
  // `:root` matches only a document's root element, and Tailwind's `:host`
  // block redeclares some of them on the host. Freeze the computed set and put
  // it on the copy's own root instead.
  function readTokens() {
    var style = getComputedStyle(document.documentElement);
    var tokens = {};
    for (var index = 0; index < style.length; index += 1) {
      var name = style[index];
      if (name.slice(0, 2) === "--") {
        tokens[name] = style.getPropertyValue(name);
      }
    }
    return tokens;
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

  function applyAppearanceAttributes(element, appearance) {
    var attributes = (appearance && appearance.attributes) || {};
    appearanceAttributes.forEach(function (name) {
      if (typeof attributes[name] === "string") {
        element.setAttribute(name, attributes[name]);
      }
    });
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
    applyAppearanceAttributes(root, appearance);
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
    // Selectors do not cross the shadow boundary, and 80-odd rules are anchored
    // on `html` (light/dark theming above all), so the copy is rooted in an
    // <html> element carrying the classes and gate attributes it was styled by,
    // plus the marker index.css hangs the shell's own rules off.
    var shellRoot = document.createElement("html");
    shellRoot.className =
      "reload-snapshot-shell " +
      (typeof snapshot.rootClass === "string" ? snapshot.rootClass : "");
    applyAppearanceAttributes(shellRoot, snapshot.appearance);
    var tokens = snapshot.tokens || {};
    Object.keys(tokens).forEach(function (name) {
      if (name.slice(0, 2) === "--" && typeof tokens[name] === "string") {
        shellRoot.style.setProperty(name, tokens[name]);
      }
    });
    // Global typography and foreground styles are applied to body, not html.
    // Recreate that inheritance boundary inside the shadow tree rather than
    // hanging the app subtree directly off the synthetic html element.
    var shellBody = document.createElement("body");
    shellBody.innerHTML = snapshot.html;
    shellRoot.appendChild(shellBody);
    shell.appendChild(shellRoot);
    document.documentElement.appendChild(overlay);
    removalTimer = setTimeout(removeOverlay, 5000);
  }

  // React drives value/checked/selected as DOM properties and cloneNode copies
  // attributes, so a populated composer or a ticked box would come back empty.
  // Secret inputs must be identified independently of their presentation type:
  // password and token fields can temporarily become type=text when revealed.
  function isSensitiveInput(input) {
    var autocomplete =
      typeof input.autocomplete === "string"
        ? input.autocomplete.toLowerCase()
        : "";
    return (
      input.type === "password" ||
      input.hasAttribute("data-reload-snapshot-sensitive") ||
      autocomplete.indexOf("password") !== -1 ||
      autocomplete.indexOf("one-time-code") !== -1 ||
      autocomplete.indexOf("cc-csc") !== -1
    );
  }

  function mirrorFieldState(original, cloned) {
    var tag = original.tagName;
    if (tag === "TEXTAREA") {
      cloned.textContent = original.value;
    } else if (tag === "INPUT") {
      if (isSensitiveInput(original)) {
        // cloneNode may retain a value attribute from an earlier controlled
        // render, so remove it as well as declining to mirror the live value.
        cloned.removeAttribute("value");
        return;
      }
      cloned.setAttribute("value", original.value);
      if (original.checked) cloned.setAttribute("checked", "");
      else cloned.removeAttribute("checked");
    } else if (tag === "OPTION") {
      if (original.selected) cloned.setAttribute("selected", "");
      else cloned.removeAttribute("selected");
    }
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
        } else {
          mirrorFieldState(original, cloned);
        }
      }
      clone
        .querySelectorAll("iframe, object, embed, script, style, link, base")
        .forEach(function (element) {
          element.remove();
      });
      clone.querySelectorAll("*").forEach(function (element) {
        // IDs are scoped to the closed shadow root, so they cannot collide with
        // the live document. Keep them for internal references such as
        // SVG fill="url(#gradient-id)" and aria-labelledby.
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
          tokens: readTokens(),
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
