// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

(function () {
  var storageKey = "unsloth.reload-snapshot.v1";
  var maxSnapshotLength = 3 * 1024 * 1024;
  var maxSnapshotAgeMs = 10 * 1000;
  var maxMaterializedMediaPixels = 1500 * 1000;
  var appearanceStorageKey = "unsloth_appearance_customization";
  var maxImportedFonts = 3;
  var maxImportedFontLength = 2200000;
  var maxImportedFontsLength = 4400000;
  var importedFontWaitMs = 250;
  var fontDataUrlPattern =
    /^data:(?:font\/(?:woff2?|ttf|otf|sfnt)|application\/(?:octet-stream|x-font-\w+|font-\w+));base64,[A-Za-z0-9+/=]+$/;
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
  // Keep this in sync with applyCustomizationToDocument. Other inline root
  // variables are transient runtime state (for example an in-progress panel
  // resize) and must not be replayed into the replacement document.
  var appearanceVariables = [
    "--background",
    "--chart-1",
    "--contrast-mix",
    "--contrast-target",
    "--control-accent",
    "--control-accent-foreground",
    "--custom-chat-font",
    "--custom-code-font",
    "--custom-code-font-size",
    "--custom-heading-font",
    "--font-heading",
    "--font-mono",
    "--font-sans",
    "--foreground",
    "--primary",
    "--primary-foreground",
    "--ui-font-scale",
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
    appearanceVariables.forEach(function (name) {
      var value = root.style.getPropertyValue(name);
      if (value) variables[name] = value;
    });
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
    appearanceVariables.forEach(function (name) {
      if (typeof variables[name] === "string") {
        root.style.setProperty(name, variables[name]);
      }
    });
    applyAppearanceAttributes(root, appearance);
  }

  function registerImportedFonts() {
    var loads = [];
    if (
      typeof FontFace !== "function" ||
      !document.fonts ||
      typeof document.fonts.add !== "function"
    ) {
      return loads;
    }
    try {
      var raw = localStorage.getItem(appearanceStorageKey);
      if (!raw || raw.length > maxImportedFontsLength + 100000) return loads;
      var persisted = JSON.parse(raw);
      var customization =
        persisted && persisted.state && persisted.state.customization;
      if (!customization || !Array.isArray(customization.importedFonts)) {
        return loads;
      }
      var selected = {};
      ["uiFont", "headingFont", "chatFont", "codeFont"].forEach(
        function (key) {
          var name = customization[key];
          if (typeof name === "string") selected[name] = true;
        },
      );
      var total = 0;
      var seen = {};
      customization.importedFonts
        .slice(0, maxImportedFonts)
        .forEach(function (font) {
          var name = font && font.name;
          var dataUrl = font && font.dataUrl;
          if (
            typeof name !== "string" ||
            !selected[name] ||
            seen[name] ||
            !name ||
            name.length > 100 ||
            /[;{}()<>"'\\/,\x60\x00-\x1f\x7f]/.test(name) ||
            typeof dataUrl !== "string" ||
            dataUrl.length > maxImportedFontLength ||
            total + dataUrl.length > maxImportedFontsLength ||
            !fontDataUrlPattern.test(dataUrl)
          ) {
            return;
          }
          total += dataUrl.length;
          seen[name] = true;
          try {
            var face = new FontFace(name, "url(" + dataUrl + ")");
            document.fonts.add(face);
            loads.push(face.load());
          } catch (error) {}
        });
    } catch (error) {}
    return loads;
  }

  function restoreScrollState(root) {
    root
      .querySelectorAll(
        "[data-reload-scroll-top], [data-reload-scroll-left]",
      )
      .forEach(function (element) {
        var top = Number(element.getAttribute("data-reload-scroll-top"));
        var left = Number(element.getAttribute("data-reload-scroll-left"));
        if (isFinite(top)) element.scrollTop = top;
        if (isFinite(left)) element.scrollLeft = left;
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

    var fontLoads = registerImportedFonts();
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
    if (fontLoads.length) shellRoot.style.visibility = "hidden";
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
    if (fontLoads.length) {
      var revealShell = function () {
        shellRoot.style.visibility = "visible";
      };
      Promise.race([
        Promise.all(
          fontLoads.map(function (load) {
            return load.catch(function () {});
          }),
        ),
        new Promise(function (resolve) {
          setTimeout(resolve, importedFontWaitMs);
        }),
      ]).then(revealShell, revealShell);
    }
    // Apply once against the current layout, then once more in the last
    // pre-paint frame after the retained stylesheets have resolved.
    restoreScrollState(shellBody);
    requestAnimationFrame(function () {
      if (overlay) restoreScrollState(shellBody);
    });
    removalTimer = setTimeout(removeOverlay, 5000);
  }

  // React drives value/checked/selected as DOM properties and cloneNode copies
  // attributes, so a populated composer or a ticked box would come back empty.
  // Secret inputs must be identified independently of their presentation type:
  // password and token fields can temporarily become type=text when revealed.
  function isSensitiveField(field) {
    var autocomplete =
      typeof field.autocomplete === "string"
        ? field.autocomplete.toLowerCase()
        : "";
    return (
      field.hasAttribute("data-reload-snapshot-sensitive") ||
      field.type === "password" ||
      autocomplete.indexOf("password") !== -1 ||
      autocomplete.indexOf("one-time-code") !== -1 ||
      autocomplete.indexOf("cc-csc") !== -1
    );
  }

  function mirrorFieldState(original, cloned) {
    var tag = original.tagName;
    if (isSensitiveField(original)) {
      // cloneNode can retain input attributes, textarea text, or a secret
      // rendered into an ordinary code/span subtree.
      cloned.removeAttribute("value");
      if (tag !== "INPUT") cloned.textContent = "";
      return;
    }
    if (tag === "TEXTAREA") {
      cloned.textContent = original.value;
    } else if (tag === "INPUT") {
      cloned.setAttribute("value", original.value);
      if (original.checked) cloned.setAttribute("checked", "");
      else cloned.removeAttribute("checked");
    } else if (tag === "OPTION") {
      if (original.selected) cloned.setAttribute("selected", "");
      else cloned.removeAttribute("selected");
    }
  }

  function hasClippingOverflow(value) {
    return (
      value === "auto" ||
      value === "scroll" ||
      value === "overlay" ||
      value === "hidden"
    );
  }

  function isScrollContainer(element, style) {
    return (
      (hasClippingOverflow(style.overflowY) &&
        element.scrollHeight > element.clientHeight) ||
      (hasClippingOverflow(style.overflowX) &&
        element.scrollWidth > element.clientWidth)
    );
  }

  function hasScrollContainerAncestor(element) {
    var ancestor = element.parentElement;
    while (ancestor && ancestor !== document.body) {
      if (isScrollContainer(ancestor, getComputedStyle(ancestor))) return true;
      ancestor = ancestor.parentElement;
    }
    return false;
  }

  function mirrorScrollState(original, cloned) {
    if (original.scrollTop) {
      cloned.setAttribute("data-reload-scroll-top", String(original.scrollTop));
    }
    if (original.scrollLeft) {
      cloned.setAttribute(
        "data-reload-scroll-left",
        String(original.scrollLeft),
      );
    }
  }

  function capturePixels(original, sourceWidth, sourceHeight, bounds) {
    if (
      !bounds ||
      bounds.bottom <= 0 ||
      bounds.right <= 0 ||
      bounds.top >= innerHeight ||
      bounds.left >= innerWidth
    ) {
      return null;
    }
    if (!sourceWidth || !sourceHeight) return null;
    try {
      var pixelRatio = window.devicePixelRatio || 1;
      var scale = Math.min(
        1,
        ((bounds.right - bounds.left) * pixelRatio) / sourceWidth,
        ((bounds.bottom - bounds.top) * pixelRatio) / sourceHeight,
      );
      var width = Math.max(1, Math.round(sourceWidth * scale));
      var height = Math.max(1, Math.round(sourceHeight * scale));
      if (width * height > maxMaterializedMediaPixels) {
        var pixelScale = Math.sqrt(
          maxMaterializedMediaPixels / (width * height),
        );
        width = Math.max(1, Math.round(width * pixelScale));
        height = Math.max(1, Math.round(height * pixelScale));
      }
      var canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      var context = canvas.getContext("2d");
      if (!context) throw new Error("Canvas 2D context unavailable");
      context.drawImage(original, 0, 0, width, height);
      var dataUrl = canvas.toDataURL("image/webp", 0.82);
      if (!dataUrl || dataUrl === "data:,") throw new Error("Empty media frame");
      return dataUrl;
    } catch (error) {
      return null;
    }
  }

  function materializeBlobMedia(original, cloned, bounds) {
    var tag = original.tagName;
    var source = original.currentSrc || original.getAttribute("src") || "";
    if (source.slice(0, 5) !== "blob:") return;

    // Audio controls remain a useful visual shell without their expiring
    // source. Images and video frames can additionally carry their rendered
    // pixels across the document boundary through a bounded data URL.
    if (tag !== "IMG" && tag !== "VIDEO") {
      cloned.removeAttribute("src");
      return;
    }
    var sourceWidth = tag === "IMG" ? original.naturalWidth : original.videoWidth;
    var sourceHeight =
      tag === "IMG" ? original.naturalHeight : original.videoHeight;
    var dataUrl = capturePixels(original, sourceWidth, sourceHeight, bounds);
    if (!dataUrl) {
      cloned.removeAttribute("src");
    } else if (tag === "IMG") {
      cloned.setAttribute("src", dataUrl);
      cloned.removeAttribute("srcset");
    } else {
      cloned.setAttribute("poster", dataUrl);
      cloned.removeAttribute("src");
    }
  }

  function materializeCanvas(original, cloned, bounds) {
    if (original.tagName !== "CANVAS") return;
    var dataUrl = capturePixels(
      original,
      original.width,
      original.height,
      bounds,
    );
    if (!dataUrl) return;
    var inlineStyle = cloned.getAttribute("style") || "";
    if (inlineStyle && inlineStyle.slice(-1) !== ";") inlineStyle += ";";
    cloned.setAttribute(
      "style",
      inlineStyle +
        "background-image:url(" +
        dataUrl +
        ");background-size:100% 100%;background-repeat:no-repeat;",
    );
  }

  function saveSnapshot() {
    if (
      document.documentElement.hasAttribute("data-reload-snapshot-private")
    ) {
      clearStoredSnapshot();
      return;
    }
    var root = document.getElementById("root");
    if (!root || !root.firstElementChild) return;
    try {
      // Clone the body's rendered surface, not just #root: dialogs, menus and
      // other primitives portal beside the app root and are part of the frame
      // the user sees. Active content is stripped before serialization below.
      var clone = document.body.cloneNode(true);
      var originalElements = Array.from(document.body.querySelectorAll("*"));
      var clonedElements = Array.from(clone.querySelectorAll("*"));
      for (var index = originalElements.length - 1; index >= 0; index -= 1) {
        var original = originalElements[index];
        var cloned = clonedElements[index];
        if (original.closest("svg")) continue;
        // ChartStyle is passive, component-generated CSS needed by the cloned
        // SVG. It has no layout box, so keep it out of rectangle pruning; every
        // unmarked style is still removed by the sanitizer below.
        if (
          original.tagName === "STYLE" &&
          original.hasAttribute("data-reload-snapshot-style")
        ) {
          continue;
        }
        var style = getComputedStyle(original);
        // A `display: contents` wrapper generates no box, so its rectangle is
        // empty however much of the viewport its children fill. Judging it by
        // that rectangle takes the whole visible subtree with it.
        var laidOut = style.display !== "contents";
        var bounds = laidOut ? original.getBoundingClientRect() : null;
        // Removing children from a scroll container changes its scroll geometry
        // and can shift the visible slice. Keep that subtree intact and restore
        // the container's offsets in the retained shell instead.
        var insideScrollContainer = hasScrollContainerAncestor(original);
        if (
          style.display === "none" ||
          style.visibility === "hidden" ||
          (!insideScrollContainer &&
            laidOut &&
            (bounds.bottom <= 0 ||
              bounds.right <= 0 ||
              bounds.top >= innerHeight ||
              bounds.left >= innerWidth))
        ) {
          cloned.remove();
        } else {
          mirrorFieldState(original, cloned);
          mirrorScrollState(original, cloned);
          materializeBlobMedia(original, cloned, bounds);
          materializeCanvas(original, cloned, bounds);
        }
      }
      clone
        .querySelectorAll("iframe, object, embed, script, style, link, base")
        .forEach(function (element) {
          if (
            element.tagName === "STYLE" &&
            element.hasAttribute("data-reload-snapshot-style")
          ) {
            return;
          }
          element.remove();
      });
      clone.querySelectorAll("*").forEach(function (element) {
        // IDs are scoped to the closed shadow root, so they cannot collide with
        // the live document. Keep them for internal references such as
        // SVG fill="url(#gradient-id)" and aria-labelledby.
        element.removeAttribute("autofocus");
        element.removeAttribute("srcdoc");
        ["src", "srcset", "poster"].forEach(function (name) {
          var value = element.getAttribute(name);
          if (value && value.indexOf("blob:") !== -1) {
            element.removeAttribute(name);
          }
        });
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
