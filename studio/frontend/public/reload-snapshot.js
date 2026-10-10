// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

(function () {
  var storageKey = "unsloth.reload-snapshot.v1";
  var maxSnapshotLength = 3 * 1024 * 1024;
  var maxInlineStylesLength = 2 * 1024 * 1024;
  var maxSnapshotAgeMs = 10 * 1000;
  var retainedStyleWaitMs = 500;
  var maxMaterializedMediaPixels = 1500 * 1000;
  var appearanceStorageKey = "unsloth_appearance_customization";
  var maxImportedFonts = 3;
  var maxImportedFontLength = 2200000;
  var maxImportedFontsLength = 4400000;
  var importedFontWaitMs = 250;
  var fontDataUrlPattern =
    /^data:(?:font\/(?:woff2?|ttf|otf|sfnt)|application\/(?:octet-stream|x-font-\w+|font-\w+));base64,[A-Za-z0-9+/=]+$/;
  var overlay = null;
  var retainedSnapshot = null;
  var removalTimer = null;
  var accountStorageKey = "unsloth.browser-account.v1";
  function readAccountMarker() {
    try {
      return localStorage.getItem(accountStorageKey);
    } catch (error) {
      return null;
    }
  }
  var initialAccountMarker = readAccountMarker();
  // Appearance gate attributes from theme-boot.js and appearance-custom-store.ts.
  var appearanceAttributes = [
    "data-chat-font",
    "data-code-font-size",
    "data-contrast-adjust",
    "data-palette",
    "data-ui-font",
    "data-ui-font-size",
  ];
  // Keep in sync with applyCustomizationToDocument; other inline vars are transient.
  var appearanceVariables = [
    "--background",
    "--chart-1",
    "--contrast-control-mix",
    "--contrast-edge-gain",
    "--contrast-fill-mix",
    "--contrast-ink-mix",
    "--contrast-ink-target",
    "--contrast-line-mix",
    "--contrast-panel-ink-target",
    "--contrast-panel-target",
    "--contrast-seam-gain",
    "--contrast-state-mix",
    "--contrast-surface-mix",
    "--contrast-target",
    "--contrast-text-mix",
    "--contrast-wash-gain",
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
    "--foreground-base",
    "--primary",
    "--primary-foreground",
    "--ui-font-size-scale",
    "--ui-interface-scale",
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

  function readInlineStyleSheets() {
    var styles = [];
    var total = 0;
    document
      .querySelectorAll("style[data-vite-dev-id]")
      .forEach(function (style) {
        var text = style.textContent;
        if (
          typeof text !== "string" ||
          !text ||
          total + text.length > maxInlineStylesLength
        ) {
          return;
        }
        total += text.length;
        styles.push(text);
      });
    return styles;
  }

  // `:root` tokens do not reach a shadow tree, so freeze the computed set onto the copy's root.
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
    var linkedStyles =
      snapshot && Array.isArray(snapshot.styles) ? snapshot.styles : [];
    var inlineStyles =
      snapshot && Array.isArray(snapshot.inlineStyles)
        ? snapshot.inlineStyles
        : [];
    if (
      navigationType() !== "reload" ||
      !snapshot ||
      typeof snapshot.createdAt !== "number" ||
      Date.now() - snapshot.createdAt > maxSnapshotAgeMs ||
      snapshot.path !== location.pathname + location.search ||
      typeof snapshot.html !== "string" ||
      !snapshot.html ||
      !linkedStyles.length &&
      !inlineStyles.length
    ) {
      return;
    }

    var fontLoads = registerImportedFonts();
    applyAppearance(snapshot.appearance);
    retainedSnapshot = snapshot;
    overlay = document.createElement("div");
    overlay.className = "reload-snapshot";
    // Dev only: index.css loads after main.tsx, so keep the host full-viewport until then.
    overlay.style.position = "fixed";
    overlay.style.inset = "0";
    overlay.style.zIndex = "2147483647";
    overlay.style.pointerEvents = "none";
    overlay.style.background = "var(--background)";
    overlay.setAttribute("aria-hidden", "true");
    // The copy carries pointer-events-auto classes, so use inert; set the attribute too.
    overlay.inert = true;
    overlay.setAttribute("inert", "");
    // Closed shadow tree keeps the duplicate shell out of document queries like `#root textarea`.
    var shell = overlay.attachShadow({ mode: "closed" });
    var pendingLinkedStyles = 0;
    var linkedStylesRestored = false;
    var shellBody;
    var restoreAfterLinkedStyles = function () {
      if (linkedStylesRestored) return;
      linkedStylesRestored = true;
      if (overlay && shellBody) restoreScrollState(shellBody);
    };
    linkedStyles.forEach(function (href) {
      if (typeof href !== "string" || !href) return;
      var link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = href;
      pendingLinkedStyles += 1;
      link.onload = function () {
        pendingLinkedStyles -= 1;
        if (pendingLinkedStyles === 0) restoreAfterLinkedStyles();
      };
      // A rebuilt bundle renames hashed CSS; drop the overlay rather than show it unstyled.
      link.onerror = removeOverlay;
      shell.appendChild(link);
    });
    inlineStyles.forEach(function (text) {
      if (typeof text !== "string" || !text) return;
      var style = document.createElement("style");
      style.textContent = text;
      shell.appendChild(style);
    });
    // Selectors do not cross the shadow boundary and many rules anchor on `html`, so root in one.
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
    shellBody = document.createElement("body");
    shellBody.innerHTML = snapshot.html;
    shellRoot.appendChild(shellBody);
    shell.appendChild(shellRoot);
    document.documentElement.appendChild(overlay);
    // Arm the fail-open timeout first so a throw below cannot strand the overlay.
    removalTimer = setTimeout(removeOverlay, 5000);
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
    restoreScrollState(shellBody);
    requestAnimationFrame(function () {
      if (overlay) restoreScrollState(shellBody);
    });
    if (pendingLinkedStyles > 0) {
      setTimeout(restoreAfterLinkedStyles, retainedStyleWaitMs);
    }
  }

  // cloneNode copies attributes, not React-driven properties. Detect secrets independently of
  // type: revealed password/token fields can become type=text.
  function isSensitiveField(field) {
    var autocomplete =
      typeof field.autocomplete === "string"
        ? field.autocomplete.toLowerCase()
        : "";
    return (
      field.hasAttribute("data-reload-snapshot-sensitive") ||
      field.type === "password" ||
      field.type === "file" ||
      autocomplete.indexOf("password") !== -1 ||
      autocomplete.indexOf("one-time-code") !== -1 ||
      autocomplete.indexOf("cc-csc") !== -1
    );
  }

  var sensitiveAttributes = ["title", "aria-label", "alt", "placeholder"];

  function mirrorFieldState(original, cloned) {
    var tag = original.tagName;
    if (isSensitiveField(original)) {
      // Secrets can also sit in value attributes, tooltips and accessible names; clear those too.
      cloned.removeAttribute("value");
      sensitiveAttributes.forEach(function (name) {
        cloned.removeAttribute(name);
      });
      if (tag !== "INPUT") cloned.replaceChildren();
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

  function nearestScrollContainer(element) {
    var ancestor = element.parentElement;
    while (ancestor && ancestor !== document.body) {
      if (isScrollContainer(ancestor, getComputedStyle(ancestor))) {
        return ancestor;
      }
      ancestor = ancestor.parentElement;
    }
    return null;
  }

  function isOutsideScrollViewport(bounds, scrollContainer) {
    var containerBounds = scrollContainer.getBoundingClientRect();
    var top = Math.max(0, containerBounds.top);
    var right = Math.min(innerWidth, containerBounds.right);
    var bottom = Math.min(innerHeight, containerBounds.bottom);
    var left = Math.max(0, containerBounds.left);
    return (
      bounds.bottom <= top ||
      bounds.right <= left ||
      bounds.top >= bottom ||
      bounds.left >= right
    );
  }

  function hasVisibleLayoutParent(element, scrollContainer) {
    var parent = element.parentElement;
    while (parent && parent !== scrollContainer) {
      var style = getComputedStyle(parent);
      if (style.display !== "contents") {
        return !isOutsideScrollViewport(
          parent.getBoundingClientRect(),
          scrollContainer,
        );
      }
      parent = parent.parentElement;
    }
    return parent === scrollContainer;
  }

  function replaceWithScrollSpacer(cloned, bounds, scrollContainer) {
    var vertical = scrollContainer.scrollHeight > scrollContainer.clientHeight;
    var axis = vertical ? "vertical" : "horizontal";
    var start = vertical ? bounds.top : bounds.left;
    var end = vertical ? bounds.bottom : bounds.right;
    var crossSize = vertical
      ? bounds.right - bounds.left
      : bounds.bottom - bounds.top;
    var next = cloned.nextElementSibling;
    if (
      next &&
      next.getAttribute("data-reload-spacer-axis") === axis
    ) {
      start = Math.min(
        start,
        Number(next.getAttribute("data-reload-spacer-start")),
      );
      end = Math.max(
        end,
        Number(next.getAttribute("data-reload-spacer-end")),
      );
      crossSize = Math.max(
        crossSize,
        Number(next.getAttribute("data-reload-spacer-cross")),
      );
      next.remove();
    }
    var size = Math.max(0, end - start);
    crossSize = Math.max(0, crossSize);
    cloned.replaceChildren();
    Array.from(cloned.attributes).forEach(function (attribute) {
      cloned.removeAttribute(attribute.name);
    });
    cloned.setAttribute("aria-hidden", "true");
    cloned.setAttribute("data-reload-spacer", "");
    cloned.setAttribute("data-reload-spacer-axis", axis);
    cloned.setAttribute("data-reload-spacer-start", String(start));
    cloned.setAttribute("data-reload-spacer-end", String(end));
    cloned.setAttribute("data-reload-spacer-cross", String(crossSize));
    cloned.setAttribute(
      "style",
      "display:block;box-sizing:border-box;flex:none;margin:0;padding:0;" +
        "border:0;overflow:hidden;" +
        (vertical
          ? "height:" +
            size +
            "px;min-height:" +
            size +
            "px;width:" +
            crossSize +
            "px;max-width:100%;"
          : "width:" +
            size +
            "px;min-width:" +
            size +
            "px;height:" +
            crossSize +
            "px;max-height:100%;"),
    );
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

  function hasSensitiveUrl(value) {
    return /[?&](?:access_token|api[-_]?key|apikey|auth|authorization|code|credential|key|secret|sig|signature|token|x-amz-credential|x-amz-signature|x-goog-signature)=/i.test(
      value,
    );
  }

  function materializeEphemeralMedia(original, cloned, bounds) {
    var tag = original.tagName;
    var source = original.currentSrc || original.getAttribute("src") || "";
    if (source.slice(0, 5) !== "blob:" && !hasSensitiveUrl(source)) return;

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

  // Pixels scale with devicePixelRatio and can pass the cap alone; drop media, keep the layout.
  function dropMaterializedMedia(clone) {
    var dropped = 0;
    clone.querySelectorAll("[src], [poster], [style]").forEach(function (el) {
      ["src", "poster"].forEach(function (name) {
        var value = el.getAttribute(name);
        if (value && value.slice(0, 5) === "data:") {
          el.removeAttribute(name);
          dropped += 1;
        }
      });
      var style = el.getAttribute("style");
      if (style && style.indexOf("url(data:") !== -1) {
        el.setAttribute(
          "style",
          style.replace(/background-image:url\(data:[^)]*\);?/g, ""),
        );
        dropped += 1;
      }
    });
    return dropped;
  }

  function saveSnapshot() {
    if (
      readAccountMarker() !== initialAccountMarker ||
      document.documentElement.hasAttribute("data-reload-snapshot-private")
    ) {
      clearStoredSnapshot();
      return;
    }
    // A second reload before the app is ready: carry the retained snapshot forward verbatim.
    if (overlay && retainedSnapshot) {
      try {
        retainedSnapshot.createdAt = Date.now();
        retainedSnapshot.path = location.pathname + location.search;
        sessionStorage.setItem(storageKey, JSON.stringify(retainedSnapshot));
      } catch (error) {
        clearStoredSnapshot();
      }
      return;
    }
    var root = document.getElementById("root");
    if (!root || !root.firstElementChild) return;
    try {
      // Clone body, not #root: portaled dialogs and menus are part of the visible frame.
      var clone = document.body.cloneNode(true);
      var originalElements = Array.from(document.body.querySelectorAll("*"));
      var clonedElements = Array.from(clone.querySelectorAll("*"));
      for (var index = originalElements.length - 1; index >= 0; index -= 1) {
        var original = originalElements[index];
        var cloned = clonedElements[index];
        if (original.closest("svg")) continue;
        // ChartStyle has no layout box, so exempt it from rectangle pruning.
        if (
          original.tagName === "STYLE" &&
          original.hasAttribute("data-reload-snapshot-style")
        ) {
          continue;
        }
        var style = getComputedStyle(original);
        // `display: contents` wrappers and closed selects paint despite empty rectangles.
        var paintsThroughSelect =
          (original.tagName === "OPTION" ||
            original.tagName === "OPTGROUP") &&
          original.closest("select");
        var laidOut = !paintsThroughSelect && style.display !== "contents";
        var bounds = laidOut ? original.getBoundingClientRect() : null;
        // Coalesce offscreen subtrees into spacers so the scroll geometry and visible slice stay put.
        var scrollContainer = nearestScrollContainer(original);
        var outsideViewport =
          laidOut &&
          (scrollContainer
            ? isOutsideScrollViewport(bounds, scrollContainer)
            : bounds.bottom <= 0 ||
              bounds.right <= 0 ||
              bounds.top >= innerHeight ||
              bounds.left >= innerWidth);
        if (
          style.display === "none" ||
          style.visibility === "hidden" ||
          outsideViewport
        ) {
          if (
            scrollContainer &&
            style.display !== "none" &&
            style.visibility !== "hidden" &&
            hasVisibleLayoutParent(original, scrollContainer)
          ) {
            replaceWithScrollSpacer(
              cloned,
              bounds,
              scrollContainer,
            );
          } else {
            cloned.remove();
          }
        } else {
          mirrorFieldState(original, cloned);
          mirrorScrollState(original, cloned);
          materializeEphemeralMedia(original, cloned, bounds);
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
        // IDs are scoped to the closed shadow root, so keep them for SVG url(#id) and aria refs.
        element.removeAttribute("autofocus");
        element.removeAttribute("srcdoc");
        // SVG <use> carries URLs on xlink:href, which a plain href lookup misses.
        [
          "src",
          "srcset",
          "poster",
          "href",
          "xlink:href",
          "action",
          "formaction",
        ].forEach(function (name) {
          var value = element.getAttribute(name);
          if (
            value &&
            (value.indexOf("blob:") !== -1 ||
              /^\s*javascript:/i.test(value) ||
              hasSensitiveUrl(value))
          ) {
            element.removeAttribute(name);
          }
        });
        Array.from(element.attributes).forEach(function (attribute) {
          if (attribute.name.toLowerCase().startsWith("on")) {
            element.removeAttribute(attribute.name);
          }
        });
      });
      clone.querySelectorAll("[data-reload-spacer]").forEach(function (element) {
        element.removeAttribute("data-reload-spacer-axis");
        element.removeAttribute("data-reload-spacer-start");
        element.removeAttribute("data-reload-spacer-end");
        element.removeAttribute("data-reload-spacer-cross");
      });
      var html = clone.innerHTML;
      if (html && html.length > maxSnapshotLength && dropMaterializedMedia(clone)) {
        html = clone.innerHTML;
      }
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
          inlineStyles: readInlineStyleSheets(),
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
    retainedSnapshot = null;
    if (!overlay) return;
    overlay.remove();
    overlay = null;
  }

  window.addEventListener("pageswap", function (event) {
    if (event.activation && event.activation.navigationType === "reload") {
      saveSnapshot();
    }
  });
  // Engines without pageswap still fire pagehide; do not register both (Chromium fires both).
  // pagehide also fires on navigation; the restore side discards those via navigationType.
  if (!("onpageswap" in window)) {
    window.addEventListener("pagehide", function (event) {
      if (!event.persisted) saveSnapshot();
    });
  }
  window.addEventListener("unsloth:app-shell-ready", function () {
    if (!overlay) return;
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        removeOverlay();
      });
    });
  });

  // Fail open: the copy is a nicety, the real document is not.
  try {
    restoreSnapshot();
  } catch (error) {
    removeOverlay();
  }
})();
