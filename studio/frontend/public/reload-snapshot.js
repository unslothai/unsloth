// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

(function () {
  var storageKey = "unsloth.reload-snapshot.v1";
  var chatHistoryDisabled =
    window.__UNSLOTH_NO_CHAT_HISTORY__ === true ||
    document.documentElement.getAttribute("data-unsloth-no-chat-history") === "true";
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
  // Appearance is inline custom properties on <html> plus these gate
  // attributes, written by theme-boot.js (mode and palette) and
  // applyCustomizationToDocument in
  // src/features/settings/stores/appearance-custom-store.ts (the rest).
  // Uncarried, the copy paints in stock colors until React restyles it.
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
    if (chatHistoryDisabled) {
      clearStoredSnapshot();
      return null;
    }
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

  // React re-applies all of these on mount, so this only brings that forward.
  // Nothing here needs undoing when the overlay goes.
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
    // Vite injects index.css only after main.tsx runs, and styles inside the
    // closed shadow tree cannot match its host. Keep the host full-viewport
    // during that development-only gap; index.css repeats this for production.
    overlay.style.position = "fixed";
    overlay.style.inset = "0";
    overlay.style.zIndex = "2147483647";
    overlay.style.pointerEvents = "none";
    overlay.style.background = "var(--background)";
    overlay.setAttribute("aria-hidden", "true");
    // pointer-events: none on the host is not enough, since the copy carries
    // the app's own pointer-events-auto classes. inert is. The property is a
    // silent expando where unsupported, so set the attribute too.
    overlay.inert = true;
    overlay.setAttribute("inert", "");
    // A closed shadow tree keeps the copy out of every document query. The
    // markup is a duplicate of the live shell, so leaving it in the page tree
    // makes `#root textarea` (and the UI tests that wait on one) ambiguous.
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
      // A rebuilt bundle renames its hashed CSS, so the shell would come back
      // unstyled. Drop it and let the real document through instead.
      link.onerror = removeOverlay;
      shell.appendChild(link);
    });
    inlineStyles.forEach(function (text) {
      if (typeof text !== "string" || !text) return;
      var style = document.createElement("style");
      style.textContent = text;
      shell.appendChild(style);
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
    // Global typography and foreground styles hang off body, not html, so the
    // copy needs that inheritance boundary rather than a bare html root.
    shellBody = document.createElement("body");
    shellBody.innerHTML = snapshot.html;
    shellRoot.appendChild(shellBody);
    shell.appendChild(shellRoot);
    document.documentElement.appendChild(overlay);
    // Arm the fail-open timeout before anything else can throw. Once the host
    // is in the document, a throw below would otherwise strand it.
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
    // Apply immediately and on the next frame for inline styles, then once
    // linked styles settle (or after a bounded wait) for their final geometry.
    restoreScrollState(shellBody);
    requestAnimationFrame(function () {
      if (overlay) restoreScrollState(shellBody);
    });
    if (pendingLinkedStyles > 0) {
      setTimeout(restoreAfterLinkedStyles, retainedStyleWaitMs);
    }
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
      // cloneNode retains input attributes, textarea text, and any secret in a
      // code/span subtree. The same value usually also sits in a tooltip or an
      // accessible name, which clearing children alone would leave behind.
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
    // Dropping a URL only costs the copy an image, so err towards dropping.
    return /[?&](?:access_token|api[-_]?key|apikey|auth|authorization|code|credential|key|secret|sig|signature|token|x-amz-credential|x-amz-signature|x-goog-signature)=/i.test(
      value,
    );
  }

  function materializeEphemeralMedia(original, cloned, bounds) {
    var tag = original.tagName;
    var source = original.currentSrc || original.getAttribute("src") || "";
    if (source.slice(0, 5) !== "blob:" && !hasSensitiveUrl(source)) return;

    // Audio controls still read as a shell without their expiring source.
    // Images and video can carry rendered pixels as a bounded data URL; if a
    // cross-origin frame refuses canvas capture, the URL is dropped, not kept.
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

  // Pixels are rasterized at devicePixelRatio, so one page costs 4x on a 2x
  // display and can pass the cap alone. Dropping them keeps the layout, which
  // is the point of the copy; dropping the snapshot puts the blank flash back.
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
      chatHistoryDisabled ||
      document.documentElement.hasAttribute("data-reload-snapshot-private")
    ) {
      clearStoredSnapshot();
      return;
    }
    // A second reload can happen before the replacement app is ready. The
    // visible frame is still the closed-shadow copy, not the loading body
    // underneath it, so carry that retained snapshot forward verbatim.
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
        // Two shapes with empty rectangles that are nonetheless visible: a
        // `display: contents` wrapper generates no box however much its
        // children fill, and a closed select still paints its selected
        // option's label. Judging either by its rectangle drops what it shows.
        var paintsThroughSelect =
          (original.tagName === "OPTION" ||
            original.tagName === "OPTGROUP") &&
          original.closest("select");
        var laidOut = !paintsThroughSelect && style.display !== "contents";
        var bounds = laidOut ? original.getBoundingClientRect() : null;
        // Removing children from a scroll container changes its scroll geometry
        // and can shift the visible slice. Fully offscreen subtrees become
        // coalesced spacers at the first laid-out level outside the viewport;
        // this preserves the slice without serializing an unbounded transcript.
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
        // IDs are scoped to the closed shadow root, so they cannot collide with
        // the live document. Keep them for internal references such as
        // SVG fill="url(#gradient-id)" and aria-labelledby.
        element.removeAttribute("autofocus");
        element.removeAttribute("srcdoc");
        // SVG <use> carries the same URLs on xlink:href, which a plain `href`
        // lookup misses. javascript: cannot be activated on an inert copy, but
        // it has no reason to be stored either.
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
  // Firefox, WebKitGTK and Safari before 18.2 do not expose pageswap, but
  // still deliver pagehide on reload. Do not register both: Chromium fires
  // pagehide after pageswap and a second full-DOM capture during unload is
  // both expensive and lower fidelity. pagehide cannot tell a reload from any
  // other unload, so on those engines a snapshot is also written when the user
  // navigates away; the restore side discards it (navigationType below).
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
    if (chatHistoryDisabled) clearStoredSnapshot();
    restoreSnapshot();
  } catch (error) {
    removeOverlay();
  }
})();
