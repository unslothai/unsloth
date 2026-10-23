// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! Browser panel pages: one native child webview per tab. Pages are untrusted:
//! - Navigations (frames too): http(s) to a public host or a local origin typed in that tab; all go via
//!   `browser_proxy`, which refuses private addresses after DNS. macOS < 14 falls back to the proxied frame.
//! - No IPC (capabilities bound to `main`); own profile, never the app's (holds sign-in).

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::webview::{
    DownloadEvent, NewWindowResponse, PageLoadEvent, PermissionKind, PermissionResponse,
};
use tauri::{
    AppHandle, Emitter, LogicalPosition, LogicalSize, Manager, Rect, Runtime, State, Url, Webview,
    WebviewBuilder, WebviewUrl,
};
use tokio::sync::watch;

const LABEL_PREFIX: &str = "unsloth-browser-";
const EVENT: &str = "unsloth-browser";
/// The app's own webview, the only caller these commands answer.
const MAIN_WEBVIEW: &str = "main";
const MAIN_WINDOW: &str = "main";
const URL_POLL: Duration = Duration::from_millis(800);
/// How far left of the window a parked page sits, in logical pixels.
const PARK_GAP: f64 = 64.0;
/// macOS 14+ data store for pages (fixed, so it persists).
#[cfg(target_os = "macos")]
const PAGE_DATA_STORE: [u8; 16] = *b"unsloth-browser1";

/// Mute tab for WebKit, which has no public mute for a view: media plays muted (in the page or not,
/// as `new Audio()` makes) and Web Audio contexts made after it are suspended. Run in the page's
/// world, so a page can undo it for itself; installed once per document, then toggled.
#[cfg(target_os = "macos")]
const MUTE_SCRIPT: &str = r#"((muted) => {
  const key = Symbol.for("unsloth.browser.mute");
  if (!window[key]) {
    let on = false;
    const silenced = new Set();
    const contexts = new Set();
    const paused = new Set();
    const pause = (context) => {
      if (context.state !== "running") return;
      paused.add(context);
      context.suspend().catch(() => {});
    };
    const silence = (media) => {
      if (!on || media.muted) return;
      media.muted = true;
      silenced.add(media);
    };
    const play = HTMLMediaElement.prototype.play;
    HTMLMediaElement.prototype.play = function (...args) {
      silence(this);
      return play.apply(this, args);
    };
    document.addEventListener("play", (event) => event.target instanceof HTMLMediaElement && silence(event.target), true);
    document.addEventListener("volumechange", (event) => event.target instanceof HTMLMediaElement && silence(event.target), true);
    for (const name of ["AudioContext", "webkitAudioContext"]) {
      const Base = window[name];
      if (typeof Base !== "function") continue;
      window[name] = class extends Base {
        constructor(...args) {
          super(...args);
          contexts.add(this);
          if (on) pause(this);
        }
        resume() {
          if (!on) return super.resume();
          paused.add(this);
          return Promise.resolve();
        }
      };
    }
    window[key] = (value) => {
      on = value;
      if (on) {
        for (const media of document.querySelectorAll("audio, video")) silence(media);
        for (const context of contexts) pause(context);
        return;
      }
      for (const media of silenced) media.muted = false;
      silenced.clear();
      for (const context of paused) context.resume().catch(() => {});
      paused.clear();
    };
  }
  window[key](muted);
})"#;

/// Pages get every plugin's init script too, but their commands are refused there (opener eats `_blank`
/// clicks, dialog/notification stub `alert`/`confirm`/`Notification`); `PAGE_RESTORE_SCRIPT` undoes them.
const PAGE_GUARD_SCRIPT: &str = r#"(() => {
  const label = window.__TAURI_INTERNALS__?.metadata?.currentWebview?.label;
  if (typeof label !== "string" || !label.startsWith("unsloth-browser-")) return;
  const kept = ["alert", "confirm", "Notification", "addEventListener"]
    .map((name) => [name, Object.getOwnPropertyDescriptor(window, name)]);
  const added = [];
  const add = window.addEventListener;
  window.addEventListener = function (...args) {
    added.push(args);
    return add.apply(this, args);
  };
  Object.defineProperty(window, "__unslothPageGuard", {
    configurable: true,
    value: () => {
      delete window.__unslothPageGuard;
      for (const [name, own] of kept) own ? Object.defineProperty(window, name, own) : delete window[name];
      for (const [type, listener, options] of added) window.removeEventListener(type, listener, options);
    },
  });
})();"#;

const PAGE_RESTORE_SCRIPT: &str = "window.__unslothPageGuard?.();";

/// Registered before the opener, dialog and notification plugins (`PAGE_GUARD_SCRIPT`).
pub fn page_guard_plugin<R: Runtime>() -> tauri::plugin::TauriPlugin<R> {
    tauri::plugin::Builder::new("unsloth-page-guard")
        .js_init_script(PAGE_GUARD_SCRIPT.to_string())
        .build()
}

/// Read after a load or title change. Pages can spoof it; it only feeds the panel's buttons.
const STATE_SCRIPT: &str = r#"(() => {
  try {
    const nav = window.navigation;
    const icon = document.querySelector("link[rel~='icon'][href], link[rel='apple-touch-icon'][href]");
    return JSON.stringify({
      back: nav && "canGoBack" in nav ? Boolean(nav.canGoBack) : history.length > 1,
      forward: nav && "canGoForward" in nav ? Boolean(nav.canGoForward) : false,
      icon: icon ? icon.href : new URL("/favicon.ico", location.href).href,
    });
  } catch { return ""; }
})()"#;

pub struct BrowserViews {
    inner: Mutex<ViewsState>,
    /// Whether the address poll has anything to watch; it sleeps on this rather than a timer.
    gate: watch::Sender<PollGate>,
}

impl Default for BrowserViews {
    fn default() -> Self {
        Self {
            inner: Mutex::default(),
            gate: watch::channel(PollGate::default()).0,
        }
    }
}

/// A view is shown, and the window it sits in is in use (focused, visible, not minimised).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PollGate {
    shown: bool,
    window_active: bool,
    /// Bumped when the shown tab changes, so a switch between two shown tabs reads at once too.
    switches: u64,
}

impl Default for PollGate {
    fn default() -> Self {
        // An unknown window state polls, as it always did.
        Self {
            shown: false,
            window_active: true,
            switches: 0,
        }
    }
}

impl PollGate {
    fn polls(self) -> bool {
        self.shown && self.window_active
    }
}

#[derive(Default)]
struct ViewsState {
    shown: Option<String>,
    urls: HashMap<String, String>,
    /// Download paths by URL, oldest first (macOS doesn't report it back; one URL can download twice).
    downloads: HashMap<String, Vec<PathBuf>>,
    /// By URL, finishes that couldn't tell which of several reservations they were (`take_finished`).
    unexplained_finishes: HashMap<String, usize>,
    download_starts: HashMap<String, VecDeque<Instant>>,
    polling: bool,
    /** Tabs the reader muted; macOS mutes each page they load, Windows the view once. */
    muted: HashSet<String>,
    /// Tabs whose view committed a page of its own since it opened or was last sent somewhere; until then a download's site is the address asked for (or the opener).
    committed: HashSet<String>,
    /// Hidden tabs shown just off the window for a screenshot (`show_for_capture`); a switch leaves them be.
    capturing: HashSet<String>,
    /// Where `browser_view_show` last put each tab, for a capture that a switch overtook (`end_capture`).
    placed: HashMap<String, Rect>,
    /// Granted local origins each tab was sent to; its pages may go back to those, never to others.
    local: HashMap<String, url::Origin>,
    local_pages: HashMap<String, LocalPage>,
}

struct LocalPage {
    origin: url::Origin,
    proxy: crate::browser_proxy::LocalProxy,
    // A distinct WebView2 environment is necessary: its proxy belongs to the profile.
    profile: tempfile::TempDir,
}

pub fn new_browser_views() -> BrowserViews {
    BrowserViews::default()
}

fn set_shown(views: &BrowserViews, inner: &mut ViewsState, shown: Option<String>) {
    if inner.shown == shown {
        return;
    }
    let visible = shown.is_some();
    inner.shown = shown;
    views.gate.send_modify(|gate| {
        gate.shown = visible;
        gate.switches = gate.switches.wrapping_add(1);
    });
}

/// Re-read the main window's state after its focus, size or visibility changed.
pub fn window_changed<R: Runtime>(app: &AppHandle<R>, focused: Option<bool>) {
    let (Some(views), Some(window)) =
        (app.try_state::<BrowserViews>(), app.get_window(MAIN_WINDOW))
    else {
        return;
    };
    // Windows reports keyboard focus moving into a child webview as the window losing focus, so
    // there only a minimised or hidden window pauses the poll; macOS and GTK report activation.
    let focused = cfg!(windows) || focused == Some(true) || window.is_focused().unwrap_or(true);
    let active =
        focused && window.is_visible().unwrap_or(true) && !window.is_minimized().unwrap_or(false);
    views
        .gate
        .send_if_modified(|gate| std::mem::replace(&mut gate.window_active, active) != active);
}

#[derive(Clone, Serialize)]
#[serde(
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    tag = "kind"
)]
enum BrowserEvent {
    Load {
        tab_id: String,
        url: String,
        loading: bool,
    },
    Title {
        tab_id: String,
        title: String,
    },
    /// The address changed without a load (history.pushState).
    Url {
        tab_id: String,
        url: String,
    },
    History {
        tab_id: String,
        can_go_back: bool,
        can_go_forward: bool,
        icon: Option<String>,
    },
    NewTab {
        tab_id: String,
        url: String,
    },
    /// An address a local view refused to go to itself; the panel offers it in a tab of its own.
    Refused {
        tab_id: String,
        url: String,
    },
    /// A link only another app opens (mailto:); the panel asks first.
    External {
        tab_id: String,
        url: String,
    },
    /// The page couldn't be protected, so its view was closed before it loaded.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    Failed {
        tab_id: String,
        error: String,
    },
    Download {
        tab_id: String,
        url: String,
        name: String,
        path: Option<String>,
        size: Option<u64>,
        done: bool,
        success: bool,
        /// A finished download's handle for Download history (browser_downloads.rs).
        download_id: Option<String>,
        /// Marked as from the internet: false if that failed (the panel warns), None where nothing marks.
        marked: Option<bool>,
        /// The DownloadPrompt `id` it was asked under; None when refused before asking.
        prompt_id: Option<String>,
    },
    /// An approved download that left no file (its save dialog was cancelled).
    DownloadCancelled {
        tab_id: String,
        url: String,
        prompt_id: String,
    },
    DownloadPrompt {
        tab_id: String,
        url: String,
        /// Page showing when the download started (the site a remembered answer is for); "" before the view showed one.
        site: String,
        name: String,
        id: String,
        /// Started from the page's context menu: the user asked for it, so save it via a dialog.
        save_as: bool,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViewBounds {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    /// The main webview's size in CSS pixels, to scale the rect by its zoom.
    viewport_width: f64,
}

impl ViewBounds {
    /// `(x, y, width, height, viewport_width)`, in the caller's CSS pixels.
    pub(crate) fn parts(&self) -> (f64, f64, f64, f64, f64) {
        (self.x, self.y, self.width, self.height, self.viewport_width)
    }
}

pub(crate) fn navigation_allowed(url: &Url) -> bool {
    match url.scheme() {
        // Documents a page makes itself, with its origin or none.
        "about" | "data" | "blob" => true,
        "http" | "https" => url.host().is_some_and(|host| !host_is_private(&host)),
        _ => false,
    }
}

/// Start of the OS ephemeral range on Linux (49152 elsewhere), where Studio's own helpers listen.
const ASSIGNED_PORTS: u16 = 32768;

const STUDIO_ADDRESS: &str = "this address is Unsloth's own";

fn loopback_origin(url: &Url) -> Option<url::Origin> {
    let loopback = match url.host()? {
        url::Host::Domain(name) => name == "localhost",
        url::Host::Ipv4(ip) => ip.is_loopback(),
        url::Host::Ipv6(ip) => ip.is_loopback(),
    };
    (loopback && matches!(url.scheme(), "http" | "https")).then(|| url.origin())
}

fn local_origin(url: &Url, own: &[u16]) -> Result<Option<url::Origin>, String> {
    let Some(origin) = loopback_origin(url) else {
        return Ok(None);
    };
    let port = url.port_or_known_default();
    if port.is_some_and(|port| {
        port >= ASSIGNED_PORTS || (8888..=8908).contains(&port) || own.contains(&port)
    }) {
        return Err(STUDIO_ADDRESS.into());
    }
    Ok(Some(origin))
}

/// The backend's port, and in a dev build the dev server the app is served from, which forwards /api to it.
fn own_ports<R: Runtime>(app: &AppHandle<R>) -> Vec<u16> {
    let backend = app
        .try_state::<crate::process::BackendState>()
        .and_then(|state| state.lock().ok()?.port);
    let dev = dev_server_port(tauri::is_dev(), app.config().build.dev_url.as_ref());
    backend.into_iter().chain(dev).collect()
}

fn dev_server_port(dev: bool, dev_url: Option<&Url>) -> Option<u16> {
    dev.then(|| dev_url?.port_or_known_default()).flatten()
}

fn grant_local(
    views: &BrowserViews,
    tab_id: &str,
    raw: &str,
    own: &[u16],
) -> Result<(), String> {
    label_for(tab_id)?;
    let url = Url::parse(raw.trim()).map_err(|_| "not a URL".to_string())?;
    let origin = local_origin(&url, own)?.ok_or("not a local address")?;
    views
        .inner
        .lock()
        .unwrap()
        .local
        .insert(tab_id.to_string(), origin);
    Ok(())
}

// A local view never becomes a public document while retaining its local proxy.
fn local_navigation_allowed(url: &Url, origin: &url::Origin) -> bool {
    matches!(url.scheme(), "about" | "data" | "blob") || url.origin() == *origin
}

fn opens_in_tab(url: &Url, local: bool) -> bool {
    (matches!(url.scheme(), "http" | "https") && navigation_allowed(url))
        || (local && loopback_origin(url).is_some())
}

pub(crate) fn host_is_private(host: &url::Host<&str>) -> bool {
    match host {
        url::Host::Ipv4(ip) => ipv4_is_private(*ip),
        url::Host::Ipv6(ip) => ipv6_is_private(*ip),
        url::Host::Domain(domain) => {
            let domain = domain.trim_end_matches('.').to_ascii_lowercase();
            // A dotless name only resolves locally.
            !domain.contains('.')
                || [
                    "localhost", "local", "internal", "lan", "home", "home.arpa", "corp", "intranet",
                ]
                    .iter()
                    .any(|suffix| domain == *suffix || domain.ends_with(&format!(".{suffix}")))
                // Written as a number the URL parser didn't normalise.
                || domain.parse::<IpAddr>().is_ok_and(ip_is_private)
        }
    }
}

pub(crate) fn ip_is_private(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => ipv4_is_private(ip),
        IpAddr::V6(ip) => ipv6_is_private(ip),
    }
}

/// Non-global IPv4 (IANA special-purpose, as backend `is_global`) plus multicast.
const NON_GLOBAL_V4: &[(u32, u32)] = &[
    (0x0000_0000, 8),  // this network
    (0x0a00_0000, 8),  // private
    (0x6440_0000, 10), // carrier-grade NAT
    (0x7f00_0000, 8),  // loopback
    (0xa9fe_0000, 16), // link-local
    (0xac10_0000, 12), // private
    (0xc000_0000, 24), // IETF protocol assignments
    (0xc000_0200, 24), // documentation
    (0xc0a8_0000, 16), // private
    (0xc612_0000, 15), // benchmarking
    (0xc633_6400, 24), // documentation
    (0xcb00_7100, 24), // documentation
    (0xe000_0000, 4),  // multicast
    (0xf000_0000, 4),  // reserved, broadcast
];

fn ipv4_is_private(ip: Ipv4Addr) -> bool {
    let ip = u32::from(ip);
    NON_GLOBAL_V4
        .iter()
        .any(|&(network, prefix)| ip >> (32 - prefix) == network >> (32 - prefix))
}

/// Global unicast (2000::/3) minus special blocks; mapped/NAT64 IPv4 is checked as IPv4.
fn ipv6_is_private(ip: Ipv6Addr) -> bool {
    if let Some(v4) = ip.to_ipv4_mapped() {
        return ipv4_is_private(v4);
    }
    let segments = ip.segments();
    if segments[..6] == [0x64, 0xff9b, 0, 0, 0, 0] {
        let [.., high, low] = segments;
        return ipv4_is_private(Ipv4Addr::from((u32::from(high) << 16) | u32::from(low)));
    }
    let [first, second, ..] = segments;
    (first & 0xe000) != 0x2000
        // 2001::/23 (Teredo, benchmarking, ORCHID...), documentation, 6to4, documentation.
        || (first == 0x2001 && second < 0x0200)
        || (first == 0x2001 && second == 0x0db8)
        || first == 0x2002
        || (first & 0xfff0) == 0x3ff0
}

/// Cap on page addresses sent to the panel, as the proxied frame caps its messages.
const MAX_URL_CHARS: usize = 8192;

fn reportable(url: &str) -> bool {
    url.len() <= MAX_URL_CHARS
}

fn is_external_handoff(url: &Url) -> bool {
    url.scheme() == "mailto"
}

/// URL prefixes (WebKit content-rule regexes, which have no `|`) a page may not request.
const BLOCKED_HOST_PREFIXES: &[&str] = &[
    r"localhost[:/]",
    r"[^/@]*\.localhost[:/]",
    r"[^/@]*\.local[:/]",
    r"127\.",
    r"0\.",
    r"10\.",
    r"192\.168\.",
    r"172\.1[6-9]\.",
    r"172\.2[0-9]\.",
    r"172\.3[01]\.",
    r"169\.254\.",
    r"100\.6[4-9]\.",
    r"100\.[7-9][0-9]\.",
    r"100\.1[01][0-9]\.",
    r"100\.12[0-7]\.",
    // IPv6 literals, private ones among them; public sites don't use them.
    r"\[",
];

/// WebKit content rules blocking private hosts (any scheme/credentials) and app schemes.
fn content_rules_json(local: Option<&url::Origin>) -> String {
    let mut filters = Vec::new();
    for prefix in BLOCKED_HOST_PREFIXES {
        filters.push(format!("^[a-z]+://{prefix}"));
        filters.push(format!("^[a-z]+://[^/@]*@{prefix}"));
    }
    for scheme in ["tauri", "ipc", "asset", "file"] {
        filters.push(format!("^{scheme}:"));
    }
    let mut rules: Vec<_> = filters
        .into_iter()
        .map(|filter| {
            serde_json::json!({
                "trigger": { "url-filter": filter },
                "action": { "type": "block" },
            })
        })
        .collect();
    if let Some(origin) = local {
        let origin = origin.ascii_serialization();
        let (scheme, authority) = origin.split_once("://").unwrap_or_default();
        let authority = regex::escape(authority);
        // Default ports leave the URL, so the scheme pins it: http://localhost must not admit :443.
        let socket = if scheme == "https" { "wss" } else { "ws" };
        for scheme_filter in [scheme, socket] {
            rules.push(serde_json::json!({
                "trigger": {
                    "url-filter": format!("^{scheme_filter}://([^/@]*@)?{authority}/"),
                    "if-top-url": [format!("^{scheme}://([^/@]*@)?{authority}/")],
                },
                "action": { "type": "ignore-previous-rules" },
            }));
        }
    }
    serde_json::Value::Array(rules).to_string()
}

#[cfg(target_os = "macos")]
mod content_rules {
    use objc2::rc::Retained;
    use objc2::MainThreadMarker;
    use objc2_foundation::{NSError, NSString};
    use objc2_web_kit::{WKContentRuleList, WKContentRuleListStore, WKWebView};
    use std::cell::RefCell;

    type Then = Box<dyn FnOnce(bool)>;

    #[derive(Default)]
    struct Rules {
        compiled: Option<Retained<WKContentRuleList>>,
        waiting: Vec<(Retained<WKWebView>, Then)>,
        compiling: bool,
    }

    thread_local! {
        // WebKit objects live on the main thread, and so does this.
        static RULES: RefCell<std::collections::HashMap<Option<url::Origin>, Rules>> = RefCell::new(std::collections::HashMap::new());
    }

    /// Add the rules to `webview`, then run `then` (the first load). Main thread only.
    pub fn protect(webview: Retained<WKWebView>, origin: Option<url::Origin>, then: Then) {
        let Some(mtm) = MainThreadMarker::new() else {
            then(false);
            return;
        };
        let compiled = RULES.with(|rules| {
            rules
                .borrow()
                .get(&origin)
                .and_then(|rules| rules.compiled.clone())
        });
        if let Some(list) = compiled {
            attach(&webview, &list);
            then(true);
            return;
        }
        let start = RULES.with(|rules| {
            let mut all = rules.borrow_mut();
            let rules = all.entry(origin.clone()).or_default();
            rules.waiting.push((webview, then));
            !std::mem::replace(&mut rules.compiling, true)
        });
        if start {
            compile(mtm, origin);
        }
    }

    fn compile(mtm: MainThreadMarker, origin: Option<url::Origin>) {
        let Some(store) = (unsafe { WKContentRuleListStore::defaultStore(mtm) }) else {
            finish(None, origin);
            return;
        };
        let identifier = NSString::from_str(&format!(
            "unsloth-browser-{}",
            origin
                .as_ref()
                .map(url::Origin::ascii_serialization)
                .unwrap_or_else(|| "public".into())
        ));
        let encoded = NSString::from_str(&super::content_rules_json(origin.as_ref()));
        let handler =
            block2::RcBlock::new(move |list: *mut WKContentRuleList, error: *mut NSError| {
                let list = unsafe { Retained::retain(list) };
                if list.is_none() {
                    let message = unsafe { error.as_ref() }
                        .map(|error| error.localizedDescription().to_string())
                        .unwrap_or_default();
                    log::warn!("browser content rules failed to compile: {message}");
                }
                finish(list, origin.clone());
            });
        unsafe {
            store.compileContentRuleListForIdentifier_encodedContentRuleList_completionHandler(
                Some(&identifier),
                Some(&encoded),
                Some(&handler),
            );
        }
    }

    fn finish(list: Option<Retained<WKContentRuleList>>, origin: Option<url::Origin>) {
        let waiting = RULES.with(|rules| {
            let mut all = rules.borrow_mut();
            let rules = all.entry(origin).or_default();
            rules.compiling = false;
            rules.compiled = list.clone();
            std::mem::take(&mut rules.waiting)
        });
        for (webview, then) in waiting {
            if let Some(list) = &list {
                attach(&webview, list);
            }
            then(list.is_some());
        }
    }

    fn attach(webview: &WKWebView, list: &WKContentRuleList) {
        unsafe {
            let controller = webview.configuration().userContentController();
            controller.removeAllContentRuleLists();
            controller.addContentRuleList(list);
        }
    }
}

fn label_for(tab_id: &str) -> Result<String, String> {
    let valid = !tab_id.is_empty()
        && tab_id.len() <= 64
        && tab_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if valid {
        Ok(format!("{LABEL_PREFIX}{tab_id}"))
    } else {
        Err("invalid tab id".into())
    }
}

fn tab_of(label: &str) -> Option<&str> {
    label.strip_prefix(LABEL_PREFIX)
}

pub(crate) fn require_main<R: Runtime>(caller: &Webview<R>) -> Result<(), String> {
    if caller.label() == MAIN_WEBVIEW {
        Ok(())
    } else {
        Err("not allowed from this webview".into())
    }
}

fn parse_page_url(
    views: &BrowserViews,
    tab_id: &str,
    raw: &str,
    own: &[u16],
) -> Result<Url, String> {
    let url = Url::parse(raw.trim()).map_err(|_| "not a URL".to_string())?;
    if let Some(origin) = local_origin(&url, own)? {
        if views.inner.lock().unwrap().local.get(tab_id) != Some(&origin) {
            return Err("local addresses open only when typed in the address bar".into());
        }
        return Ok(url);
    }
    if !matches!(url.scheme(), "http" | "https") || !navigation_allowed(&url) {
        return Err("only public web addresses open in the browser".into());
    }
    // Leaving the local origin ends its grant, however the tab's next page is opened.
    views.inner.lock().unwrap().local.remove(tab_id);
    Ok(url)
}

fn emit<R: Runtime>(app: &AppHandle<R>, event: BrowserEvent) {
    let _ = app.emit_to(MAIN_WEBVIEW, EVENT, event);
}

pub(crate) fn emit_download_done<R: Runtime>(
    app: &AppHandle<R>,
    tab_id: &str,
    url: &Url,
    path: &Path,
    download_id: Option<String>,
    marked: Option<bool>,
    prompt_id: &str,
) {
    emit(
        app,
        BrowserEvent::Download {
            tab_id: tab_id.to_string(),
            url: url.to_string(),
            name: path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default(),
            path: None,
            size: std::fs::metadata(path).ok().map(|m| m.len()),
            done: true,
            success: true,
            download_id,
            marked,
            prompt_id: Some(prompt_id.to_string()),
        },
    );
}

pub(crate) fn emit_download_cancelled<R: Runtime>(
    app: &AppHandle<R>,
    tab_id: &str,
    url: &Url,
    prompt_id: &str,
) {
    emit(
        app,
        BrowserEvent::DownloadCancelled {
            tab_id: tab_id.to_string(),
            url: url.to_string(),
            prompt_id: prompt_id.to_string(),
        },
    );
}

pub(crate) fn emit_download_failed<R: Runtime>(
    app: &AppHandle<R>,
    tab_id: &str,
    url: &Url,
    name: &str,
    prompt_id: Option<&str>,
) {
    emit(
        app,
        BrowserEvent::Download {
            tab_id: tab_id.to_string(),
            url: url.to_string(),
            name: name.to_string(),
            path: None,
            size: None,
            done: true,
            success: false,
            download_id: None,
            marked: None,
            prompt_id: prompt_id.map(str::to_string),
        },
    );
}

pub(crate) fn view<R: Runtime>(app: &AppHandle<R>, tab_id: &str) -> Result<Webview<R>, String> {
    let label = label_for(tab_id)?;
    app.get_webview(&label).ok_or_else(|| "no such tab".into())
}

fn refresh_history<R: Runtime>(webview: &Webview<R>) {
    let Some(tab_id) = tab_of(webview.label()).map(str::to_string) else {
        return;
    };
    let app = webview.app_handle().clone();
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    let page = webview.clone();
    let _ = webview.eval_with_callback(STATE_SCRIPT, move |result| {
        let Ok(inner) = serde_json::from_str::<String>(&result) else {
            return;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&inner) else {
            return;
        };
        let icon = value
            .get("icon")
            .and_then(|v| v.as_str())
            .and_then(|icon| Url::parse(icon).ok())
            .filter(|icon| matches!(icon.scheme(), "http" | "https") && navigation_allowed(icon))
            .filter(|icon| icon.as_str().len() <= 2048)
            .map(String::from);
        // WebKit before Safari 26.2 and WebKitGTK lack the Navigation API: ask the engine.
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            let (app, tab_id) = (app.clone(), tab_id.clone());
            let _ = page.with_webview(move |platform| {
                #[cfg(target_os = "macos")]
                // Safety: wry's live view, read on the main thread.
                let (can_go_back, can_go_forward) = unsafe {
                    let view = &*(platform.inner() as *const objc2_web_kit::WKWebView);
                    (view.canGoBack(), view.canGoForward())
                };
                #[cfg(target_os = "linux")]
                let (can_go_back, can_go_forward) = {
                    use webkit2gtk::WebViewExt;
                    let view = platform.inner();
                    (view.can_go_back(), view.can_go_forward())
                };
                emit(
                    &app,
                    BrowserEvent::History {
                        tab_id,
                        can_go_back,
                        can_go_forward,
                        icon,
                    },
                );
            });
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        emit(
            &app,
            BrowserEvent::History {
                tab_id: tab_id.clone(),
                can_go_back: value.get("back").and_then(|v| v.as_bool()).unwrap_or(false),
                can_go_forward: value
                    .get("forward")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                icon,
            },
        );
    });
}

// A page picks its title, so it can't be large enough to stall the UI; the frame path's cap.
const MAX_TITLE_CHARS: usize = 1024;

fn bounded_title(title: String) -> String {
    match title.char_indices().nth(MAX_TITLE_CHARS) {
        Some((end, _)) => title[..end].to_string(),
        None => title,
    }
}

// Pages can download without a click: cap concurrent and per-minute starts to protect the disk.
const MAX_DOWNLOADS_IN_FLIGHT: usize = 3;
const DOWNLOADS_PER_WINDOW: usize = 5;
const DOWNLOAD_WINDOW: Duration = Duration::from_secs(60);

fn download_allowed(in_flight: usize, starts: &mut VecDeque<Instant>, now: Instant) -> bool {
    while starts
        .front()
        .is_some_and(|start| now.duration_since(*start) >= DOWNLOAD_WINDOW)
    {
        starts.pop_front();
    }
    if in_flight >= MAX_DOWNLOADS_IN_FLIGHT || starts.len() >= DOWNLOADS_PER_WINDOW {
        return false;
    }
    starts.push_back(now);
    true
}

/// A free name in `dir` for a download, from the name the page suggested; `reserved` holds the
/// destinations of downloads still in flight, which don't exist on disk yet.
fn download_destination(dir: &Path, suggested: &Path, reserved: &HashSet<&Path>) -> PathBuf {
    let name = suggested
        .file_name()
        .and_then(|name| name.to_str())
        .map(crate::native_file_dialogs::safe_download_name)
        .unwrap_or_else(|| "download".into());
    // Windows and macOS file systems ignore case, so `Report.pdf` and `report.pdf` are one file.
    let same = |a: &Path, b: &Path| {
        if cfg!(any(windows, target_os = "macos")) {
            a.to_string_lossy().to_lowercase() == b.to_string_lossy().to_lowercase()
        } else {
            a == b
        }
    };
    let free = |path: &Path| !path.exists() && !reserved.iter().any(|taken| same(taken, path));
    let candidate = dir.join(&name);
    if free(&candidate) {
        return candidate;
    }
    let path = Path::new(&name);
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("download");
    let ext = path.extension().and_then(|s| s.to_str());
    (1..10_000)
        .map(|n| match ext {
            Some(ext) => dir.join(format!("{stem} ({n}).{ext}")),
            None => dir.join(format!("{stem} ({n})")),
        })
        .find(|p| free(p))
        .unwrap_or(candidate)
}

/// Where a download came from, for the file's internet mark: no credentials, query or fragment,
/// which can carry tokens. Web addresses only.
#[cfg_attr(not(windows), allow(dead_code))]
fn sanitized_source(url: &Url) -> Option<String> {
    if !matches!(url.scheme(), "http" | "https") {
        return None;
    }
    let mut clean = url.clone();
    let _ = clean.set_username("");
    let _ = clean.set_password(None);
    clean.set_query(None);
    clean.set_fragment(None);
    Some(clean.to_string())
}

/// Mark a download as from the internet, so Gatekeeper or SmartScreen checks it. Whether that
/// worked (FAT, exFAT and some network drives keep no mark); None where there is no mark.
pub(crate) fn mark_downloaded(path: &Path, url: &Url) -> Option<bool> {
    #[cfg(target_os = "macos")]
    {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let value = format!("0081;{now:08x};Unsloth;");
        let (Ok(path), Ok(name)) = (
            CString::new(path.as_os_str().as_bytes()),
            CString::new("com.apple.quarantine"),
        ) else {
            return Some(false);
        };
        let _ = url;
        let result = unsafe {
            libc::setxattr(
                path.as_ptr(),
                name.as_ptr(),
                value.as_ptr().cast(),
                value.len(),
                0,
                0,
            )
        };
        Some(result == 0)
    }
    #[cfg(windows)]
    {
        let mut stream = path.as_os_str().to_owned();
        stream.push(":Zone.Identifier");
        let host = sanitized_source(url)
            .map(|source| format!("HostUrl={source}\r\n"))
            .unwrap_or_default();
        Some(std::fs::write(stream, format!("[ZoneTransfer]\r\nZoneId=3\r\n{host}")).is_ok())
    }
    #[cfg(not(any(target_os = "macos", windows)))]
    {
        let _ = (path, url);
        None
    }
}

#[cfg(target_os = "macos")]
fn macos_at_least(major: isize) -> bool {
    use objc2_foundation::{NSOperatingSystemVersion, NSProcessInfo};
    NSProcessInfo::processInfo().isOperatingSystemAtLeastVersion(NSOperatingSystemVersion {
        majorVersion: major,
        minorVersion: 0,
        patchVersion: 0,
    })
}

/// WebKit's default names no browser version, which sites such as Gmail refuse as outdated.
#[cfg(target_os = "macos")]
fn page_user_agent() -> Option<String> {
    use objc2_foundation::{ns_string, NSBundle, NSString};
    let safari = NSBundle::bundleWithPath(ns_string!("/Applications/Safari.app"))?;
    let version = safari
        .objectForInfoDictionaryKey(ns_string!("CFBundleShortVersionString"))?
        .downcast::<NSString>()
        .ok()?;
    Some(format!(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{version} Safari/605.1.15"
    ))
}

/// Pages ask before using a capture device. With no handler wry grants every WKWebView capture
/// request on macOS without a prompt, so any page could open the microphone the app holds for
/// dictation. `Default` is WebKit's own per-site prompt there and the platform default elsewhere
/// (WebView2's prompt, WebKitGTK's deny), so only the silent grant goes away.
fn page_permission(_kind: PermissionKind) -> PermissionResponse {
    PermissionResponse::Default
}

/// Pages' own profile, proxied via `browser_proxy` (macOS: set on first load). Every
/// browser-panel webview is built through here, so none can miss the permission handler.
fn with_page_profile<R: Runtime>(
    builder: WebviewBuilder<R>,
    app: &AppHandle<R>,
    local: Option<&LocalPage>,
) -> Result<WebviewBuilder<R>, String> {
    let builder = builder.on_permission_request(|_, kind| page_permission(kind));
    let proxy = match local {
        Some(local) => local.proxy.address,
        None => crate::browser_proxy::address()?,
    };
    #[cfg(target_os = "macos")]
    {
        let _ = (app, proxy);
        // Before macOS 14 the only other store is a private one.
        if !browser_view_supported() {
            return Err("native pages need macOS 14".into());
        }
        let builder = if local.is_some() {
            builder.incognito(true)
        } else {
            builder.data_store_identifier(PAGE_DATA_STORE)
        };
        Ok(match page_user_agent() {
            Some(agent) => builder.user_agent(&agent),
            None => builder,
        })
    }
    #[cfg(not(target_os = "macos"))]
    {
        #[cfg(windows)]
        let builder = builder.additional_browser_args(&page_browser_args(&proxy.to_string()));
        #[cfg(not(windows))]
        let builder = builder
            .proxy_url(Url::parse(&format!("http://{proxy}")).map_err(|error| error.to_string())?);
        if let Some(local) = local {
            return Ok(builder
                .incognito(true)
                .data_directory(local.profile.path().to_path_buf()));
        }
        Ok(match app.path().app_local_data_dir() {
            Ok(dir) => builder.data_directory(dir.join("browser-profile")),
            Err(_) => builder.incognito(true),
        })
    }
}

/// WebView2 arguments for pages' own environment. They replace wry's defaults, so repeat those
/// except SmartScreen, which stays on as in Edge since these are open-web pages; Chromium bypasses
/// the proxy for loopback otherwise.
#[cfg(any(windows, test))]
fn page_browser_args(proxy: &str) -> String {
    format!(
        "--disable-features=msWebOOUI,msPdfOOUI --proxy-server=http://{proxy} \
         --proxy-bypass-list=<-loopback>"
    )
}

/// The page's address. wry's `url()` panics on macOS while WebKit has none (a failed load).
async fn page_url<R: Runtime>(webview: &Webview<R>) -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        webview
            .with_webview(move |platform| {
                let view = platform.inner() as *const objc2_web_kit::WKWebView;
                // Safety: wry's live view, read on the main thread.
                let url = unsafe { view.as_ref().and_then(|view| view.URL()) }
                    .and_then(|url| url.absoluteString())
                    .map(|url| url.to_string());
                let _ = sender.send(url);
            })
            .ok()?;
        receiver.await.ok().flatten()
    }
    #[cfg(not(target_os = "macos"))]
    webview.url().ok().map(|url| url.to_string())
}

/// Poll the shown tab's address, which pushState changes without a load. Started once; it only
/// runs while a view is shown in a window in use.
fn start_url_poll<R: Runtime>(app: &AppHandle<R>) {
    let state = app.state::<BrowserViews>();
    {
        let mut inner = state.inner.lock().unwrap();
        if inner.polling {
            return;
        }
        inner.polling = true;
    }
    window_changed(app, None);
    let gate = state.gate.subscribe();
    let app = app.clone();
    tauri::async_runtime::spawn(run_url_poll(gate, move || poll_url(app.clone())));
}

/// Read now and every `URL_POLL` while the gate is open; while it is shut, wait on it with no
/// timer armed. A change while open reads at once (a tab switch or the window coming back).
async fn run_url_poll<F, Fut>(mut gate: watch::Receiver<PollGate>, mut poll: F)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    loop {
        if !gate.borrow_and_update().polls() {
            if gate.changed().await.is_err() {
                return;
            }
            continue;
        }
        poll().await;
        tokio::select! {
            _ = tokio::time::sleep(URL_POLL) => {}
            changed = gate.changed() => {
                if changed.is_err() {
                    return;
                }
            }
        }
    }
}

async fn poll_url<R: Runtime>(app: AppHandle<R>) {
    let state = app.state::<BrowserViews>();
    let Some(tab_id) = state.inner.lock().unwrap().shown.clone() else {
        return;
    };
    let Ok(webview) = view(&app, &tab_id) else {
        return;
    };
    let Some(url) = page_url(&webview).await.filter(|url| reportable(url)) else {
        return;
    };
    let changed = {
        let mut inner = state.inner.lock().unwrap();
        // Switched or closed while the address was read: it belongs to a view no longer shown.
        if inner.shown.as_deref() != Some(tab_id.as_str()) {
            return;
        }
        let last = inner.urls.insert(tab_id.clone(), url.clone());
        last.as_deref() != Some(url.as_str())
    };
    if changed {
        emit(&app, BrowserEvent::Url { tab_id, url });
        refresh_history(&webview);
    }
}

/// Stage a tab's download and ask about it. False refuses it.
pub(crate) fn download_requested<R: Runtime>(
    webview: &Webview<R>,
    download_tab: &str,
    url: Url,
    destination: &mut PathBuf,
    save_as: bool,
) -> bool {
    let app = webview.app_handle();
    if crate::browser_downloads::unanswered(app, download_tab)
        >= crate::browser_downloads::MAX_UNANSWERED_PER_TAB
    {
        let name = destination
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        emit_download_failed(app, download_tab, &url, &name, None);
        return false;
    }
    let Some((id, staging)) = crate::browser_downloads::staging_dir(app) else {
        return false;
    };
    let path = {
        let state = app.state::<BrowserViews>();
        let mut inner = state.inner.lock().unwrap();
        // macOS reports no path when a download finishes, so two of one URL at
        // once couldn't be told apart: one at a time.
        let busy = cfg!(target_os = "macos") && inner.downloads.contains_key(url.as_str());
        let in_flight = inner.downloads.values().map(Vec::len).sum();
        let starts = inner
            .download_starts
            .entry(download_tab.to_string())
            .or_default();
        if busy || !download_allowed(in_flight, starts, Instant::now()) {
            drop(inner);
            let _ = std::fs::remove_dir_all(&staging);
            if !busy {
                let name = destination
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default();
                emit_download_failed(app, download_tab, &url, &name, None);
            }
            return false;
        }
        let path = download_destination(&staging, destination, &HashSet::new());
        inner
            .downloads
            .entry(url.to_string())
            .or_default()
            .push(path.clone());
        path
    };
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    *destination = path.clone();
    crate::browser_downloads::add_pending(
        app,
        id.clone(),
        download_tab.to_string(),
        url.clone(),
        path,
    );
    // "" before the view showed its own page: the frontend knows the tab's address or opener.
    let committed = app
        .state::<BrowserViews>()
        .inner
        .lock()
        .unwrap()
        .committed
        .contains(download_tab);
    let site = match webview.url() {
        Ok(page) if committed => page.to_string(),
        _ => String::new(),
    };
    emit(
        app,
        BrowserEvent::DownloadPrompt {
            tab_id: download_tab.to_string(),
            url: url.to_string(),
            site,
            name,
            id,
            save_as,
        },
    );
    true
}

/// Settle a finished download. Needs no view: its tab may have closed meanwhile.
pub(crate) fn download_finished<R: Runtime>(
    app: &AppHandle<R>,
    url: Url,
    path: Option<PathBuf>,
    success: bool,
) {
    let (recorded, retired) = {
        let state = app.state::<BrowserViews>();
        let mut inner = state.inner.lock().unwrap();
        let ViewsState {
            downloads,
            unexplained_finishes,
            ..
        } = &mut *inner;
        let key = url.to_string();
        let mut unexplained = unexplained_finishes.remove(&key).unwrap_or(0);
        let pending = downloads.entry(key.clone()).or_default();
        let taken = take_finished(pending, &mut unexplained, path.as_deref());
        if pending.is_empty() {
            downloads.remove(&key);
        } else if unexplained > 0 {
            unexplained_finishes.insert(key, unexplained);
        }
        taken
    };
    // The reserved path first: the engine may report another spelling of it.
    crate::browser_downloads::finished(app, [recorded.as_deref(), path.as_deref()], success);
    for staged in retired {
        crate::browser_downloads::finished(app, [Some(&staged), None], false);
    }
}

fn close_view<R: Runtime>(page: &Webview<R>) -> tauri::Result<()> {
    #[cfg(target_os = "macos")]
    crate::browser_context_downloads::forget(page);
    #[cfg(target_os = "linux")]
    crate::browser_linux::detach(page)?;
    let result = page.close();
    if let Some(tab) = tab_of(page.label()) {
        page.state::<BrowserViews>()
            .inner
            .lock()
            .unwrap()
            .local_pages
            .remove(tab);
    }
    result
}

#[cfg(target_os = "linux")]
pub fn close_all<R: Runtime>(app: &AppHandle<R>) {
    for page in browser_views(app) {
        if let Err(error) = close_view(&page) {
            log::warn!("Could not close browser page before exit: {error}");
        }
    }
}

async fn create_view<R: Runtime>(
    caller: &Webview<R>,
    tab_id: &str,
    url: Url,
    bounds: &ViewBounds,
) -> Result<Webview<R>, String> {
    let app = caller.app_handle().clone();
    let label = label_for(tab_id)?;
    let tab = tab_id.to_string();

    let origin = loopback_origin(&url);
    let local = origin
        .clone()
        .map(|origin| -> Result<LocalPage, String> {
            Ok(LocalPage {
                proxy: crate::browser_proxy::LocalProxy::new(origin.clone())?,
                origin,
                profile: tempfile::tempdir().map_err(|error| error.to_string())?,
            })
        })
        .transpose()?;
    let nav_origin = origin.clone();
    let nav_app = app.clone();
    let nav_tab = tab.clone();
    let load_tab = tab.clone();
    let title_tab = tab.clone();
    let window_app = app.clone();
    let window_tab = tab.clone();
    let window_local = origin.is_some();
    let download_tab = tab.clone();

    let initial = Url::parse("about:blank").unwrap();
    let deferred = url;

    let builder = WebviewBuilder::new(&label, WebviewUrl::External(initial))
        .on_navigation(move |url| {
            // An address too long to report would leave the bar showing the last page's: refused.
            if nav_origin.as_ref().map_or_else(
                || navigation_allowed(url),
                |origin| local_navigation_allowed(url, origin),
            ) && reportable(url.as_str())
            {
                return true;
            }
            if !reportable(url.as_str()) {
                return false;
            }
            // Frames navigate here too, so the panel asks rather than opening a tab.
            if nav_origin.is_some() && opens_in_tab(url, true) {
                emit(
                    &nav_app,
                    BrowserEvent::Refused {
                        tab_id: nav_tab.clone(),
                        url: url.to_string(),
                    },
                );
            } else if is_external_handoff(url) {
                emit(
                    &nav_app,
                    BrowserEvent::External {
                        tab_id: nav_tab.clone(),
                        url: url.to_string(),
                    },
                );
            }
            false
        })
        .on_page_load(move |webview, payload| {
            let app = webview.app_handle();
            if payload.url().scheme() == "about" || !reportable(payload.url().as_str()) {
                return;
            }
            let url = payload.url().to_string();
            let loading = matches!(payload.event(), PageLoadEvent::Started);
            {
                let state = app.state::<BrowserViews>();
                let mut inner = state.inner.lock().unwrap();
                inner.urls.insert(load_tab.clone(), url.clone());
                // Started is a commit (WebKit) or content loading (WebView2): never a download.
                if loading {
                    inner.committed.insert(load_tab.clone());
                }
            }
            emit(
                app,
                BrowserEvent::Load {
                    tab_id: load_tab.clone(),
                    url,
                    loading,
                },
            );
            // Each page starts unmuted: mute it as it commits, and again once it has loaded.
            #[cfg(target_os = "macos")]
            if app
                .state::<BrowserViews>()
                .inner
                .lock()
                .unwrap()
                .muted
                .contains(&load_tab)
            {
                let _ = webview.eval(format!("{MUTE_SCRIPT}(true)"));
            }
            if !loading {
                refresh_history(&webview);
            }
        })
        .on_document_title_changed(move |webview, title| {
            emit(
                webview.app_handle(),
                BrowserEvent::Title {
                    tab_id: title_tab.clone(),
                    title: bounded_title(title),
                },
            );
            refresh_history(&webview);
        })
        .on_new_window(move |url, _features| {
            if !reportable(url.as_str()) {
                return NewWindowResponse::Deny;
            }
            // A tab instead of a window, with no opener to script.
            if opens_in_tab(&url, window_local) {
                emit(
                    &window_app,
                    BrowserEvent::NewTab {
                        tab_id: window_tab.clone(),
                        url: url.to_string(),
                    },
                );
            } else if is_external_handoff(&url) {
                emit(
                    &window_app,
                    BrowserEvent::External {
                        tab_id: window_tab.clone(),
                        url: url.to_string(),
                    },
                );
            }
            NewWindowResponse::Deny
        })
        .on_download(move |webview, event| match event {
            DownloadEvent::Requested { url, destination } => {
                download_requested(&webview, &download_tab, url, destination, false)
            }
            DownloadEvent::Finished { url, path, success } => {
                download_finished(webview.app_handle(), url, path, success);
                true
            }
            _ => false,
        })
        .initialization_script(PAGE_RESTORE_SCRIPT)
        .zoom_hotkeys_enabled(false)
        .devtools(cfg!(debug_assertions))
        .focused(false);
    let builder = with_page_profile(builder, &app, local.as_ref())?;

    let (position, size) = logical_rect(caller, bounds);
    let window = caller.window();
    let webview = window
        .add_child(builder, position, size)
        .map_err(|error| error.to_string())?;
    #[cfg(target_os = "linux")]
    if let Err(error) = crate::browser_linux::attach(&webview, position, size).await {
        let _ = webview.close();
        return Err(error);
    }
    #[cfg(target_os = "macos")]
    crate::browser_layer::place(&webview);
    #[cfg(target_os = "macos")]
    crate::browser_context_downloads::watch(&webview, &tab);
    // A muted tab whose view closed (four at most stay open) opens muted again.
    if app
        .state::<BrowserViews>()
        .inner
        .lock()
        .unwrap()
        .muted
        .contains(&tab)
    {
        let _ = apply_mute(&webview, true);
    }
    #[cfg(windows)]
    if let Some(origin) = origin {
        if let Err(error) = protect_local_frames(&webview, origin).await {
            let _ = close_view(&webview);
            return Err(error);
        }
    }
    if let Some(local) = local {
        app.state::<BrowserViews>()
            .inner
            .lock()
            .unwrap()
            .local_pages
            .insert(tab.clone(), local);
    }
    load_when_protected(&webview, deferred);
    app.state::<BrowserViews>()
        .inner
        .lock()
        .unwrap()
        .placed
        .insert(
            tab,
            Rect {
                position: position.into(),
                size: size.into(),
            },
        );
    start_url_poll(&app);
    Ok(webview)
}

// Wry's Windows navigation callback covers only the top document, so frames are filtered here.
#[cfg(windows)]
async fn protect_local_frames<R: Runtime>(
    page: &Webview<R>,
    origin: url::Origin,
) -> Result<(), String> {
    use webview2_com::{
        Microsoft::Web::WebView2::Win32::{
            ICoreWebView2_22, COREWEBVIEW2_WEB_RESOURCE_CONTEXT_DOCUMENT,
            COREWEBVIEW2_WEB_RESOURCE_REQUEST_SOURCE_KINDS_ALL,
        },
        WebResourceRequestedEventHandler,
    };
    use windows_core::{w, Interface, PWSTR};
    let (send, receive) = tokio::sync::oneshot::channel();
    page.with_webview(move |platform| {
        let result = (|| -> windows_core::Result<()> {
            let view = unsafe { platform.controller().CoreWebView2()? };
            let environment = platform.environment();
            let filtered = view.cast::<ICoreWebView2_22>()?;
            unsafe {
                filtered.AddWebResourceRequestedFilterWithRequestSourceKinds(
                    w!("*"),
                    COREWEBVIEW2_WEB_RESOURCE_CONTEXT_DOCUMENT,
                    COREWEBVIEW2_WEB_RESOURCE_REQUEST_SOURCE_KINDS_ALL,
                )?;
                let mut token = 0;
                view.add_WebResourceRequested(
                    &WebResourceRequestedEventHandler::create(Box::new(move |_, args| {
                        let Some(args) = args else { return Ok(()) };
                        let request = args.Request()?;
                        let mut uri = PWSTR::null();
                        request.Uri(&mut uri)?;
                        let uri = webview2_com::take_pwstr(uri);
                        if !Url::parse(&uri)
                            .is_ok_and(|url| local_navigation_allowed(&url, &origin))
                        {
                            let response = environment.CreateWebResourceResponse(
                                None,
                                403,
                                w!("Forbidden"),
                                w!("Content-Length: 0\r\n"),
                            )?;
                            args.SetResponse(&response)?;
                        }
                        Ok(())
                    })),
                    &mut token,
                )?;
            }
            Ok(())
        })()
        .map_err(|error| error.to_string());
        let _ = send.send(result);
    })
    .map_err(|error| error.to_string())?;
    tokio::time::timeout(Duration::from_secs(10), receive)
        .await
        .map_err(|_| "local page protection timed out".to_string())?
        .map_err(|_| "local page protection did not finish".to_string())?
}

#[cfg(target_os = "macos")]
fn load_when_protected<R: Runtime>(webview: &Webview<R>, url: Url) {
    // Off main thread: `protect` can finish inside `with_webview`, holding `navigate`'s lock.
    let local = tab_of(webview.label()).and_then(|tab| {
        webview
            .state::<BrowserViews>()
            .inner
            .lock()
            .unwrap()
            .local_pages
            .get(tab)
            .map(|page| (page.proxy.address, page.origin.clone()))
    });
    let proxy = local
        .as_ref()
        .map(|(proxy, _)| Ok(*proxy))
        .unwrap_or_else(crate::browser_proxy::address);
    let origin = local.map(|(_, origin)| origin);
    let page = webview.clone();
    let load = move |protected: bool| {
        tauri::async_runtime::spawn(async move {
            if protected {
                let _ = page.navigate(url);
                return;
            }
            // WebKit sends loopback requests around the proxy; only the rules stop them.
            let tab_id = tab_of(page.label()).unwrap_or_default().to_string();
            let _ = close_view(&page);
            emit(
                page.app_handle(),
                BrowserEvent::Failed {
                    tab_id,
                    error: "the page's protections couldn't be set up, so it wasn't opened".into(),
                },
            );
        });
    };
    let _ = webview.with_webview(move |platform| {
        let raw = platform.inner() as *mut objc2_web_kit::WKWebView;
        // Safety: wry's view, alive with its Tauri webview; retained for the callback.
        let Some(view) = (unsafe { objc2::rc::Retained::retain(raw) }) else {
            return;
        };
        // Never unproxied.
        if !proxy.is_ok_and(|proxy| page_proxy::route(&view, proxy)) {
            log::error!("browser page not loaded: its proxy couldn't be set");
            load(false);
            return;
        }
        content_rules::protect(view, origin, Box::new(load));
    });
}

#[cfg(target_os = "macos")]
mod page_proxy {
    use objc2::rc::Retained;
    use objc2::runtime::AnyObject;
    use objc2_foundation::{ns_string, NSArray, NSObjectNSKeyValueCoding};
    use objc2_web_kit::WKWebView;
    use std::ffi::{c_char, CString};
    use std::net::SocketAddr;

    type CreateHost = unsafe extern "C" fn(*const c_char, *const c_char) -> *mut AnyObject;
    type CreateConnect = unsafe extern "C" fn(*mut AnyObject, *mut AnyObject) -> *mut AnyObject;

    /// Point the data store at the proxy. Network calls (macOS 14+) are looked up at run time;
    /// linking them would stop older macOS launching the app.
    pub fn route(view: &WKWebView, proxy: SocketAddr) -> bool {
        let (Ok(ip), Ok(port)) = (
            CString::new(proxy.ip().to_string()),
            CString::new(proxy.port().to_string()),
        ) else {
            return false;
        };
        unsafe {
            libc::dlopen(
                c"/System/Library/Frameworks/Network.framework/Network".as_ptr(),
                libc::RTLD_LAZY,
            );
            let create_host = libc::dlsym(libc::RTLD_DEFAULT, c"nw_endpoint_create_host".as_ptr());
            let create_connect = libc::dlsym(
                libc::RTLD_DEFAULT,
                c"nw_proxy_config_create_http_connect".as_ptr(),
            );
            if create_host.is_null() || create_connect.is_null() {
                return false;
            }
            let create_host: CreateHost = std::mem::transmute(create_host);
            let create_connect: CreateConnect = std::mem::transmute(create_connect);
            // Both return +1 references.
            let Some(endpoint) = Retained::from_raw(create_host(ip.as_ptr(), port.as_ptr())) else {
                return false;
            };
            let config =
                create_connect(Retained::as_ptr(&endpoint).cast_mut(), std::ptr::null_mut());
            let Some(config) = Retained::from_raw(config) else {
                return false;
            };
            let configs = NSArray::from_retained_slice(&[config]);
            view.configuration()
                .websiteDataStore()
                .setValue_forKey(Some(&configs), ns_string!("proxyConfigurations"));
        }
        true
    }
}

#[cfg(not(target_os = "macos"))]
fn load_when_protected<R: Runtime>(webview: &Webview<R>, url: Url) {
    let _ = webview.navigate(url);
}

fn logical_rect<R: Runtime>(
    caller: &Webview<R>,
    bounds: &ViewBounds,
) -> (LogicalPosition<f64>, LogicalSize<f64>) {
    // Read once: each is a round trip to the window, and this runs on every resize frame.
    let main = caller
        .bounds()
        .ok()
        .zip(caller.window().scale_factor().ok())
        .map(|(rect, factor)| {
            (
                rect.position.to_logical::<f64>(factor),
                rect.size.to_logical::<f64>(factor).width,
            )
        });
    let origin = main
        .map(|(origin, _)| origin)
        .unwrap_or(LogicalPosition::new(0.0, 0.0));
    let scale = main
        .map(|(_, width)| width)
        .filter(|width| *width > 0.0 && bounds.viewport_width > 0.0)
        .map(|width| width / bounds.viewport_width)
        .unwrap_or(1.0);
    (
        LogicalPosition::new(origin.x + bounds.x * scale, origin.y + bounds.y * scale),
        LogicalSize::new(
            (bounds.width * scale).max(1.0),
            (bounds.height * scale).max(1.0),
        ),
    )
}

fn browser_views<R: Runtime>(app: &AppHandle<R>) -> Vec<Webview<R>> {
    app.webviews()
        .into_iter()
        .filter(|(label, _)| label.starts_with(LABEL_PREFIX))
        .map(|(_, webview)| webview)
        .collect()
}

/// Native pages need per-webview proxy support: macOS 14+, Windows, and Linux.
#[tauri::command]
pub fn browser_view_supported() -> bool {
    #[cfg(target_os = "macos")]
    return macos_at_least(14);
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    return true;
    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    false
}

/// Whether pages sit under the app's webview (`browser_layer`). Elsewhere covered pages are snapshotted.
/// Asked once per load of the app, so it also drops input state a reload left behind.
#[tauri::command]
pub fn browser_view_layered<R: Runtime>(webview: Webview<R>) -> Result<bool, String> {
    require_main(&webview)?;
    #[cfg(target_os = "macos")]
    {
        if crate::browser_layer::set_input(false, &[]) {
            let _ = webview.run_on_main_thread(crate::browser_layer::reroute_moves);
        }
        Ok(browser_view_supported())
    }
    #[cfg(not(target_os = "macos"))]
    Ok(false)
}

/// The panel UI over the page: all of it while a menu or dialog is open, else clickable rects.
#[tauri::command]
pub fn browser_view_input<R: Runtime>(
    webview: Webview<R>,
    blocked: bool,
    exclude: Vec<ViewBounds>,
) -> Result<(), String> {
    require_main(&webview)?;
    #[cfg(target_os = "macos")]
    if crate::browser_layer::set_input(blocked, &exclude) {
        let _ = webview.run_on_main_thread(crate::browser_layer::reroute_moves);
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (blocked, exclude);
    Ok(())
}

/// Show a tab's page at `bounds` (created at `url` first time), hide the rest; `None` hides all.
/// `parked` moves the page just off-window instead of hiding it, so it keeps painting under its snapshot.
/// Async: creating a webview from a sync command deadlocks on Windows (Tauri known issue).
#[tauri::command]
pub async fn browser_view_show<R: Runtime>(
    webview: Webview<R>,
    state: State<'_, BrowserViews>,
    tab_id: Option<String>,
    url: Option<String>,
    bounds: Option<ViewBounds>,
    parked: Option<bool>,
) -> Result<(), String> {
    require_main(&webview)?;
    let app = webview.app_handle().clone();
    let parked = parked.unwrap_or(false);
    let target = match (&tab_id, &bounds) {
        (Some(tab_id), Some(bounds)) => Some(match app.get_webview(&label_for(tab_id)?) {
            Some(view) => {
                let (mut position, size) = logical_rect(&webview, bounds);
                if parked {
                    position.x = -size.width - PARK_GAP;
                }
                let rect = Rect {
                    position: position.into(),
                    size: size.into(),
                };
                state
                    .inner
                    .lock()
                    .unwrap()
                    .placed
                    .insert(tab_id.clone(), rect);
                set_view_bounds(&view, rect).await?;
                view
            }
            None if parked => return Err("no page to park".into()),
            None => {
                let url = parse_page_url(
                    &state,
                    tab_id,
                    url.as_deref().ok_or("no address to open")?,
                    &own_ports(&app),
                )?;
                #[cfg(target_os = "macos")]
                crate::browser_layer::install(&webview);
                #[cfg(target_os = "linux")]
                crate::browser_linux::install(&webview).await?;
                create_view(&webview, tab_id, url, bounds).await?
            }
        }),
        _ => None,
    };
    let shown = tab_id.filter(|_| target.is_some());
    // A resize only moves the shown view; showing it and hiding the rest is for a switch.
    if state.inner.lock().unwrap().shown == shown {
        return Ok(());
    }
    if let Some(view) = &target {
        view.show().map_err(|error| error.to_string())?;
    }
    let capturing: HashSet<String> = state
        .inner
        .lock()
        .unwrap()
        .capturing
        .iter()
        .filter_map(|tab_id| label_for(tab_id).ok())
        .collect();
    for view in browser_views(&app) {
        if Some(view.label()) != target.as_ref().map(|view| view.label())
            && !capturing.contains(view.label())
        {
            let _ = view.hide();
        }
    }
    set_shown(&state, &mut state.inner.lock().unwrap(), shown);
    Ok(())
}

async fn set_view_bounds<R: Runtime>(page: &Webview<R>, rect: Rect) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    {
        let scale = page
            .window()
            .scale_factor()
            .map_err(|error| error.to_string())?;
        crate::browser_linux::set_bounds(
            page,
            rect.position.to_logical(scale),
            rect.size.to_logical(scale),
        )
        .await
    }
    #[cfg(not(target_os = "linux"))]
    page.set_bounds(rect).map_err(|error| error.to_string())
}

/// Opens a tab's page without showing it: the agent at work in a chat that isn't on screen. Made
/// just off the window and hidden, so nothing flashes and the page on screen stays.
#[tauri::command]
pub async fn browser_view_open_background<R: Runtime>(
    webview: Webview<R>,
    tab_id: String,
    url: String,
    bounds: ViewBounds,
) -> Result<(), String> {
    require_main(&webview)?;
    let label = label_for(&tab_id)?;
    if webview.app_handle().get_webview(&label).is_some() {
        return Ok(());
    }
    let url = parse_page_url(
        &webview.state::<BrowserViews>(),
        &tab_id,
        &url,
        &own_ports(webview.app_handle()),
    )?;
    #[cfg(target_os = "macos")]
    crate::browser_layer::install(&webview);
    #[cfg(target_os = "linux")]
    crate::browser_linux::install(&webview).await?;
    let off_window = ViewBounds {
        x: -bounds.width - PARK_GAP,
        ..bounds
    };
    let page = create_view(&webview, &tab_id, url, &off_window).await?;
    if let Err(error) = page.hide() {
        // The frontend never records a view that failed to open, so it would never close this one.
        let _ = close_view(&page);
        return Err(error.to_string());
    }
    Ok(())
}

/// A hidden page paints nothing to capture (WebView2 doesn't render it; WebKit's snapshot waits
/// on a paint that never comes), so it shows just off the window for the shot. True if it did.
pub(crate) async fn show_for_capture<R: Runtime>(
    views: &BrowserViews,
    page: &Webview<R>,
    tab_id: &str,
) -> bool {
    let placed = {
        let mut inner = views.inner.lock().unwrap();
        if inner.shown.as_deref() == Some(tab_id) || !inner.capturing.insert(tab_id.to_string()) {
            return false;
        }
        inner.placed.get(tab_id).copied()
    };
    // A GTK view opened hidden has not received its requested allocation yet.
    let size = placed
        .or_else(|| page.bounds().ok())
        .zip(page.window().scale_factor().ok())
        .map(|(rect, factor)| rect.size.to_logical::<f64>(factor))
        .unwrap_or(LogicalSize::new(1024.0, 768.0));
    let position = LogicalPosition::new(-size.width - PARK_GAP, 0.0);
    let _ = set_view_bounds(
        page,
        Rect {
            position: position.into(),
            size: size.into(),
        },
    )
    .await;
    if page.show().is_err() {
        views.inner.lock().unwrap().capturing.remove(tab_id);
        return false;
    }
    true
}

/// Hides a page `show_for_capture` showed. One that became the shown page meanwhile goes back
/// where the switch put it, not off the window.
pub(crate) async fn end_capture<R: Runtime>(views: &BrowserViews, page: &Webview<R>, tab_id: &str) {
    let rect = {
        let mut inner = views.inner.lock().unwrap();
        inner.capturing.remove(tab_id);
        if inner.shown.as_deref() != Some(tab_id) {
            let _ = page.hide();
            return;
        }
        inner.placed.get(tab_id).copied()
    };
    if let Some(rect) = rect {
        let _ = set_view_bounds(page, rect).await;
    }
}

#[tauri::command]
pub async fn browser_view_navigate<R: Runtime>(
    webview: Webview<R>,
    tab_id: String,
    url: String,
) -> Result<(), String> {
    require_main(&webview)?;
    let views = webview.state::<BrowserViews>();
    let url = parse_page_url(&views, &tab_id, &url, &own_ports(webview.app_handle()))?;
    let page = view(webview.app_handle(), &tab_id)?;
    let origin = loopback_origin(&url);
    let previous = views
        .inner
        .lock()
        .unwrap()
        .local_pages
        .get(&tab_id)
        .map(|page| page.origin.clone());
    views.inner.lock().unwrap().committed.remove(&tab_id);
    if previous != origin {
        let rect = views
            .inner
            .lock()
            .unwrap()
            .placed
            .get(&tab_id)
            .copied()
            .ok_or("no page bounds")?;
        let scale = webview.window().scale_factor().map_err(|e| e.to_string())?;
        let main = webview.bounds().map_err(|e| e.to_string())?;
        let position = rect.position.to_logical::<f64>(scale);
        let main_position = main.position.to_logical::<f64>(scale);
        let size = rect.size.to_logical::<f64>(scale);
        let bounds = ViewBounds {
            x: position.x - main_position.x,
            y: position.y - main_position.y,
            width: size.width,
            height: size.height,
            viewport_width: main.size.to_logical::<f64>(scale).width,
        };
        close_view(&page).map_err(|e| e.to_string())?;
        views.inner.lock().unwrap().local_pages.remove(&tab_id);
        let page = create_view(&webview, &tab_id, url, &bounds).await?;
        // Linux attaches new views hidden, and `browser_view_show` won't show a tab already shown.
        if views.inner.lock().unwrap().shown.as_deref() == Some(tab_id.as_str()) {
            page.show().map_err(|error| error.to_string())?;
        } else {
            let _ = page.hide();
        }
        return Ok(());
    }
    page.navigate(url).map_err(|error| error.to_string())
}

/// Only the address bar grants access; the agent must reuse this tab.
#[tauri::command]
pub fn browser_view_grant<R: Runtime>(
    webview: Webview<R>,
    tab_id: String,
    url: String,
) -> Result<(), String> {
    require_main(&webview)?;
    grant_local(
        &webview.state::<BrowserViews>(),
        &tab_id,
        &url,
        &own_ports(webview.app_handle()),
    )
}

#[tauri::command]
pub fn browser_view_action<R: Runtime>(
    webview: Webview<R>,
    tab_id: String,
    action: String,
) -> Result<(), String> {
    require_main(&webview)?;
    let page = view(webview.app_handle(), &tab_id)?;
    #[cfg(target_os = "linux")]
    if matches!(action.as_str(), "back" | "forward" | "stop") {
        return page
            .with_webview(move |platform| {
                use webkit2gtk::WebViewExt;
                match action.as_str() {
                    "back" => platform.inner().go_back(),
                    "forward" => platform.inner().go_forward(),
                    _ => platform.inner().stop_loading(),
                }
            })
            .map_err(|error| error.to_string());
    }
    let result = match action.as_str() {
        "back" => page.eval("history.back()"),
        "forward" => page.eval("history.forward()"),
        "reload" => page.reload(),
        "stop" => page.eval("window.stop()"),
        "focus" => page.set_focus(),
        // Key focus back to the panel, e.g. for an annotation comment.
        "blur" => webview.set_focus(),
        _ => return Err("unknown action".into()),
    };
    result.map_err(|error| error.to_string())
}

#[tauri::command]
pub fn browser_view_zoom<R: Runtime>(
    webview: Webview<R>,
    tab_id: String,
    zoom: f64,
) -> Result<(), String> {
    require_main(&webview)?;
    if !(0.25..=5.0).contains(&zoom) {
        return Err("zoom out of range".into());
    }
    view(webview.app_handle(), &tab_id)?
        .set_zoom(zoom)
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub async fn browser_view_find<R: Runtime>(
    webview: Webview<R>,
    tab_id: String,
    query: String,
    backwards: bool,
) -> Result<bool, String> {
    require_main(&webview)?;
    if query.is_empty() || query.len() > 1000 {
        return Ok(false);
    }
    let page = view(webview.app_handle(), &tab_id)?;
    let query = serde_json::to_string(&query).map_err(|error| error.to_string())?;
    let script = format!(
        "(() => {{ try {{ return Boolean(window.find({query}, false, {backwards}, true, false, false, false)); }} catch {{ return false; }} }})()"
    );
    let (tx, rx) = tokio::sync::oneshot::channel();
    let tx = Mutex::new(Some(tx));
    page.eval_with_callback(script, move |result| {
        if let Some(tx) = tx.lock().unwrap().take() {
            let _ = tx.send(result.trim() == "true");
        }
    })
    .map_err(|error| error.to_string())?;
    Ok(tokio::time::timeout(Duration::from_secs(5), rx)
        .await
        .ok()
        .and_then(Result::ok)
        .unwrap_or(false))
}

// Same cap as the page shell's `install` (routes/browser.py).
const MAX_ANNOTATE_CODE: usize = 262_144;
const MAX_ANNOTATE_NUMBERS: usize = 500;

/// Commands for a page's annotate code (`_ANNOTATE_JS` in routes/browser.py).
#[derive(Debug, Deserialize)]
#[serde(tag = "command", rename_all = "camelCase")]
pub enum AnnotateCommand {
    Install { code: String },
    Start { color: String },
    Stop,
    Forget { id: u32 },
    Number { numbers: Vec<(u32, u32)> },
    Poll,
}

// Installs the annotate code under a non-enumerable key. `post` queues reports for the panel to
// poll, keeping only the latest `rects`. Each call returns whether the code is installed (a
// navigation drops it) and the reports since the last call.
const ANNOTATE_SCRIPT: &str = r#"(() => {
  const key = Symbol.for("unsloth.annotate");
  const command = __COMMAND__;
  let shell = window[key];
  if (!shell && command.command === "install") {
    const queue = [];
    const post = (message) => {
      if (!message || message.type !== "annotate") return;
      if (message.event === "rects") {
        const at = queue.findIndex((queued) => queued.event === "rects");
        if (at !== -1) queue.splice(at, 1);
      }
      if (queue.length < 200) queue.push(message);
    };
    try {
      shell = { queue, annotation: (function (post) {
__CODE__
      })(post) };
      Object.defineProperty(window, key, { value: shell, configurable: true });
    } catch { shell = null; }
  }
  if (!shell) return JSON.stringify({ installed: false, events: [] });
  const { annotation, queue } = shell;
  if (command.command === "start") annotation.start(command.color);
  else if (command.command === "stop") annotation.stop();
  else if (command.command === "forget") annotation.forget(command.id);
  else if (command.command === "number") annotation.number(command.numbers);
  return JSON.stringify({ installed: true, events: queue.splice(0) });
})()"#;

fn annotate_script(command: &AnnotateCommand) -> Result<String, String> {
    let (args, code) = match command {
        AnnotateCommand::Install { code } => {
            if code.len() > MAX_ANNOTATE_CODE {
                return Err("annotate code too large".into());
            }
            (serde_json::json!({ "command": "install" }), code.as_str())
        }
        AnnotateCommand::Start { color } => (
            serde_json::json!({ "command": "start", "color": color.chars().take(64).collect::<String>() }),
            "",
        ),
        AnnotateCommand::Stop => (serde_json::json!({ "command": "stop" }), ""),
        AnnotateCommand::Forget { id } => {
            (serde_json::json!({ "command": "forget", "id": id }), "")
        }
        AnnotateCommand::Number { numbers } => {
            let numbers: Vec<_> = numbers.iter().take(MAX_ANNOTATE_NUMBERS).collect();
            (
                serde_json::json!({ "command": "number", "numbers": numbers }),
                "",
            )
        }
        AnnotateCommand::Poll => (serde_json::json!({ "command": "poll" }), ""),
    };
    // Spliced in, not compiled in the page, so a CSP without 'unsafe-eval' can't block it.
    // Single pass, so neither part can inject the other's placeholder.
    let (head, rest) = ANNOTATE_SCRIPT
        .split_once("__COMMAND__")
        .ok_or("bad annotate script")?;
    let (middle, tail) = rest.split_once("__CODE__").ok_or("bad annotate script")?;
    Ok(format!("{head}{args}{middle}{code}{tail}"))
}

/// Runs an annotate command in a tab's page and returns `{ installed, events }`. The panel
/// validates events like frame messages, since the page can write to the queue.
#[tauri::command]
pub async fn browser_view_annotate<R: Runtime>(
    webview: Webview<R>,
    tab_id: String,
    command: AnnotateCommand,
) -> Result<serde_json::Value, String> {
    require_main(&webview)?;
    let page = view(webview.app_handle(), &tab_id)?;
    let script = annotate_script(&command)?;
    let (tx, rx) = tokio::sync::oneshot::channel();
    let tx = Mutex::new(Some(tx));
    page.eval_with_callback(script, move |result| {
        if let Some(tx) = tx.lock().unwrap().take() {
            let _ = tx.send(result);
        }
    })
    .map_err(|error| error.to_string())?;
    let result = tokio::time::timeout(Duration::from_secs(5), rx)
        .await
        .map_err(|_| "page did not answer".to_string())?
        .map_err(|error| error.to_string())?;
    // The script returns a JSON string, which the engine JSON-encodes again.
    let inner = serde_json::from_str::<String>(&result).map_err(|error| error.to_string())?;
    serde_json::from_str(&inner).map_err(|error| error.to_string())
}

// The panel's browser agent (frontend agent/page-agent.ts): installed on first use per page, then
// each call runs one command and returns `{ installed, result }`.
const AGENT_SCRIPT: &str = r#"(() => {
  const key = Symbol.for("unsloth.agent");
  const command = __COMMAND__;
  if (!window[key] && command.install) {
    try {
__CODE__
    } catch (error) {}
  }
  const agent = window[key];
  if (!agent) return JSON.stringify({ installed: false, result: null });
  return JSON.stringify({ installed: true, result: agent.run(command.op) });
})()"#;
const MAX_AGENT_OP: usize = 65_536;

#[derive(Debug, Deserialize)]
pub struct AgentCommand {
    op: serde_json::Map<String, serde_json::Value>,
    code: Option<String>,
}

fn agent_script(command: &AgentCommand) -> Result<String, String> {
    let code = command.code.as_deref().unwrap_or("");
    if code.len() > MAX_ANNOTATE_CODE {
        return Err("agent code too large".into());
    }
    let args = serde_json::json!({ "install": !code.is_empty(), "op": command.op }).to_string();
    if args.len() > MAX_AGENT_OP {
        return Err("agent command too large".into());
    }
    let (head, rest) = AGENT_SCRIPT
        .split_once("__COMMAND__")
        .ok_or("bad agent script")?;
    let (middle, tail) = rest.split_once("__CODE__").ok_or("bad agent script")?;
    Ok(format!("{head}{args}{middle}{code}{tail}"))
}

/// Runs a browser agent command in a tab's page. The panel checks the result, since the page
/// shares the agent's world and can write to it.
#[tauri::command]
pub async fn browser_view_agent<R: Runtime>(
    webview: Webview<R>,
    tab_id: String,
    command: AgentCommand,
) -> Result<serde_json::Value, String> {
    require_main(&webview)?;
    let page = view(webview.app_handle(), &tab_id)?;
    let script = agent_script(&command)?;
    eval_json(&page, script).await
}

// Spliced past the CSP; the document check is its own statement since spliced code can close any wrapper.
const EVAL_SCRIPT: &str = r#"(() => {
  const agent = window[Symbol.for("unsloth.agent")];
  if (!agent || typeof agent.owns !== "function" || agent.owns(__DOC__) !== true) throw new Error("not the approved document");
})();
JSON.stringify({ installed: true, result: window[Symbol.for("unsloth.agent")].evaluate(__ID__, __DOC__, async () => {
__CODE__
}) });"#;
// browser_agent.py allows 8000 characters, of up to 4 bytes each.
const MAX_EVAL_CODE: usize = 32_000;

fn eval_script(id: &str, doc: &str, code: &str) -> Result<String, String> {
    if doc.len() < 2
        || doc.len() > 16
        || !doc.starts_with('r')
        || !doc[1..].chars().all(|c| c.is_ascii_digit())
    {
        return Err("bad document ref".into());
    }
    if id.is_empty()
        || id.len() > 64
        || !id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err("bad script id".into());
    }
    if code.len() > MAX_EVAL_CODE {
        return Err("script too large".into());
    }
    // Single pass over the template, so the code can't inject a placeholder.
    let mut out = String::with_capacity(EVAL_SCRIPT.len() + code.len() + 96);
    let mut rest = EVAL_SCRIPT;
    for (marker, value) in [
        ("__DOC__", format!("\"{doc}\"")),
        ("__ID__", format!("\"{id}\"")),
        ("__DOC__", format!("\"{doc}\"")),
        ("__CODE__", code.to_string()),
    ] {
        let (head, tail) = rest.split_once(marker).ok_or("bad eval script")?;
        out.push_str(head);
        out.push_str(&value);
        rest = tail;
    }
    out.push_str(rest);
    Ok(out)
}

#[tauri::command]
pub async fn browser_view_eval<R: Runtime>(
    webview: Webview<R>,
    tab_id: String,
    id: String,
    doc: String,
    code: String,
) -> Result<serde_json::Value, String> {
    require_main(&webview)?;
    let page = view(webview.app_handle(), &tab_id)?;
    let script = eval_script(&id, &doc, &code)?;
    eval_json(&page, script).await
}

async fn eval_json<R: Runtime>(
    page: &Webview<R>,
    script: String,
) -> Result<serde_json::Value, String> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let tx = Mutex::new(Some(tx));
    page.eval_with_callback(script, move |result| {
        if let Some(tx) = tx.lock().unwrap().take() {
            let _ = tx.send(result);
        }
    })
    .map_err(|error| error.to_string())?;
    let result = tokio::time::timeout(Duration::from_secs(5), rx)
        .await
        .map_err(|_| "page did not answer".to_string())?
        .map_err(|error| error.to_string())?;
    let inner = serde_json::from_str::<String>(&result).map_err(|error| error.to_string())?;
    serde_json::from_str(&inner).map_err(|error| error.to_string())
}

/// Mute or unmute a tab's page. Windows mutes the whole view (Web Audio too) and keeps it muted
/// across loads; macOS mutes the page's media, and each page it loads after (see `MUTE_SCRIPT`).
#[tauri::command]
pub async fn browser_view_mute<R: Runtime>(
    webview: Webview<R>,
    state: State<'_, BrowserViews>,
    tab_id: String,
    muted: bool,
) -> Result<(), String> {
    require_main(&webview)?;
    label_for(&tab_id)?;
    {
        let mut inner = state.inner.lock().unwrap();
        if muted {
            inner.muted.insert(tab_id.clone());
        } else {
            inner.muted.remove(&tab_id);
        }
    }
    // No view yet: it is muted as it opens (`create_view`).
    let Ok(page) = view(webview.app_handle(), &tab_id) else {
        return Ok(());
    };
    apply_mute(&page, muted)
}

#[cfg(target_os = "macos")]
fn apply_mute<R: Runtime>(page: &Webview<R>, muted: bool) -> Result<(), String> {
    page.eval(format!("{MUTE_SCRIPT}({muted})"))
        .map_err(|error| error.to_string())
}

#[cfg(windows)]
fn apply_mute<R: Runtime>(page: &Webview<R>, muted: bool) -> Result<(), String> {
    page.with_webview(move |platform| {
        use webview2_com::Microsoft::Web::WebView2::Win32::ICoreWebView2_8;
        use windows_core::Interface;
        let _ = unsafe {
            platform
                .controller()
                .CoreWebView2()
                .and_then(|webview| webview.cast::<ICoreWebView2_8>())
                .and_then(|webview| webview.SetIsMuted(muted))
        };
    })
    .map_err(|error| error.to_string())
}

#[cfg(target_os = "linux")]
fn apply_mute<R: Runtime>(page: &Webview<R>, muted: bool) -> Result<(), String> {
    page.with_webview(move |platform| {
        use webkit2gtk::WebViewExt;
        platform.inner().set_is_muted(muted);
    })
    .map_err(|error| error.to_string())
}

#[cfg(not(any(target_os = "macos", windows, target_os = "linux")))]
fn apply_mute<R: Runtime>(_page: &Webview<R>, _muted: bool) -> Result<(), String> {
    Ok(())
}

#[tauri::command]
pub fn browser_view_close<R: Runtime>(
    webview: Webview<R>,
    state: State<'_, BrowserViews>,
    tab_id: String,
    forget_local: Option<bool>,
) -> Result<(), String> {
    require_main(&webview)?;
    {
        let mut inner = state.inner.lock().unwrap();
        inner.urls.remove(&tab_id);
        inner.committed.remove(&tab_id);
        inner.download_starts.remove(&tab_id);
        inner.placed.remove(&tab_id);
        inner.local_pages.remove(&tab_id);
        if forget_local.unwrap_or(true) {
            inner.local.remove(&tab_id);
        }
        // `muted` stays: a pruned view reopens muted; unmuting is what forgets it.
        if inner.shown.as_deref() == Some(tab_id.as_str()) {
            set_shown(&state, &mut inner, None);
        }
    }
    if let Ok(page) = view(webview.app_handle(), &tab_id) {
        close_view(&page).map_err(|error| error.to_string())?;
    }
    Ok(())
}

#[tauri::command]
pub async fn browser_view_clear_data<R: Runtime>(
    webview: Webview<R>,
    state: State<'_, BrowserViews>,
    close_views: Option<bool>,
) -> Result<(), String> {
    require_main(&webview)?;
    let app = webview.app_handle().clone();
    // An account switch closes the pages first, so none can write the old account's data back.
    let closing = close_views.unwrap_or(false);
    let local_tabs = {
        let mut inner = state.inner.lock().unwrap();
        inner.local.clear();
        let tabs: Vec<_> = inner.local_pages.keys().cloned().collect();
        inner.local_pages.clear();
        tabs
    };
    for tab in local_tabs {
        if let Ok(page) = view(&app, &tab) {
            let _ = close_view(&page);
        }
    }
    if closing {
        {
            let mut inner = state.inner.lock().unwrap();
            inner.urls.clear();
            inner.committed.clear();
            set_shown(&state, &mut inner, None);
        }
        for page in browser_views(&app) {
            let _ = close_view(&page);
        }
    }
    let live = if closing {
        None
    } else {
        browser_views(&app).into_iter().next()
    };
    let hidden = live.is_none();
    let page = match live {
        Some(page) => page,
        None => {
            let builder = with_page_profile(
                WebviewBuilder::new(
                    format!("{LABEL_PREFIX}clear"),
                    WebviewUrl::External(Url::parse("about:blank").unwrap()),
                )
                .focused(false),
                &app,
                None,
            )?;
            #[cfg(target_os = "linux")]
            crate::browser_linux::install(&webview).await?;
            let page = webview
                .window()
                .add_child(
                    builder,
                    LogicalPosition::new(0.0, 0.0),
                    LogicalSize::new(1.0, 1.0),
                )
                .map_err(|error| error.to_string())?;
            let _ = page.hide();
            page
        }
    };
    let result = clear_profile(&page).await;
    if hidden {
        let _ = close_view(&page);
    }
    result
}

const CLEAR_TIMEOUT: Duration = Duration::from_secs(30);

/// Clear a page's profile, resolving once the engine reports it done: wry's own call starts the
/// clear and returns before it finishes.
async fn clear_profile<R: Runtime>(page: &Webview<R>) -> Result<(), String> {
    #[cfg(any(target_os = "macos", windows, target_os = "linux"))]
    {
        let (done, finished) = tokio::sync::oneshot::channel::<Result<(), String>>();
        let done = Mutex::new(Some(done));
        let finish: ClearFinish = std::sync::Arc::new(move |result| {
            if let Some(done) = done.lock().unwrap().take() {
                let _ = done.send(result);
            }
        });
        page.with_webview(move |platform| platform_clear(platform, finish))
            .map_err(|error| error.to_string())?;
        match tokio::time::timeout(CLEAR_TIMEOUT, finished).await {
            Ok(Ok(result)) => result,
            _ => Err("Clearing browsing data didn't finish".into()),
        }
    }
    #[cfg(not(any(target_os = "macos", windows, target_os = "linux")))]
    {
        page.clear_all_browsing_data()
            .map_err(|error| error.to_string())?;
        tokio::time::sleep(Duration::from_secs(2)).await;
        Ok(())
    }
}

#[cfg(any(target_os = "macos", windows, target_os = "linux"))]
type ClearFinish = std::sync::Arc<dyn Fn(Result<(), String>) + Send + Sync>;

#[cfg(target_os = "macos")]
fn platform_clear(platform: tauri::webview::PlatformWebview, finish: ClearFinish) {
    use objc2_foundation::NSDate;
    use objc2_web_kit::{WKWebView, WKWebsiteDataStore};
    let Some(mtm) = objc2::MainThreadMarker::new() else {
        return finish(Err("Not on the main thread".into()));
    };
    unsafe {
        let view = &*(platform.inner() as *const WKWebView);
        let store = view.configuration().websiteDataStore();
        let types = WKWebsiteDataStore::allWebsiteDataTypes(mtm);
        let date = NSDate::dateWithTimeIntervalSince1970(0.0);
        let handler = block2::RcBlock::new(move || finish(Ok(())));
        store.removeDataOfTypes_modifiedSince_completionHandler(&types, &date, &handler);
    }
}

#[cfg(target_os = "linux")]
fn platform_clear(platform: tauri::webview::PlatformWebview, finish: ClearFinish) {
    use webkit2gtk::{WebViewExt, WebsiteDataManagerExtManual, WebsiteDataTypes};
    let Some(manager) = platform.inner().website_data_manager() else {
        return finish(Err("Browser profile is unavailable".into()));
    };
    manager.clear(
        WebsiteDataTypes::all(),
        glib::TimeSpan::from_seconds(0),
        None::<&gtk::gio::Cancellable>,
        move |result| finish(result.map_err(|error| error.to_string())),
    );
}

#[cfg(windows)]
fn platform_clear(platform: tauri::webview::PlatformWebview, finish: ClearFinish) {
    use webview2_com::ClearBrowsingDataCompletedHandler;
    use webview2_com::Microsoft::Web::WebView2::Win32::{ICoreWebView2Profile2, ICoreWebView2_13};
    use windows_core::Interface;
    let callback = finish.clone();
    let started = unsafe {
        platform
            .controller()
            .CoreWebView2()
            .and_then(|webview| webview.cast::<ICoreWebView2_13>())
            .and_then(|webview| webview.Profile())
            .and_then(|profile| profile.cast::<ICoreWebView2Profile2>())
            .and_then(|profile| {
                profile.ClearBrowsingDataAll(&ClearBrowsingDataCompletedHandler::create(Box::new(
                    move |result| {
                        callback(result.map_err(|error| error.to_string()));
                        Ok(())
                    },
                )))
            })
    };
    if let Err(error) = started {
        finish(Err(error.to_string()));
    }
}

/// The reservation a finished download of one URL settles, and any it retires. No path: the first on
/// macOS, which reports none and runs downloads of one URL in turn. Elsewhere a missing path (WebView2
/// sends none for a failed download) or one matching none stands for the sole reservation only, else
/// settling one of several would hand it another download's result. Such a finish is counted instead:
/// once every other download has finished, the reservations left are exactly the counted ones, and
/// they are retired as failed so none holds a download slot or waits forever.
fn take_finished(
    pending: &mut Vec<PathBuf>,
    unexplained: &mut usize,
    path: Option<&Path>,
) -> (Option<PathBuf>, Vec<PathBuf>) {
    let sole = (pending.len() == 1).then_some(0);
    let index = match path {
        None if cfg!(target_os = "macos") => Some(0),
        None => sole,
        Some(path) => pending
            .iter()
            .position(|p| crate::browser_downloads::same_path(p, path))
            .or(sole),
    };
    let recorded = match index.filter(|&index| index < pending.len()) {
        Some(index) => Some(pending.remove(index)),
        None => {
            if !pending.is_empty() {
                *unexplained += 1;
            }
            None
        }
    };
    let mut retired = Vec::new();
    if *unexplained > 0 && pending.len() <= *unexplained {
        retired = std::mem::take(pending);
        *unexplained = 0;
    }
    (recorded, retired)
}

#[cfg(test)]
mod tests {
    #[test]
    fn page_capture_requests_are_never_granted_silently() {
        for kind in [
            PermissionKind::Microphone,
            PermissionKind::Camera,
            PermissionKind::DisplayCapture,
            PermissionKind::Other,
        ] {
            assert!(
                !matches!(page_permission(kind), PermissionResponse::Allow),
                "{kind:?} must not be granted without asking"
            );
        }
    }

    #[test]
    fn every_page_webview_gets_the_permission_handler() {
        // A Windows checkout may carry CRLF line endings.
        let source = include_str!("browser_webview.rs").replace("\r\n", "\n");
        let code = &source[..source.find("#[cfg(test)]\nmod tests").unwrap()];
        let start = code.find("fn with_page_profile").unwrap();
        let body = &code[start..start + code[start..].find("\n}\n").unwrap()];
        assert!(
            body.contains(".on_permission_request("),
            "with_page_profile must install the page permission handler"
        );
        let builders = code.matches("WebviewBuilder::new(").count();
        let profiled = code.matches("with_page_profile(").count();
        assert!(builders > 0);
        assert_eq!(
            builders,
            profiled,
            "every page WebviewBuilder must go through with_page_profile"
        );
    }

    #[test]
    fn a_finished_path_matching_no_reservation_settles_only_a_sole_one() {
        let (a, b) = (PathBuf::from("/s/a.zip"), PathBuf::from("/s/b.zip"));
        let mut n = 0;
        let mut two = vec![a.clone(), b.clone()];
        let found = take_finished(&mut two, &mut n, Some(Path::new("/s/b.zip")));
        assert_eq!(found, (Some(b.clone()), vec![]));
        let mut two = vec![a.clone(), b.clone()];
        assert_eq!(
            take_finished(&mut two, &mut n, Some(Path::new("/elsewhere"))),
            (None, vec![])
        );
        assert_eq!((two.len(), n), (2, 1));
        // The other one finishes: the one left is the unexplained finish, retired as failed.
        assert_eq!(
            take_finished(&mut two, &mut n, Some(Path::new("/s/b.zip"))),
            (Some(b.clone()), vec![a.clone()])
        );
        assert_eq!((two.len(), n), (0, 0));
        let mut one = vec![b.clone()];
        assert_eq!(
            take_finished(&mut one, &mut n, Some(Path::new("/elsewhere"))),
            (Some(b.clone()), vec![])
        );
        assert_eq!(take_finished(&mut Vec::new(), &mut n, None), (None, vec![]));
        assert_eq!(n, 0);
    }

    #[test]
    fn a_finish_without_a_path_takes_the_first_only_where_downloads_run_in_turn() {
        let (a, b) = (PathBuf::from("/s/a.zip"), PathBuf::from("/s/b.zip"));
        let mut n = 0;
        let mut two = vec![a.clone(), b.clone()];
        if cfg!(target_os = "macos") {
            assert_eq!(take_finished(&mut two, &mut n, None), (Some(a), vec![]));
            return;
        }
        // WebView2: a failed download reports no path. Which one is unknown until the other ends.
        assert_eq!(take_finished(&mut two, &mut n, None), (None, vec![]));
        assert_eq!(
            take_finished(&mut two, &mut n, Some(Path::new("/s/a.zip"))),
            (Some(a), vec![b])
        );
        let mut one = vec![PathBuf::from("/s/c.zip")];
        assert!(take_finished(&mut one, &mut n, None).0.is_some());
    }

    use super::*;

    #[test]
    fn the_internet_mark_keeps_where_not_secrets() {
        let source = |url: &str| sanitized_source(&Url::parse(url).unwrap());
        assert_eq!(
            source("https://user:pass@example.com:8443/f/a.exe?sig=secret#x").as_deref(),
            Some("https://example.com:8443/f/a.exe")
        );
        assert_eq!(source("file:///etc/passwd"), None);
    }

    mod url_poll {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        const TEN_MINUTES: Duration = Duration::from_secs(600);

        /// Runs the poll for `span` from `start`, returning the reads it made.
        async fn reads(
            start: PollGate,
            span: Duration,
        ) -> (watch::Sender<PollGate>, Arc<AtomicUsize>) {
            let (gate, receiver) = watch::channel(start);
            let count = Arc::new(AtomicUsize::new(0));
            let counter = count.clone();
            tokio::spawn(run_url_poll(receiver, move || {
                counter.fetch_add(1, Ordering::SeqCst);
                async {}
            }));
            tokio::time::sleep(span).await;
            (gate, count)
        }

        fn gate(shown: bool, window_active: bool) -> PollGate {
            PollGate {
                shown,
                window_active,
                switches: 0,
            }
        }

        #[tokio::test(start_paused = true)]
        async fn polls_every_800_ms_only_while_a_view_shows_in_a_window_in_use() {
            // One at once, then 600 s / 0.8 s = 750 (just past the last one, so no tie with it).
            let span = TEN_MINUTES + Duration::from_millis(1);
            assert_eq!(
                reads(gate(true, true), span).await.1.load(Ordering::SeqCst),
                751
            );
            for idle in [gate(true, false), gate(false, true), gate(false, false)] {
                assert_eq!(
                    reads(idle, TEN_MINUTES).await.1.load(Ordering::SeqCst),
                    0,
                    "{idle:?}"
                );
            }
        }

        #[tokio::test(start_paused = true)]
        async fn coming_back_reads_at_once_then_resumes_the_interval() {
            let (gate_tx, count) = reads(gate(true, false), TEN_MINUTES).await;
            gate_tx.send_modify(|gate| gate.window_active = true);
            tokio::time::sleep(Duration::from_millis(1)).await;
            assert_eq!(count.load(Ordering::SeqCst), 1);
            tokio::time::sleep(URL_POLL * 10).await;
            assert_eq!(count.load(Ordering::SeqCst), 11);
            gate_tx.send_modify(|gate| gate.window_active = false);
            tokio::time::sleep(TEN_MINUTES).await;
            assert_eq!(count.load(Ordering::SeqCst), 11);
        }

        #[tokio::test(start_paused = true)]
        async fn rapid_flips_read_once_per_opening_and_dropping_the_gate_ends_the_poll() {
            let (gate_tx, count) = reads(gate(false, true), Duration::from_millis(1)).await;
            for _ in 0..5 {
                gate_tx.send_modify(|gate| gate.shown = true);
                tokio::time::sleep(Duration::from_millis(1)).await;
                gate_tx.send_modify(|gate| gate.shown = false);
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            assert_eq!(count.load(Ordering::SeqCst), 5);
            gate_tx.send_modify(|gate| gate.shown = true);
            drop(gate_tx);
            tokio::time::sleep(TEN_MINUTES).await;
            // The read made as it opened, then nothing: the poll ended with its gate.
            assert_eq!(count.load(Ordering::SeqCst), 6);
        }

        #[test]
        fn shown_is_tracked_with_the_gate() {
            let views = BrowserViews::default();
            let mut inner = ViewsState::default();
            set_shown(&views, &mut inner, Some("a".into()));
            assert!(views.gate.borrow().polls());
            set_shown(&views, &mut inner, None);
            assert!(!views.gate.borrow().polls());
            assert_eq!(inner.shown, None);
        }

        #[tokio::test(start_paused = true)]
        async fn switching_between_shown_tabs_reads_at_once() {
            let views = BrowserViews::default();
            let mut inner = ViewsState::default();
            set_shown(&views, &mut inner, Some("a".into()));
            let count = Arc::new(AtomicUsize::new(0));
            let counter = count.clone();
            tokio::spawn(run_url_poll(views.gate.subscribe(), move || {
                counter.fetch_add(1, Ordering::SeqCst);
                async {}
            }));
            tokio::time::sleep(Duration::from_millis(1)).await;
            assert_eq!(count.load(Ordering::SeqCst), 1);
            set_shown(&views, &mut inner, Some("b".into()));
            tokio::time::sleep(Duration::from_millis(1)).await;
            assert_eq!(count.load(Ordering::SeqCst), 2);
            // Showing the same tab again is not a switch.
            set_shown(&views, &mut inner, Some("b".into()));
            tokio::time::sleep(Duration::from_millis(1)).await;
            assert_eq!(count.load(Ordering::SeqCst), 2);
        }
    }

    fn allowed(url: &str) -> bool {
        navigation_allowed(&Url::parse(url).unwrap())
    }

    #[test]
    fn page_addresses_are_bounded() {
        let long = format!("https://example.com/{}", "a".repeat(MAX_URL_CHARS));
        assert!(!reportable(&long));
        assert!(reportable("https://example.com/"));
    }

    #[test]
    fn windows_pages_keep_smartscreen_and_the_proxy() {
        let args = page_browser_args("127.0.0.1:9");
        assert!(!args.contains("SmartScreen"), "{args}");
        assert!(args.contains("--proxy-server=http://127.0.0.1:9"), "{args}");
        assert!(args.contains("--proxy-bypass-list=<-loopback>"), "{args}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mac_pages_send_the_installed_safaris_user_agent() {
        // plutil reads binary and XML plists alike.
        let out = std::process::Command::new("/usr/bin/plutil")
            .args(["-extract", "CFBundleShortVersionString", "raw", "-o", "-"])
            .arg("/Applications/Safari.app/Contents/Info.plist")
            .output()
            .unwrap();
        assert!(out.status.success(), "{out:?}");
        let version = String::from_utf8(out.stdout).unwrap();
        let version = version.trim();
        assert!(!version.is_empty());
        assert_eq!(
            page_user_agent().unwrap(),
            format!(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{version} Safari/605.1.15"
            )
        );
    }

    #[test]
    fn only_public_web_pages_load() {
        for url in [
            "https://example.com/",
            "https://93.184.216.34/",
            "https://[2606:4700::1111]/",
            "https://[64:ff9b::808:808]/",
            "about:blank",
            "data:,hi",
        ] {
            assert!(allowed(url), "{url}");
        }
        for url in [
            "tauri://localhost/",
            "http://ipc.localhost/plugin",
            "http://localhost:8888/",
            "http://127.1/",
            "http://0x7f000001/",
            "http://0.0.0.0:8080/",
            "http://[::ffff:127.0.0.1]/",
            "http://192.168.1.1/",
            "http://169.254.169.254/",
            "http://100.100.1.1/",
            "http://[fd00::1]/",
            "http://[64:ff9b::7f00:1]/",
            "http://198.18.0.1/",
            "http://192.0.0.8/",
            "http://224.0.0.1/",
            "http://240.0.0.1/",
            "http://[2001:db8::1]/",
            "http://[2002:7f00:1::1]/",
            "http://[ff02::1]/",
            "http://printer.local/",
            "http://router/",
            "file:///etc/passwd",
            "javascript:alert(1)",
        ] {
            assert!(!allowed(url), "{url}");
        }
    }

    #[test]
    fn a_local_address_opens_once_typed_and_only_at_that_origin() {
        let backend: &[u16] = &[8888];
        let views = new_browser_views();
        assert!(parse_page_url(&views, "a", "http://localhost:17860/", backend).is_err());
        grant_local(&views, "a", "http://localhost:17860", backend).unwrap();
        assert!(parse_page_url(&views, "a", "http://LOCALHOST:17860/app?x=1", backend).is_ok());
        for other in [
            "https://localhost:17860/",
            "http://localhost:17861/",
            "http://127.0.0.1:17860/",
            "http://[::1]:17860/",
            "http://localhost./",
            "http://app.localhost:17860/",
        ] {
            assert!(
                parse_page_url(&views, "a", other, backend).is_err(),
                "{other}"
            );
        }
        assert!(!allowed("http://localhost:17860/"));
        assert!(parse_page_url(&views, "b", "http://localhost:17860/", backend).is_err());
        views.inner.lock().unwrap().local.remove("a");
        assert!(parse_page_url(&views, "a", "http://localhost:17860/", backend).is_err());
    }

    #[test]
    fn opening_a_public_page_ends_the_tabs_local_grant() {
        let views = new_browser_views();
        grant_local(&views, "a", "http://localhost:17860", &[]).unwrap();
        assert!(parse_page_url(&views, "a", "http://localhost./", &[]).is_err());
        assert!(parse_page_url(&views, "a", "http://localhost:17860/", &[]).is_ok());
        assert!(parse_page_url(&views, "a", "https://example.com/", &[]).is_ok());
        assert!(parse_page_url(&views, "a", "http://localhost:17860/", &[]).is_err());
    }

    #[test]
    fn the_dev_server_the_app_is_served_from_is_never_granted() {
        let dev_url = Url::parse("http://localhost:5173").unwrap();
        assert_eq!(dev_server_port(true, Some(&dev_url)), Some(5173));
        assert_eq!(dev_server_port(false, Some(&dev_url)), None);
        assert_eq!(dev_server_port(true, None), None);
        let views = new_browser_views();
        assert_eq!(
            grant_local(&views, "a", "http://localhost:5173/", &[8888, 5173]),
            Err(STUDIO_ADDRESS.into())
        );
        // A release build is served from its own scheme, so a user's dev server on 5173 opens.
        grant_local(&views, "a", "http://localhost:5173/", &[8888]).unwrap();
    }

    #[test]
    fn studio_and_non_loopback_addresses_are_never_granted() {
        let views = new_browser_views();
        for url in [
            "http://localhost:8888/",
            "http://127.0.0.1:52011/",
            "http://[::1]:60000/",
        ] {
            assert_eq!(
                grant_local(&views, "a", url, &[8888]),
                Err(STUDIO_ADDRESS.into()),
                "{url}"
            );
            assert_eq!(
                parse_page_url(&views, "a", url, &[8888]),
                Err(STUDIO_ADDRESS.into()),
                "{url}"
            );
        }
        for port in 8888..=8908 {
            assert!(grant_local(&views, "a", &format!("http://localhost:{port}/"), &[]).is_err());
        }
        for url in [
            "http://localtest.me:3000/",
            "http://192.168.1.10:3000/",
            "http://[::ffff:127.0.0.1]:3000/",
            "http://studio.local:3000/",
            "ws://localhost:3000/",
            "file:///etc/passwd",
        ] {
            assert!(grant_local(&views, "a", url, &[]).is_err(), "{url}");
        }
    }

    #[test]
    fn content_rules_let_a_granted_origin_load_only_its_own_pages() {
        let exceptions = |granted: &str| -> Vec<serde_json::Value> {
            let origin = Url::parse(granted).unwrap().origin();
            let rules: serde_json::Value =
                serde_json::from_str(&content_rules_json(Some(&origin))).unwrap();
            rules
                .as_array()
                .unwrap()
                .iter()
                .filter(|rule| rule["action"]["type"] == "ignore-previous-rules")
                .cloned()
                .collect()
        };
        let excepted = |granted: &str, url: &str| {
            exceptions(granted).iter().any(|rule| {
                let filter = rule["trigger"]["url-filter"].as_str().unwrap();
                assert!(!filter.contains('|') && !filter.contains('{'), "{filter}");
                regex::Regex::new(filter).unwrap().is_match(url)
            })
        };
        for rule in exceptions("http://127.0.0.1:17870/") {
            assert_eq!(
                rule["trigger"]["if-top-url"],
                serde_json::json!([r"^http://([^/@]*@)?127\.0\.0\.1:17870/"])
            );
        }
        for url in [
            "http://127.0.0.1:17870/app.js",
            "ws://127.0.0.1:17870/hmr",
            "http://u@127.0.0.1:17870/",
        ] {
            assert!(excepted("http://127.0.0.1:17870/", url), "{url}");
        }
        for url in ["http://127.0.0.1:17871/", "http://localhost:17870/"] {
            assert!(!excepted("http://127.0.0.1:17870/", url), "{url}");
        }
        assert!(excepted("http://localhost/", "http://localhost/app.js"));
        assert!(excepted("http://localhost/", "ws://localhost/hmr"));
        assert!(!excepted("http://localhost/", "https://localhost/"));
        assert!(!excepted("http://localhost/", "wss://localhost/"));
        assert!(excepted("https://localhost/", "wss://localhost/"));
        assert!(!excepted("https://localhost/", "http://localhost/"));
    }

    #[test]
    fn local_views_do_not_host_public_documents_or_other_local_origins() {
        let origin = Url::parse("http://localhost:17860/").unwrap().origin();
        for raw in [
            "http://localhost:17860/app",
            "about:blank",
            "blob:http://localhost:17860/id",
        ] {
            assert!(local_navigation_allowed(&Url::parse(raw).unwrap(), &origin));
        }
        for raw in [
            "https://example.com/",
            "http://localhost:17861/",
            "http://127.0.0.1:17860/",
            "https://localhost:17860/",
            "file:///etc/passwd",
        ] {
            assert!(!local_navigation_allowed(
                &Url::parse(raw).unwrap(),
                &origin
            ));
        }
        let rules: serde_json::Value = serde_json::from_str(&content_rules_json(None)).unwrap();
        assert!(rules
            .as_array()
            .unwrap()
            .iter()
            .all(|rule| rule["action"]["type"] == "block"));
    }

    #[test]
    fn what_a_view_refuses_is_offered_in_a_tab_that_applies_its_own_rules() {
        for raw in [
            "https://example.com/",
            "http://localhost:17861/",
            "http://127.0.0.1:17860/",
            "http://localhost:8888/",
        ] {
            assert!(opens_in_tab(&Url::parse(raw).unwrap(), true), "{raw}");
        }
        for raw in ["http://192.168.1.1/", "file:///etc/passwd", "mailto:a@b.co"] {
            assert!(!opens_in_tab(&Url::parse(raw).unwrap(), true), "{raw}");
        }
        assert!(opens_in_tab(
            &Url::parse("https://example.com/").unwrap(),
            false
        ));
        assert!(!opens_in_tab(
            &Url::parse("http://localhost:17860/").unwrap(),
            false
        ));
        assert_eq!(
            serde_json::to_value(BrowserEvent::Refused {
                tab_id: "t1".into(),
                url: "https://example.com/".into(),
            })
            .unwrap(),
            serde_json::json!({ "kind": "refused", "tabId": "t1", "url": "https://example.com/" })
        );
    }

    #[test]
    fn only_mail_and_phone_links_go_to_other_apps() {
        assert!(is_external_handoff(&Url::parse("mailto:a@b.co").unwrap()));
        assert!(!is_external_handoff(&Url::parse("tel:+1555").unwrap()));
        assert!(!is_external_handoff(&Url::parse("zoommtg://join").unwrap()));
    }

    #[test]
    fn tab_ids_are_checked() {
        assert_eq!(label_for("main").unwrap(), "unsloth-browser-main");
        for bad in ["", "../x", "a b", &"x".repeat(65)] {
            assert!(label_for(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn content_rules_block_private_hosts() {
        let rules: serde_json::Value = serde_json::from_str(&content_rules_json(None)).unwrap();
        let filters: Vec<regex::Regex> = rules
            .as_array()
            .unwrap()
            .iter()
            .map(|rule| {
                let filter = rule["trigger"]["url-filter"].as_str().unwrap();
                // WebKit's rule regexes have no alternation or counted repetition.
                assert!(!filter.contains('|') && !filter.contains('{'), "{filter}");
                regex::RegexBuilder::new(filter)
                    .case_insensitive(true)
                    .build()
                    .unwrap()
            })
            .collect();
        let blocked = |url: &str| filters.iter().any(|filter| filter.is_match(url));
        for url in [
            "ws://127.0.0.1:8080/",
            "https://user:pw@127.0.0.1/",
            "http://studio.localhost/",
            "http://172.20.0.1/",
            "http://[::1]:8888/",
            "tauri://localhost/index.html",
        ] {
            assert!(blocked(url), "{url}");
        }
        for url in [
            "https://challenges.cloudflare.com/turnstile/v0/api.js",
            "https://172.217.1.1/",
            "https://localhost.example.com/",
            "https://example.com/?next=http://127.0.0.1/",
        ] {
            assert!(!blocked(url), "{url}");
        }
    }

    #[test]
    fn ipc_shapes_match_the_panel() {
        let event = serde_json::to_value(BrowserEvent::History {
            tab_id: "t1".into(),
            can_go_back: true,
            can_go_forward: false,
            icon: None,
        })
        .unwrap();
        assert_eq!(
            event,
            serde_json::json!({ "kind": "history", "tabId": "t1", "canGoBack": true, "canGoForward": false, "icon": null })
        );
        let bounds: ViewBounds = serde_json::from_value(serde_json::json!({
            "x": 1, "y": 2, "width": 3, "height": 4, "viewportWidth": 5
        }))
        .unwrap();
        assert_eq!(bounds.viewport_width, 5.0);
    }

    #[test]
    fn annotate_commands_match_the_panel() {
        let parse = |value| serde_json::from_value::<AnnotateCommand>(value).unwrap();
        assert!(matches!(
            parse(serde_json::json!({ "command": "number", "numbers": [[3, 1]] })),
            AnnotateCommand::Number { numbers } if numbers == vec![(3, 1)]
        ));
        assert!(matches!(
            parse(serde_json::json!({ "command": "poll" })),
            AnnotateCommand::Poll
        ));
        assert!(serde_json::from_value::<AnnotateCommand>(
            serde_json::json!({ "command": "eval" })
        )
        .is_err());
    }

    #[test]
    fn annotate_code_is_spliced_once() {
        let script = annotate_script(&AnnotateCommand::Install {
            code: "return __COMMAND__;".into(),
        })
        .unwrap();
        assert!(script.contains(r#"const command = {"command":"install"};"#));
        assert!(script.contains("return __COMMAND__;"));
        let start = annotate_script(&AnnotateCommand::Start {
            color: "__CODE__".into(),
        })
        .unwrap();
        assert!(start.contains(r#""color":"__CODE__""#));
        assert!(annotate_script(&AnnotateCommand::Install {
            code: "x".repeat(MAX_ANNOTATE_CODE + 1)
        })
        .is_err());
    }

    #[test]
    fn agent_code_and_command_are_spliced_once() {
        let mut op = serde_json::Map::new();
        op.insert("op".into(), "find".into());
        op.insert("query".into(), "__CODE__".into());
        let script = agent_script(&AgentCommand {
            op: op.clone(),
            code: Some("window.x = '__COMMAND__';".into()),
        })
        .unwrap();
        assert!(script.contains(
            r#"const command = {"install":true,"op":{"op":"find","query":"__CODE__"}};"#
        ));
        assert!(script.contains("window.x = '__COMMAND__';"));
        let bare = agent_script(&AgentCommand { op, code: None }).unwrap();
        assert!(bare.contains(r#""install":false"#));
        assert!(agent_script(&AgentCommand {
            op: serde_json::Map::new(),
            code: Some("x".repeat(MAX_ANNOTATE_CODE + 1)),
        })
        .is_err());
    }

    #[test]
    fn eval_code_is_spliced_once_and_ids_are_plain() {
        let script = eval_script("s1-a_B", "r100001", "return '__ID__' + '__DOC__';").unwrap();
        let check = script
            .find(r#"agent.owns("r100001") !== true) throw"#)
            .unwrap();
        let run = script
            .find(r#".evaluate("s1-a_B", "r100001", async () => {"#)
            .unwrap();
        assert!(check < run && script[check..run].contains("})();"));
        assert!(script.contains("return '__ID__' + '__DOC__';"));
        assert!(eval_script("a\"); evil(\"", "r1", "").is_err());
        assert!(eval_script("", "r1", "").is_err());
        assert!(eval_script("s1", "r1\"", "").is_err());
        assert!(eval_script("s1", "", "").is_err());
        assert!(eval_script("s1", "r1", &"x".repeat(MAX_EVAL_CODE + 1)).is_err());
    }

    #[test]
    fn titles_are_bounded() {
        assert_eq!(
            bounded_title("é".repeat(5000)).chars().count(),
            MAX_TITLE_CHARS
        );
        assert_eq!(bounded_title("short".into()), "short");
    }

    #[test]
    fn pages_get_a_few_downloads_at_once_and_a_minute() {
        let start = Instant::now();
        let mut starts = VecDeque::new();
        assert!(!download_allowed(
            MAX_DOWNLOADS_IN_FLIGHT,
            &mut starts,
            start
        ));
        for _ in 0..DOWNLOADS_PER_WINDOW {
            assert!(download_allowed(0, &mut starts, start));
        }
        assert!(!download_allowed(0, &mut starts, start));
        assert!(download_allowed(0, &mut starts, start + DOWNLOAD_WINDOW));
    }

    #[test]
    fn downloads_get_a_free_safe_name() {
        let dir = tempfile::tempdir().unwrap();
        let none = HashSet::new();
        let name = |suggested: &str| download_destination(dir.path(), Path::new(suggested), &none);
        std::fs::write(name("report.pdf"), b"x").unwrap();
        assert_eq!(name("report.pdf"), dir.path().join("report (1).pdf"));
        let first = dir.path().join("report (1).pdf");
        let reserved = HashSet::from([first.as_path()]);
        assert_eq!(
            download_destination(dir.path(), Path::new("report.pdf"), &reserved),
            dir.path().join("report (2).pdf")
        );
        assert_eq!(name(".."), dir.path().join("download"));
        assert_eq!(name("/x/evil\u{7}name.sh"), dir.path().join("evil_name.sh"));
    }

    #[cfg(any(windows, target_os = "macos"))]
    #[test]
    fn reserved_download_names_ignore_case() {
        let dir = tempfile::tempdir().unwrap();
        let taken = dir.path().join("Report.pdf");
        let reserved = HashSet::from([taken.as_path()]);
        assert_eq!(
            download_destination(dir.path(), Path::new("report.pdf"), &reserved),
            dir.path().join("report (1).pdf")
        );
    }
}
