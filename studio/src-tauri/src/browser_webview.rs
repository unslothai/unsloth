// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! Browser panel pages: one native child webview per tab. Pages are untrusted:
//! - Navigations (frames too) must be http(s) to a public host; all requests go via
//!   `browser_proxy`, which refuses private addresses after DNS. macOS < 14 can't proxy a
//!   webview, so the panel uses its proxied frame there.
//! - No IPC (capabilities bound to `main`); own profile, never the app's (holds sign-in).

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::webview::{DownloadEvent, NewWindowResponse, PageLoadEvent};
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
    download_starts: HashMap<String, VecDeque<Instant>>,
    polling: bool,
    /** Tabs the reader muted; macOS mutes each page they load, Windows the view once. */
    muted: HashSet<String>,
    /// Tabs whose view committed a page of its own since it opened or was last sent somewhere; until then a download's site is the address asked for (or the opener).
    committed: HashSet<String>,
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
    /// A link only another app opens (mailto:); the panel asks first.
    External {
        tab_id: String,
        url: String,
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
    },
    DownloadPrompt {
        tab_id: String,
        url: String,
        /// Page showing when the download started (the site a remembered answer is for); "" before the view showed one.
        site: String,
        name: String,
        id: String,
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

pub(crate) fn host_is_private(host: &url::Host<&str>) -> bool {
    match host {
        url::Host::Ipv4(ip) => ipv4_is_private(*ip),
        url::Host::Ipv6(ip) => ipv6_is_private(*ip),
        url::Host::Domain(domain) => {
            let domain = domain.trim_end_matches('.').to_ascii_lowercase();
            // A dotless name only resolves locally.
            !domain.contains('.')
                || ["localhost", "local", "internal", "lan", "home.arpa", "intranet"]
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
fn content_rules_json() -> String {
    let mut filters = Vec::new();
    for prefix in BLOCKED_HOST_PREFIXES {
        filters.push(format!("^[a-z]+://{prefix}"));
        filters.push(format!("^[a-z]+://[^/@]*@{prefix}"));
    }
    for scheme in ["tauri", "ipc", "asset", "file"] {
        filters.push(format!("^{scheme}:"));
    }
    let rules: Vec<_> = filters
        .into_iter()
        .map(|filter| {
            serde_json::json!({
                "trigger": { "url-filter": filter },
                "action": { "type": "block" },
            })
        })
        .collect();
    serde_json::Value::Array(rules).to_string()
}

#[cfg(target_os = "macos")]
mod content_rules {
    use objc2::rc::Retained;
    use objc2::MainThreadMarker;
    use objc2_foundation::{NSError, NSString};
    use objc2_web_kit::{WKContentRuleList, WKContentRuleListStore, WKWebView};
    use std::cell::RefCell;

    type Then = Box<dyn FnOnce()>;

    #[derive(Default)]
    struct Rules {
        compiled: Option<Retained<WKContentRuleList>>,
        waiting: Vec<(Retained<WKWebView>, Then)>,
        compiling: bool,
    }

    thread_local! {
        // WebKit objects live on the main thread, and so does this.
        static RULES: RefCell<Rules> = RefCell::new(Rules::default());
    }

    /// Add the rules to `webview`, then run `then` (the first load). Main thread only.
    pub fn protect(webview: Retained<WKWebView>, then: Then) {
        let Some(mtm) = MainThreadMarker::new() else {
            then();
            return;
        };
        let compiled = RULES.with(|rules| rules.borrow().compiled.clone());
        if let Some(list) = compiled {
            unsafe {
                webview
                    .configuration()
                    .userContentController()
                    .addContentRuleList(&list)
            };
            then();
            return;
        }
        let start = RULES.with(|rules| {
            let mut rules = rules.borrow_mut();
            rules.waiting.push((webview, then));
            !std::mem::replace(&mut rules.compiling, true)
        });
        if !start {
            return;
        }
        let Some(store) = (unsafe { WKContentRuleListStore::defaultStore(mtm) }) else {
            finish(None);
            return;
        };
        let identifier = NSString::from_str("unsloth-browser-private-hosts");
        let encoded = NSString::from_str(&super::content_rules_json());
        let handler =
            block2::RcBlock::new(move |list: *mut WKContentRuleList, error: *mut NSError| {
                let list = unsafe { Retained::retain(list) };
                if list.is_none() {
                    let message = unsafe { error.as_ref() }
                        .map(|error| error.localizedDescription().to_string())
                        .unwrap_or_default();
                    log::warn!("browser content rules failed to compile: {message}");
                }
                finish(list);
            });
        unsafe {
            store.compileContentRuleListForIdentifier_encodedContentRuleList_completionHandler(
                Some(&identifier),
                Some(&encoded),
                Some(&handler),
            );
        }
    }

    fn finish(list: Option<Retained<WKContentRuleList>>) {
        let waiting = RULES.with(|rules| {
            let mut rules = rules.borrow_mut();
            rules.compiling = false;
            rules.compiled = list.clone();
            std::mem::take(&mut rules.waiting)
        });
        for (webview, then) in waiting {
            if let Some(list) = &list {
                unsafe {
                    webview
                        .configuration()
                        .userContentController()
                        .addContentRuleList(list)
                };
            }
            // Without rules the navigation policy still holds; load rather than hang.
            then();
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

fn parse_page_url(raw: &str) -> Result<Url, String> {
    let url = Url::parse(raw.trim()).map_err(|_| "not a URL".to_string())?;
    if !matches!(url.scheme(), "http" | "https") || !navigation_allowed(&url) {
        return Err("only public web addresses open in the browser".into());
    }
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
        },
    );
}

pub(crate) fn emit_download_failed<R: Runtime>(
    app: &AppHandle<R>,
    tab_id: &str,
    url: &Url,
    name: &str,
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

/// Pages' own profile, proxied via `browser_proxy` (macOS: set on first load).
fn with_page_profile<R: Runtime>(
    builder: WebviewBuilder<R>,
    app: &AppHandle<R>,
) -> Result<WebviewBuilder<R>, String> {
    let proxy = crate::browser_proxy::address()?;
    #[cfg(target_os = "macos")]
    {
        let _ = (app, proxy);
        // Before macOS 14 the only other store is a private one.
        if !browser_view_supported() {
            return Err("native pages need macOS 14".into());
        }
        Ok(builder.data_store_identifier(PAGE_DATA_STORE))
    }
    #[cfg(not(target_os = "macos"))]
    {
        #[cfg(windows)]
        let builder = builder.additional_browser_args(&page_browser_args(&proxy.to_string()));
        #[cfg(not(windows))]
        let builder = builder
            .proxy_url(Url::parse(&format!("http://{proxy}")).map_err(|error| error.to_string())?);
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

fn create_view<R: Runtime>(
    caller: &Webview<R>,
    tab_id: &str,
    url: Url,
    bounds: &ViewBounds,
) -> Result<Webview<R>, String> {
    let app = caller.app_handle().clone();
    let label = label_for(tab_id)?;
    let tab = tab_id.to_string();

    let nav_app = app.clone();
    let nav_tab = tab.clone();
    let load_tab = tab.clone();
    let title_tab = tab.clone();
    let window_app = app.clone();
    let window_tab = tab.clone();
    let download_tab = tab.clone();

    #[cfg(target_os = "macos")]
    let (initial, deferred) = (Url::parse("about:blank").unwrap(), Some(url));
    #[cfg(not(target_os = "macos"))]
    let (initial, deferred): (Url, Option<Url>) = (url, None);

    let builder = WebviewBuilder::new(&label, WebviewUrl::External(initial))
        .on_navigation(move |url| {
            // An address too long to report would leave the bar showing the last page's: refused.
            if navigation_allowed(url) && reportable(url.as_str()) {
                return true;
            }
            if is_external_handoff(url) && reportable(url.as_str()) {
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
            if navigation_allowed(&url) && matches!(url.scheme(), "http" | "https") {
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
        .on_download(move |webview, event| {
            let app = webview.app_handle();
            match event {
                DownloadEvent::Requested { url, destination } => {
                    if crate::browser_downloads::unanswered(app, &download_tab)
                        >= crate::browser_downloads::MAX_UNANSWERED_PER_TAB
                    {
                        let name = destination
                            .file_name()
                            .map(|name| name.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        emit_download_failed(app, &download_tab, &url, &name);
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
                        let busy =
                            cfg!(target_os = "macos") && inner.downloads.contains_key(url.as_str());
                        let in_flight = inner.downloads.values().map(Vec::len).sum();
                        let starts = inner
                            .download_starts
                            .entry(download_tab.clone())
                            .or_default();
                        if busy || !download_allowed(in_flight, starts, Instant::now()) {
                            drop(inner);
                            let _ = std::fs::remove_dir_all(&staging);
                            if !busy {
                                let name = destination
                                    .file_name()
                                    .map(|name| name.to_string_lossy().into_owned())
                                    .unwrap_or_default();
                                emit_download_failed(app, &download_tab, &url, &name);
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
                        download_tab.clone(),
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
                        .contains(&download_tab);
                    let site = match webview.url() {
                        Ok(page) if committed => page.to_string(),
                        _ => String::new(),
                    };
                    emit(
                        app,
                        BrowserEvent::DownloadPrompt {
                            tab_id: download_tab.clone(),
                            url: url.to_string(),
                            site,
                            name,
                            id,
                        },
                    );
                    true
                }
                DownloadEvent::Finished { url, path, success } => {
                    let recorded = {
                        let state = app.state::<BrowserViews>();
                        let mut inner = state.inner.lock().unwrap();
                        let pending = inner.downloads.entry(url.to_string()).or_default();
                        let recorded = take_reservation(pending, path.as_deref());
                        if pending.is_empty() {
                            inner.downloads.remove(url.as_str());
                        }
                        recorded
                    };
                    // The reserved path first: the engine may report another spelling of it.
                    crate::browser_downloads::finished(
                        app,
                        [recorded.as_deref(), path.as_deref()],
                        success,
                    );
                    true
                }
                _ => false,
            }
        })
        .zoom_hotkeys_enabled(false)
        .devtools(cfg!(debug_assertions))
        .focused(false);
    let builder = with_page_profile(builder, &app)?;

    let (position, size) = logical_rect(caller, bounds);
    let window = caller.window();
    let webview = window
        .add_child(builder, position, size)
        .map_err(|error| error.to_string())?;
    // A muted tab whose view closed (four at most stay open) opens muted again.
    if app.state::<BrowserViews>().inner.lock().unwrap().muted.contains(&tab) {
        let _ = apply_mute(&webview, true);
    }
    if let Some(url) = deferred {
        load_when_protected(&webview, url);
    }
    start_url_poll(&app);
    Ok(webview)
}

#[cfg(target_os = "macos")]
fn load_when_protected<R: Runtime>(webview: &Webview<R>, url: Url) {
    // Off main thread: `protect` can finish inside `with_webview`, holding `navigate`'s lock.
    let page = webview.clone();
    let load = move || {
        tauri::async_runtime::spawn(async move {
            let _ = page.navigate(url);
        });
    };
    let _ = webview.with_webview(move |platform| {
        let raw = platform.inner() as *mut objc2_web_kit::WKWebView;
        // Safety: wry's view, alive with its Tauri webview; retained for the callback.
        let Some(view) = (unsafe { objc2::rc::Retained::retain(raw) }) else {
            return;
        };
        // Never unproxied.
        if !crate::browser_proxy::address().is_ok_and(|proxy| page_proxy::route(&view, proxy)) {
            log::error!("browser page not loaded: its proxy couldn't be set");
            return;
        }
        content_rules::protect(view, Box::new(load));
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

/// Whether pages open in native webviews: macOS 14+ (per-webview proxy) and Windows. On Linux
/// Tauri packs child webviews into the window's GtkBox, which ignores their bounds, so the panel
/// uses its proxied frame there.
#[tauri::command]
pub fn browser_view_supported() -> bool {
    #[cfg(target_os = "macos")]
    return macos_at_least(14);
    #[cfg(target_os = "windows")]
    return true;
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    false
}

/// Show a tab's page at `bounds` (created at `url` first time), hide the rest; `None` hides all.
/// Async: creating a webview from a sync command deadlocks on Windows (Tauri known issue).
#[tauri::command]
pub async fn browser_view_show<R: Runtime>(
    webview: Webview<R>,
    state: State<'_, BrowserViews>,
    tab_id: Option<String>,
    url: Option<String>,
    bounds: Option<ViewBounds>,
) -> Result<(), String> {
    require_main(&webview)?;
    let app = webview.app_handle().clone();
    let target = match (&tab_id, &bounds) {
        (Some(tab_id), Some(bounds)) => Some(match app.get_webview(&label_for(tab_id)?) {
            Some(view) => {
                let (position, size) = logical_rect(&webview, bounds);
                view.set_bounds(Rect {
                    position: position.into(),
                    size: size.into(),
                })
                .map_err(|error| error.to_string())?;
                view
            }
            None => {
                let url = parse_page_url(url.as_deref().ok_or("no address to open")?)?;
                create_view(&webview, tab_id, url, bounds)?
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
    for view in browser_views(&app) {
        if Some(view.label()) != target.as_ref().map(|view| view.label()) {
            let _ = view.hide();
        }
    }
    set_shown(&state, &mut state.inner.lock().unwrap(), shown);
    Ok(())
}

#[tauri::command]
pub fn browser_view_navigate<R: Runtime>(
    webview: Webview<R>,
    tab_id: String,
    url: String,
) -> Result<(), String> {
    require_main(&webview)?;
    let url = parse_page_url(&url)?;
    let page = view(webview.app_handle(), &tab_id)?;
    webview
        .state::<BrowserViews>()
        .inner
        .lock()
        .unwrap()
        .committed
        .remove(&tab_id);
    page.navigate(url).map_err(|error| error.to_string())
}

#[tauri::command]
pub fn browser_view_action<R: Runtime>(
    webview: Webview<R>,
    tab_id: String,
    action: String,
) -> Result<(), String> {
    require_main(&webview)?;
    let page = view(webview.app_handle(), &tab_id)?;
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

#[cfg(not(any(target_os = "macos", windows)))]
fn apply_mute<R: Runtime>(_page: &Webview<R>, _muted: bool) -> Result<(), String> {
    Ok(())
}

#[tauri::command]
pub fn browser_view_close<R: Runtime>(
    webview: Webview<R>,
    state: State<'_, BrowserViews>,
    tab_id: String,
) -> Result<(), String> {
    require_main(&webview)?;
    {
        let mut inner = state.inner.lock().unwrap();
        inner.urls.remove(&tab_id);
        inner.committed.remove(&tab_id);
        inner.download_starts.remove(&tab_id);
        // `muted` stays: a pruned view reopens muted; unmuting is what forgets it.
        if inner.shown.as_deref() == Some(tab_id.as_str()) {
            set_shown(&state, &mut inner, None);
        }
    }
    if let Ok(page) = view(webview.app_handle(), &tab_id) {
        page.close().map_err(|error| error.to_string())?;
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
    if closing {
        {
            let mut inner = state.inner.lock().unwrap();
            inner.urls.clear();
            inner.committed.clear();
            set_shown(&state, &mut inner, None);
        }
        for page in browser_views(&app) {
            let _ = page.close();
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
            )?;
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
        let _ = page.close();
    }
    result
}

const CLEAR_TIMEOUT: Duration = Duration::from_secs(30);

/// Clear a page's profile, resolving once the engine reports it done: wry's own call starts the
/// clear and returns before it finishes.
async fn clear_profile<R: Runtime>(page: &Webview<R>) -> Result<(), String> {
    #[cfg(any(target_os = "macos", windows))]
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
    #[cfg(not(any(target_os = "macos", windows)))]
    {
        page.clear_all_browsing_data()
            .map_err(|error| error.to_string())?;
        tokio::time::sleep(Duration::from_secs(2)).await;
        Ok(())
    }
}

#[cfg(any(target_os = "macos", windows))]
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

/// The path reserved for a finished download of one URL. No path: the first on macOS, which reports
/// none and runs downloads of one URL in turn. Elsewhere a missing path (WebView2 sends none for a
/// failed download) or one matching none stands for the sole reservation only, else settling one of
/// several would hand it another download's result.
fn take_reservation(pending: &mut Vec<PathBuf>, path: Option<&Path>) -> Option<PathBuf> {
    let sole = (pending.len() == 1).then_some(0);
    let index = match path {
        None if cfg!(target_os = "macos") => Some(0),
        None => sole,
        Some(path) => pending
            .iter()
            .position(|p| crate::browser_downloads::same_path(p, path))
            .or(sole),
    };
    index
        .filter(|&index| index < pending.len())
        .map(|index| pending.remove(index))
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_finished_path_matching_no_reservation_settles_only_a_sole_one() {
        let (a, b) = (PathBuf::from("/s/a.zip"), PathBuf::from("/s/b.zip"));
        let mut two = vec![a.clone(), b.clone()];
        assert_eq!(
            take_reservation(&mut two, Some(Path::new("/s/b.zip"))),
            Some(b.clone())
        );
        let mut two = vec![a.clone(), b.clone()];
        assert_eq!(
            take_reservation(&mut two, Some(Path::new("/elsewhere"))),
            None
        );
        assert_eq!(two.len(), 2);
        // No path: macOS runs one URL's downloads in turn, so the first; elsewhere it could be any.
        let expected = cfg!(target_os = "macos").then(|| a.clone());
        assert_eq!(take_reservation(&mut two, None), expected);
        let mut one = vec![b.clone()];
        assert_eq!(take_reservation(&mut one, None), Some(b.clone()));
        let mut one = vec![b.clone()];
        assert_eq!(
            take_reservation(&mut one, Some(Path::new("/elsewhere"))),
            Some(b)
        );
        assert_eq!(take_reservation(&mut Vec::new(), None), None);
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
        let rules: serde_json::Value = serde_json::from_str(&content_rules_json()).unwrap();
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
