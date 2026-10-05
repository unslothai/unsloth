// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! agent operations on the browser pane, run by a page runtime that holds no Studio authority.
use crate::browser_webview::{current, BrowserState};
use base64::Engine;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::webview::PlatformWebview;
use tauri::{Manager, Webview};
use tokio::sync::oneshot;

const RUNTIME: &str = include_str!("browser_agent.js");
const RUNTIME_OPS: [&str; 14] = [
    "activation",
    "snapshot",
    "read",
    "find",
    "inspect",
    "locate",
    "click",
    "type",
    "select",
    "press",
    "scroll",
    "status",
    "marks",
    "highlight",
];
const EVAL_TIMEOUT: Duration = Duration::from_secs(5);
// shorter than EVAL_TIMEOUT so work the page starts can still report back.
const PAGE_DEADLINE: Duration = Duration::from_secs(4);
const SHOT_TIMEOUT: Duration = Duration::from_secs(8);
const MAX_REQUEST_BYTES: usize = 64 * 1024;
const MAX_REPLY_BYTES: usize = 2 * 1024 * 1024;
const MAX_PNG_BYTES: usize = 24 * 1024 * 1024;
const NO_RESPONSE: &str = "The page did not respond, so the action may or may not have happened. Take a snapshot before trying again";
const LOADING: &str = "The page is still loading";
const NO_RESULT: &str = "The page did not return a result; it may be navigating";
const UNREADABLE: &str = "The page returned an unreadable result";

pub fn runtime_name() -> String {
    format!("__unslothAgent_{:016x}", rand::random::<u64>())
}

/// document-start script; the runtime itself returns early in subframes.
pub fn init_script(name: &str) -> String {
    invoke_runtime(RUNTIME, &js_string(name))
}

// tolerate a trailing `;` or line comment after the function expression.
fn invoke_runtime(source: &str, argument: &str) -> String {
    let source = source.trim_end().trim_end_matches(';').trim_end();
    format!("{source}\n({argument});")
}

// a JSON string is valid JS except for U+2028/U+2029 in older engines.
fn js_string(value: &str) -> String {
    Value::String(value.into())
        .to_string()
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029")
}

const NATIVE_OPS: [&str; 2] = ["screenshot", "check_url"];

fn checked_op(op: &str) -> Result<&'static str, String> {
    RUNTIME_OPS
        .iter()
        .chain(&NATIVE_OPS)
        .find(|known| **known == op)
        .copied()
        .ok_or_else(|| "Unknown browser agent operation".into())
}

fn checked_args(args: Value) -> Result<Value, String> {
    let args = match args {
        Value::Null => json!({}),
        Value::Object(_) => args,
        _ => return Err("Browser agent arguments must be an object".into()),
    };
    if args.to_string().len() > MAX_REQUEST_BYTES {
        return Err("Browser agent request is too large".into());
    }
    Ok(args)
}

fn epoch_ms(after: Duration) -> u128 {
    (SystemTime::now() + after)
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// a null result means the runtime is missing; the deadline stops a timed-out call acting later.
fn page_script(name: &str, request: &str, install: bool, deadline_ms: u128) -> String {
    let install = if install {
        format!("if(!a){{{}a=window[n];}}", invoke_runtime(RUNTIME, "n"))
    } else {
        String::new()
    };
    format!(
        "(function(){{try{{if(Date.now()>{deadline_ms})return {expired};\
         var n={name},a=window[n];{install}\
         return a&&typeof a.run==='function'?a.run({request}):null;\
         }}catch(e){{return {failed};}}}})()",
        name = js_string(name),
        request = js_string(request),
        expired = js_string(
            r#"{"ok":false,"code":"expired","error":"The page was busy too long; nothing was done"}"#
        ),
        failed = js_string(r#"{"ok":false,"error":"The browser agent failed on this page"}"#),
    )
}

#[derive(Debug, PartialEq)]
enum Reply {
    Missing,
    Done(Value),
}

/// engines JSON-encode the script's value, so a JS string arrives as a quoted JSON literal.
fn decode_reply(raw: &str) -> Result<Reply, String> {
    if raw.len() > MAX_REPLY_BYTES {
        return Err("The page returned too much data".into());
    }
    let raw = raw.trim();
    // exceptions and teardown arrive as an empty string on WebKit.
    if raw.is_empty() {
        return Err(NO_RESULT.into());
    }
    let reply = match serde_json::from_str(raw).map_err(|_| UNREADABLE)? {
        Value::Null => return Ok(Reply::Missing),
        Value::String(inner) => serde_json::from_str(&inner).map_err(|_| UNREADABLE)?,
        decoded => decoded,
    };
    match reply.get("ok") {
        Some(Value::Bool(_)) => Ok(Reply::Done(reply)),
        _ => Err(UNREADABLE.into()),
    }
}

type Slot<T> = Arc<Mutex<Option<oneshot::Sender<Result<T, String>>>>>;

fn fulfil<T>(slot: &Slot<T>, value: Result<T, String>) {
    if let Some(send) = slot.lock().ok().and_then(|mut slot| slot.take()) {
        let _ = send.send(value);
    }
}

/// runs `start` on the engine's thread; it answers through the slot, now or from a handler.
async fn native<T: Send + 'static>(
    view: &Webview,
    limit: Duration,
    start: impl FnOnce(&PlatformWebview, Slot<T>) + Send + 'static,
) -> Result<T, String> {
    let (send, receive) = oneshot::channel();
    let slot: Slot<T> = Arc::new(Mutex::new(Some(send)));
    view.with_webview(move |platform| start(&platform, slot))
        .map_err(|e| e.to_string())?;
    tokio::time::timeout(limit, receive)
        .await
        .map_err(|_| "Browser engine did not answer")?
        .map_err(|_| "Browser engine dropped the request")?
}

async fn eval(view: &Webview, script: String) -> Result<String, String> {
    let (send, receive) = oneshot::channel();
    let slot = Mutex::new(Some(send));
    view.eval_with_callback(script, move |raw| {
        if let Some(send) = slot.lock().ok().and_then(|mut slot| slot.take()) {
            let _ = send.send(raw);
        }
    })
    .map_err(|e| e.to_string())?;
    // a dropped sender means WebKit discarded the callback of a script queued for the first commit.
    match tokio::time::timeout(EVAL_TIMEOUT, receive).await {
        Ok(Ok(raw)) => Ok(raw),
        Ok(Err(_)) => Err(LOADING.into()),
        Err(_) => Err(NO_RESPONSE.into()),
    }
}

async fn run(view: &Webview, name: &str, op: &str, args: &Value) -> Result<Value, String> {
    let request = json!({ "op": op, "args": args }).to_string();
    for install in [false, true] {
        let script = page_script(name, &request, install, epoch_ms(PAGE_DEADLINE));
        if let Reply::Done(reply) = decode_reply(&eval(view, script).await?)? {
            return Ok(reply);
        }
    }
    Err("The browser agent could not start on this page".into())
}

#[derive(Debug, PartialEq)]
struct Shot {
    data: String,
    width: u32,
    height: u32,
}

fn png_size(png: &[u8]) -> Option<(u32, u32)> {
    const SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
    if png.len() < 24 || &png[..8] != SIGNATURE || &png[12..16] != b"IHDR" {
        return None;
    }
    let width = u32::from_be_bytes(png[16..20].try_into().ok()?);
    let height = u32::from_be_bytes(png[20..24].try_into().ok()?);
    Some((width, height))
}

#[cfg_attr(all(windows, not(test)), allow(dead_code))]
fn shot_from_png(png: &[u8]) -> Result<Shot, String> {
    if png.len() > MAX_PNG_BYTES {
        return Err("The screenshot is too large".into());
    }
    let (width, height) = png_size(png).ok_or("The browser engine returned an invalid image")?;
    Ok(Shot {
        data: base64::engine::general_purpose::STANDARD.encode(png),
        width,
        height,
    })
}

// the CDP reply is already base64, so only the header is decoded for the size.
#[cfg_attr(not(any(windows, test)), allow(dead_code))]
fn shot_from_cdp(reply: &str) -> Result<Shot, String> {
    let data = serde_json::from_str::<Value>(reply)
        .ok()
        .and_then(|reply| reply.get("data")?.as_str().map(str::to_owned))
        .ok_or("The browser engine returned no image")?;
    if data.len() > MAX_PNG_BYTES / 3 * 4 + 4 {
        return Err("The screenshot is too large".into());
    }
    let head = base64::engine::general_purpose::STANDARD
        .decode(
            data.get(..32)
                .ok_or("The browser engine returned an invalid image")?,
        )
        .map_err(|_| "The browser engine returned an invalid image")?;
    let (width, height) = png_size(&head).ok_or("The browser engine returned an invalid image")?;
    Ok(Shot {
        data,
        width,
        height,
    })
}

#[cfg(target_os = "linux")]
fn capture(platform: &PlatformWebview, slot: Slot<Shot>) {
    use webkit2gtk::{SnapshotOptions, SnapshotRegion, WebViewExt};
    platform.inner().snapshot(
        SnapshotRegion::Visible,
        SnapshotOptions::NONE,
        None::<&webkit2gtk::gio::Cancellable>,
        move |surface| {
            let shot = surface.map_err(|e| e.to_string()).and_then(|surface| {
                let mut png = Vec::new();
                surface.write_to_png(&mut png).map_err(|e| e.to_string())?;
                shot_from_png(&png)
            });
            fulfil(&slot, shot);
        },
    );
}

#[cfg(windows)]
fn devtools(platform: &PlatformWebview, method: &str, params: String, slot: Slot<String>) {
    use webview2_com::CallDevToolsProtocolMethodCompletedHandler;
    use windows_core::HSTRING;
    let reply = slot.clone();
    let handler =
        CallDevToolsProtocolMethodCompletedHandler::create(Box::new(move |result, json| {
            fulfil(&reply, result.map(|()| json).map_err(|e| e.to_string()));
            Ok(())
        }));
    // bound, not inlined: the call reads both strings before returning.
    let method = HSTRING::from(method);
    let params = HSTRING::from(params);
    let started = unsafe {
        platform
            .controller()
            .CoreWebView2()
            .and_then(|core| core.CallDevToolsProtocolMethod(&method, &params, &handler))
    };
    if let Err(error) = started {
        fulfil(&slot, Err(error.to_string()));
    }
}

#[cfg(target_os = "macos")]
fn capture(platform: &PlatformWebview, slot: Slot<Shot>) {
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSBitmapImageFileType, NSBitmapImageRep, NSImage};
    use objc2_foundation::{NSDictionary, NSError};
    use objc2_web_kit::{WKSnapshotConfiguration, WKWebView};
    let Some(main_thread) = MainThreadMarker::new() else {
        return fulfil(&slot, Err("Browser engine is not on its thread".into()));
    };
    let encode = |image: &NSImage| -> Result<Shot, String> {
        let tiff = image
            .TIFFRepresentation()
            .ok_or("WebKit returned an empty snapshot")?;
        let bitmap = NSBitmapImageRep::imageRepWithData(&tiff).ok_or("Unreadable snapshot")?;
        let png = unsafe {
            bitmap.representationUsingType_properties(
                NSBitmapImageFileType::PNG,
                &NSDictionary::new(),
            )
        }
        .ok_or("Could not encode the snapshot")?;
        shot_from_png(&png.to_vec())
    };
    let done = block2::RcBlock::new(move |image: *mut NSImage, error: *mut NSError| {
        let shot = match unsafe { image.as_ref() } {
            Some(image) => encode(image),
            None => Err(unsafe { error.as_ref() }
                .map(|error| error.localizedDescription().to_string())
                .unwrap_or_else(|| "WebKit returned no snapshot".into())),
        };
        fulfil(&slot, shot);
    });
    unsafe {
        let view = &*platform.inner().cast::<WKWebView>();
        let config = WKSnapshotConfiguration::new(main_thread);
        view.takeSnapshotWithConfiguration_completionHandler(Some(&config), &done);
    }
}

async fn screenshot(
    app: &tauri::AppHandle,
    view: &Webview,
    name: &str,
    visible: bool,
    args: &Value,
) -> Result<Value, String> {
    let marks = args.get("marks").and_then(Value::as_bool).unwrap_or(false);
    // the labels are refs of this document; a page that refuses the overlay still gets a plain shot.
    let marked = if marks {
        run(view, name, "marks", &json!({ "on": true }))
            .await
            .ok()
            .filter(|reply| reply["ok"] == true)
    } else {
        None
    };
    let shot = {
        let state = app.state::<BrowserState>();
        let _operation = state.operations.lock().await;
        #[cfg(not(windows))]
        let shot = native(view, SHOT_TIMEOUT, capture).await;
        #[cfg(windows)]
        let shot = native(view, SHOT_TIMEOUT, |platform, slot| {
            devtools(
                platform,
                "Page.captureScreenshot",
                r#"{"format":"png"}"#.into(),
                slot,
            )
        })
        .await
        .and_then(|reply| shot_from_cdp(&reply));
        shot
    };
    if marks {
        let _ = run(view, name, "marks", &json!({ "on": false })).await;
    }
    Ok(match shot {
        Ok(shot) => json!({
            "ok": true,
            "mime": "image/png",
            "data": shot.data,
            "width": shot.width,
            "height": shot.height,
            "docId": marked.map(|reply| reply["docId"].clone()).unwrap_or(Value::Null),
        }),
        Err(error) if !visible => json!({
            "ok": false,
            "error": format!("The browser pane is hidden, so it cannot be captured ({error})"),
        }),
        Err(error) => json!({ "ok": false, "error": format!("Screenshot failed: {error}") }),
    })
}

/// asks the runtime's click probe where trusted presses, releases and clicks near a point land.
#[cfg(any(target_os = "linux", windows))]
async fn probe(
    view: &Webview,
    name: &str,
    x: f64,
    y: f64,
    step: &str,
    target: &Value,
) -> Result<Value, String> {
    run(view, name, "probe", &json!({ "x": x, "y": y, "step": step, "ref": target })).await
}

/// tries a trusted click; None means none reached the element, so a synthetic click is safe.
#[cfg(any(target_os = "linux", windows))]
async fn trusted_click(
    app: &tauri::AppHandle,
    view: &Webview,
    name: &str,
    args: &Value,
) -> Option<Value> {
    let info = run(view, name, "inspect", args).await.ok()?;
    let spot = run(view, name, "locate", args).await.ok()?;
    if info["ok"] != true
        // the runtime's own click refuses these with guidance; a native click would not.
        || info["disabled"] == true
        || matches!(info["tag"].as_str(), Some("select" | "option"))
        || spot["ok"] != true
        || spot["visible"] != true
        || !spot["occludedBy"].is_null()
        // the probe listens on the top window, so a frame's click would go unseen and be repeated.
        || spot["inFrame"] != false
    {
        return None;
    }
    let x = spot["center"]["x"]
        .as_f64()
        .filter(|v| v.is_finite() && *v >= 0.0)?;
    let y = spot["center"]["y"]
        .as_f64()
        .filter(|v| v.is_finite() && *v >= 0.0)?;
    let document = match probe(view, name, x, y, "arm", &args["ref"]).await {
        Ok(armed) if armed["ok"] == true && armed["armed"] == true => armed["docId"].clone(),
        _ => return None,
    };
    let _ = run(view, name, "highlight", &json!({ "ref": args["ref"] })).await;
    let clicked = json!({ "ok": true, "description": info["description"], "trusted": true });
    let sent = {
        let state = app.state::<BrowserState>();
        let _operation = state.operations.lock().await;
        pointer(view, x, y).await
    };
    // scripts can overtake mouse events WebKit forwards one at a time, so wait for the release.
    let mut seen = Value::Null;
    for _ in 0..if sent.is_ok() { 50 } else { 1 } {
        tokio::time::sleep(Duration::from_millis(40)).await;
        match probe(view, name, x, y, "read", &Value::Null).await {
            Ok(reply) if reply["ok"] == true && reply["docId"] == document => seen = reply,
            // a replaced document or a busy page may mean the click landed, so never repeat it.
            _ => return Some(clicked),
        }
        if seen["up"] == true {
            break;
        }
    }
    let _ = probe(view, name, x, y, "disarm", &Value::Null).await;
    if seen["down"] == true && seen["up"] != true {
        return Some(clicked);
    }
    match seen["hit"].as_str() {
        Some("on") => Some(clicked),
        Some("other") => Some(json!({
            "ok": false,
            "code": "intercepted",
            "error": format!(
                "The click hit a <{}> instead of {} (the page changed under the pointer). Take a new snapshot before retrying.",
                seen["tag"].as_str().unwrap_or("element"),
                info["description"].as_str().unwrap_or("the element"),
            ),
        })),
        // nothing arrived, or press and release split across elements so only an ancestor saw it.
        _ => None,
    }
}

// built like WebKit's own GTK test event sender so it is handled exactly like hardware input.
#[cfg(target_os = "linux")]
fn send_pointer(
    window: &gdk::Window,
    device: &gdk::Device,
    (x, y): (f64, f64),
    kind: gdk::ffi::GdkEventType,
) {
    use glib::translate::{from_glib_full, ToGlibPtr};
    let (root_x, root_y) = window.root_coords(x.round() as i32, y.round() as i32);
    unsafe {
        let raw = gdk::ffi::gdk_event_new(kind);
        if kind == gdk::ffi::GDK_MOTION_NOTIFY {
            let motion = &mut (*raw).motion;
            motion.window = window.to_glib_full();
            (motion.x, motion.y) = (x, y);
            (motion.x_root, motion.y_root) = (root_x.into(), root_y.into());
        } else {
            let button = &mut (*raw).button;
            button.window = window.to_glib_full();
            button.button = 1;
            if kind == gdk::ffi::GDK_BUTTON_RELEASE {
                button.state = gdk::ffi::GDK_BUTTON1_MASK;
            }
            (button.x, button.y) = (x, y);
            (button.x_root, button.y_root) = (root_x.into(), root_y.into());
        }
        let mut event: gdk::Event = from_glib_full(raw);
        event.set_device(Some(device));
        gtk::main_do_event(&mut event);
    }
}

#[cfg(target_os = "linux")]
fn click_at(platform: &PlatformWebview, x: f64, y: f64, slot: Slot<()>) {
    use gdk::ffi::{GDK_BUTTON_PRESS, GDK_BUTTON_RELEASE, GDK_MOTION_NOTIFY};
    use gtk::prelude::*;
    use webkit2gtk::WebViewExt;
    let page = platform.inner();
    let Some(window) = page
        .window()
        .filter(|window| page.is_visible() && window.is_viewable())
    else {
        return fulfil(&slot, Err("The browser pane is not on screen".into()));
    };
    let Some(device) = window
        .display()
        .default_seat()
        .and_then(|seat| seat.pointer())
    else {
        return fulfil(&slot, Err("No pointer device".into()));
    };
    let at = (x * page.zoom_level(), y * page.zoom_level());
    // hand focus back after WebKit takes it on press so Studio typing never lands in the page.
    let focus = page
        .toplevel()
        .and_then(|top| top.downcast::<gtk::Window>().ok())
        .and_then(|top| top.focused_widget())
        .filter(|widget| widget != page.upcast_ref::<gtk::Widget>());
    let restore = move || {
        if let Some(widget) = &focus {
            widget.grab_focus();
        }
    };
    // presses close in time and place become a double click in WebKit, so agent clicks are spaced.
    let settings = gtk::Settings::default();
    let span = settings
        .as_ref()
        .map_or(400, |s| s.gtk_double_click_time())
        .max(0) as u64
        + 50;
    let near = settings
        .as_ref()
        .map_or(5, |s| s.gtk_double_click_distance()) as f64;
    let wait = LAST_PRESS
        .get()
        .filter(|(_, (x, y))| (x - at.0).abs() < near && (y - at.1).abs() < near)
        .map_or(Duration::ZERO, |(when, _)| {
            Duration::from_millis(span).saturating_sub(when.elapsed())
        });
    glib::timeout_add_local_once(wait, move || {
        send_pointer(&window, &device, at, GDK_MOTION_NOTIFY);
        send_pointer(&window, &device, at, GDK_BUTTON_PRESS);
        LAST_PRESS.set(Some((std::time::Instant::now(), at)));
        restore();
        glib::timeout_add_local_once(Duration::from_millis(50), move || {
            send_pointer(&window, &device, at, GDK_BUTTON_RELEASE);
            restore();
            fulfil(&slot, Ok(()));
        });
    });
}

#[cfg(target_os = "linux")]
thread_local! {
    static LAST_PRESS: std::cell::Cell<Option<(std::time::Instant, (f64, f64))>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(target_os = "linux")]
async fn pointer(view: &Webview, x: f64, y: f64) -> Result<(), String> {
    native(view, EVAL_TIMEOUT, move |platform, slot| {
        click_at(platform, x, y, slot)
    })
    .await
}

#[cfg(windows)]
async fn pointer(view: &Webview, x: f64, y: f64) -> Result<(), String> {
    for (kind, button, buttons, count) in [
        ("mouseMoved", "none", 0, 0),
        ("mousePressed", "left", 1, 1),
        ("mouseReleased", "left", 0, 1),
    ] {
        let params = json!({
            "type": kind, "x": x, "y": y, "button": button, "buttons": buttons, "clickCount": count,
        })
        .to_string();
        native(view, EVAL_TIMEOUT, move |platform, slot| {
            devtools(platform, "Input.dispatchMouseEvent", params, slot)
        })
        .await?;
        if kind == "mousePressed" {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
    Ok(())
}

async fn click(
    app: &tauri::AppHandle,
    view: &Webview,
    session_id: &str,
    name: &str,
    args: &Value,
) -> Result<Value, String> {
    let popups = current(app, session_id)?.1.popups;
    #[cfg(any(target_os = "linux", windows))]
    let native = trusted_click(app, view, name, args).await;
    #[cfg(not(any(target_os = "linux", windows)))]
    let native: Option<Value> = None;
    let mut reply = match native {
        Some(reply) => reply,
        None => {
            let mut reply = run(view, name, "click", args).await?;
            if reply["ok"] == true {
                reply["trusted"] = false.into();
            }
            reply
        }
    };
    // window.open is recorded before the page replies, but link targets and WebView2 report later.
    let mut popup = current(app, session_id)?.1;
    if popup.popups == popups && reply["ok"] == true {
        tokio::time::sleep(Duration::from_millis(120)).await;
        popup = current(app, session_id)?.1;
    }
    if popup.popups != popups {
        reply["popupUrl"] = popup.popup_url.into();
    }
    Ok(reply)
}

/// the pane's guard reads only the host as written, so this resolves it to catch private addresses.
async fn check_url(args: &Value) -> Result<Value, String> {
    let raw = args["url"].as_str().ok_or("check_url needs a url")?;
    let url = match crate::browser_webview::website(raw) {
        Ok(url) => url,
        Err(error) => return Ok(json!({ "ok": false, "code": "blocked", "error": error })),
    };
    let host = url.host_str().unwrap_or_default().to_string();
    let port = url.port_or_known_default().unwrap_or(443);
    let lookup = tokio::task::spawn_blocking(move || {
        use std::net::ToSocketAddrs;
        (host.as_str(), port)
            .to_socket_addrs()
            .map(|addrs| addrs.map(|addr| addr.ip()).collect::<Vec<_>>())
    });
    let addresses = match tokio::time::timeout(Duration::from_secs(3), lookup).await {
        Ok(Ok(Ok(addresses))) => addresses,
        // unresolvable here is the page's problem to report, not a private address.
        _ => return Ok(json!({ "ok": true, "resolved": false })),
    };
    if addresses
        .iter()
        .any(|ip| crate::browser_webview::is_private_host(&ip.to_string()))
    {
        return Ok(json!({
            "ok": false,
            "code": "private_address",
            "error": "This address leads to a private network or this computer, which the browser agent may not use",
        }));
    }
    Ok(json!({ "ok": true, "resolved": true }))
}

#[tauri::command]
pub async fn desktop_browser_agent(
    webview: Webview,
    app: tauri::AppHandle,
    session_id: String,
    op: String,
    args: Value,
) -> Result<Value, String> {
    crate::native_intents::ensure_main_window(&webview)?;
    let op = checked_op(&op)?;
    let args = checked_args(args)?;
    if op == "check_url" {
        return check_url(&args).await;
    }
    let state = app.state::<BrowserState>();
    let _agent = state.agent.lock().await;
    let (view, snapshot) = current(&app, &session_id)?;
    if !snapshot.committed {
        return Err(LOADING.into());
    }
    let name = &state.agent_name;
    match op {
        "screenshot" => screenshot(&app, &view, name, snapshot.visible, &args).await,
        "click" => click(&app, &view, &session_id, name, &args).await,
        _ => run(&view, name, op, &args).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_names_are_random_hex_globals() {
        let name = runtime_name();
        let suffix = name.strip_prefix("__unslothAgent_").expect("prefix");
        assert_eq!(suffix.len(), 16);
        assert!(suffix.bytes().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(name, runtime_name());
    }

    #[test]
    fn only_contract_operations_are_accepted() {
        for op in RUNTIME_OPS.iter().chain(&NATIVE_OPS) {
            assert_eq!(checked_op(op), Ok(*op));
        }
        for op in ["", "Click", "eval", "navigate", "__proto__", "click ", "probe"] {
            assert!(checked_op(op).is_err(), "{op}");
        }
    }

    #[test]
    fn arguments_must_be_a_bounded_object() {
        assert_eq!(checked_args(Value::Null), Ok(json!({})));
        assert_eq!(checked_args(json!({"ref": "e1"})), Ok(json!({"ref": "e1"})));
        assert!(checked_args(json!(["e1"])).is_err());
        assert!(checked_args(json!("e1")).is_err());
        let huge = "x".repeat(MAX_REQUEST_BYTES);
        assert!(checked_args(json!({ "text": huge })).is_err());
    }

    #[test]
    fn engine_replies_are_decoded_twice() {
        let inner = r#"{"ok":true,"text":"a \"quoted\" / line\nnext"}"#;
        let expected = Reply::Done(serde_json::from_str(inner).unwrap());
        // the shape from WebKitGTK (jsc to_json) and WebView2 (ExecuteScript).
        assert_eq!(
            decode_reply(&Value::String(inner.into()).to_string()),
            Ok(expected)
        );
        // the shape from NSJSONSerialization, which escapes '/' and may pad.
        let mac = r#" "{\"ok\":false,\"error\":\"a\/b\"}" "#;
        assert_eq!(
            decode_reply(mac),
            Ok(Reply::Done(json!({"ok": false, "error": "a/b"})))
        );
        // an engine that already decoded the string.
        assert_eq!(
            decode_reply(r#"{"ok":true}"#),
            Ok(Reply::Done(json!({"ok": true})))
        );
    }

    #[test]
    fn missing_runtime_and_failures_are_distinguished() {
        assert_eq!(decode_reply("null"), Ok(Reply::Missing));
        assert_eq!(decode_reply(""), Err(NO_RESULT.into()));
        for raw in [
            "undefined",
            "\"not json\"",
            "\"[1]\"",
            "\"{}\"",
            "42",
            r#""{\"ok\":1}""#,
        ] {
            assert_eq!(decode_reply(raw), Err(UNREADABLE.into()), "{raw}");
        }
        let huge = Value::String(format!(
            r#"{{"ok":true,"t":"{}"}}"#,
            "x".repeat(MAX_REPLY_BYTES)
        ));
        assert!(decode_reply(&huge.to_string()).is_err());
    }

    #[test]
    fn page_script_embeds_only_literals() {
        let request = json!({"op": "type", "args": {"text": "a'\"\u{2028}</script>"}}).to_string();
        let script = page_script("__unslothAgent_00ff", &request, false, 1234);
        assert!(script.contains(r#"var n="__unslothAgent_00ff""#));
        assert!(script.contains("Date.now()>1234"));
        assert!(script.contains(&js_string(&request)));
        assert!(!script.contains('\u{2028}'));
        assert!(!script.contains(RUNTIME.trim()));
        let install = page_script("__unslothAgent_00ff", &request, true, 1234);
        assert!(install.contains(&invoke_runtime(RUNTIME, "n")));
    }

    #[test]
    fn runtime_is_called_even_with_a_trailing_semicolon() {
        for source in [
            "(function (N) {})",
            "(function (N) {});\n",
            "(function (N) {}) ;  ",
        ] {
            assert_eq!(
                invoke_runtime(source, "\"x\""),
                "(function (N) {})\n(\"x\");"
            );
        }
        assert_eq!(invoke_runtime("(f) // end", "n"), "(f) // end\n(n);");
    }

    const PNG_HEAD: &[u8] = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR\0\0\x05\x00\0\0\x02\xd0\x08\x06\0\0\0";

    #[test]
    fn png_size_reads_the_header() {
        assert_eq!(png_size(PNG_HEAD), Some((1280, 720)));
        assert_eq!(png_size(&PNG_HEAD[..20]), None);
        assert_eq!(png_size(b"GIF89a.................."), None);
        let shot = shot_from_png(PNG_HEAD).unwrap();
        assert_eq!((shot.width, shot.height), (1280, 720));
        assert!(shot_from_png(b"not a png").is_err());
    }

    #[test]
    fn devtools_screenshots_keep_their_base64() {
        let data = base64::engine::general_purpose::STANDARD.encode(PNG_HEAD);
        let shot = shot_from_cdp(&json!({ "data": data }).to_string()).unwrap();
        assert_eq!(
            shot,
            Shot {
                data,
                width: 1280,
                height: 720
            }
        );
        assert!(shot_from_cdp("{}").is_err());
        assert!(shot_from_cdp(r#"{"data":"short"}"#).is_err());
        assert!(shot_from_cdp(r#"{"data":"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"}"#).is_err());
    }
}
