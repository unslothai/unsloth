//! Transcript insertion into the focused app.
//!
//! The transcript goes through the clipboard so layout, IME state and rich text
//! fields all behave the same as a real paste. The previous clipboard contents are
//! saved first and restored a moment later, unless the user asked to keep the
//! transcript around or they replaced the clipboard themselves in the meantime.

use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use arboard::{Clipboard, ImageData};

/// The paste target needs a moment to consume the clipboard before it is safe to
/// put the previous contents back.
const RESTORE_DELAY: Duration = Duration::from_millis(450);
/// A real space key after the paste is more reliable than a trailing space in the
/// clipboard: many fields trim trailing whitespace when pasting. 65ms gives async
/// Electron/WebView2 editors time to finish inserting the clipboard before Space lands.
const SPACE_DELAY: Duration = Duration::from_millis(65);
/// Enter must land after the target has inserted the paste, not while it is
/// still handling Ctrl+V.
const SUBMIT_DELAY: Duration = Duration::from_millis(120);
/// Brief pause after releasing physically held shortcut modifiers before sending the
/// paste chord, so the target window registers a clean modifier state.
#[cfg(target_os = "windows")]
const MODIFIER_RELEASE_SETTLE: Duration = Duration::from_millis(12);

static PASTE_EPOCH: AtomicU64 = AtomicU64::new(0);
static PENDING_RESTORE: LazyLock<Mutex<Option<(SavedClipboard, String)>>> =
    LazyLock::new(|| Mutex::new(None));

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PasteMethod {
    CtrlV,
    CtrlShiftV,
    ShiftInsert,
    Direct,
}

impl PasteMethod {
    pub fn from_option(raw: Option<&str>) -> Self {
        match raw.unwrap_or("ctrl_v") {
            "ctrl_shift_v" => Self::CtrlShiftV,
            "shift_insert" => Self::ShiftInsert,
            "direct" => Self::Direct,
            _ => Self::CtrlV,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubmitKey {
    None,
    Enter,
    CtrlEnter,
    ShiftEnter,
}

impl SubmitKey {
    pub fn from_options(raw: Option<&str>, legacy_submit: bool) -> Self {
        match raw.unwrap_or_default() {
            "enter" => Self::Enter,
            "ctrl_enter" => Self::CtrlEnter,
            "shift_enter" => Self::ShiftEnter,
            "none" => Self::None,
            _ if legacy_submit => Self::Enter,
            _ => Self::None,
        }
    }
}

enum SavedClipboard {
    Text(String),
    Image {
        width: usize,
        height: usize,
        bytes: Vec<u8>,
    },
}

/// Types `text` into the focused window and returns `"pasted"`, or `"copied"` when
/// the platform has no way to synthesize a paste: the transcript then waits on the
/// clipboard for a manual paste. With `paste` off the transcript is only copied,
/// which is the way out of apps that refuse synthesized keystrokes. `copy_last_transcript`
/// keeps the finished text on the clipboard after an automatic paste, without copying
/// it a second time. `submit_key` sends Enter, Ctrl+Enter or Shift+Enter after a
/// successful paste, for chat boxes that send or add a line on that chord.
#[tauri::command]
pub async fn voice_typing_insert(
    text: String,
    copy_last_transcript: bool,
    paste: bool,
    submit: Option<bool>,
    submit_key: Option<String>,
    trailing_space: bool,
    paste_method: Option<String>,
) -> Result<String, String> {
    let text = text.trim_end().to_string();
    let characters = text.chars().count();
    let method = PasteMethod::from_option(paste_method.as_deref());
    let submit_key = SubmitKey::from_options(submit_key.as_deref(), submit.unwrap_or(false));

    // Direct Unicode typing bypasses the clipboard unless the user explicitly asked
    // to keep the last transcript on the clipboard or `paste` is disabled.
    if paste && method == PasteMethod::Direct {
        if copy_last_transcript {
            if let Ok(mut clipboard) = Clipboard::new() {
                let _ = clipboard.set_text(clipboard_text(&text, trailing_space, false));
            }
        }
        let direct_payload = clipboard_text(&text, trailing_space, false);
        #[cfg(target_os = "windows")]
        if release_held_modifiers() {
            tokio::time::sleep(MODIFIER_RELEASE_SETTLE).await;
        }
        if direct_type(&direct_payload) {
            if submit_key != SubmitKey::None {
                tokio::time::sleep(SUBMIT_DELAY).await;
                let _ = send_submit_key(submit_key);
            }
            log::info!("Voice typing insert: direct typed ({characters} characters)");
            return Ok("pasted".to_string());
        }
        log::warn!("Voice typing insert: direct typing failed; falling back to clipboard paste");
    }

    let copy_text = clipboard_text(&text, trailing_space, paste);
    let epoch = PASTE_EPOCH.fetch_add(1, Ordering::SeqCst) + 1;
    let mut clipboard =
        Clipboard::new().map_err(|error| format!("Clipboard unavailable: {error}"))?;

    if paste && !copy_last_transcript {
        let mut guard = PENDING_RESTORE.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            if let Some(saved) = save(&mut clipboard) {
                *guard = Some((saved, text.clone()));
            }
        } else if let Some((_, ref mut latest_text)) = *guard {
            *latest_text = text.clone();
        }
    } else if copy_last_transcript || !paste {
        // Explicit clipboard copy supersedes any pending restore from an earlier paste.
        let mut guard = PENDING_RESTORE.lock().unwrap_or_else(|e| e.into_inner());
        *guard = None;
    }

    clipboard
        .set_text(copy_text)
        .map_err(|error| format!("Clipboard unavailable: {error}"))?;

    if !paste {
        log::info!("Voice typing insert: copied to the clipboard ({characters} characters)");
        return Ok("copied".to_string());
    }

    #[cfg(target_os = "windows")]
    if release_held_modifiers() {
        tokio::time::sleep(MODIFIER_RELEASE_SETTLE).await;
    }

    if !paste_chord_with(method) {
        // The transcript is now the manual-paste path; give it the space it would
        // have received as a key.
        let _ = clipboard.set_text(clipboard_text(&text, trailing_space, false));
        let mut guard = PENDING_RESTORE.lock().unwrap_or_else(|e| e.into_inner());
        *guard = None;
        log::warn!("Voice typing insert: paste key could not be sent; transcript left on the clipboard");
        return Ok("copied".to_string());
    }

    // The clipboard handle is not held across the delay; on some platforms it is
    // not `Send`, and the restore opens its own handle anyway.
    drop(clipboard);

    if trailing_space && !ends_with_space(&text) {
        // Async command on a worker thread: sleeping here delays nothing else.
        tokio::time::sleep(SPACE_DELAY).await;
        if space_key() {
            log::info!("Voice typing insert: trailing space sent");
        } else {
            log::warn!("Voice typing insert: trailing space could not be sent");
        }
    }

    if submit_key != SubmitKey::None {
        tokio::time::sleep(SUBMIT_DELAY).await;
        if send_submit_key(submit_key) {
            log::info!("Voice typing insert: submitted with {submit_key:?}");
        } else {
            log::warn!("Voice typing insert: {submit_key:?} could not be sent");
        }
    }

    if !copy_last_transcript {
        // Restore on a worker thread: rewriting the clipboard immediately can
        // race the paste that is still being delivered. Coordinated by `epoch` so
        // back-to-back pastes never overwrite the original saved clipboard.
        std::thread::spawn(move || {
            std::thread::sleep(RESTORE_DELAY);
            if PASTE_EPOCH.load(Ordering::SeqCst) != epoch {
                return;
            }
            let pending = {
                let mut guard = PENDING_RESTORE.lock().unwrap_or_else(|e| e.into_inner());
                guard.take()
            };
            let Some((saved, expected_text)) = pending else {
                return;
            };
            let Ok(mut clipboard) = Clipboard::new() else {
                return;
            };
            // Only restore when the clipboard still holds our transcript; if the
            // user copied something else meanwhile, their clipboard wins.
            let still_ours = clipboard
                .get_text()
                .ok()
                .map(|current| current.trim_end() == expected_text.as_str())
                .unwrap_or(false);
            if still_ours {
                restore(&mut clipboard, saved);
            }
        });
    }

    log::info!("Voice typing insert: pasted ({characters} characters)");
    Ok("pasted".to_string())
}

/// What goes on the clipboard. A manual paste must carry the trailing space, but a
/// synthesized paste sends a real space key instead so a field that trims trailing
/// whitespace cannot drop it.
fn clipboard_text(text: &str, trailing_space: bool, paste: bool) -> String {
    if trailing_space && !paste && !ends_with_space(text) {
        format!("{text} ")
    } else {
        text.to_string()
    }
}

fn ends_with_space(text: &str) -> bool {
    text.chars().last().is_some_and(char::is_whitespace)
}

fn save(clipboard: &mut Clipboard) -> Option<SavedClipboard> {
    if let Ok(text) = clipboard.get_text() {
        if !text.is_empty() {
            return Some(SavedClipboard::Text(text));
        }
    }
    if let Ok(image) = clipboard.get_image() {
        return Some(SavedClipboard::Image {
            width: image.width,
            height: image.height,
            bytes: image.bytes.into_owned(),
        });
    }
    // Nothing representable (copied files, for instance): never clear what we
    // cannot rebuild, so the transcript simply stays in place.
    None
}

fn restore(clipboard: &mut Clipboard, saved: SavedClipboard) {
    match saved {
        SavedClipboard::Text(text) => {
            let _ = clipboard.set_text(text);
        }
        SavedClipboard::Image {
            width,
            height,
            bytes,
        } => {
            let _ = clipboard.set_image(ImageData {
                width,
                height,
                bytes: Cow::Borrowed(&bytes),
            });
        }
    }
}

#[cfg(target_os = "windows")]
fn send_virtual_keys(keys: &[(u16, bool)]) -> bool {
    use std::mem::size_of;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP,
    };

    let inputs: Vec<INPUT> = keys
        .iter()
        .map(|&(vk, up)| INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: vk,
                    wScan: 0,
                    dwFlags: if up { KEYEVENTF_KEYUP } else { 0 },
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        })
        .collect();
    let sent = unsafe {
        SendInput(
            inputs.len() as u32,
            inputs.as_ptr(),
            size_of::<INPUT>() as i32,
        )
    };
    sent == inputs.len() as u32
}

#[cfg(target_os = "windows")]
fn release_held_modifiers() -> bool {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, VK_CONTROL, VK_LCONTROL, VK_LMENU, VK_LSHIFT, VK_LWIN, VK_MENU,
        VK_RCONTROL, VK_RMENU, VK_RSHIFT, VK_RWIN, VK_SHIFT,
    };

    let candidates: [u16; 11] = [
        VK_LSHIFT,
        VK_RSHIFT,
        VK_SHIFT,
        VK_LMENU,
        VK_RMENU,
        VK_MENU,
        VK_LWIN,
        VK_RWIN,
        VK_LCONTROL,
        VK_RCONTROL,
        VK_CONTROL,
    ];
    let mut releases: Vec<(u16, bool)> = Vec::new();
    for vk in candidates {
        let down = unsafe { (GetAsyncKeyState(vk as i32) as u16 & 0x8000) != 0 };
        if down {
            releases.push((vk, true));
        }
    }
    if releases.is_empty() {
        return false;
    }
    let _ = send_virtual_keys(&releases);
    true
}

#[cfg(target_os = "windows")]
fn direct_type(text: &str) -> bool {
    use std::mem::size_of;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE,
    };

    if text.is_empty() {
        return true;
    }
    let utf16: Vec<u16> = text.encode_utf16().collect();
    let mut inputs: Vec<INPUT> = Vec::with_capacity(utf16.len() * 2);
    for unit in utf16 {
        for up in [false, true] {
            inputs.push(INPUT {
                r#type: INPUT_KEYBOARD,
                Anonymous: INPUT_0 {
                    ki: KEYBDINPUT {
                        wVk: 0,
                        wScan: unit,
                        dwFlags: KEYEVENTF_UNICODE | if up { KEYEVENTF_KEYUP } else { 0 },
                        time: 0,
                        dwExtraInfo: 0,
                    },
                },
            });
        }
    }
    let sent = unsafe {
        SendInput(
            inputs.len() as u32,
            inputs.as_ptr(),
            size_of::<INPUT>() as i32,
        )
    };
    sent == inputs.len() as u32
}

#[cfg(target_os = "windows")]
fn paste_chord_with(method: PasteMethod) -> bool {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{VK_CONTROL, VK_INSERT, VK_SHIFT, VK_V};

    match method {
        PasteMethod::CtrlShiftV => send_virtual_keys(&[
            (VK_CONTROL, false),
            (VK_SHIFT, false),
            (VK_V, false),
            (VK_V, true),
            (VK_SHIFT, true),
            (VK_CONTROL, true),
        ]),
        PasteMethod::ShiftInsert => send_virtual_keys(&[
            (VK_SHIFT, false),
            (VK_INSERT, false),
            (VK_INSERT, true),
            (VK_SHIFT, true),
        ]),
        PasteMethod::CtrlV | PasteMethod::Direct => send_virtual_keys(&[
            (VK_CONTROL, false),
            (VK_V, false),
            (VK_V, true),
            (VK_CONTROL, true),
        ]),
    }
}

#[cfg(target_os = "windows")]
fn space_key() -> bool {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::VK_SPACE;

    send_virtual_keys(&[(VK_SPACE, false), (VK_SPACE, true)])
}

#[cfg(target_os = "windows")]
fn send_submit_key(key: SubmitKey) -> bool {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        VK_CONTROL, VK_RETURN, VK_SHIFT,
    };

    match key {
        SubmitKey::None => true,
        SubmitKey::Enter => send_virtual_keys(&[(VK_RETURN, false), (VK_RETURN, true)]),
        SubmitKey::CtrlEnter => send_virtual_keys(&[
            (VK_CONTROL, false),
            (VK_RETURN, false),
            (VK_RETURN, true),
            (VK_CONTROL, true),
        ]),
        SubmitKey::ShiftEnter => send_virtual_keys(&[
            (VK_SHIFT, false),
            (VK_RETURN, false),
            (VK_RETURN, true),
            (VK_SHIFT, true),
        ]),
    }
}

#[cfg(target_os = "macos")]
fn direct_type(text: &str) -> bool {
    let escaped = text.replace('\\', "\\\\").replace('"', "\\\"");
    std::process::Command::new("osascript")
        .args([
            "-e",
            &format!("tell application \"System Events\" to keystroke \"{escaped}\""),
        ])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(target_os = "macos")]
fn paste_chord_with(method: PasteMethod) -> bool {
    let script = match method {
        PasteMethod::CtrlShiftV => {
            "tell application \"System Events\" to keystroke \"v\" using {command down, shift down}"
        }
        _ => "tell application \"System Events\" to keystroke \"v\" using command down",
    };
    std::process::Command::new("osascript")
        .args(["-e", script])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(target_os = "macos")]
fn space_key() -> bool {
    std::process::Command::new("osascript")
        .args(["-e", "tell application \"System Events\" to key code 49"])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(target_os = "macos")]
fn send_submit_key(key: SubmitKey) -> bool {
    let script = match key {
        SubmitKey::None => return true,
        SubmitKey::Enter => "tell application \"System Events\" to key code 36",
        SubmitKey::CtrlEnter => {
            "tell application \"System Events\" to key code 36 using control down"
        }
        SubmitKey::ShiftEnter => {
            "tell application \"System Events\" to key code 36 using shift down"
        }
    };
    std::process::Command::new("osascript")
        .args(["-e", script])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(target_os = "linux")]
fn direct_type(text: &str) -> bool {
    if std::process::Command::new("wtype")
        .args(["--", text])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
    {
        return true;
    }
    std::process::Command::new("xdotool")
        .args(["type", "--clearmodifiers", "--", text])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(target_os = "linux")]
fn paste_chord_with(method: PasteMethod) -> bool {
    let attempts: [(&str, &[&str]); 3] = match method {
        PasteMethod::CtrlShiftV => [
            (
                "wtype",
                &["-M", "ctrl", "-M", "shift", "v", "-m", "shift", "-m", "ctrl"],
            ),
            ("xdotool", &["key", "--clearmodifiers", "ctrl+shift+v"]),
            (
                "ydotool",
                &["key", "29:1", "42:1", "47:1", "47:0", "42:0", "29:0"],
            ),
        ],
        PasteMethod::ShiftInsert => [
            ("wtype", &["-M", "shift", "-k", "Insert", "-m", "shift"]),
            ("xdotool", &["key", "--clearmodifiers", "shift+Insert"]),
            ("ydotool", &["key", "42:1", "110:1", "110:0", "42:0"]),
        ],
        PasteMethod::CtrlV | PasteMethod::Direct => [
            ("wtype", &["-M", "ctrl", "v", "-m", "ctrl"]),
            ("xdotool", &["key", "--clearmodifiers", "ctrl+v"]),
            ("ydotool", &["key", "29:1", "47:1", "47:0", "29:0"]),
        ],
    };
    for (program, args) in attempts {
        if std::process::Command::new(program)
            .args(args)
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
        {
            return true;
        }
    }
    false
}

#[cfg(target_os = "linux")]
fn space_key() -> bool {
    let attempts: [(&str, &[&str]); 3] = [
        ("wtype", &["space"]),
        ("xdotool", &["key", "--clearmodifiers", "space"]),
        ("ydotool", &["key", "57:1", "57:0"]),
    ];
    for (program, args) in attempts {
        if std::process::Command::new(program)
            .args(args)
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
        {
            return true;
        }
    }
    false
}

#[cfg(target_os = "linux")]
fn send_submit_key(key: SubmitKey) -> bool {
    let attempts: &[(&str, &[&str])] = match key {
        SubmitKey::None => return true,
        SubmitKey::Enter => &[
            ("wtype", &["Return"]),
            ("xdotool", &["key", "--clearmodifiers", "Return"]),
            ("ydotool", &["key", "28:1", "28:0"]),
        ],
        SubmitKey::CtrlEnter => &[
            (
                "wtype",
                &["-M", "ctrl", "-k", "Return", "-m", "ctrl"],
            ),
            ("xdotool", &["key", "--clearmodifiers", "ctrl+Return"]),
            ("ydotool", &["key", "29:1", "28:1", "28:0", "29:0"]),
        ],
        SubmitKey::ShiftEnter => &[
            (
                "wtype",
                &["-M", "shift", "-k", "Return", "-m", "shift"],
            ),
            ("xdotool", &["key", "--clearmodifiers", "shift+Return"]),
            ("ydotool", &["key", "42:1", "28:1", "28:0", "42:0"]),
        ],
    };
    for &(program, args) in attempts {
        if std::process::Command::new(program)
            .args(args)
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
        {
            return true;
        }
    }
    false
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
fn direct_type(_text: &str) -> bool {
    false
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
fn paste_chord_with(_method: PasteMethod) -> bool {
    false
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
fn space_key() -> bool {
    false
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
fn send_submit_key(_key: SubmitKey) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::{clipboard_text, ends_with_space, PasteMethod, SubmitKey};

    #[test]
    fn copy_only_keeps_the_trailing_space_for_a_manual_paste() {
        assert_eq!(clipboard_text("hello", true, false), "hello ");
        assert_eq!(clipboard_text("hello", false, false), "hello");
    }

    #[test]
    fn a_real_paste_sends_the_space_as_a_key_instead() {
        // The clipboard copy stays exact, so the target cannot trim the space away.
        assert_eq!(clipboard_text("hello", true, true), "hello");
    }

    #[test]
    fn existing_whitespace_is_never_doubled() {
        assert!(ends_with_space("hello "));
        assert!(ends_with_space("hello\n"));
        assert!(!ends_with_space("hello"));
        assert_eq!(clipboard_text("hello ", true, false), "hello ");
    }

    #[test]
    fn parses_submit_keys_with_legacy_boolean() {
        assert_eq!(SubmitKey::from_options(None, false), SubmitKey::None);
        assert_eq!(SubmitKey::from_options(None, true), SubmitKey::Enter);
        assert_eq!(
            SubmitKey::from_options(Some("ctrl_enter"), false),
            SubmitKey::CtrlEnter
        );
        assert_eq!(
            SubmitKey::from_options(Some("shift_enter"), false),
            SubmitKey::ShiftEnter
        );
        assert_eq!(
            SubmitKey::from_options(Some("none"), true),
            SubmitKey::None
        );
    }

    #[test]
    fn parses_paste_methods_with_safe_default() {
        assert_eq!(PasteMethod::from_option(None), PasteMethod::CtrlV);
        assert_eq!(PasteMethod::from_option(Some("ctrl_v")), PasteMethod::CtrlV);
        assert_eq!(
            PasteMethod::from_option(Some("ctrl_shift_v")),
            PasteMethod::CtrlShiftV
        );
        assert_eq!(
            PasteMethod::from_option(Some("shift_insert")),
            PasteMethod::ShiftInsert
        );
        assert_eq!(PasteMethod::from_option(Some("direct")), PasteMethod::Direct);
        assert_eq!(PasteMethod::from_option(Some("unknown")), PasteMethod::CtrlV);
    }
}
