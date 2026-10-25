//! Transcript insertion into the focused app.
//!
//! The transcript goes through the clipboard so layout, IME state and rich text
//! fields all behave the same as a real paste. The previous clipboard contents are
//! saved first and restored a moment later, unless the user asked to keep the
//! transcript around or they replaced the clipboard themselves in the meantime.

use std::borrow::Cow;
use std::time::Duration;

use arboard::{Clipboard, ImageData};

/// The paste target needs a moment to consume the clipboard before it is safe to
/// put the previous contents back.
const RESTORE_DELAY: Duration = Duration::from_millis(350);
/// Enter must land after the target has inserted the paste, not while it is
/// still handling Ctrl+V.
const SUBMIT_DELAY: Duration = Duration::from_millis(120);

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
/// which is the way out of apps that refuse synthesized keystrokes. `submit` sends
/// Enter after a successful paste, for chat boxes that send on Enter.
#[tauri::command]
pub async fn voice_typing_insert(
    text: String,
    keep_on_clipboard: bool,
    paste: bool,
    submit: bool,
) -> Result<String, String> {
    let mut clipboard =
        Clipboard::new().map_err(|error| format!("Clipboard unavailable: {error}"))?;
    let saved = if paste { save(&mut clipboard) } else { None };
    clipboard
        .set_text(text.clone())
        .map_err(|error| format!("Clipboard unavailable: {error}"))?;
    let characters = text.chars().count();

    if !paste {
        log::info!("Voice typing insert: copied to the clipboard ({characters} characters)");
        return Ok("copied".to_string());
    }

    if !paste_chord() {
        log::warn!("Voice typing insert: paste key could not be sent; transcript left on the clipboard");
        return Ok("copied".to_string());
    }

    // The clipboard handle is not held across the submit delay; on some
    // platforms it is not `Send`, and the restore opens its own handle anyway.
    drop(clipboard);

    if submit {
        // Async command on a worker thread: sleeping here delays nothing else.
        tokio::time::sleep(SUBMIT_DELAY).await;
        if enter_key() {
            log::info!("Voice typing insert: submitted with Enter");
        } else {
            log::warn!("Voice typing insert: Enter could not be sent");
        }
    }

    if !keep_on_clipboard {
        if let Some(saved) = saved {
            // Restore on a worker thread: rewriting the clipboard immediately can
            // race the paste that is still being delivered.
            std::thread::spawn(move || {
                std::thread::sleep(RESTORE_DELAY);
                let Ok(mut clipboard) = Clipboard::new() else {
                    return;
                };
                // Only restore when the clipboard still holds our transcript; if the
                // user copied something else meanwhile, their clipboard wins.
                let still_ours = clipboard.get_text().ok().as_deref() == Some(text.as_str());
                if still_ours {
                    restore(&mut clipboard, saved);
                }
            });
        }
    }

    log::info!("Voice typing insert: pasted ({characters} characters)");
    Ok("pasted".to_string())
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
fn paste_chord() -> bool {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{VK_CONTROL, VK_V};

    send_virtual_keys(&[
        (VK_CONTROL, false),
        (VK_V, false),
        (VK_V, true),
        (VK_CONTROL, true),
    ])
}

#[cfg(target_os = "windows")]
fn enter_key() -> bool {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::VK_RETURN;

    send_virtual_keys(&[(VK_RETURN, false), (VK_RETURN, true)])
}

#[cfg(target_os = "macos")]
fn paste_chord() -> bool {
    // System Events requires the Accessibility permission; if it is missing this
    // fails and the transcript stays on the clipboard for a manual paste.
    std::process::Command::new("osascript")
        .args([
            "-e",
            "tell application \"System Events\" to keystroke \"v\" using command down",
        ])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(target_os = "macos")]
fn enter_key() -> bool {
    std::process::Command::new("osascript")
        .args(["-e", "tell application \"System Events\" to key code 36"])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(target_os = "linux")]
fn paste_chord() -> bool {
    // Wayland first, then the X11 tools; whichever is installed and can reach the
    // compositor wins, and a miss just falls through to the manual-paste path.
    let attempts: [(&str, &[&str]); 3] = [
        ("wtype", &["-M", "ctrl", "v", "-m", "ctrl"]),
        ("xdotool", &["key", "--clearmodifiers", "ctrl+v"]),
        ("ydotool", &["key", "29:1", "47:1", "47:0", "29:0"]),
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
fn enter_key() -> bool {
    let attempts: [(&str, &[&str]); 3] = [
        ("wtype", &["Return"]),
        ("xdotool", &["key", "--clearmodifiers", "Return"]),
        ("ydotool", &["key", "28:1", "28:0"]),
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

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
fn paste_chord() -> bool {
    false
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
fn enter_key() -> bool {
    false
}
