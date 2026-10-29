//! tao's `extern "C"` `sendEvent:` overrides abort on any Objective-C exception beneath them
//! (macOS 27.0 NSCampoLightweightUIController throws one); catch it at AppKit's `sendEvent:`.
//! Upstream fix: tauri-apps/tao#1354.

use std::ffi::CStr;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use log::{error, warn};
use objc2::exception::Exception;
use objc2::rc::Retained;
use objc2::runtime::{AnyClass, AnyObject, Imp, Sel};
use objc2::{class, msg_send, sel};

type SendEvent = unsafe extern "C-unwind" fn(*mut AnyObject, Sel, *mut AnyObject);

/// NSWindow also covers TaoApp's Cmd+key-up path, which calls the key window's `sendEvent:`.
const GUARDED_CLASSES: [&CStr; 2] = [c"NSApplication", c"NSWindow"];
static APPKIT_SEND_EVENT: [OnceLock<SendEvent>; 2] = [OnceLock::new(), OnceLock::new()];
static CAUGHT_EXCEPTIONS: AtomicU64 = AtomicU64::new(0);

/// Call on the main thread before the event loop starts.
pub fn install() {
    install_guard::<0>();
    install_guard::<1>();
}

fn install_guard<const I: usize>() {
    let name = GUARDED_CLASSES[I];
    if APPKIT_SEND_EVENT[I].get().is_some() {
        return;
    }
    let Some((class, method)) = AnyClass::get(name)
        .and_then(|class| Some((class, class.instance_method(sel!(sendEvent:))?)))
    else {
        warn!("{name:?} has no sendEvent:; AppKit exceptions beneath it stay fatal");
        return;
    };
    // SAFETY: Same signature, and the original is stored before the swap can call it.
    unsafe {
        let appkit: SendEvent = std::mem::transmute::<Imp, SendEvent>(method.implementation());
        let _ = APPKIT_SEND_EVENT[I].set(appkit);
        objc2::ffi::class_replaceMethod(
            (class as *const AnyClass).cast_mut(),
            sel!(sendEvent:),
            std::mem::transmute::<SendEvent, Imp>(guarded_send_event::<I>),
            objc2::ffi::method_getTypeEncoding(method),
        );
    }
}

unsafe extern "C-unwind" fn guarded_send_event<const I: usize>(
    this: *mut AnyObject,
    cmd: Sel,
    event: *mut AnyObject,
) {
    if let Some(appkit) = APPKIT_SEND_EVENT[I].get() {
        // SAFETY: AppKit's arguments, forwarded to AppKit's implementation.
        unsafe { dispatch(*appkit, this, cmd, event) };
    }
}

/// Catches Objective-C exceptions only. Rust panics pass through and stay fatal.
unsafe fn dispatch(send_event: SendEvent, this: *mut AnyObject, cmd: Sel, event: *mut AnyObject) {
    // SAFETY: Upheld by the caller.
    let result =
        objc2::exception::catch(AssertUnwindSafe(|| unsafe { send_event(this, cmd, event) }));
    if let Err(exception) = result {
        // Null only in tests, which must not touch NSApp off the main thread.
        unsafe { report(!this.is_null(), exception) };
    }
}

/// Like `-[NSApplication run]`, honoring NSApplicationCrashOnExceptions. Powers of two only.
unsafe fn report(to_app: bool, exception: Option<Retained<Exception>>) {
    let count = CAUGHT_EXCEPTIONS.fetch_add(1, Ordering::Relaxed) + 1;
    if !count.is_power_of_two() {
        return;
    }
    let Some(exception) = exception else {
        error!("Caught a nil AppKit exception during event dispatch (#{count})");
        return;
    };
    error!("Caught an AppKit exception during event dispatch (#{count}): {exception:?}");
    if !to_app {
        return;
    }
    // Skip nil: msg_send! panics on it in debug builds, and a panic here aborts.
    let app: Option<Retained<AnyObject>> =
        unsafe { msg_send![class!(NSApplication), sharedApplication] };
    if let Some(app) = app {
        let () = unsafe { msg_send![&*app, reportException: &*exception] };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_foundation::NSString;

    static REACHED_APPKIT: AtomicU64 = AtomicU64::new(0);

    unsafe extern "C-unwind" fn throwing_send_event(_: *mut AnyObject, _: Sel, _: *mut AnyObject) {
        REACHED_APPKIT.fetch_add(1, Ordering::SeqCst);
        let name = NSString::from_str("NSInternalInconsistencyException");
        let reason = NSString::from_str("simulated NSCampoLightweightUIController assertion");
        let exception: Retained<AnyObject> = unsafe {
            msg_send![
                class!(NSException),
                exceptionWithName: &*name,
                reason: &*reason,
                userInfo: std::ptr::null::<AnyObject>()
            ]
        };
        // SAFETY: An NSException instance.
        objc2::exception::throw(unsafe { Retained::cast_unchecked::<Exception>(exception) });
    }

    extern "C" fn tao_send_event(send_event: SendEvent) {
        unsafe {
            dispatch(
                send_event,
                std::ptr::null_mut(),
                sel!(sendEvent:),
                std::ptr::null_mut(),
            )
        };
    }

    #[test]
    fn appkit_exceptions_beneath_an_extern_c_frame_do_not_abort() {
        for _ in 0..5 {
            tao_send_event(throwing_send_event);
        }
        assert_eq!(REACHED_APPKIT.load(Ordering::SeqCst), 5);
        assert_eq!(CAUGHT_EXCEPTIONS.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn install_wraps_appkit_send_event_once() {
        let method = |name: &CStr| {
            AnyClass::get(name)
                .and_then(|class| class.instance_method(sel!(sendEvent:)))
                .expect("AppKit class implements sendEvent:")
        };
        let appkit = GUARDED_CLASSES.map(|name| method(name).implementation() as usize);
        install();
        install();
        let guarded: [Imp; 2] = unsafe {
            [
                std::mem::transmute::<SendEvent, Imp>(guarded_send_event::<0>),
                std::mem::transmute::<SendEvent, Imp>(guarded_send_event::<1>),
            ]
        };
        for (i, name) in GUARDED_CLASSES.into_iter().enumerate() {
            assert_eq!(method(name).implementation() as usize, guarded[i] as usize);
            let stored = *APPKIT_SEND_EVENT[i].get().expect("original stored");
            assert_eq!(stored as usize, appkit[i]);
        }
    }
}
