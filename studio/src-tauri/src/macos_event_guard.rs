//! Stops an AppKit exception during event dispatch from aborting the app.
//!
//! tao's `sendEvent:` override is `extern "C"`, so an Objective-C exception raised beneath it
//! aborts with "panic in a function that cannot unwind". macOS 27.0 raises one from the Siri
//! selected-text affordance (NSCampoLightweightUIController). Catching it at AppKit's own
//! `sendEvent:` restores stock AppKit behavior: report and keep running.
//! Upstream fix: tauri-apps/tao#1354.

use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use log::{error, warn};
use objc2::exception::Exception;
use objc2::rc::Retained;
use objc2::runtime::{AnyClass, AnyObject, Imp, Sel};
use objc2::{msg_send, sel};

type SendEvent = unsafe extern "C-unwind" fn(*mut AnyObject, Sel, *mut AnyObject);

static APPKIT_SEND_EVENT: OnceLock<SendEvent> = OnceLock::new();
static CAUGHT_EXCEPTIONS: AtomicU64 = AtomicU64::new(0);

/// Wraps `-[NSApplication sendEvent:]`. Call on the main thread before the event loop starts.
pub fn install() {
    if APPKIT_SEND_EVENT.get().is_some() {
        return;
    }
    let Some((class, method)) = AnyClass::get(c"NSApplication")
        .and_then(|class| Some((class, class.instance_method(sel!(sendEvent:))?)))
    else {
        warn!(
            "NSApplication has no sendEvent:; AppKit exceptions during event dispatch stay fatal"
        );
        return;
    };
    // SAFETY: Same signature, and the original is stored before the swap can call it.
    unsafe {
        let appkit: SendEvent = std::mem::transmute::<Imp, SendEvent>(method.implementation());
        let _ = APPKIT_SEND_EVENT.set(appkit);
        objc2::ffi::class_replaceMethod(
            (class as *const AnyClass).cast_mut(),
            sel!(sendEvent:),
            std::mem::transmute::<SendEvent, Imp>(guarded_send_event),
            objc2::ffi::method_getTypeEncoding(method),
        );
    }
}

unsafe extern "C-unwind" fn guarded_send_event(
    this: *mut AnyObject,
    cmd: Sel,
    event: *mut AnyObject,
) {
    if let Some(appkit) = APPKIT_SEND_EVENT.get() {
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
        unsafe { report(this, exception) };
    }
}

/// Reports like `-[NSApplication run]`, honoring NSApplicationCrashOnExceptions. Only the 1st,
/// 2nd, 4th, 8th, ... occurrence is reported so a per-event exception cannot flood the logs.
unsafe fn report(app: *mut AnyObject, exception: Option<Retained<Exception>>) {
    let count = CAUGHT_EXCEPTIONS.fetch_add(1, Ordering::Relaxed) + 1;
    if !count.is_power_of_two() {
        return;
    }
    let Some(exception) = exception else {
        error!("Caught a nil AppKit exception during event dispatch (#{count})");
        return;
    };
    error!("Caught an AppKit exception during event dispatch (#{count}): {exception:?}");
    // Skip nil: msg_send! panics on it in debug builds, and a panic here aborts.
    if let Some(app) = unsafe { app.as_ref() } {
        let () = unsafe { msg_send![app, reportException: &*exception] };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2::class;
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

    /// Stands in for tao's `extern "C"` override. Without the catch, this aborts the test.
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
        let method = AnyClass::get(c"NSApplication")
            .and_then(|class| class.instance_method(sel!(sendEvent:)))
            .expect("NSApplication implements sendEvent:");
        let appkit = method.implementation();
        install();
        install();
        let guarded: Imp = unsafe { std::mem::transmute::<SendEvent, Imp>(guarded_send_event) };
        assert_eq!(method.implementation() as usize, guarded as usize);
        let stored = *APPKIT_SEND_EVENT.get().expect("original stored");
        assert_eq!(stored as usize, appkit as usize);
    }
}
