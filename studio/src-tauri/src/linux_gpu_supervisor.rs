// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! Startup supervisor for GPU rendering on NVIDIA Wayland.
//!
//! WebKitGTK's GPU path can die at launch on NVIDIA Wayland: GTK3 logs
//! `Error 71 (Protocol error) dispatching to Wayland display.` and calls `_exit(1)`
//! (gdk/wayland/gdkeventsource.c, gtk-3-24, lines 203-205). Nothing in-process survives that,
//! so on hosts where the GPU path is risky `main()` becomes a small parent that runs the real
//! app as a child, and relaunches it once on the CPU path if the child dies that way before it
//! has shown a window.
//!
//! The parent never builds Tauri or GTK, so it never holds the single-instance D-Bus name.
//! Parent and child talk over an inherited `AF_UNIX` socket whose fd is named by
//! [`SUPERVISOR_FD_ENV`]:
//! - child to parent: `S` (a window was shown), `R` (a window has been on screen for
//!   [`READY_GRACE`]), `E` (GDK logged a Wayland protocol error);
//! - parent to child: `T` (the parent got SIGTERM; please shut down cleanly).

use std::convert::Infallible;
use std::ffi::{CStr, OsString};
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::os::unix::net::UnixStream;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::process::{Command, ExitStatus};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

/// Names the inherited supervisor socket fd in the child.
pub const SUPERVISOR_FD_ENV: &str = "UNSLOTH_GPU_SUPERVISOR_FD";
/// Set on the CPU relaunch; its value is the reason (`error71`).
pub const FALLBACK_ENV: &str = "UNSLOTH_GPU_FALLBACK";
/// Debug builds only: `1` makes a supervised child fake the GTK Error 71 exit.
#[cfg(debug_assertions)]
pub const TEST_ERROR71_ENV: &str = "UNSLOTH_TEST_GPU_ERROR71";

const MSG_SHOWN: u8 = b'S';
const MSG_READY: u8 = b'R';
const MSG_ERROR71: u8 = b'E';
const MSG_TERMINATE: u8 = b'T';
const FALLBACK_REASON: &str = "error71";
/// How long a window must stay up before the child counts the GPU path as working.
const READY_GRACE: Duration = Duration::from_secs(2);

static CHANNEL: OnceLock<UnixStream> = OnceLock::new();

// ---------------------------------------------------------------------------------------------
// Child side
// ---------------------------------------------------------------------------------------------

/// What the parent handed this process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChildChannel {
    /// A valid supervisor socket was inherited and is now held in a static.
    pub supervised: bool,
    /// The value of [`FALLBACK_ENV`], if set: the parent relaunched us on the CPU path.
    pub fallback: Option<String>,
}

/// Call first in `main()`, while the process is still single-threaded. Consumes and removes
/// both env vars so the backend and any `process::restart` grandchild start clean, checks the
/// fd is a socket (anything else is ignored and left alone), and marks it close-on-exec.
pub fn take_child_channel() -> ChildChannel {
    let (stream, fallback) = channel_from_env();
    let supervised = match stream {
        Some(stream) => CHANNEL.set(stream).is_ok() || CHANNEL.get().is_some(),
        None => CHANNEL.get().is_some(),
    };
    ChildChannel {
        supervised,
        fallback,
    }
}

fn channel_from_env() -> (Option<UnixStream>, Option<String>) {
    let fd = std::env::var_os(SUPERVISOR_FD_ENV);
    std::env::remove_var(SUPERVISOR_FD_ENV);
    let fallback = std::env::var_os(FALLBACK_ENV).map(|value| value.to_string_lossy().into_owned());
    std::env::remove_var(FALLBACK_ENV);

    let stream = fd
        .and_then(|fd| fd.to_str().and_then(|fd| fd.trim().parse::<RawFd>().ok()))
        .filter(|fd| *fd > 2 && fd_is_socket(*fd))
        .map(|fd| {
            set_cloexec(fd, true);
            // SAFETY: the fd is an open socket inherited from the parent and nothing else in
            // this process owns it.
            unsafe { UnixStream::from_raw_fd(fd) }
        });
    (stream, fallback)
}

fn fd_is_socket(fd: RawFd) -> bool {
    // SAFETY: fstat writes into a zeroed, correctly sized buffer.
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut stat) } != 0 {
        return false;
    }
    stat.st_mode & libc::S_IFMT == libc::S_IFSOCK
}

/// Async-signal-safe (only `fcntl`), so it may run in `pre_exec`.
fn set_cloexec(fd: RawFd, cloexec: bool) -> bool {
    // SAFETY: plain fcntl on an fd; failure is reported through the return value.
    unsafe {
        let flags = libc::fcntl(fd, libc::F_GETFD);
        if flags < 0 {
            return false;
        }
        let flags = if cloexec {
            flags | libc::FD_CLOEXEC
        } else {
            flags & !libc::FD_CLOEXEC
        };
        libc::fcntl(fd, libc::F_SETFD, flags) == 0
    }
}

/// Best effort: a dead parent must not hurt the child, so errors (and SIGPIPE) are ignored.
fn send(byte: u8) {
    if let Some(stream) = CHANNEL.get() {
        // SAFETY: one byte from a valid buffer; MSG_NOSIGNAL turns a closed peer into EPIPE.
        unsafe {
            libc::send(
                stream.as_raw_fd(),
                (&byte as *const u8).cast(),
                1,
                libc::MSG_NOSIGNAL,
            );
        }
    }
}

/// GTK3 reports a fatal Wayland error with one of (gdk/wayland/gdkeventsource.c, gtk-3-24):
/// - `"Error %d (%s) dispatching to Wayland display."` (203-205)
/// - `"Error flushing display: %s"` (67)
/// - `"Error reading events from display: %s"` (96)
///
/// each followed by `_exit(1)`. Only EPROTO (71) is the GPU-buffer failure; EPIPE and
/// "Lost connection to Wayland compositor." mean the compositor went away and do not count.
pub fn is_wayland_protocol_error(msg: &str) -> bool {
    if msg.contains("Error 71 (") && msg.contains("dispatching to Wayland display") {
        return true;
    }
    let strerror = msg
        .strip_prefix("Error flushing display: ")
        .or_else(|| msg.strip_prefix("Error reading events from display: "));
    match strerror {
        // GLib translates g_strerror, so also compare against this locale's EPROTO text.
        Some(text) => {
            let text = text.trim();
            text == "Protocol error" || localized_eproto().is_some_and(|eproto| text == eproto)
        }
        None => false,
    }
}

fn localized_eproto() -> Option<String> {
    // SAFETY: g_strerror returns a valid, static, NUL-terminated UTF-8 string.
    let text = unsafe { CStr::from_ptr(glib::ffi::g_strerror(libc::EPROTO)) };
    text.to_str().ok().map(str::to_owned)
}

/// Child only (no-op without a channel): reports GDK's Wayland protocol error to the parent
/// before GTK's `_exit(1)`, then hands the message to GLib's default handler as before.
pub fn install_gdk_hook() {
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    if CHANNEL.get().is_none() {
        return;
    }
    INSTALLED.call_once(|| {
        glib::log_set_handler(
            Some("Gdk"),
            glib::LogLevels::LEVEL_MESSAGE,
            false,
            false,
            |domain, level, message| {
                if is_wayland_protocol_error(message) {
                    send(MSG_ERROR71);
                }
                glib::log_default_handler(domain, level, Some(message));
            },
        );
        // GTK 3 builds with G_LOG_USE_STRUCTURED, so its own g_message bypasses the handler above
        // and only reaches the writer. The writer can be set once per process, before any logging.
        glib::log_set_writer_func(|level, fields| {
            let field = |key: &str| {
                fields
                    .iter()
                    .find(|field| field.key() == key)
                    .and_then(|field| field.value_str())
            };
            if field("GLIB_DOMAIN") == Some("Gdk")
                && field("MESSAGE").is_some_and(is_wayland_protocol_error)
            {
                send(MSG_ERROR71);
            }
            glib::log_writer_default(level, fields)
        });
    });
}

/// Call when a `--hidden` launch decides to show its window after all (an in-app relaunch).
/// The window's first frame is where Error 71 strikes, so the CPU relaunch must already know
/// not to come back hidden; [`note_window_shown`] still arms the ready timer later.
pub fn note_window_will_show() {
    send(MSG_SHOWN);
}

/// Call whenever the main window is shown. The first call arms a one-shot timer; if the
/// process is still alive [`READY_GRACE`] later it tells the parent the GPU path works.
pub fn note_window_shown() {
    static ARMED: AtomicBool = AtomicBool::new(false);
    if CHANNEL.get().is_none() || ARMED.swap(true, Ordering::SeqCst) {
        return;
    }
    // A CPU relaunch after this must not come back hidden: the user has seen the window.
    send(MSG_SHOWN);
    let spawned = std::thread::Builder::new()
        .name("gpu-supervisor-ready".to_string())
        .spawn(|| {
            std::thread::sleep(READY_GRACE);
            send(MSG_READY);
        });
    if spawned.is_err() {
        // No timer thread: say ready now rather than risk a needless CPU fallback later.
        send(MSG_READY);
    }
}

/// Child only: calls `on_terminate` once on the parent's `T` or a closed/broken control stream.
/// A dead supervisor must not strand the app in its separate process group, even if the parent
/// died before this reader began. Unread window-status messages can turn close into ECONNRESET.
pub fn start_control_reader(on_terminate: impl Fn() + Send + 'static) {
    let Some(stream) = CHANNEL.get() else {
        return;
    };
    let Ok(mut reader) = stream.try_clone() else {
        return;
    };
    let _ = std::thread::Builder::new()
        .name("gpu-supervisor-control".to_string())
        .spawn(move || {
            let mut fired = false;
            let mut buf = [0u8; 16];
            loop {
                match reader.read(&mut buf) {
                    Ok(n) if n > 0 => {
                        if !fired && buf[..n].contains(&MSG_TERMINATE) {
                            fired = true;
                            on_terminate();
                        }
                    }
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                    _ => {
                        if !fired {
                            on_terminate();
                        }
                        return;
                    }
                }
            }
        });
}

/// Debug builds: whether [`TEST_ERROR71_ENV`] asks for the crash simulation, so the caller can
/// make supervision eligible on any host.
pub fn test_hook_requested() -> bool {
    #[cfg(debug_assertions)]
    {
        std::env::var_os(TEST_ERROR71_ENV).is_some_and(|value| value == "1")
    }
    #[cfg(not(debug_assertions))]
    {
        false
    }
}

/// Debug builds: in a supervised child with [`TEST_ERROR71_ENV`]=1, log GTK's exact Error 71
/// line through GLib (so the real [`install_gdk_hook`] path runs) and exit 1 like GTK does.
#[cfg(debug_assertions)]
pub fn maybe_simulate_error71() {
    if CHANNEL.get().is_none() || !test_hook_requested() {
        return;
    }
    // Use GTK's exact structured API with NUL-terminated fields. Older GLib default writers
    // do not reliably print the explicitly sized, non-NUL fields in log_structured!.
    // SAFETY: all pointers are static C strings, and the format has no variadic arguments.
    unsafe {
        glib::ffi::g_log_structured_standard(
            c"Gdk".as_ptr(),
            glib::ffi::G_LOG_LEVEL_MESSAGE,
            c"linux_gpu_supervisor.rs".as_ptr(),
            c"0".as_ptr(),
            c"maybe_simulate_error71".as_ptr(),
            c"Error 71 (Protocol error) dispatching to Wayland display.".as_ptr(),
        );
    }
    std::process::exit(1);
}

// ---------------------------------------------------------------------------------------------
// Parent side
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The GPU child died of a Wayland protocol error before it was ready: relaunch on CPU.
    FallbackToCpu,
    /// Anything else: pass the child's status on unchanged.
    Propagate,
}

pub fn outcome(status: &ExitStatus, saw_ready: bool, saw_error71: bool) -> Outcome {
    if status.code() == Some(1) && saw_error71 && !saw_ready {
        Outcome::FallbackToCpu
    } else {
        Outcome::Propagate
    }
}

/// The child currently running, for the signal watcher.
#[derive(Default)]
struct Current {
    pid: Option<libc::pid_t>,
    control: Option<UnixStream>,
}

struct Shared {
    current: Mutex<Current>,
    terminations: AtomicUsize,
    /// Any termination signal: start no further child.
    /// Read under `current`'s lock by [`run_attempt`], so a request cannot slip between the
    /// check and the spawn.
    stopping: AtomicBool,
}

impl Shared {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            current: Mutex::new(Current::default()),
            terminations: AtomicUsize::new(0),
            stopping: AtomicBool::new(false),
        })
    }

    /// The child has its own process group, so both PID-directed and group-directed signals
    /// reach it exactly once through this watcher.
    fn on_interrupt(&self, signal: libc::c_int) {
        // A following SIGTERM is a repeat even when the first request was INT/HUP/QUIT.
        self.terminations.fetch_add(1, Ordering::SeqCst);
        let current = self.current.lock().unwrap_or_else(|e| e.into_inner());
        self.stopping.store(true, Ordering::SeqCst);
        if let Some(pid) = current.pid {
            // SAFETY: our own child, still registered under the same lock as spawn/reap.
            unsafe {
                libc::kill(pid, signal);
            }
        }
    }

    fn on_sigterm(&self) {
        let count = self.terminations.fetch_add(1, Ordering::SeqCst) + 1;
        let current = self.current.lock().unwrap_or_else(|e| e.into_inner());
        self.stopping.store(true, Ordering::SeqCst);
        let Some(pid) = current.pid else {
            return;
        };
        let asked = count == 1
            && current
                .control
                .as_ref()
                .is_some_and(|mut control| control.write_all(&[MSG_TERMINATE]).is_ok());
        if !asked {
            // SAFETY: plain kill(2) on our own, not yet reaped, child.
            unsafe {
                libc::kill(pid, libc::SIGTERM);
            }
        }
    }
}

/// The running executable by its real path, so the child keeps the app's process name (`comm`
/// comes from the exec'd path; `/proc/self/exe` would name it "exe"). Falls back to
/// `/proc/self/exe` once the file is gone, e.g. a dpkg upgrade replaced it under us.
fn app_program(resolved: std::io::Result<std::path::PathBuf>) -> std::path::PathBuf {
    match resolved {
        Ok(path) if path.is_file() => path,
        _ => std::path::PathBuf::from("/proc/self/exe"),
    }
}

/// The real app, re-executed with the same argv[0] and arguments, so single-instance and
/// deep-link hand-off see identical argv.
fn app_command(fallback: bool, shown: bool) -> Command {
    let mut args = std::env::args_os();
    let arg0 = args
        .next()
        .unwrap_or_else(|| OsString::from("unsloth-studio"));
    let mut command = Command::new(app_program(std::env::current_exe()));
    command.arg0(arg0).args(relaunch_args(args, shown));
    if fallback {
        command.env(FALLBACK_ENV, FALLBACK_REASON);
        #[cfg(debug_assertions)]
        command.env_remove(TEST_ERROR71_ENV);
    }
    command
}

/// A login start's `--hidden` is dropped once its window was shown, so the CPU relaunch shows
/// the window the user had just opened instead of going back to the tray.
fn relaunch_args(args: impl Iterator<Item = OsString>, shown: bool) -> Vec<OsString> {
    args.filter(|arg| !(shown && arg == "--hidden")).collect()
}

/// Run the app as a supervised child and never return on success: the process exits with the
/// child's status (or re-raises its signal).
///
/// Returns `Err` only when the first child cannot be spawned. The caller should then log it and
/// carry on running the app in-process, unsupervised; the parent's signal watcher is closed.
/// `on_fallback` runs once, before the CPU relaunch (persist the "GPU unsafe" flag there).
pub fn run_parent(on_fallback: impl FnOnce(), log: impl Fn(&str)) -> Result<Infallible, io::Error> {
    let shared = Shared::new();
    // Watch signals before the first spawn, so a request in that window is not lost.
    let watcher = install_parent_signals(&shared);
    match supervise(app_command, &shared, on_fallback, &log) {
        Ok(status) => propagate(status),
        Err(error) => {
            if let Some(handle) = watcher {
                handle.close();
            }
            Err(error)
        }
    }
}

/// Forward interrupts to the child's separate process group via its PID. It does not receive
/// the original terminal/session group delivery, which would otherwise look like a repeat and
/// skip backend cleanup. SIGTERM retains the graceful shutdown control message.
fn install_parent_signals(shared: &Arc<Shared>) -> Option<signal_hook::iterator::Handle> {
    match signal_hook::iterator::Signals::new([
        libc::SIGTERM,
        libc::SIGINT,
        libc::SIGQUIT,
        libc::SIGHUP,
    ]) {
        Ok(mut signals) => {
            let handle = signals.handle();
            let shared = Arc::clone(shared);
            let _ = std::thread::Builder::new()
                .name("gpu-supervisor-signals".to_string())
                .spawn(move || {
                    for signal in signals.forever() {
                        if signal == libc::SIGTERM {
                            shared.on_sigterm();
                        } else {
                            shared.on_interrupt(signal);
                        }
                    }
                });
            Some(handle)
        }
        Err(error) => {
            eprintln!("GPU supervisor: could not watch termination signals: {error}");
            None
        }
    }
}

struct Attempt {
    status: ExitStatus,
    saw_shown: bool,
    saw_ready: bool,
    saw_error71: bool,
}

/// What a supervised process exits with when it was asked to stop before a child ran.
fn terminated_status() -> ExitStatus {
    ExitStatus::from_raw((128 + libc::SIGTERM) << 8)
}

/// Spawns `make_command(false, false)`, and on [`Outcome::FallbackToCpu`] exactly one
/// `make_command(true, shown)`. Returns the status to propagate. A termination request before
/// a spawn means that child is never started.
fn supervise(
    mut make_command: impl FnMut(bool, bool) -> Command,
    shared: &Shared,
    on_fallback: impl FnOnce(),
    log: &dyn Fn(&str),
) -> io::Result<ExitStatus> {
    let Some(first) = run_attempt(make_command(false, false), shared)? else {
        log("GPU supervisor: asked to stop before the app started");
        return Ok(terminated_status());
    };
    let decision = outcome(&first.status, first.saw_ready, first.saw_error71);
    if decision == Outcome::Propagate {
        return Ok(first.status);
    }
    if shared.stopping.load(Ordering::SeqCst) {
        log("GPU supervisor: child hit Wayland Error 71 while terminating; not relaunching");
        return Ok(first.status);
    }

    on_fallback();
    log("GPU supervisor: the app exited with Wayland Error 71 before its window was ready");
    log("GPU supervisor: GPU rendering marked unsafe on this host; relaunching with CPU rendering");
    match run_attempt(make_command(true, first.saw_shown), shared) {
        Ok(None) => {
            log("GPU supervisor: asked to stop before the CPU relaunch; not relaunching");
            Ok(first.status)
        }
        Ok(Some(second)) => {
            log(&format!(
                "GPU supervisor: CPU relaunch exited with {}",
                describe(&second.status)
            ));
            Ok(second.status)
        }
        Err(error) => {
            log(&format!(
                "GPU supervisor: CPU relaunch failed to start: {error}"
            ));
            Ok(first.status)
        }
    }
}

/// `Ok(None)` when a termination request arrived before the spawn.
fn run_attempt(mut command: Command, shared: &Shared) -> io::Result<Option<Attempt>> {
    // std creates both ends close-on-exec; only the child's copy is made inheritable.
    let (mut parent_end, child_end) = UnixStream::pair()?;
    let child_fd = child_end.as_raw_fd();
    command.env(SUPERVISOR_FD_ENV, child_fd.to_string());
    command.process_group(0);
    // SAFETY: the closure only calls async-signal-safe fcntl(2) and signal(2).
    unsafe {
        command.pre_exec(move || {
            if !set_cloexec(child_fd, false) {
                return Err(io::Error::last_os_error());
            }
            for signal in [libc::SIGINT, libc::SIGQUIT, libc::SIGHUP, libc::SIGTERM] {
                libc::signal(signal, libc::SIG_DFL);
            }
            Ok(())
        });
    }

    let mut child = {
        // Hold the lock across spawn so SIGTERM cannot miss a child that is just starting.
        let mut current = shared.current.lock().unwrap_or_else(|e| e.into_inner());
        if shared.stopping.load(Ordering::SeqCst) {
            return Ok(None);
        }
        let child = command.spawn()?;
        current.pid = Some(child.id() as libc::pid_t);
        current.control = parent_end.try_clone().ok();
        child
    };
    drop(child_end);

    let status = loop {
        match child.wait() {
            Ok(status) => break status,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => {
                *shared.current.lock().unwrap_or_else(|e| e.into_inner()) = Current::default();
                return Err(error);
            }
        }
    };
    *shared.current.lock().unwrap_or_else(|e| e.into_inner()) = Current::default();

    // The child writes before it exits, so everything it sent is already queued. Draining
    // after the wait cannot race the exit, and cannot block on a descendant holding the fd.
    let mut saw_shown = false;
    let mut saw_ready = false;
    let mut saw_error71 = false;
    parent_end.set_nonblocking(true)?;
    let mut buf = [0u8; 64];
    loop {
        match parent_end.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                saw_shown |= buf[..n].contains(&MSG_SHOWN);
                saw_ready |= buf[..n].contains(&MSG_READY);
                saw_error71 |= buf[..n].contains(&MSG_ERROR71);
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
    Ok(Some(Attempt {
        status,
        saw_shown,
        saw_ready,
        saw_error71,
    }))
}

fn describe(status: &ExitStatus) -> String {
    match (status.code(), status.signal()) {
        (Some(code), _) => format!("code {code}"),
        (None, Some(signal)) => format!("signal {signal}"),
        _ => "an unknown status".to_string(),
    }
}

/// Exit like the child did: same code, or die of the same signal.
fn propagate(status: ExitStatus) -> ! {
    if let Some(code) = status.code() {
        std::process::exit(code);
    }
    if let Some(signal) = status.signal() {
        // SAFETY: restore the default action and re-raise so our own parent sees the signal.
        unsafe {
            libc::signal(signal, libc::SIG_DFL);
            let mut set: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&mut set);
            libc::sigaddset(&mut set, signal);
            libc::pthread_sigmask(libc::SIG_UNBLOCK, &set, std::ptr::null_mut());
            libc::raise(signal);
        }
        std::process::exit(128 + signal);
    }
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::os::fd::IntoRawFd;

    /// Tests that read or write process env hold this.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn exited(code: i32) -> ExitStatus {
        ExitStatus::from_raw(code << 8)
    }

    fn signaled(signal: i32) -> ExitStatus {
        ExitStatus::from_raw(signal)
    }

    #[test]
    fn outcome_truth_table() {
        // E then exit 1 before R.
        assert_eq!(outcome(&exited(1), false, true), Outcome::FallbackToCpu);
        // R then E + exit 1: the GPU path worked, so the crash is just propagated.
        assert_eq!(outcome(&exited(1), true, true), Outcome::Propagate);
        // Exit 1 without E.
        assert_eq!(outcome(&exited(1), false, false), Outcome::Propagate);
        // Other codes, even with E.
        assert_eq!(outcome(&exited(0), false, true), Outcome::Propagate);
        assert_eq!(outcome(&exited(0), false, false), Outcome::Propagate);
        assert_eq!(outcome(&exited(2), false, true), Outcome::Propagate);
        // Killed by a signal.
        assert_eq!(
            outcome(&signaled(libc::SIGKILL), false, true),
            Outcome::Propagate
        );
        assert_eq!(signaled(libc::SIGKILL).signal(), Some(libc::SIGKILL));
    }

    #[test]
    fn classifier_matches_gtk_protocol_errors_only() {
        assert!(is_wayland_protocol_error(
            "Error 71 (Protocol error) dispatching to Wayland display."
        ));
        // Translated strerror, same errno.
        assert!(is_wayland_protocol_error(
            "Error 71 (Protokollfehler) dispatching to Wayland display."
        ));
        assert!(is_wayland_protocol_error(
            "Error flushing display: Protocol error"
        ));
        assert!(is_wayland_protocol_error(
            "Error reading events from display: Protocol error"
        ));
        if let Some(eproto) = localized_eproto() {
            assert!(is_wayland_protocol_error(&format!(
                "Error flushing display: {eproto}"
            )));
        }

        assert!(!is_wayland_protocol_error(
            "Error 32 (Broken pipe) dispatching to Wayland display."
        ));
        assert!(!is_wayland_protocol_error(
            "Error flushing display: Broken pipe"
        ));
        assert!(!is_wayland_protocol_error(
            "Error reading events from display: Connection reset by peer"
        ));
        assert!(!is_wayland_protocol_error(
            "Lost connection to Wayland compositor."
        ));
        assert!(!is_wayland_protocol_error("Error 71 (Protocol error)"));
        assert!(!is_wayland_protocol_error(""));
    }

    #[test]
    fn channel_from_env_accepts_sockets_and_clears_env() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (ours, theirs) = UnixStream::pair().unwrap();
        let fd = theirs.into_raw_fd();
        set_cloexec(fd, false);
        std::env::set_var(SUPERVISOR_FD_ENV, fd.to_string());
        std::env::set_var(FALLBACK_ENV, "error71");

        let (stream, fallback) = channel_from_env();
        assert_eq!(fallback.as_deref(), Some("error71"));
        assert!(std::env::var_os(SUPERVISOR_FD_ENV).is_none());
        assert!(std::env::var_os(FALLBACK_ENV).is_none());
        let mut stream = stream.expect("socket fd accepted");
        assert_eq!(stream.as_raw_fd(), fd);
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
        assert_ne!(flags & libc::FD_CLOEXEC, 0, "fd marked close-on-exec");
        stream.write_all(b"R").unwrap();
        let mut byte = [0u8; 1];
        (&ours).read_exact(&mut byte).unwrap();
        assert_eq!(&byte, b"R");
    }

    #[test]
    fn channel_from_env_ignores_non_sockets_and_garbage() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let file = std::fs::File::open("/proc/self/exe").unwrap();
        std::env::set_var(SUPERVISOR_FD_ENV, file.as_raw_fd().to_string());
        let (stream, fallback) = channel_from_env();
        assert!(stream.is_none());
        assert!(fallback.is_none());
        assert!(std::env::var_os(SUPERVISOR_FD_ENV).is_none());
        // The fd we were wrongly pointed at is left open.
        assert!(unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFD) } >= 0);

        for value in ["", "abc", "-1", "1", "999999"] {
            std::env::set_var(SUPERVISOR_FD_ENV, value);
            assert!(channel_from_env().0.is_none(), "{value:?}");
        }
    }

    /// Runs `supervise` with `bash -c <script>` children; returns (status, spawn flags, fallback
    /// calls, log lines). Not sh: dash rejects `>&N` once N has two digits, which a parallel test
    /// run reaches.
    fn fake_supervise(scripts: &[&str]) -> (ExitStatus, Vec<bool>, usize, Vec<String>) {
        let spawns = RefCell::new(Vec::new());
        let fallbacks = Cell::new(0);
        let logs = RefCell::new(Vec::new());
        let shared = Shared::new();
        let status = supervise(
            |fallback, _shown| {
                let index = spawns.borrow().len().min(scripts.len() - 1);
                spawns.borrow_mut().push(fallback);
                let mut command = Command::new("bash");
                command.arg("-c").arg(scripts[index]);
                if fallback {
                    command.env(FALLBACK_ENV, FALLBACK_REASON);
                }
                command
            },
            &shared,
            || fallbacks.set(fallbacks.get() + 1),
            &|line| logs.borrow_mut().push(line.to_string()),
        )
        .expect("bash spawns");
        (
            status,
            spawns.into_inner(),
            fallbacks.get(),
            logs.into_inner(),
        )
    }

    #[test]
    fn fake_child_error71_falls_back_once() {
        let (status, spawns, fallbacks, logs) = fake_supervise(&[
            "printf E >&$UNSLOTH_GPU_SUPERVISOR_FD; exit 1",
            // The CPU child sees the fallback env and a fresh socket.
            "[ \"$UNSLOTH_GPU_FALLBACK\" = error71 ] && printf x >&$UNSLOTH_GPU_SUPERVISOR_FD && exit 7; exit 9",
        ]);
        assert_eq!(spawns, vec![false, true]);
        assert_eq!(fallbacks, 1);
        assert_eq!(status.code(), Some(7));
        assert!(logs.iter().any(|line| line.contains("CPU rendering")));
    }

    #[test]
    fn fake_child_error71_twice_still_spawns_only_twice() {
        let (status, spawns, fallbacks, _) =
            fake_supervise(&["printf E >&$UNSLOTH_GPU_SUPERVISOR_FD; exit 1"]);
        assert_eq!(spawns.len(), 2);
        assert_eq!(fallbacks, 1);
        assert_eq!(status.code(), Some(1));
    }

    #[test]
    fn fake_child_ready_then_error71_propagates() {
        let (status, spawns, fallbacks, _) =
            fake_supervise(&["printf RE >&$UNSLOTH_GPU_SUPERVISOR_FD; exit 1"]);
        assert_eq!(spawns, vec![false]);
        assert_eq!(fallbacks, 0);
        assert_eq!(status.code(), Some(1));
    }

    #[test]
    fn fake_child_plain_exits_propagate() {
        for (script, code) in [("exit 0", 0), ("exit 1", 1), ("exit 3", 3)] {
            let (status, spawns, fallbacks, _) = fake_supervise(&[script]);
            assert_eq!(spawns.len(), 1, "{script}");
            assert_eq!(fallbacks, 0, "{script}");
            assert_eq!(status.code(), Some(code), "{script}");
        }
    }

    #[test]
    fn fake_child_signal_propagates() {
        let (status, spawns, fallbacks, _) =
            fake_supervise(&["printf E >&$UNSLOTH_GPU_SUPERVISOR_FD; kill -KILL $$"]);
        assert_eq!(spawns.len(), 1);
        assert_eq!(fallbacks, 0);
        assert_eq!(status.signal(), Some(libc::SIGKILL));
    }

    #[test]
    fn fake_child_gets_inheritable_socket_only() {
        // The fd is a socket in the child, and the parent's own end is not inherited.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (status, _, _, _) = fake_supervise(&[
            "fd=$UNSLOTH_GPU_SUPERVISOR_FD; [ -S /proc/$$/fd/$fd ] || exit 4; \
             for f in /proc/$$/fd/*; do n=${f##*/}; \
             case $n in 0|1|2|$fd) ;; *) [ -S $f ] && exit 5;; esac; done; exit 0",
        ]);
        assert_eq!(status.code(), Some(0));
    }

    /// Re-execs this test binary as the supervised child (the windows_job.rs pattern), so the
    /// real take_child_channel -> install_gdk_hook -> g_log_structured_standard -> exit(1) runs.
    #[cfg(debug_assertions)]
    #[test]
    fn real_gdk_hook_drives_fallback_end_to_end() {
        const CHILD_MARKER: &str = "UNSLOTH_GPU_SUPERVISOR_TEST_CHILD";
        const NAME: &str = "linux_gpu_supervisor::tests::real_gdk_hook_drives_fallback_end_to_end";
        if std::env::var_os(CHILD_MARKER).is_some() {
            let channel = take_child_channel();
            assert!(channel.supervised);
            assert!(std::env::var_os(SUPERVISOR_FD_ENV).is_none());
            install_gdk_hook();
            maybe_simulate_error71(); // exits 1 when the hook env is set
            let code = if channel.fallback.as_deref() == Some(FALLBACK_REASON) {
                0
            } else {
                3
            };
            std::process::exit(code);
        }

        let exe = std::env::current_exe().unwrap();
        let spawns = RefCell::new(Vec::new());
        let fallbacks = Cell::new(0);
        let status = supervise(
            |fallback, _shown| {
                spawns.borrow_mut().push(fallback);
                let mut command = Command::new(&exe);
                command
                    .args([NAME, "--exact", "--nocapture", "--test-threads=1"])
                    .env(CHILD_MARKER, "1");
                if fallback {
                    command
                        .env(FALLBACK_ENV, FALLBACK_REASON)
                        .env_remove(TEST_ERROR71_ENV);
                } else {
                    command.env(TEST_ERROR71_ENV, "1");
                }
                command
            },
            &Shared::new(),
            || fallbacks.set(fallbacks.get() + 1),
            &|_| {},
        )
        .unwrap();
        assert_eq!(spawns.into_inner(), vec![false, true]);
        assert_eq!(fallbacks.get(), 1);
        assert_eq!(status.code(), Some(0));
    }

    #[test]
    fn sigterm_asks_child_then_kills() {
        let shared = Shared::new();
        let logs = RefCell::new(Vec::new());
        let shared_for_thread = Arc::clone(&shared);
        let killer = std::thread::spawn(move || {
            // Wait for the child to start, then deliver two "SIGTERM"s to the parent logic.
            loop {
                if shared_for_thread.current.lock().unwrap().pid.is_some() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            std::thread::sleep(Duration::from_millis(200));
            shared_for_thread.on_sigterm();
        });
        // The child exits 143 once it reads T, like the real app.exit(143).
        let status = supervise(
            |_, _| {
                let mut command = Command::new("bash");
                command.arg("-c").arg(
                    "c=$(dd bs=1 count=1 <&$UNSLOTH_GPU_SUPERVISOR_FD 2>/dev/null); \
                     [ \"$c\" = T ] && exit 143; exit 2",
                );
                command
            },
            &shared,
            || panic!("no fallback"),
            &|line| logs.borrow_mut().push(line.to_string()),
        )
        .unwrap();
        killer.join().unwrap();
        assert_eq!(status.code(), Some(143));

        // Second SIGTERM: a real signal to the child.
        let shared = Shared::new();
        shared.terminations.store(1, Ordering::SeqCst);
        let shared_for_thread = Arc::clone(&shared);
        let killer = std::thread::spawn(move || {
            loop {
                if shared_for_thread.current.lock().unwrap().pid.is_some() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            std::thread::sleep(Duration::from_millis(200));
            shared_for_thread.on_sigterm();
        });
        let status = supervise(
            |_, _| {
                let mut command = Command::new("bash");
                command.arg("-c").arg("exec sleep 30");
                command
            },
            &shared,
            || panic!("no fallback"),
            &|_| {},
        )
        .unwrap();
        killer.join().unwrap();
        assert_eq!(status.signal(), Some(libc::SIGTERM));
    }

    #[test]
    fn child_helpers_are_noops_without_channel() {
        // Tests never set CHANNEL (channel_from_env does not touch the static).
        assert!(CHANNEL.get().is_none());
        note_window_shown();
        note_window_will_show();
        start_control_reader(|| panic!("no channel"));
        install_gdk_hook();
        #[cfg(debug_assertions)]
        maybe_simulate_error71();
    }

    #[test]
    fn app_command_strips_test_hook_on_fallback() {
        let command = app_command(true, false);
        assert_eq!(
            std::path::Path::new(command.get_program()),
            std::env::current_exe().unwrap()
        );
        let envs: Vec<_> = command.get_envs().collect();
        assert!(envs.contains(&(
            std::ffi::OsStr::new(FALLBACK_ENV),
            Some(std::ffi::OsStr::new("error71"))
        )));
        #[cfg(debug_assertions)]
        assert!(envs.contains(&(std::ffi::OsStr::new(TEST_ERROR71_ENV), None)));
        assert!(app_command(false, false).get_envs().next().is_none());
        assert_eq!(
            app_command(false, false).get_args().count(),
            std::env::args_os().count() - 1
        );
    }

    #[test]
    fn the_child_keeps_the_app_name_and_falls_back_once_the_file_is_gone() {
        let exe = std::env::current_exe().unwrap();
        assert_eq!(app_program(Ok(exe.clone())), exe);
        let gone = std::path::PathBuf::from("/nonexistent/unsloth-studio (deleted)");
        assert_eq!(
            app_program(Ok(gone)),
            std::path::PathBuf::from("/proc/self/exe")
        );
        assert_eq!(
            app_program(Err(std::io::Error::other("no exe"))),
            std::path::PathBuf::from("/proc/self/exe")
        );
    }

    #[test]
    fn a_shown_window_drops_hidden_from_the_cpu_relaunch() {
        let args = || {
            ["--hidden", "unsloth://open"]
                .map(OsString::from)
                .into_iter()
        };
        assert_eq!(relaunch_args(args(), false), args().collect::<Vec<_>>());
        assert_eq!(
            relaunch_args(args(), true),
            vec![OsString::from("unsloth://open")]
        );

        // The parent learns about the shown window from the crashed child's S.
        let shown = RefCell::new(Vec::new());
        let status = supervise(
            |fallback, was_shown| {
                shown.borrow_mut().push(was_shown);
                let mut command = Command::new("bash");
                let script = if fallback {
                    "exit 0"
                } else {
                    "printf SE >&$UNSLOTH_GPU_SUPERVISOR_FD; exit 1"
                };
                command.arg("-c").arg(script);
                command
            },
            &Shared::new(),
            || {},
            &|_| {},
        )
        .unwrap();
        assert_eq!(status.code(), Some(0));
        assert_eq!(shown.into_inner(), vec![false, true]);
    }

    #[test]
    fn a_termination_before_the_first_spawn_starts_nothing() {
        let shared = Shared::new();
        shared.on_sigterm();
        let spawns = Cell::new(0);
        let status = supervise(
            |_, _| {
                spawns.set(spawns.get() + 1);
                Command::new("true")
            },
            &shared,
            || panic!("no fallback"),
            &|_| {},
        )
        .unwrap();
        assert_eq!(spawns.get(), 1, "the command is built, never spawned");
        assert_eq!(status.code(), Some(128 + libc::SIGTERM));
    }

    #[test]
    fn a_termination_between_the_crash_and_the_cpu_relaunch_stops_the_relaunch() {
        // SIGTERM lands while on_fallback runs: no child is registered at that moment.
        let shared = Shared::new();
        let spawned = RefCell::new(Vec::new());
        let logs = RefCell::new(Vec::new());
        let status = supervise(
            |fallback, _| {
                let mut command = Command::new("bash");
                // A CPU child that did start would exit 7.
                let script = if fallback {
                    "exit 7"
                } else {
                    "printf E >&$UNSLOTH_GPU_SUPERVISOR_FD; exit 1"
                };
                command.arg("-c").arg(script);
                spawned.borrow_mut().push(fallback);
                command
            },
            &shared,
            || shared.on_sigterm(),
            &|line| logs.borrow_mut().push(line.to_string()),
        )
        .unwrap();
        assert_eq!(status.code(), Some(1));
        assert!(logs
            .borrow()
            .iter()
            .any(|line| line.contains("not relaunching")));
        assert_eq!(shared.current.lock().unwrap().pid, None);
    }

    #[test]
    fn interrupts_before_a_spawn_stop_the_supervisor() {
        for signal in [libc::SIGINT, libc::SIGHUP, libc::SIGQUIT] {
            let shared = Shared::new();
            shared.on_interrupt(signal);
            assert!(shared.stopping.load(Ordering::SeqCst));
            assert!(run_attempt(Command::new("false"), &shared)
                .unwrap()
                .is_none());
        }
    }

    /// Separate processes exercise real signal delivery without installing handlers in the
    /// parallel test runner. A duplicate signal would interrupt the child's simulated cleanup.
    #[test]
    fn pid_and_group_signals_reach_the_child_once() {
        const ROLE: &str = "UNSLOTH_SUPERVISOR_SIGNAL_TEST";
        const STATE: &str = "UNSLOTH_SUPERVISOR_SIGNAL_TEST_STATE";
        const NAME: &str =
            "linux_gpu_supervisor::tests::pid_and_group_signals_reach_the_child_once";

        if std::env::var_os(ROLE).is_some() {
            let channel = take_child_channel();
            if !channel.supervised {
                run_parent(|| panic!("no fallback"), |_| {}).unwrap();
            }
            let state = std::path::PathBuf::from(std::env::var_os(STATE).unwrap());
            let received = Arc::new(AtomicUsize::new(0));
            let on_signal = {
                let received = Arc::clone(&received);
                let state = state.clone();
                move |signal: libc::c_int| {
                    if received.fetch_add(1, Ordering::SeqCst) != 0 {
                        std::process::exit(90);
                    }
                    let state = state.clone();
                    std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_millis(200));
                        std::fs::write(state.join("cleaned"), signal.to_string()).unwrap();
                        std::process::exit(128 + signal);
                    });
                }
            };
            let mut signals = signal_hook::iterator::Signals::new([
                libc::SIGINT,
                libc::SIGHUP,
                libc::SIGQUIT,
                libc::SIGTERM,
            ])
            .unwrap();
            let on_control = on_signal.clone();
            start_control_reader(move || on_control(libc::SIGTERM));
            // A real GUI reports its window before the parent drains the channel. If the
            // supervisor is killed with this unread data, Linux resets the peer, not just EOF.
            note_window_will_show();
            // SAFETY: getpgrp only reads the calling process's group ID.
            let pgid = unsafe { libc::getpgrp() };
            std::fs::write(
                state.join("ready"),
                format!("{} {pgid}", std::process::id()),
            )
            .unwrap();
            for signal in signals.forever() {
                on_signal(signal);
            }
            unreachable!();
        }

        for signal in [libc::SIGINT, libc::SIGHUP, libc::SIGQUIT, libc::SIGTERM] {
            for group in [false, true] {
                check_signal(signal, group, false);
            }
        }
        // An uncatchable group signal must not strand the app in its separate group.
        check_signal(libc::SIGKILL, true, true);

        fn check_signal(signal: libc::c_int, group: bool, parent_death: bool) {
            let state = tempfile::tempdir().unwrap();
            let mut supervisor = Command::new(std::env::current_exe().unwrap())
                .args([NAME, "--exact", "--nocapture"])
                .env(ROLE, "1")
                .env(STATE, state.path())
                .process_group(0)
                .spawn()
                .unwrap();
            let supervisor_pid = supervisor.id() as libc::pid_t;
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            let mut child_pid = None;
            let checked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let ready = loop {
                    if let Ok(ready) = std::fs::read_to_string(state.path().join("ready")) {
                        break ready;
                    }
                    assert!(
                        supervisor.try_wait().unwrap().is_none(),
                        "supervisor exited early"
                    );
                    assert!(
                        std::time::Instant::now() < deadline,
                        "child did not become ready"
                    );
                    std::thread::sleep(Duration::from_millis(10));
                };
                let ids: Vec<libc::pid_t> = ready
                    .split_whitespace()
                    .map(|id| id.parse().unwrap())
                    .collect();
                child_pid = Some(ids[0]);
                assert_eq!(ids[0], ids[1], "child leads its own process group");
                assert_ne!(
                    ids[1], supervisor_pid,
                    "child cannot receive the original group signal"
                );
                let target = if group {
                    -supervisor_pid
                } else {
                    supervisor_pid
                };
                // SAFETY: the test created this isolated process group and waited for its child.
                assert_eq!(unsafe { libc::kill(target, signal) }, 0);
                let status = loop {
                    if let Some(status) = supervisor.try_wait().unwrap() {
                        break status;
                    }
                    assert!(
                        std::time::Instant::now() < deadline,
                        "supervisor ignored signal {signal}"
                    );
                    std::thread::sleep(Duration::from_millis(10));
                };
                if parent_death {
                    assert_eq!(status.signal(), Some(libc::SIGKILL));
                } else {
                    assert_eq!(
                        status.code(),
                        Some(128 + signal),
                        "signal {signal}, group={group}"
                    );
                }
                let cleaned = loop {
                    if let Ok(cleaned) = std::fs::read_to_string(state.path().join("cleaned")) {
                        break cleaned;
                    }
                    assert!(
                        std::time::Instant::now() < deadline,
                        "child cleanup did not finish"
                    );
                    std::thread::sleep(Duration::from_millis(10));
                };
                let expected = if parent_death { libc::SIGTERM } else { signal };
                assert_eq!(cleaned, expected.to_string());
            }));
            if checked.is_err() {
                let _ = supervisor.kill();
                if let Some(pid) = child_pid {
                    // SAFETY: only the child PID recorded for this test can be killed.
                    unsafe { libc::kill(pid, libc::SIGKILL) };
                }
            }
            let _ = supervisor.wait();
            if let Err(error) = checked {
                std::panic::resume_unwind(error);
            }
        }
    }
}
