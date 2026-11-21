//! PATH from the login shell for a GUI launch. Adapted from tauri-apps/fix-path-env-rs
//! (MIT OR Apache-2.0) `fix()`, except unsloth#12678: a `-c` shell never loads $HISTFILE
//! but saves history on exit, so the probe disables that save and `exec`s past exit hooks.

const DELIMITER: &str = "_SHELL_ENV_DELIMITER_";

/// Keep the old command; an exception list so a renamed bash/zsh still gets the fix.
const NON_POSIX_SHELLS: &[&str] = &[
    "fish", "nu", "nushell", "csh", "tcsh", "xonsh", "elvish", "pwsh", "ion", "murex",
];

#[cfg_attr(windows, allow(dead_code))]
fn env_command() -> String {
    format!("echo -n \"{DELIMITER}\"; env; echo -n \"{DELIMITER}\"; exit")
}

#[cfg_attr(windows, allow(dead_code))]
fn probe_command(shell: &str) -> String {
    let name = std::path::Path::new(shell)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if NON_POSIX_SHELLS.contains(&name) {
        return env_command();
    }
    // RCS covers zsh's readonly HISTFILE; subshell first: dash exits on a readonly unset.
    format!(
        "if [ -n \"${{ZSH_VERSION-}}\" ]; then eval 'unsetopt RCS' 2>/dev/null || :; fi; \
         (unset HISTFILE) 2>/dev/null && unset HISTFILE; \
         exec /bin/sh -c 'printf \"%s\" \"{DELIMITER}\"; env; printf \"%s\" \"{DELIMITER}\"'"
    )
}

#[cfg(test)]
fn path_from_output(stdout: &[u8]) -> Option<String> {
    env_block_path(stdout)?
}

/// `None` when there is no delimited block (the shell did not run the command),
/// `Some(None)` when it ran but printed no PATH.
#[cfg_attr(windows, allow(dead_code))]
fn env_block_path(stdout: &[u8]) -> Option<Option<String>> {
    let stdout = String::from_utf8_lossy(stdout);
    let env = stdout.split(DELIMITER).nth(1)?;
    let env = strip_ansi_escapes::strip(env);
    Some(
        String::from_utf8_lossy(&env)
            .split('\n')
            .filter_map(|line| line.strip_prefix("PATH=").map(str::to_owned))
            .last(),
    )
}

#[cfg(not(windows))]
const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

#[cfg(not(windows))]
const DRAIN_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

#[cfg(not(windows))]
const DRAIN_MAX_BYTES: usize = 1 << 20;

#[cfg(not(windows))]
#[derive(Debug, PartialEq)]
struct TimedOut;

/// `Command::output()`, but a job left holding stdout cannot delay launch; killed at `timeout`.
#[cfg(not(windows))]
fn run_login_shell(
    shell: &str,
    command: &str,
    timeout: std::time::Duration,
) -> Result<Option<Vec<u8>>, TimedOut> {
    use std::io::Read;
    use std::process::Stdio;
    use std::sync::mpsc::{self, RecvTimeoutError};
    use std::time::{Duration, Instant};

    let mut cmd = std::process::Command::new(shell);
    cmd.arg("-ilc")
        .arg(command)
        // Oh My Zsh's auto-update prompt can block the shell forever.
        .env("DISABLE_AUTO_UPDATE", "true")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    if let Some(home) = dirs::home_dir() {
        cmd.current_dir(home);
    }
    let Ok(mut child) = cmd.spawn() else {
        return Ok(None);
    };
    let Some(mut stdout) = child.stdout.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return Ok(None);
    };
    // Bounded so a nonstop writer blocks; the dropped receiver later closes the pipe.
    let (sender, chunks) = mpsc::sync_channel::<Vec<u8>>(64);
    // Builder, not thread::spawn: a refused thread must not panic launch.
    let reader = std::thread::Builder::new().spawn(move || {
        let mut buffer = [0u8; 8192];
        loop {
            match stdout.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => {
                    if sender.send(buffer[..read].to_vec()).is_err() {
                        break;
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(_) => break,
            }
        }
    });
    if reader.is_err() {
        let _ = child.kill();
        let _ = child.wait();
        return Ok(None);
    }

    let deadline = Instant::now() + timeout;
    let mut out = Vec::new();
    let mut status = None;
    let mut drain_until = None;
    let mut eof = false;
    loop {
        if status.is_none() {
            status = child.try_wait().ok().flatten();
            if status.is_some() {
                drain_until = Some(Instant::now() + DRAIN_GRACE);
            }
        }
        if status.is_some() && (eof || env_block_complete(&out)) {
            break;
        }
        let now = Instant::now();
        if drain_until.is_some_and(|until| now >= until) {
            break;
        }
        if status.is_none() && now >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(TimedOut);
        }
        if eof {
            std::thread::sleep(Duration::from_millis(1));
            continue;
        }
        match chunks.recv_timeout(Duration::from_millis(20)) {
            Ok(chunk) => out.extend_from_slice(&chunk),
            Err(RecvTimeoutError::Disconnected) => eof = true,
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
    let mut drained = 0;
    while drained < DRAIN_MAX_BYTES {
        let Ok(chunk) = chunks.try_recv() else { break };
        drained += chunk.len();
        out.extend_from_slice(&chunk);
    }
    let success = status.is_some_and(|status| status.success());
    // A cut-off PATH is not a result.
    if success && !eof && !env_block_complete(&out) {
        return Err(TimedOut);
    }
    Ok(success.then_some(out))
}

#[cfg(not(windows))]
fn env_block_complete(stdout: &[u8]) -> bool {
    let delimiter = DELIMITER.as_bytes();
    stdout
        .windows(delimiter.len())
        .filter(|window| *window == delimiter)
        .nth(1)
        .is_some()
}

/// `Ok(None)` when the shell rejected `command`; `Ok(Some(None))` when it ran without PATH.
#[cfg(not(windows))]
fn read_login_shell_path(
    shell: &str,
    command: &str,
    timeout: std::time::Duration,
) -> Result<Option<Option<String>>, TimedOut> {
    Ok(run_login_shell(shell, command, timeout)?.and_then(|stdout| env_block_path(&stdout)))
}

/// Old command only if the shell rejected the probe (Plan 9 rc, a renamed fish); a probe
/// that ran without PATH (`export -n PATH`) or timed out must not rerun the history-rewriting one.
#[cfg(not(windows))]
fn login_shell_path(shell: &str, timeout: std::time::Duration) -> Option<String> {
    let probe = probe_command(shell);
    match read_login_shell_path(shell, &probe, timeout).ok()? {
        Some(path) => path,
        None if probe != env_command() => read_login_shell_path(shell, &env_command(), timeout)
            .ok()
            .flatten()
            .flatten(),
        None => None,
    }
}

/// Best effort: on any failure PATH stays as the process inherited it.
pub fn fix_path() {
    #[cfg(windows)]
    {
        // std::process::Command can miss a GUI app's PATH until it is set again.
        if let Ok(path) = std::env::var("PATH") {
            std::env::set_var("PATH", path);
        }
    }
    #[cfg(not(windows))]
    {
        let default_shell = if cfg!(target_os = "macos") {
            "/bin/zsh"
        } else {
            "/bin/sh"
        };
        let shell = std::env::var("SHELL").unwrap_or_else(|_| default_shell.into());
        if let Some(path) = login_shell_path(&shell, PROBE_TIMEOUT) {
            std::env::set_var("PATH", path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posix_shells_unset_histfile_first() {
        for shell in [
            "/bin/zsh",
            "/bin/bash",
            "/opt/homebrew/bin/bash",
            "/bin/sh",
            "dash",
            "/usr/local/bin/zsh5",
            "/opt/bash-5.2/bin/bash-5.2",
        ] {
            assert!(
                probe_command(shell)
                    .contains("(unset HISTFILE) 2>/dev/null && unset HISTFILE; exec "),
                "{shell}"
            );
        }
        for shell in ["/opt/homebrew/bin/fish", "/usr/bin/nu", "/bin/tcsh"] {
            assert_eq!(probe_command(shell), env_command(), "{shell}");
        }
    }

    #[test]
    fn path_comes_from_between_the_delimiters() {
        let out = format!(
            "motd PATH=/rc/noise\n{DELIMITER}HOME=/h\nPATH=/a:/b=c\n\x1b[0mX=1\n{DELIMITER}"
        );
        assert_eq!(path_from_output(out.as_bytes()).as_deref(), Some("/a:/b=c"));
        assert_eq!(path_from_output(b"no delimiter"), None);
    }

    /// A shell that cannot parse the probe still yields PATH through the old command.
    #[cfg(not(windows))]
    #[test]
    fn a_shell_rejecting_the_probe_falls_back() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("unsloth-12678-fb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let shell = dir.join("notposix");
        std::fs::write(
            &shell,
            "#!/bin/sh\ncase \"$2\" in if*) exit 2 ;; esac\nPATH=/from/fallback:$PATH exec /bin/sh -c \"$2\"\n",
        )
        .unwrap();
        std::fs::set_permissions(&shell, std::fs::Permissions::from_mode(0o755)).unwrap();
        let shell = shell.to_str().unwrap();
        assert_eq!(
            read_login_shell_path(shell, &probe_command(shell), PROBE_TIMEOUT),
            Ok(None)
        );
        assert_eq!(
            read_login_shell_path(shell, "true", PROBE_TIMEOUT),
            Ok(None)
        );
        let path = login_shell_path(shell, PROBE_TIMEOUT).unwrap_or_default();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(path.starts_with("/from/fallback:"), "{path:?}");
    }

    /// A probe that ran without PATH (`export -n PATH`) is not retried with the old command.
    #[cfg(not(windows))]
    #[test]
    fn a_probe_without_path_is_not_retried() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("unsloth-12678-np-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let shell = dir.join("nopath");
        std::fs::write(
            &shell,
            format!("#!/bin/sh\necho run >> \"$0.calls\"\nprintf %s {DELIMITER}X=1{DELIMITER}\n"),
        )
        .unwrap();
        std::fs::set_permissions(&shell, std::fs::Permissions::from_mode(0o755)).unwrap();
        let path = login_shell_path(shell.to_str().unwrap(), PROBE_TIMEOUT);
        let calls = std::fs::read_to_string(dir.join("nopath.calls")).unwrap_or_default();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(path, None);
        assert_eq!(calls.lines().count(), 1, "{calls:?}");
    }

    #[cfg(not(windows))]
    fn fake_shell(name: &str, body: &str) -> (std::path::PathBuf, String) {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("unsloth-shell-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let shell = dir.join(name);
        std::fs::write(
            &shell,
            format!("#!/bin/sh\necho run >> \"$0.calls\"\n{body}"),
        )
        .unwrap();
        std::fs::set_permissions(&shell, std::fs::Permissions::from_mode(0o755)).unwrap();
        let shell = shell.to_str().unwrap().to_owned();
        (dir, shell)
    }

    #[cfg(not(windows))]
    #[test]
    fn a_background_job_holding_stdout_does_not_hold_launch() {
        let (dir, shell) = fake_shell(
            "agent",
            "sleep 30 &\nPATH=/from/rc:$PATH exec /bin/sh -c \"$2\"\n",
        );
        let started = std::time::Instant::now();
        let path = login_shell_path(&shell, PROBE_TIMEOUT).unwrap_or_default();
        let elapsed = started.elapsed();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(path.starts_with("/from/rc:"), "{path:?}");
        assert!(elapsed < std::time::Duration::from_secs(10), "{elapsed:?}");
    }

    #[cfg(not(windows))]
    #[test]
    fn a_shell_that_never_finishes_times_out_once() {
        let (dir, shell) = fake_shell("hang", "exec sleep 30\n");
        let timeout = std::time::Duration::from_millis(300);
        let started = std::time::Instant::now();
        assert_eq!(
            read_login_shell_path(&shell, &probe_command(&shell), timeout),
            Err(TimedOut)
        );
        let path = login_shell_path(&shell, timeout);
        let elapsed = started.elapsed();
        let calls = std::fs::read_to_string(format!("{shell}.calls")).unwrap_or_default();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(path, None);
        assert_eq!(calls.lines().count(), 2, "{calls:?}");
        assert!(elapsed < std::time::Duration::from_secs(10), "{elapsed:?}");
    }

    #[cfg(not(windows))]
    #[test]
    fn a_cut_off_block_after_the_grace_is_not_a_path() {
        let (dir, shell) = fake_shell(
            "cutoff",
            &format!("sleep 30 &\nprintf %s '{DELIMITER}PATH=/cut/off'\nexit 0\n"),
        );
        let started = std::time::Instant::now();
        let path = login_shell_path(&shell, PROBE_TIMEOUT);
        let elapsed = started.elapsed();
        let calls = std::fs::read_to_string(format!("{shell}.calls")).unwrap_or_default();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(path, None);
        assert_eq!(calls.lines().count(), 1, "{calls:?}");
        assert!(elapsed < std::time::Duration::from_secs(10), "{elapsed:?}");
    }

    /// `head -c` caps the writer: an unbounded-queue regression fails without eating host memory.
    #[cfg(not(windows))]
    #[test]
    fn a_job_writing_nonstop_does_not_hold_launch() {
        let (dir, shell) = fake_shell("spew", "yes | head -c 268435456 &\nexit 0\n");
        let started = std::time::Instant::now();
        let result = read_login_shell_path(&shell, &probe_command(&shell), PROBE_TIMEOUT);
        let elapsed = started.elapsed();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(result, Err(TimedOut));
        assert!(elapsed < std::time::Duration::from_secs(10), "{elapsed:?}");
    }

    #[cfg(not(windows))]
    #[test]
    fn a_rejecting_shell_with_a_held_stdout_still_falls_back() {
        let (dir, shell) = fake_shell(
            "heldfb",
            "sleep 30 &\ncase \"$2\" in if*) exit 2 ;; esac\nPATH=/from/fallback:$PATH exec /bin/sh -c \"$2\"\n",
        );
        let started = std::time::Instant::now();
        let path = login_shell_path(&shell, PROBE_TIMEOUT).unwrap_or_default();
        let elapsed = started.elapsed();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(path.starts_with("/from/fallback:"), "{path:?}");
        assert!(elapsed < std::time::Duration::from_secs(10), "{elapsed:?}");
    }

    #[cfg(not(windows))]
    fn run_probe_in(home: &std::path::Path, shell: &str, command: &str) -> Option<String> {
        let out = std::process::Command::new(shell)
            .arg("-ilc")
            .arg(command)
            .env_clear()
            .env("HOME", home)
            .env("ZDOTDIR", home)
            .env("PATH", "/usr/bin:/bin")
            .current_dir(home)
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| path_from_output(&out.stdout))
            .flatten()
    }

    /// unsloth#12678 against real shells: the old command loses the history, this one keeps it.
    #[cfg(not(windows))]
    #[test]
    fn reading_path_leaves_the_history_file_alone() {
        const ZSH: &str = "HISTFILE=$HOME/.zsh_history\nHISTSIZE=2000\nSAVEHIST=1000\n\
                           unsetopt APPEND_HISTORY\nprint -s plugin-entry\n";
        const BASH: &str = "HISTSIZE=100000\nHISTFILESIZE=100000\nhistory -s plugin-entry\n";
        // (shell, rc file, history file, rc body, extra file)
        let cases = [
            (
                "bash",
                ".bash_profile",
                ".bash_history",
                BASH.to_string(),
                None,
            ),
            (
                "bash",
                ".bash_profile",
                ".bash_history",
                format!("{BASH}trap 'history -w ~/.bash_history' EXIT\n"),
                None,
            ),
            ("zsh", ".zshrc", ".zsh_history", ZSH.to_string(), None),
            (
                "zsh",
                ".zshrc",
                ".zsh_history",
                ZSH.to_string(),
                Some((".zlogout", "fc -W $HOME/.zsh_history\n")),
            ),
            // Readonly HISTFILE: RCS still stops the save; the failed unset must not abort.
            (
                "zsh",
                ".zshrc",
                ".zsh_history",
                format!("{ZSH}typeset -r HISTFILE SAVEHIST\nsetopt ERR_EXIT\n"),
                None,
            ),
        ];
        let history: String = (0..3000).map(|i| format!("echo history-{i}\n")).collect();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("unsloth-12678-{}-{nanos}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let mut ran = 0;
        for (index, (shell, rc, hist, body, extra)) in cases.into_iter().enumerate() {
            let Some(shell) = ["/bin", "/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"]
                .iter()
                .map(|dir| format!("{dir}/{shell}"))
                .find(|path| std::path::Path::new(path).exists())
            else {
                eprintln!("{shell} not installed, skipped");
                continue;
            };
            ran += 1;
            for (arm, command) in [("before", env_command()), ("after", probe_command(&shell))] {
                let home = root.join(format!("{arm}-{index}"));
                std::fs::create_dir_all(&home).unwrap();
                std::fs::write(home.join(rc), format!("export PATH=/from/rc:$PATH\n{body}"))
                    .unwrap();
                if let Some((name, text)) = extra {
                    std::fs::write(home.join(name), text).unwrap();
                }
                std::fs::write(home.join(hist), &history).unwrap();
                let path = run_probe_in(&home, &shell, &command).unwrap_or_default();
                let kept = std::fs::read_to_string(home.join(hist)).unwrap() == history;
                assert_eq!(
                    kept,
                    arm == "after",
                    "case {index} ({shell}) {arm}: kept = {kept}"
                );
                if arm == "after" {
                    assert!(
                        path.starts_with("/from/rc:"),
                        "case {index} ({shell}): {path:?}"
                    );
                }
            }
        }
        let _ = std::fs::remove_dir_all(&root);
        assert!(ran > 0, "no POSIX shell found to test against");
    }
}
