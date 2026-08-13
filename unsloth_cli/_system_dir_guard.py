# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows system-folder guard for the `unsloth` console script.

Running from C:\\Windows\\System32 breaks Unsloth: the folder is unwritable for a
normal user, and cwd-relative paths (`./models`, an output dir,
`unsloth_compiled_cache`) would resolve inside the Windows tree.

Two ways in. "Run as administrator" opens a terminal there, which is a mistake
and still stops with an actionable error. And "Run Unsloth at login" starts the
desktop from an HKCU Run value, which carries no working directory, so it and
every CLI child inherit System32 (issue #8510). That one is not the user's
mistake: the desktop's own commands take no paths from the user, so they move to
~/.unsloth rather than leaving a tray icon and no server.

Imports stay at `os`, since `unsloth_cli/__init__.py` runs this before the
command modules, which resolve STUDIO_HOME against the working directory.
"""

import os as _os

# Set by Unsloth Desktop on every CLI child it owns (process.rs,
# apply_managed_cli_context). Forging it grants nothing: anyone who can set a
# child's environment can set its working directory, and the move lands inside
# the caller's own account.
DESKTOP_MANAGED_ENV = "UNSLOTH_DESKTOP_MANAGED"

# The directory process.rs pins, so a desktop that predates that fix lands where
# a current one would have put it.
WORK_DIR_NAME = ".unsloth"


def windows_root(
    environ,
    pathmod = _os.path,
    isdir = None,
):
    """Where Windows is installed, for messages."""
    return windows_roots(environ, pathmod, isdir)[0]


def windows_roots(
    environ,
    pathmod = _os.path,
    isdir = None,
):
    """Every real Windows directory.

    Candidates are checked, not trusted. Trusting one variable lets whoever sets
    it aim the guard somewhere harmless; trusting all of them is worse, since a
    WINDIR aimed at the user's profile would make their ordinary folders look
    like system ones. So a directory counts only if it holds System32.
    """
    if isdir is None:
        isdir = pathmod.isdir
    system_root = environ.get("SystemRoot")
    candidates = [system_root, environ.get("WINDIR"), r"C:\Windows"]

    roots = []
    for value in candidates:
        if value and value not in roots and isdir(pathmod.join(value, "System32")):
            roots.append(value)
    if roots:
        return roots
    # Nothing on this machine looks like a Windows installation. Keep the guard
    # alive, on SystemRoot or the default, never on a user-settable value.
    return [system_root or r"C:\Windows"]


def _strip_extended_prefix(path):
    r"""Drop the \\?\ (and \\?\UNC\) form so it compares like an ordinary path.

    The prefix is matched case-insensitively: the object manager accepts
    \\?\unc\server\share, and reading it as a relative path would reject a
    profile that Windows itself resolves.
    """
    lowered = path.lower()
    if lowered.startswith("\\\\?\\unc\\"):
        return "\\\\" + path[8:]
    if lowered.startswith("\\\\?\\"):
        return path[4:]
    return path


def _normalize(path, pathmod):
    return pathmod.normcase(pathmod.normpath(_strip_extended_prefix(path)))


def system_dirs(windir, pathmod = _os.path):
    """The Windows folders Unsloth refuses to run from."""
    # SysWOW64 too: a 32-bit elevated shell opens there, same unwritable folder.
    return [_normalize(pathmod.join(windir, name), pathmod) for name in ("System32", "SysWOW64")]


def is_system_dir(
    cwd,
    windir,
    pathmod = _os.path,
    sep = _os.sep,
):
    """True for a system folder itself or anything under it.

    `windir` may be a single directory or several candidates. The separator keeps
    the match on a path boundary, so C:\\Windows2\\System32x is an ordinary folder.
    """
    if not cwd:
        return False
    roots = [windir] if isinstance(windir, str) else list(windir)
    normalized = _normalize(cwd, pathmod)
    return any(
        normalized == directory or normalized.startswith(directory + sep)
        for root in roots
        for directory in system_dirs(root, pathmod)
    )


def _is_rooted(path, pathmod):
    """Absolute, or at least rooted at a drive.

    "." and "C:sub" name no directory on their own, so they are no escape at all
    (pin_relative_overrides resolves the drive-relative form separately, since
    only Windows knows where drive C currently sits). A leading separator is
    drive-relative rather than absolute, but it can never resolve back inside
    System32.
    """
    stripped = _strip_extended_prefix(path)
    return pathmod.isabs(stripped) or stripped.startswith(("\\", "/"))


def _is_fully_qualified(path, pathmod):
    r"""Whether the value names one directory whatever the process does next.

    Narrower than _is_rooted: "\cache" is rooted, but only to the drive of the
    current directory, so moving to a profile on another drive silently moves it
    too. Windows calls these drive-relative and root-relative; both have to be
    resolved before the process leaves.

    Spelled out rather than deferred to isabs(), which answered True for a
    leading separator until Python 3.13 and False after it: the folder a value
    names cannot depend on the interpreter running the guard.
    """
    stripped = _strip_extended_prefix(path)
    if stripped.startswith(("\\\\", "//")):
        # A UNC share names its own root.
        return True
    drive, rest = pathmod.splitdrive(stripped)
    return bool(drive) and rest.startswith(("\\", "/"))


def _outside_windows(candidate, windirs, pathmod, sep):
    if not candidate or not _is_rooted(candidate, pathmod):
        return False
    norm = _normalize(candidate, pathmod)
    for windir in windirs:
        windir_norm = _normalize(windir, pathmod)
        if norm == windir_norm or norm.startswith(windir_norm + sep):
            return False
    return True


def safe_user_dir(
    environ,
    windir,
    pathmod = _os.path,
    sep = _os.sep,
    expanduser = None,
    allow_public = False,
):
    """First home outside the Windows tree, or None.

    SYSTEM's USERPROFILE is C:\\Windows\\System32\\config\\systemprofile, so a
    naive pick would send the caller straight back into the rejected folder.

    %PUBLIC% is only offered as a suggestion a human can type. Moving there
    automatically would put one account's caches, scans and outputs in a folder
    every other account on the machine can read and write.
    """
    if expanduser is None:
        expanduser = pathmod.expanduser
    windirs = [windir] if isinstance(windir, str) else list(windir)
    public = (environ.get("PUBLIC") or "").strip()
    candidates = [environ.get("USERPROFILE")]
    if allow_public:
        candidates.append(public)
    candidates.append(expanduser("~"))
    for candidate in candidates:
        if not _outside_windows(candidate, windirs, pathmod, sep):
            continue
        # USERPROFILE and ~ can name the public profile themselves, so the check
        # is on the folder, not on which variable it came from.
        if (
            not allow_public
            and public
            and _normalize(candidate, pathmod) == _normalize(public, pathmod)
        ):
            continue
        return candidate
    return None


# Commands the desktop runs that take no path from a user: Studio resolves its
# venv, llama.cpp, auth, pid files and logs from the Studio home. `update` is
# here so a desktop that predates the Rust-side fix can still upgrade from the
# tray. Everything else keeps the hard error, `studio run` above all: it declares
# its own --api-only and takes --model ./x plus a raw llama-server tail.
#
# Matched whole, not on the first word, since `studio update --local <path>`
# resolves that path against the working directory.
_STUDIO_COMMANDS = (
    ("provision-desktop-auth",),
    ("desktop-capabilities",),
    ("desktop-capabilities", "--json"),
    ("update",),
)
_HELP_FLAGS = ("-h", "--help", "--version", "-V")
_API_ONLY_FLAGS = ("--api-only", "-H", "--host", "-p", "--port")


def _is_desktop_backend_launch(rest):
    """`studio --api-only -H 127.0.0.1 -p 8888` and nothing else.

    Matching --api-only anywhere would also match `studio run --model ./m.gguf
    --api-only`, which is a user command with user paths.
    """
    if "--api-only" not in rest:
        return False
    expects_value = False
    for arg in rest:
        if expects_value:
            expects_value = False
            continue
        if arg not in _API_ONLY_FLAGS:
            return False
        expects_value = arg != "--api-only"
    return True


# Subcommands that take a path from the caller, by argument or by environment.
# `run` takes --model and a raw llama-server tail; `update --local` installs
# from a checkout the caller names.
_PATH_TAKING_STUDIO_COMMANDS = ("run", "update")


def _takes_a_path(rest):
    """Whether this `studio` invocation can carry a caller's path.

    Deliberately blunt: the bare forms the desktop runs carry no path, so
    anything else is treated as though it might.
    """
    if not rest:
        return False
    if rest[0] in _PATH_TAKING_STUDIO_COMMANDS:
        return tuple(rest) not in _STUDIO_COMMANDS
    return any(not arg.startswith("-") for arg in rest[1:])


def is_relocatable_invocation(argv, environ):
    """True when this invocation is desktop-managed or provably cwd-independent.

    The argv arm matters on its own: it fixes users whose installed desktop build
    predates the Rust-side working-directory fix and so sets no marker.
    """
    args = [arg for arg in argv if arg]
    if not args:
        return False
    # Click handles top-level -h/--help/--version eagerly, before the callback
    # this runs from, so that case never actually arrives here. `studio --help`
    # does, and printing help is worth as little as it costs.
    if all(arg in _HELP_FLAGS for arg in args):
        return True
    if args[0] != "studio":
        # Everything the backend spawns inherits the marker, so it authorises
        # the studio commands the desktop runs and nothing else: rebasing
        # `train --dataset .\data.json` under a stray one would be worse than
        # the refusal it replaced.
        return False
    rest = args[1:]
    if environ.get(DESKTOP_MANAGED_ENV) == "1" and not _takes_a_path(rest):
        # The marker is for a desktop build whose command shape this CLI does
        # not know yet. It is inherited by the backend and everything below it,
        # so it must not widen the set to commands that carry a path: `studio
        # run --model .\local.gguf` from a marked shell would be rebased.
        return True
    if rest and all(arg in _HELP_FLAGS for arg in rest):
        return True
    if _is_desktop_backend_launch(rest):
        return True
    return tuple(rest) in _STUDIO_COMMANDS


# Path overrides the caller may have written relative to the folder being left.
# Studio resolves them with Path.resolve(), which anchors a relative value to the
# working directory, so moving first would silently retarget them.
_RELATIVE_PATH_ENV = (
    # Studio roots: storage_roots.py.
    "UNSLOTH_STUDIO_HOME",
    "STUDIO_HOME",
    "UNSLOTH_STUDIO_DOCUMENTS_HOME",
    "UNSLOTH_STUDIO_PROJECTS_HOME",
    "UNSLOTH_STUDIO_SANDBOX_HOME",
    # `studio update` reads it, and that command relocates.
    "STUDIO_LOCAL_REPO",
    # Engine and tool locations the user may point somewhere of their own.
    "UNSLOTH_LLAMA_CPP_PATH",
    "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR",
    "UNSLOTH_SD_CPP_PATH",
    "UNSLOTH_WHISPER_CPP_PATH",
    "LLAMA_SERVER_PATH",
    "WHISPER_SERVER_PATH",
    "SD_CLI_PATH",
    "SD_SERVER_PATH",
    # GPU SDK roots, joined with bin/ for DLL discovery.
    "CUDA_PATH",
    "HIP_PATH",
    "HIP_PATH_57",
    "ROCM_PATH",
    "MLX_HOSTFILE",
    "OLLAMA_MODELS",
    "DG_VISUAL_BIN",
    "UNSLOTH_DG_SHIM",
    # Caches.
    "UNSLOTH_COMPILE_LOCATION",
    "TORCHINDUCTOR_CACHE_DIR",
    "UNSLOTH_DIFFUSION_COMPILE_CACHE_DIR",
    "UNSLOTH_DIFFUSION_COND_CACHE_DIR",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_XET_CACHE",
    "HF_DATASETS_CACHE",
    "HF_ASSETS_CACHE",
    "TRANSFORMERS_CACHE",
    "SENTENCE_TRANSFORMERS_HOME",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "UNSLOTH_STUDIO_CHILD_RECORD",
    "UNSLOTH_LLAMA_INSTALLER",
    "CUDA_HOME",
    "CUDA_ROOT",
)

# The separator is Windows', not the host's: this guard only ever runs on
# Windows, and os.pathsep would split "D:\\shared" apart anywhere else.
_PATH_LIST_SEPARATOR = ";"

# Values holding several separated directories, anchored entry by entry.
_PATH_LIST_ENV = (
    "UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH",
    "CUDA_RUNTIME_DLL_DIR",
)


def pin_relative_overrides(
    environ,
    cwd,
    pathmod = _os.path,
    abspath = None,
):
    """Rewrite relative path overrides so they keep meaning the folder they did.

    Returns the names that were pinned. A `~` value is left alone: expanduser
    does not consult the working directory.
    """
    pinned = []
    for name in _RELATIVE_PATH_ENV:
        value = (environ.get(name) or "").strip()
        anchored = _anchor(value, cwd, pathmod, abspath)
        if anchored is not None:
            environ[name] = anchored
            pinned.append(name)
    for name in _PATH_LIST_ENV:
        raw = environ.get(name) or ""
        if not raw.strip():
            continue
        # A list authorises or searches several directories, so each entry is
        # anchored on its own; one relative entry is enough to change what the
        # whole list means.
        entries = raw.split(_PATH_LIST_SEPARATOR)
        anchored_entries = [_anchor(e.strip(), cwd, pathmod, abspath) or e for e in entries]
        if anchored_entries != entries:
            environ[name] = _PATH_LIST_SEPARATOR.join(anchored_entries)
            pinned.append(name)
    return pinned


def _anchor(
    value,
    cwd,
    pathmod,
    abspath = None,
):
    """The value rewritten to name the same folder from anywhere, or None.

    None means it needs no rewriting: it is empty, it starts with `~` (expanduser
    does not consult the working directory), or it is already fully qualified.
    """
    value = (value or "").strip()
    if not value or value.startswith("~") or _is_fully_qualified(value, pathmod):
        return None
    if pathmod.splitdrive(value)[0] or value.startswith(("\\", "/")):
        # "D:cache" is the current directory on drive D and "\cache" is the root
        # of the current drive: both depend on process state that join() cannot
        # see, so ask the OS. A failure reaches the caller, which then declines
        # to move at all.
        return (abspath or pathmod.abspath)(value)
    return pathmod.join(cwd, value)


def relocation_target(
    environ,
    windir,
    pathmod = _os.path,
    sep = _os.sep,
    expanduser = None,
    makedirs = _os.makedirs,
    home_isdir = None,
):
    """Where a desktop-managed command should run instead, or None."""
    home = safe_user_dir(environ, windir, pathmod, sep, expanduser)
    if not home:
        return None
    if home_isdir is None:
        home_isdir = pathmod.isdir
    # A profile that has not mounted yet still has a writable parent, so makedirs
    # would build an empty second one that shadows the real profile when it
    # arrives. Report it unavailable, as the Rust half does.
    if not home_isdir(home):
        return None
    work_dir = pathmod.join(home, WORK_DIR_NAME)
    try:
        makedirs(work_dir, exist_ok = True)
    except OSError:
        # An unwritable home is a broken profile, and Studio must write there
        # anyway. Stop, as the Rust half does, rather than failing later.
        return None
    return work_dir


def blocked_message(
    cwd,
    argv,
    environ,
    windir,
    pathmod = _os.path,
    sep = _os.sep,
    expanduser = None,
):
    """The error shown to someone who ran Unsloth from a system folder by hand."""
    # allow_public here only: a person can sensibly `cd C:\Users\Public`, but
    # relocating there automatically would share one account's state with every
    # other account on the machine.
    home = safe_user_dir(environ, windir, pathmod, sep, expanduser, allow_public = True)
    if home:
        # Quote it, or C:\Users\Jane Doe reaches Set-Location as two arguments.
        # PowerShell single quotes are verbatim ('' escapes an apostrophe); cmd
        # needs double quotes once extensions are off. " is not legal in a path.
        home_ps = "'" + home.replace("'", "''") + "'"
        home_cmd = '"' + home + '"'
        cd_lines = (
            f"    cd {home_ps}          (PowerShell)\n" f"    cd /d {home_cmd}       (cmd.exe)\n"
        )
    else:
        cd_lines = f"    (any folder outside {windir if isinstance(windir, str) else windir[0]})\n"
    rendered_argv = " ".join((f'"{arg}"' if " " in arg else arg) for arg in argv)
    retry = ("unsloth " + rendered_argv).rstrip()
    return (
        f"Unsloth cannot run from {cwd}\n"
        "\n"
        "That is a Windows system folder. Windows blocks writes here, and any\n"
        "relative path you pass would resolve inside the Windows folder.\n"
        "Opening a terminal with 'Run as administrator' starts you in a folder like\n"
        "this one, which is how most people end up here.\n"
        "\n"
        "Change to a normal folder and run the command again:\n"
        f"{cd_lines}"
        f"    {retry}"
    )


def check_working_directory(
    argv,
    environ,
    platform,
    getcwd = _os.getcwd,
    chdir = _os.chdir,
    pathmod = _os.path,
    sep = _os.sep,
    expanduser = None,
    makedirs = _os.makedirs,
    isdir = None,
    abspath = None,
    home_isdir = None,
):
    """Decide what to do about the current working directory.

    Returns (message, colour, fatal). `fatal` is the caller's cue to exit 1;
    a message with fatal False is a warning printed after a successful move.
    """
    if platform != "win32":
        return None, None, False

    windirs = windows_roots(environ, pathmod, isdir)
    windir = windirs[0]
    try:
        cwd = getcwd()
    except OSError:
        # The launch directory was deleted or its drive went away: nothing to
        # compare, so say that rather than name a folder they were never in.
        return (
            (
                "Unsloth cannot determine its current folder. It may have been deleted,\n"
                "or it may be on a drive that is no longer available.\n"
                "Change to a folder that exists and run the command again."
            ),
            "red",
            True,
        )

    if not is_system_dir(cwd, windirs, pathmod, sep):
        return None, None, False

    if not is_relocatable_invocation(argv, environ):
        return blocked_message(cwd, argv, environ, windirs, pathmod, sep, expanduser), "red", True

    target = relocation_target(environ, windirs, pathmod, sep, expanduser, makedirs, home_isdir)
    if target is not None:
        try:
            # Before moving, or a relative override the caller wrote would end up
            # naming a folder under the new directory instead of theirs.
            pin_relative_overrides(environ, cwd, pathmod, abspath)
        except Exception:
            # An environment we cannot pin is one we must not move underneath.
            target = None
    if target is not None:
        try:
            chdir(target)
        except OSError:
            target = None
        else:
            # Confirm the move landed outside the Windows tree instead of
            # trusting chdir not to have raised.
            try:
                if is_system_dir(getcwd(), windirs, pathmod, sep):
                    target = None
            except OSError:
                target = None
    if target is None:
        # Fail closed: nowhere usable outside the Windows tree. This text lands
        # in the desktop's logs, so it describes that case, not a shell.
        return (
            (
                f"Unsloth cannot run from {cwd}, and no folder outside {windir} was\n"
                "available to run from instead. Check that the user profile for this\n"
                "account exists and is writable."
            ),
            "red",
            True,
        )

    return (
        (
            f"Unsloth was started from {cwd}, which is a Windows system folder,\n"
            f"so it switched to {target} instead.\n"
            "This happens when Unsloth Desktop is started by 'Run Unsloth at login'."
        ),
        "yellow",
        False,
    )
