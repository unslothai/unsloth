"""
Shared test harness utilities for PR #4507 spawn behavior tests.

Provides context managers to:
- Force spawn start method on Linux (simulating Windows/macOS)
- Create temporary Python modules in temp directories
- Override environment variables
- Temporarily modify sys.path
"""

import contextlib
import multiprocessing
import os
import sys
import tempfile
import textwrap


@contextlib.contextmanager
def force_spawn_context():
    """Force both multiprocess (dill-based) and stdlib multiprocessing to use spawn.

    This simulates the Windows/macOS environment on Linux, where the default
    start method is 'spawn' rather than 'fork'.

    Restores the original start method on exit.
    """
    # Save originals
    _orig_stdlib_ctx = multiprocessing.context._default_context._actual_context
    _orig_stdlib_method = multiprocessing.get_start_method()

    _has_multiprocess = False
    _orig_mp_method = None
    try:
        import multiprocess
        _has_multiprocess = True
        _orig_mp_method = multiprocess.get_start_method()
    except ImportError:
        pass

    try:
        # Force stdlib multiprocessing to spawn
        multiprocessing.context._default_context._actual_context = (
            multiprocessing.context._concrete_contexts['spawn']
        )

        # Force multiprocess (dill-based, used by datasets) to spawn
        if _has_multiprocess:
            multiprocess.context._force_start_method('spawn')

        yield

    finally:
        # Restore stdlib
        multiprocessing.context._default_context._actual_context = _orig_stdlib_ctx

        # Restore multiprocess
        if _has_multiprocess:
            if _orig_mp_method is not None:
                multiprocess.context._force_start_method(_orig_mp_method)
            else:
                # Reset to default (fork on Linux)
                multiprocess.context._force_start_method('fork')


@contextlib.contextmanager
def temp_module_in_dir(module_name="fake_module", content=None):
    """Create a temporary Python module file in a temp directory.

    Args:
        module_name: Name of the module (without .py extension).
        content: Python source code for the module. Defaults to a simple
                 module that defines MARKER = True.

    Yields:
        (tmpdir, module_name) -- the directory containing the module and
        the module name (for import).
    """
    if content is None:
        content = textwrap.dedent("""\
            # Auto-generated fake module for PR #4507 testing
            MARKER = True

            def transform(text):
                return text.upper()
        """)

    tmpdir = tempfile.mkdtemp(prefix="pr4507_module_")
    module_path = os.path.join(tmpdir, f"{module_name}.py")
    try:
        with open(module_path, "w") as f:
            f.write(content)
        yield tmpdir, module_name
    finally:
        # Cleanup
        try:
            os.unlink(module_path)
            os.rmdir(tmpdir)
        except OSError:
            pass


@contextlib.contextmanager
def env_override(**kwargs):
    """Temporarily set or unset environment variables.

    Pass value=None to unset a variable.

    Example:
        with env_override(PYTHONPATH="/tmp/test", UNSLOTH_FOO=None):
            ...
    """
    originals = {}
    for key in kwargs:
        originals[key] = os.environ.get(key)  # None if not set

    try:
        for key, value in kwargs.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, orig_value in originals.items():
            if orig_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = orig_value


@contextlib.contextmanager
def sys_path_prepend(path):
    """Temporarily prepend a path to sys.path. Removes it on exit."""
    sys.path.insert(0, path)
    try:
        yield
    finally:
        try:
            sys.path.remove(path)
        except ValueError:
            pass


def simulate_pythonpath_propagation(platform, compile_location, current_pythonpath, cwd):
    """Pure-function reproduction of trainer.py lines 21-30.

    This extracts the PYTHONPATH propagation logic from trainer.py into a
    testable pure function, avoiding the need to import trainer.py (which
    pulls in torch, unsloth, etc.).

    Args:
        platform: sys.platform value ('win32', 'darwin', 'linux')
        compile_location: UNSLOTH_COMPILE_LOCATION env var value
        current_pythonpath: Current PYTHONPATH env var value ('' if unset)
        cwd: Current working directory (for resolving relative paths)

    Returns:
        (new_pythonpath, resolved_cache_path) tuple.
        On linux, returns (current_pythonpath, compile_location) unchanged.
    """
    if platform not in ('win32', 'darwin'):
        return current_pythonpath, compile_location

    cache = compile_location
    if not os.path.isabs(cache):
        cache = os.path.join(cwd, cache)

    pp = current_pythonpath
    if cache not in pp.split(os.pathsep):
        pp = cache + (os.pathsep + pp if pp else '')

    return pp, cache
