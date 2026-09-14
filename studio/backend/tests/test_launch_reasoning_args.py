import struct
import subprocess
from unittest.mock import patch

import pytest

from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend


_REAL_POPEN = subprocess.Popen
_MODEL_FILENAME = "model.gguf"
_MODEL_ARCHITECTURE = "llama"
_ARCHITECTURE_METADATA_KEY = "general.architecture"
_GGUF_STRING_LENGTH_FORMAT = "<Q"
_GGUF_METADATA_TYPE_FORMAT = "<I"
_GGUF_HEADER_FORMAT = "<IIQQ"
_GGUF_MAGIC = 0x46554747
_GGUF_VERSION = 3
_GGUF_TENSOR_COUNT = 0
_GGUF_METADATA_COUNT = 1
_GGUF_STRING_METADATA_TYPE = 8
_MODEL_SIZE_BYTES = 1024
_FAKE_BINARY = "/fake/llama-server"
_FAKE_PROCESS_PID = 123
_MODEL_IDENTIFIER = "test"
_REASONING_STYLE = "enable_thinking"
_REASONING_CAPABILITY = "supports_reasoning_flag"
_PROBE_INCONCLUSIVE_CAPABILITY = "mtp_probe_inconclusive"
_REASONING_FLAG = "--reasoning"
_LLAMA_REASONING_ENV = "LLAMA_ARG_REASONING"
_NO_REASONING_PRESERVE_FLAG = "--no-reasoning-preserve"
_CHAT_TEMPLATE_KWARGS_FLAG = "--chat-template-kwargs"
_ENABLE_THINKING_KWARG = "enable_thinking"
_REASONING_ON = "on"
_REASONING_OFF = "off"
_REASONING_AUTO = "auto"
_SERVER_COMMAND = "llama-server"
_MODERN_HELP = "--reasoning VALUE\n"
_MISSING_BINARY_FILENAME = "missing"
_FAKE_BINARY_CONTENT = "stub"
_UTF8_ENCODING = "utf-8"
_SUBPROCESS_RUN_TARGET = "core.inference.llama_cpp.subprocess.run"
_PRESERVE_ONLY_KWARGS = '{"preserve_thinking": false}'
_PRESERVE_DEFAULT_KWARGS = '{"preserve_thinking": true}'
_REASONING_EFFORT_KWARGS = '{"reasoning_effort": "high"}'
_OLD_KWARGS = '{"enable_thinking": false, "preserve_thinking": false}'
_REASONING_TEMPLATE = (
    "{% if enable_thinking %}thinking{% endif %}{% if preserve_thinking %}history{% endif %}"
)


def _load_backend(tmp_path):
    gguf = tmp_path / _MODEL_FILENAME
    encoded_architecture = _MODEL_ARCHITECTURE.encode()
    encoded_key = _ARCHITECTURE_METADATA_KEY.encode()
    string = lambda value: struct.pack(_GGUF_STRING_LENGTH_FORMAT, len(value)) + value
    metadata = (
        string(encoded_key)
        + struct.pack(_GGUF_METADATA_TYPE_FORMAT, _GGUF_STRING_METADATA_TYPE)
        + string(encoded_architecture)
    )
    gguf.write_bytes(
        struct.pack(
            _GGUF_HEADER_FORMAT,
            _GGUF_MAGIC,
            _GGUF_VERSION,
            _GGUF_TENSOR_COUNT,
            _GGUF_METADATA_COUNT,
        )
        + metadata
    )

    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda *args, **kwargs: []
    backend._get_gpu_free_memory = lambda *args, **kwargs: []
    backend._read_gguf_metadata = lambda path: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda path: _MODEL_SIZE_BYTES
    backend._mmproj_vram_bytes = lambda path: 0
    backend._resolve_launch_mmproj_path = lambda **kwargs: None
    backend._apu_ram_shortfall_message = lambda *args, **kwargs: None
    backend._amd_apu_wants_unified_memory = lambda *args, **kwargs: False
    backend._find_llama_server_binary = lambda include_denied = False: _FAKE_BINARY
    backend._is_vulkan_backend = lambda binary = None: False
    backend._wait_for_health = lambda *args, **kwargs: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda detected: True
    return backend, gguf


def _launch(backend, gguf, **load_kwargs):
    captured = {}

    def fake_popen(command, **kwargs):
        if command[0] != _FAKE_BINARY:
            return _REAL_POPEN(command, **kwargs)
        captured["cmd"] = list(command)
        return type(
            "Process",
            (),
            {
                "pid": _FAKE_PROCESS_PID,
                "stdout": (),
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf),
                model_identifier = _MODEL_IDENTIFIER,
                **load_kwargs,
            )
        )
    return captured


@pytest.fixture(autouse = True)
def _no_inherited_reasoning_env(monkeypatch):
    """Every assertion here is about what the LAUNCH decides, not the shell.

    The helper stands aside whenever LLAMA_ARG_REASONING is explicit, so a
    developer (or a CI image) exporting it turned the three ``--reasoning on``
    assertions red for reasons that had nothing to do with the code under test.
    Tests that want an override set it themselves, after this has cleared it.
    """
    monkeypatch.delenv(_LLAMA_REASONING_ENV, raising = False)


def _backend(style = _REASONING_STYLE, supports_preserve = False):
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._reasoning_style = style
    backend._architecture = None
    backend._supports_preserve_thinking = supports_preserve
    backend._preserve_thinking_default = False
    backend._reasoning_default = True
    return backend


@pytest.mark.parametrize(
    ("thinking_default", "reasoning_override"),
    ((True, _REASONING_OFF), (False, _REASONING_ON)),
)
def test_modern_launch_honors_reasoning_env_override(
    monkeypatch, thinking_default, reasoning_override
):
    monkeypatch.setenv(_LLAMA_REASONING_ENV, reasoning_override)
    backend = _backend()
    command = [_SERVER_COMMAND]

    backend._append_launch_reasoning_args(
        command,
        thinking_default,
        {_REASONING_CAPABILITY: True},
    )

    assert _REASONING_FLAG not in command
    assert backend._reasoning_default is (reasoning_override == _REASONING_ON)


def test_modern_launch_treats_auto_env_as_absent_override(monkeypatch):
    monkeypatch.setenv(_LLAMA_REASONING_ENV, _REASONING_AUTO)
    command = [_SERVER_COMMAND]

    _backend()._append_launch_reasoning_args(
        command,
        False,
        {_REASONING_CAPABILITY: True},
    )

    assert command == [_SERVER_COMMAND, _REASONING_FLAG, _REASONING_OFF]


def test_inconclusive_probe_preserves_explicit_reasoning_env(monkeypatch):
    monkeypatch.setenv(_LLAMA_REASONING_ENV, _REASONING_ON)
    backend = _backend()
    command = [_SERVER_COMMAND]

    backend._append_launch_reasoning_args(
        command,
        False,
        {
            _REASONING_CAPABILITY: False,
            _PROBE_INCONCLUSIVE_CAPABILITY: True,
        },
    )

    assert command == [_SERVER_COMMAND]
    assert backend._reasoning_default is True


def test_launch_reasoning_args_use_modern_flag_with_old_binary_fallback(tmp_path, monkeypatch):
    assert (
        LlamaCppBackend.probe_server_capabilities(str(tmp_path / _MISSING_BINARY_FILENAME)).get(
            _REASONING_CAPABILITY, False
        )
        is False
    )
    binary = tmp_path / _SERVER_COMMAND
    binary.write_text(_FAKE_BINARY_CONTENT, encoding = _UTF8_ENCODING)
    monkeypatch.setattr(
        LlamaCppBackend,
        "_llama_server_env_for_binary",
        classmethod(lambda cls, path: {}),
    )
    monkeypatch.setattr(
        _SUBPROCESS_RUN_TARGET,
        lambda *args, **kwargs: __import__("subprocess").CompletedProcess(
            args[0], 0, _MODERN_HELP, ""
        ),
    )
    LlamaCppBackend._capability_cache.clear()
    modern_capabilities = LlamaCppBackend.probe_server_capabilities(str(binary))
    assert modern_capabilities[_REASONING_CAPABILITY] is True

    backend = _backend(supports_preserve = True)

    modern_command = [_SERVER_COMMAND]
    backend._append_launch_reasoning_args(modern_command, True, modern_capabilities)
    assert modern_command == [
        _SERVER_COMMAND,
        _REASONING_FLAG,
        _REASONING_ON,
        _CHAT_TEMPLATE_KWARGS_FLAG,
        _PRESERVE_ONLY_KWARGS,
    ]
    assert _NO_REASONING_PRESERVE_FLAG not in modern_command

    old_command = [_SERVER_COMMAND]
    backend._append_launch_reasoning_args(old_command, False, {_REASONING_CAPABILITY: False})
    assert old_command == [_SERVER_COMMAND, _CHAT_TEMPLATE_KWARGS_FLAG, _OLD_KWARGS]


def test_modern_launch_preserves_model_thinking_default():
    backend = _backend(supports_preserve = True)
    backend._preserve_thinking_default = True
    command = [_SERVER_COMMAND]

    backend._append_launch_reasoning_args(command, True, {_REASONING_CAPABILITY: True})

    assert command == [
        _SERVER_COMMAND,
        _REASONING_FLAG,
        _REASONING_ON,
        _CHAT_TEMPLATE_KWARGS_FLAG,
        _PRESERVE_DEFAULT_KWARGS,
    ]


def test_modern_launch_keeps_reasoning_effort_in_template_kwargs():
    command = [_SERVER_COMMAND]

    _backend(style = "reasoning_effort")._append_launch_reasoning_args(
        command,
        True,
        {_REASONING_CAPABILITY: True},
    )

    assert _REASONING_FLAG not in command
    assert command == [_SERVER_COMMAND, _CHAT_TEMPLATE_KWARGS_FLAG, _REASONING_EFFORT_KWARGS]


def test_load_command_uses_reasoning_flag(tmp_path):
    backend, gguf = _load_backend(tmp_path)
    backend.probe_server_capabilities = lambda binary = None: {
        _REASONING_CAPABILITY: True,
        _PROBE_INCONCLUSIVE_CAPABILITY: False,
    }

    command = _launch(
        backend,
        gguf,
        chat_template_override = _REASONING_TEMPLATE,
    )["cmd"]

    assert _REASONING_FLAG in command
    assert command[command.index(_REASONING_FLAG) + 1] == _REASONING_ON
    kwargs = command[command.index(_CHAT_TEMPLATE_KWARGS_FLAG) + 1]
    assert _ENABLE_THINKING_KWARG not in kwargs
    assert _NO_REASONING_PRESERVE_FLAG not in command


@pytest.mark.parametrize(
    ("env_value", "expected_thinking"),
    (
        ("true", True), ("1", True), ("enabled", True),
        ("false", False), ("0", False), ("disabled", False),
    ),
)
def test_modern_launch_honors_every_spelling_llama_server_accepts(
    monkeypatch, env_value, expected_thinking
):
    """llama-server's is_truthy/is_falsey take more than on/off.

    Reading only on/off classified LLAMA_ARG_REASONING=true as "no override", so
    the launch appended its own --reasoning, and llama-server prints "will be
    overwritten by command line argument" and uses ours. The operator's choice
    has to survive whichever spelling they used.
    """
    monkeypatch.setenv(_LLAMA_REASONING_ENV, env_value)
    backend = _backend()
    command = [_SERVER_COMMAND]

    backend._append_launch_reasoning_args(
        command,
        not expected_thinking,
        {_REASONING_CAPABILITY: True},
    )

    assert command == [_SERVER_COMMAND]
    assert backend._reasoning_default is expected_thinking


@pytest.mark.parametrize("env_value", ("-1", _REASONING_AUTO))
def test_modern_launch_treats_every_auto_spelling_as_absent_override(monkeypatch, env_value):
    """`-1` is is_autoy's other spelling of auto, so it is not an operator choice."""
    monkeypatch.setenv(_LLAMA_REASONING_ENV, env_value)
    backend = _backend()
    command = [_SERVER_COMMAND]

    backend._append_launch_reasoning_args(command, True, {_REASONING_CAPABILITY: True})

    assert command == [_SERVER_COMMAND, _REASONING_FLAG, _REASONING_ON]
    assert backend._reasoning_default is True


@pytest.mark.parametrize("env_value", ("ON", " on ", "bogus"))
def test_unreadable_env_is_left_alone_without_claiming_a_default(monkeypatch, env_value):
    """Spellings upstream rejects are still the operator's, and still not a default.

    The parse in common/arg.cpp is case-sensitive and does not strip, so all
    three abort llama-server at startup whatever we append. Appending our own
    --reasoning could not rescue that launch (the environment is parsed first and
    raises), and a future llama.cpp that widens the vocabulary would find the
    operator's value quietly overwritten. So the launch stands aside, and because
    the polarity is unreadable it must NOT record a reasoning default.
    """
    monkeypatch.setenv(_LLAMA_REASONING_ENV, env_value)
    backend = _backend()
    backend._reasoning_default = False
    command = [_SERVER_COMMAND]

    backend._append_launch_reasoning_args(command, False, {_REASONING_CAPABILITY: True})

    assert command == [_SERVER_COMMAND]
    assert backend._reasoning_default is False


@pytest.mark.parametrize(
    ("env_value", "expected_kwargs"),
    (
        (_REASONING_ON, '{"enable_thinking": true}'),
        (_REASONING_OFF, '{"enable_thinking": false}'),
        ("false", '{"enable_thinking": false}'),
    ),
)
def test_old_binary_carries_explicit_env_on_the_channel_it_understands(
    monkeypatch, env_value, expected_kwargs
):
    """A build whose --help positively lacks --reasoning ignores the env too.

    The launch recorded the operator's default but shipped the model-derived one,
    so `unsloth start --reasoning off` was silently dropped on such a build and
    reasoning_default described a server that was never started that way.
    """
    monkeypatch.setenv(_LLAMA_REASONING_ENV, env_value)
    backend = _backend()
    command = [_SERVER_COMMAND]

    backend._append_launch_reasoning_args(
        command,
        env_value == _REASONING_OFF,
        {_REASONING_CAPABILITY: False, _PROBE_INCONCLUSIVE_CAPABILITY: False},
    )

    assert command == [_SERVER_COMMAND, _CHAT_TEMPLATE_KWARGS_FLAG, expected_kwargs]
    assert backend._reasoning_default is (env_value in (_REASONING_ON,))


def test_old_binary_without_env_keeps_the_model_default(monkeypatch):
    """No override means the old fallback is untouched, deprecation warning and all."""
    backend = _backend(supports_preserve = True)
    command = [_SERVER_COMMAND]

    backend._append_launch_reasoning_args(
        command,
        False,
        {_REASONING_CAPABILITY: False, _PROBE_INCONCLUSIVE_CAPABILITY: False},
    )

    assert command == [_SERVER_COMMAND, _CHAT_TEMPLATE_KWARGS_FLAG, _OLD_KWARGS]


def test_old_binary_never_receives_the_flag_it_cannot_parse(monkeypatch):
    """Emitting --reasoning to a pre-b20297 build aborts the launch outright."""
    for env_value in (None, _REASONING_ON, _REASONING_OFF, _REASONING_AUTO, "true", "bogus"):
        for inconclusive in (True, False):
            if env_value is None:
                monkeypatch.delenv(_LLAMA_REASONING_ENV, raising = False)
            else:
                monkeypatch.setenv(_LLAMA_REASONING_ENV, env_value)
            command = [_SERVER_COMMAND]
            _backend()._append_launch_reasoning_args(
                command,
                True,
                {
                    _REASONING_CAPABILITY: False,
                    _PROBE_INCONCLUSIVE_CAPABILITY: inconclusive,
                },
            )
            assert _REASONING_FLAG not in command


@pytest.mark.parametrize(
    ("env_value", "expected_kwargs", "expected_default"),
    (
        (_REASONING_OFF, '{"reasoning_effort": "low"}', False),
        (_REASONING_ON, _REASONING_EFFORT_KWARGS, True),
    ),
)
def test_explicit_env_moves_the_reasoning_effort_ladder(
    monkeypatch, env_value, expected_kwargs, expected_default
):
    """An effort dial has no boolean for --reasoning, so the ladder has to carry it.

    gpt-oss-style templates never read enable_thinking, so LLAMA_ARG_REASONING
    does nothing to them whatever the binary. Leaving the ladder on the model's
    own default launched a thinking server while Studio recorded thinking off.
    """
    monkeypatch.setenv(_LLAMA_REASONING_ENV, env_value)
    backend = _backend(style = "reasoning_effort")
    command = [_SERVER_COMMAND]

    backend._append_launch_reasoning_args(
        command,
        env_value == _REASONING_OFF,
        {_REASONING_CAPABILITY: True},
    )

    assert _REASONING_FLAG not in command
    assert command == [_SERVER_COMMAND, _CHAT_TEMPLATE_KWARGS_FLAG, expected_kwargs]
    assert backend._reasoning_default is expected_default


def test_preserve_thinking_survives_every_reasoning_channel(monkeypatch):
    """#9096's independent preserve default must ride along on all four paths."""
    cases = (
        (None, {_REASONING_CAPABILITY: True}),
        (_REASONING_ON, {_REASONING_CAPABILITY: True}),
        (None, {_REASONING_CAPABILITY: False, _PROBE_INCONCLUSIVE_CAPABILITY: False}),
        (_REASONING_ON, {_REASONING_CAPABILITY: False, _PROBE_INCONCLUSIVE_CAPABILITY: True}),
    )
    for env_value, caps in cases:
        if env_value is None:
            monkeypatch.delenv(_LLAMA_REASONING_ENV, raising = False)
        else:
            monkeypatch.setenv(_LLAMA_REASONING_ENV, env_value)
        backend = _backend(supports_preserve = True)
        backend._preserve_thinking_default = True
        command = [_SERVER_COMMAND]

        backend._append_launch_reasoning_args(command, True, caps)

        kwargs = command[command.index(_CHAT_TEMPLATE_KWARGS_FLAG) + 1]
        assert '"preserve_thinking": true' in kwargs
