# Unsloth's in-process AsyncGRPO server against TRL's own startup check, `_check_sampling_replay`.
# TRL >= 1.19 reads `vllm_config.model_config.logprobs_mode` from `/server_info` before the first rollout whenever
# score centering meets temperature scaling; a missing key is a KeyError at startup.

import asyncio
import inspect
import types

import pytest

from tests.test_async_grpo_fast_inference import _FakeLLM, _Sampling, _get, _model, server_module


def _check_sampling_replay():
    try:
        from trl.trainer.async_grpo_trainer import AsyncRolloutLoop
        return AsyncRolloutLoop._check_sampling_replay
    except Exception:
        pass
    try:
        from trl.experimental.async_grpo.async_rollout_worker import _AsyncRolloutLoop
        return _AsyncRolloutLoop._check_sampling_replay
    except Exception:
        pytest.skip("TRL AsyncGRPO with _check_sampling_replay not installed")


def _serve(logprobs_mode):
    llm = _FakeLLM()
    if logprobs_mode is not None:
        llm.llm_engine.vllm_config.model_config.logprobs_mode = logprobs_mode
    return server_module.UnslothAsyncVLLMServer(_model(llm), sampling_params = _Sampling)


def _run_trl_check(server, temperature):
    aiohttp = pytest.importorskip("aiohttp")
    check = _check_sampling_replay()

    async def run():
        async with aiohttp.ClientSession() as session:
            loop = types.SimpleNamespace(
                config = types.SimpleNamespace(score_centering = True, temperature = temperature),
                sampling_replay = False,
                session = session,
                vllm_server_url = server.url,
            )
            await check(loop)

    asyncio.run(run())
    return check


def _reads_logprobs_mode(check):
    if "logprobs_mode" not in inspect.getsource(check):
        pytest.skip("this TRL does not check logprobs_mode")


@pytest.mark.parametrize("logprobs_mode", ["processed_logprobs", "raw_logprobs", None])
def test_server_info_reports_engine_logprobs_mode(logprobs_mode):
    server = _serve(logprobs_mode)
    try:
        model_config = _get(server, "/server_info?config_format=json")[1]["vllm_config"][
            "model_config"
        ]
    finally:
        server.stop()
    assert model_config["dtype"] == "torch.bfloat16"
    assert model_config["logprobs_mode"] == (logprobs_mode or "raw_logprobs")
    assert model_config["return_sampling_mask"] is False


def test_trl_check_passes_with_processed_logprobs():
    print("unsloth server module:", server_module.__file__)
    server = _serve("processed_logprobs")
    try:
        check = _run_trl_check(server, temperature = 0.7)
    finally:
        server.stop()
    print("trl check:", inspect.getsourcefile(check))


@pytest.mark.parametrize("logprobs_mode", ["raw_logprobs", None])
def test_trl_check_raises_its_own_error_with_raw_logprobs(logprobs_mode):
    _reads_logprobs_mode(_check_sampling_replay())
    server = _serve(logprobs_mode)
    try:
        with pytest.raises(RuntimeError, match = "--logprobs-mode processed_logprobs"):
            _run_trl_check(server, temperature = 0.7)
    finally:
        server.stop()
