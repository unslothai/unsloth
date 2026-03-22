# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI argument parsing tests for unsloth-cli.py."""

from pathlib import Path
import importlib.util

import pytest


def _load_cli_module():
    root = Path(__file__).resolve().parents[1]
    cli_path = root / "unsloth-cli.py"
    spec = importlib.util.spec_from_file_location("unsloth_cli", cli_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_defaults_asft():
    cli = _load_cli_module()
    parser = cli.build_parser()
    args = parser.parse_args([])

    assert args.asft is False
    assert args.asft_mode == "asft"
    assert args.kl_weight == 0.0
    assert args.reference_policy == "disable_adapter"
    assert args.asft_streaming == "off"
    assert args.ref_microbatch_size is None
    assert args.seq_chunk_size is None


def test_cli_asft_streaming_flag_defaults_auto():
    cli = _load_cli_module()
    parser = cli.build_parser()
    args = parser.parse_args(["--asft_streaming"])

    assert args.asft_streaming == "auto"


def test_cli_asft_streaming_value():
    cli = _load_cli_module()
    parser = cli.build_parser()
    args = parser.parse_args(["--asft_streaming", "batch"])

    assert args.asft_streaming == "batch"


def test_cli_asft_options_parsed():
    cli = _load_cli_module()
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--asft",
            "--asft_mode",
            "sft+kl",
            "--kl_weight",
            "0.2",
            "--reference_policy",
            "frozen_copy",
        ]
    )

    assert args.asft is True
    assert args.asft_mode == "sft+kl"
    assert args.kl_weight == pytest.approx(0.2)
    assert args.reference_policy == "frozen_copy"
