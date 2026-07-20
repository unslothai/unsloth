    def test_rocm_pin_family_mismatch_helper(self):
        """_rocm_pin_family_mismatch: exact rocm compare, else the 2.11 line."""
        f = stack_mod._rocm_pin_family_mismatch
        base = "https://download.pytorch.org/whl"
        amd = "https://repo.amd.com/rocm/whl"
        # Exact rocm version comparison.
        assert f(f"{base}/rocm7.2", "2.11.0+rocm7.2") is False
        assert f(f"{base}/rocm7.2", "2.10.0+rocm6.4") is True
        assert f(f"{base}/rocm6.4", "2.10.0+rocm6.4") is False
        # rocm7.2 is KNOWN-2.11. A +rocm7.2 wheel whose RELEASE drifted off 2.11 shares the
        # tag but violates the spec -> mismatch (a plain version compare would accept it).
        assert f(f"{base}/rocm7.2", "2.12.0+rocm7.2") is True
        assert f(f"{base}/rocm7.2", "2.13.0+rocm7.2") is True
        assert f(f"{base}/rocm7.2", "2.11.5+rocm7.2") is False  # patch on 2.11 is in-spec
        # An UNKNOWN newer rocm (not on the 2.11 allowlist) is not floored to 2.11, so a
        # matching rocm version at any release line is NOT a mismatch on this branch.
        assert f(f"{base}/rocm8.0", "2.12.0+rocm8.0") is False
        # gfx pin (2.11 line) vs installed release line.
        assert f(f"{amd}/gfx1151", "2.10.0+rocm6.4") is True
        assert f(f"{amd}/gfx1151", "2.11.0+rocm7.13.0") is False
        # rocm7.2 pin vs an untagged (no +rocm) wheel: a CPU/CUDA build never
        # satisfies a ROCm pin, regardless of its release line -> always a mismatch.
        assert f(f"{base}/rocm7.2", "2.10.0") is True
        assert f(f"{base}/rocm7.2", "2.11.0") is True
        assert f(f"{base}/rocm6.4", "2.10.0") is True
        # A 2.11-allowlist gfx pin over a GENERIC (two-part +rocm7.2) 2.11 wheel is a
        # mismatch -- the user wants AMD's per-arch (three-part) wheel, not generic.
        assert f(f"{amd}/gfx1151", "2.11.0+rocm7.2") is True
        assert f(f"{amd}/gfx120X-all", "2.11.0+rocm7.2") is True
        # ...but an already-installed per-arch (three-part) wheel is NOT re-flagged
        # (no reinstall loop once the correct gfx wheel is present).
        assert f(f"{amd}/gfx120X-all", "2.11.0+rocm7.13.0") is False
        assert f(f"{amd}/gfx1150", "2.11.0+rocm7.13.0") is False
        # A NON-2.11 gfx pin (gfx110X-all/gfx90a/gfx908) tracks the default <2.11 spec: a
        # correct 2.10+rocm wheel is NOT a mismatch, a 2.11 build is.
        assert f(f"{amd}/gfx110X-all", "2.10.0+rocm6.4") is False
        assert f(f"{amd}/gfx90a", "2.10.0+rocm6.3") is False
        assert f(f"{amd}/gfx908", "2.10.0+rocm7.0") is False
        assert f(f"{amd}/gfx110X-all", "2.11.0+rocm7.2") is True
        # A non-2.11 gfx pin over an untagged (no +rocm) wheel is a mismatch even
        # when torch is already <2.11: a CPU/CUDA build never satisfies the ROCm pin.
        assert f(f"{amd}/gfx110X-all", "2.10.0") is True
        assert f(f"{amd}/gfx90a", "2.10.0") is True

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install_try", return_value = True)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (7, 2))
    def test_rocm_pin_mismatch_over_installed_rocm_reinstalls(
        self, mock_ver, mock_gpu, mock_nvidia, mock_pip, mock_pip_try
    ):
        """A rocm7.2 pin over an already-installed OLDER +rocm6.4 build must reinstall,
        even though has_hip_torch is True (the ROCm analogue of the CUDA cuXXX mismatch)."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        # HIP marker present (has_hip_torch=True) + installed +rocm6.4 wheel.
        mock_probe.stdout = b"6.4.12345|2.10.0+rocm6.4\n"
        env = {"UNSLOTH_TORCH_INDEX_FAMILY": "rocm7.2"}
        with patch.dict(stack_mod.os.environ, env, clear = False):
            stack_mod.os.environ.pop("UNSLOTH_TORCH_INDEX_URL", None)
            with patch("os.path.isdir", return_value = True):
                with patch("subprocess.run", return_value = mock_probe):
                    _ensure_rocm_torch()
        torch_call = str(mock_pip.call_args_list[0])
        assert "rocm7.2" in torch_call
        assert "torch>=2.11.0,<2.12.0" in torch_call

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install_try", return_value = True)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (6, 4))
    def test_gfx_pin_over_installed_pre211_rocm_reinstalls(
        self, mock_ver, mock_gpu, mock_nvidia, mock_pip, mock_pip_try
    ):
        """A gfx* pin (2.11 line) over an installed pre-2.11 +rocm6.4 build reinstalls."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"6.4.12345|2.10.0+rocm6.4\n"
        env = {"UNSLOTH_TORCH_INDEX_URL": "https://repo.amd.com/rocm/whl/gfx1151"}
        with patch.dict(stack_mod.os.environ, env, clear = False):
            stack_mod.os.environ.pop("UNSLOTH_TORCH_INDEX_FAMILY", None)
            with patch("os.path.isdir", return_value = True):
                with patch("subprocess.run", return_value = mock_probe):
                    with patch.object(
                        stack_mod, "_detect_amd_gfx_codes", side_effect = AssertionError
                    ):
                        _ensure_rocm_torch()
        torch_call = str(mock_pip.call_args_list[0])
        assert "gfx1151" in torch_call
        assert "torch>=2.11.0,<2.12.0" in torch_call

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install_try", return_value = True)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (7, 2))
    def test_rocm_pin_matches_installed_no_torch_reinstall(
        self, mock_ver, mock_gpu, mock_nvidia, mock_pip, mock_pip_try
    ):
        """A rocm7.2 pin over an already-matching +rocm7.2 build must NOT reinstall torch
        (no false reinstall of a correct ROCm venv)."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"7.2.12345|2.11.0+rocm7.2\n"
        env = {"UNSLOTH_TORCH_INDEX_FAMILY": "rocm7.2"}
        with patch.dict(stack_mod.os.environ, env, clear = False):
            stack_mod.os.environ.pop("UNSLOTH_TORCH_INDEX_URL", None)
            with patch("os.path.isdir", return_value = True):
                with patch("subprocess.run", return_value = mock_probe):
                    _ensure_rocm_torch()
        # No torch reinstall: any pip_install call must not target a torch index.
        for _call in mock_pip.call_args_list:
            _args = [str(a) for a in _call.args]
            if "--index-url" in _args:
                _url = _args[_args.index("--index-url") + 1]
                assert "rocm7.2" not in _url or "torch" not in " ".join(
                    _args
                ), "torch must not be reinstalled when the pin already matches"
        # A torch reinstall would pass torch>=... as a positional; assert none did.
        assert not any(
            any(str(a).startswith("torch") for a in _c.args) for _c in mock_pip.call_args_list
        )

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install_try", return_value = True)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (6, 4))
    def test_non211_gfx_pin_over_210_rocm_no_reinstall(
        self, mock_ver, mock_gpu, mock_nvidia, mock_pip, mock_pip_try
    ):
        """A gfx110X-all pin (NOT in the 2.11 allowlist) over a correct 2.10+rocm
        wheel must NOT be flagged stale -- the install path uses the default <2.11
        specs for that arch, so re-flagging would reinstall-loop on every update."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"6.4.12345|2.10.0+rocm6.4\n"
        env = {"UNSLOTH_TORCH_INDEX_URL": "https://repo.amd.com/rocm/whl/gfx110X-all"}
        with patch.dict(stack_mod.os.environ, env, clear = False):
            stack_mod.os.environ.pop("UNSLOTH_TORCH_INDEX_FAMILY", None)
            with patch("os.path.isdir", return_value = True):
                with patch("subprocess.run", return_value = mock_probe):
                    _ensure_rocm_torch()
        # has_hip_torch True + no mismatch -> torch must NOT be reinstalled.
        assert not any(
            any(str(a).startswith("torch") for a in _c.args) for _c in mock_pip.call_args_list
        )

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install_try", return_value = True)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (7, 2))
    def test_gfx_pin_over_generic_rocm211_reinstalls(
        self, mock_ver, mock_gpu, mock_nvidia, mock_pip, mock_pip_try
    ):
        """A gfx1151 pin over a GENERIC (two-part +rocm7.2) 2.11 wheel must reinstall
        the AMD per-arch wheel -- even though both are torch 2.11, the generic wheel
        is not the per-arch build the user pinned (Strix stays off the generic wheel)."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"7.2.12345|2.11.0+rocm7.2\n"
        env = {"UNSLOTH_TORCH_INDEX_URL": "https://repo.amd.com/rocm/whl/gfx1151"}
        with patch.dict(stack_mod.os.environ, env, clear = False):
            stack_mod.os.environ.pop("UNSLOTH_TORCH_INDEX_FAMILY", None)
            with patch("os.path.isdir", return_value = True):
                with patch("subprocess.run", return_value = mock_probe):
                    with patch.object(
                        stack_mod, "_detect_amd_gfx_codes", side_effect = AssertionError
                    ):
                        _ensure_rocm_torch()
        torch_call = str(mock_pip.call_args_list[0])
        assert "gfx1151" in torch_call
        assert "torch>=2.11.0,<2.12.0" in torch_call

