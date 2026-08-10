from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "NaviCache4Wan2.2" / "navicache_generate.py"


def load_public_module():
    spec = importlib.util.spec_from_file_location("navicache_wan22_public", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeModel:
    def __init__(self):
        self.native_calls = []
        self.forward = self.native_forward

    def native_forward(self, x, t, context, seq_len, y=None):
        branch_offset = 10.0 if context == "cond" else 20.0
        self.native_calls.append({"context": context, "y": y})
        return [item + branch_offset for item in x]


def config(module, **overrides):
    values = {
        "threshold": 0.05,
        "ret_steps": 0,
        "kalman_q": 0.05,
        "kalman_r": 0.05,
        "sample_steps": 4,
    }
    values.update(overrides)
    return module.NaviCacheConfig(**values)


def call_pair(model, value, y=None):
    raw = [torch.tensor([float(value)])]
    cond = model.forward(raw, None, "cond", 1, y=y)[0]
    uncond = model.forward(raw, None, "uncond", 1, y=y)[0]
    return cond, uncond


def test_reset_restores_every_prompt_local_state():
    module = load_public_module()
    model = FakeModel()
    module.install_navicache(model, config(module))
    module.reset_navicache_state(model)

    model._nc_forward_count = 7
    model._nc_accumulated_error = 3.0
    model._nc_k = torch.tensor(2.0)
    model._nc_kalman_k = torch.tensor(4.0)
    model._nc_kalman_p = torch.tensor(5.0)
    model._nc_cache_even = [torch.tensor([1.0])]
    model._nc_cache_odd = [torch.tensor([2.0])]
    model._nc_prev_input_even = [torch.tensor([3.0])]
    model._nc_prev_prev_input_even = [torch.tensor([4.0])]
    model._nc_prev_output_even = [torch.tensor([5.0])]
    model._nc_prev_output_odd = [torch.tensor([6.0])]

    module.reset_navicache_state(model)

    assert model._nc_forward_count == 0
    assert model._nc_compute_count == 0
    assert model._nc_skip_count == 0
    assert model._nc_accumulated_error == 0.0
    assert model._nc_should_compute is True
    assert model._nc_k is None
    assert model._nc_kalman_k == 0.0
    assert model._nc_kalman_p == 1.0
    assert model._nc_cache_even is None and model._nc_cache_odd is None
    assert model._nc_prev_input_even is None
    assert model._nc_prev_prev_input_even is None
    assert model._nc_prev_output_even is None
    assert model._nc_prev_output_odd is None


def test_ret_steps_force_both_cfg_forwards_to_compute():
    module = load_public_module()
    model = FakeModel()
    module.install_navicache(model, config(module, ret_steps=2, threshold=100.0))
    module.reset_navicache_state(model)

    call_pair(model, 1.0)
    call_pair(model, 2.0)

    assert len(model.native_calls) == 4
    assert model._nc_compute_count == 4
    assert model._nc_skip_count == 0


def test_final_cfg_pair_is_forced_to_compute():
    module = load_public_module()
    model = FakeModel()
    module.install_navicache(model, config(module, threshold=100.0))
    module.reset_navicache_state(model)
    model._nc_forward_count = 6
    model._nc_should_compute = False
    model._nc_cache_even = [torch.tensor([10.0])]
    model._nc_cache_odd = [torch.tensor([20.0])]

    call_pair(model, 4.0)

    assert len(model.native_calls) == 2
    assert model._nc_compute_count == 2
    assert model._nc_skip_count == 0


def test_conditional_decision_controls_pair_and_branch_caches_stay_separate():
    module = load_public_module()
    model = FakeModel()
    module.install_navicache(model, config(module, threshold=100.0))
    module.reset_navicache_state(model)

    call_pair(model, 1.0)
    call_pair(model, 2.0)
    cond, uncond = call_pair(model, 3.0)

    assert len(model.native_calls) == 4
    assert model._nc_compute_count == 4
    assert model._nc_skip_count == 2
    assert torch.equal(cond, torch.tensor([13.0]))
    assert torch.equal(uncond, torch.tensor([23.0]))


def test_accumulated_error_uses_normalized_predicted_change():
    module = load_public_module()
    model = FakeModel()
    module.install_navicache(model, config(module, threshold=0.9))
    module.reset_navicache_state(model)
    model._nc_forward_count = 2
    model._nc_accumulated_error = torch.tensor(0.1)
    model._nc_k = torch.tensor(2.0)
    model._nc_prev_input_even = [torch.tensor([1.0])]
    model._nc_prev_output_even = [torch.tensor([4.0])]
    model._nc_cache_even = [torch.tensor([10.0])]
    model._nc_cache_odd = [torch.tensor([20.0])]

    call_pair(model, 2.0)

    assert model._nc_accumulated_error.item() == pytest.approx(0.6)
    assert model._nc_skip_count == 2
    assert len(model.native_calls) == 0


def test_exact_compute_corrects_kalman_from_k_zero_and_p_one():
    module = load_public_module()
    model = FakeModel()
    options = config(module, threshold=0.0)
    module.install_navicache(model, options)
    module.reset_navicache_state(model)

    call_pair(model, 1.0)
    call_pair(model, 2.0)

    expected_gain = 1.05 / (1.05 + 0.05 + 1e-8)
    assert float(model._nc_kalman_k) == pytest.approx(expected_gain)
    assert float(model._nc_kalman_p) == pytest.approx((1.0 - expected_gain) * 1.05)
    assert float(model._nc_k) == pytest.approx(expected_gain)


def test_threshold_zero_matches_native_and_never_skips():
    module = load_public_module()
    model = FakeModel()
    module.install_navicache(model, config(module, threshold=0.0))
    module.reset_navicache_state(model)

    outputs = [call_pair(model, step + 1.0) for step in range(4)]

    expected = [
        (torch.tensor([step + 10.0]), torch.tensor([step + 20.0]))
        for step in range(1, 5)
    ]
    assert all(
        torch.equal(actual_cond, expected_cond)
        and torch.equal(actual_uncond, expected_uncond)
        for (actual_cond, actual_uncond), (expected_cond, expected_uncond) in zip(outputs, expected)
    )
    assert len(model.native_calls) == 8
    assert model._nc_skip_count == 0


def test_consecutive_prompts_do_not_inherit_cache_or_kalman_state():
    module = load_public_module()
    model = FakeModel()
    module.install_navicache(model, config(module, threshold=100.0))
    module.reset_navicache_state(model)
    call_pair(model, 1.0)
    call_pair(model, 2.0)
    call_pair(model, 3.0)
    assert model._nc_skip_count == 2

    module.reset_navicache_state(model)
    call_pair(model, 100.0)

    assert model._nc_forward_count == 2
    assert model._nc_compute_count == 2
    assert model._nc_skip_count == 0
    assert model._nc_cache_even is not None and model._nc_cache_odd is not None


def test_native_forward_receives_y_keyword_and_not_clip_fea():
    module = load_public_module()
    model = FakeModel()
    module.install_navicache(model, config(module, threshold=0.0))
    module.reset_navicache_state(model)
    y = [torch.tensor([9.0])]

    call_pair(model, 1.0, y=y)

    assert [call["y"] for call in model.native_calls] == [y, y]


def test_public_cli_defaults_match_verified_configuration():
    module = load_public_module()
    parser = module.build_parser()
    args = parser.parse_args(["--ckpt_dir", "weights"])

    assert args.task == "ti2v-5B"
    assert args.size == "1280*704"
    assert args.frame_num == 121
    assert args.sample_steps == 50
    assert args.sample_shift == 5.0
    assert args.sample_guide_scale == 5.0
    assert args.base_seed == 42
    assert args.navicache_thresh == 0.05
    assert args.navicache_align_steps == 10
    assert args.navicache_process_noise == 0.05
    assert args.navicache_measurement_noise == 0.05


def test_public_cli_maps_repository_standard_navicache_names():
    module = load_public_module()
    args = module.build_parser().parse_args(
        [
            "--ckpt_dir",
            "weights",
            "--navicache_thresh",
            "0.07",
            "--navicache_align_steps",
            "8",
            "--navicache_process_noise",
            "0.03",
            "--navicache_measurement_noise",
            "0.04",
        ]
    )

    assert args.navicache_thresh == 0.07
    assert args.navicache_align_steps == 8
    assert args.navicache_process_noise == 0.03
    assert args.navicache_measurement_noise == 0.04


@pytest.mark.parametrize(
    "removed_name",
    [
        "--navicache_ret_steps",
        "--navicache_kalman_q",
        "--navicache_kalman_r",
    ],
)
def test_public_cli_rejects_removed_wan22_only_names(removed_name):
    module = load_public_module()
    parser = module.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--ckpt_dir", "weights", removed_name, "1"])
