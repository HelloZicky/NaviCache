# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Copyright 2026 The NaviCache Authors. All rights reserved.
#
# The CLI and generation flow are adapted from Wan2.2 generate.py at commit
# 42bf4cfaa384bc21833865abc2f9e6c0e67233dc (Apache License 2.0). NaviCache
# adds only the model-forward cache hook and its command-line options.

"""Generate Wan2.2-TI2V-5B videos with NaviCache.

Place this file in the root of an official Wan2.2 checkout, or invoke it by
absolute path while the current directory is the Wan2.2 repository root.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import types
from datetime import datetime

import torch
import torch.distributed as dist


DEFAULT_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight "
    "intensely on a spotlighted stage."
)


class NaviCacheConfig:
    """Prompt-local NaviCache parameters shared by the public hook and tests."""

    def __init__(
        self,
        *,
        threshold: float = 0.05,
        ret_steps: int = 10,
        kalman_q: float = 0.05,
        kalman_r: float = 0.05,
        sample_steps: int = 50,
    ) -> None:
        if threshold < 0:
            raise ValueError("navicache threshold must be non-negative")
        if ret_steps < 0:
            raise ValueError("navicache ret_steps must be non-negative")
        if sample_steps <= 0:
            raise ValueError("sample_steps must be positive")
        if ret_steps > sample_steps:
            raise ValueError("navicache ret_steps cannot exceed sample_steps")
        if kalman_q < 0 or kalman_r < 0:
            raise ValueError("Kalman Q and R must be non-negative")
        self.threshold = float(threshold)
        self.ret_steps = int(ret_steps)
        self.kalman_q = float(kalman_q)
        self.kalman_r = float(kalman_r)
        self.sample_steps = int(sample_steps)


def _mean_abs_cat(tensors):
    """Mean absolute value over a list of tensors, matching the formal runner."""
    return torch.cat([tensor.flatten() for tensor in tensors]).abs().mean()


def reset_navicache_state(model) -> None:
    """Reset every prompt-dependent NaviCache state value on ``model``."""
    model._nc_forward_count = 0
    model._nc_compute_count = 0
    model._nc_skip_count = 0
    model._nc_accumulated_error = 0.0
    model._nc_should_compute = True
    model._nc_k = None
    model._nc_kalman_k = 0.0
    model._nc_kalman_p = 1.0
    model._nc_prev_input_even = None
    model._nc_prev_prev_input_even = None
    model._nc_prev_output_even = None
    model._nc_prev_output_odd = None
    model._nc_cache_even = None
    model._nc_cache_odd = None


def _navicache_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
):
    """Wan2.2 forward hook with prompt-local residual reuse.

    Wan2.2's formal TI2V interface uses ``y`` and has no ``clip_fea``
    parameter. ``clip_fea`` remains accepted by this wrapper only so accidental
    calls fail at the native boundary rather than being forwarded incorrectly.
    """
    config = self._navicache_config
    raw_input = [tensor.clone() for tensor in x]
    is_cond_forward = self._nc_forward_count % 2 == 0

    # The conditional/even call makes one decision for the entire CFG pair.
    if is_cond_forward:
        first_alignment_forwards = config.ret_steps * 2
        final_pair_start = config.sample_steps * 2 - 2
        if (
            self._nc_forward_count < first_alignment_forwards
            or self._nc_forward_count >= final_pair_start
        ):
            self._nc_should_compute = True
            self._nc_accumulated_error = 0.0
        elif (
            self._nc_prev_input_even is not None
            and self._nc_prev_output_even is not None
            and self._nc_k is not None
        ):
            input_change = _mean_abs_cat(
                [
                    current - previous
                    for current, previous in zip(raw_input, self._nc_prev_input_even)
                ]
            )
            output_norm = _mean_abs_cat(self._nc_prev_output_even)
            self._nc_accumulated_error += (
                self._nc_k * input_change / (output_norm + 1e-8)
            )
            self._nc_should_compute = bool(
                self._nc_accumulated_error >= config.threshold
            )
            if self._nc_should_compute:
                self._nc_accumulated_error = 0.0
        else:
            self._nc_should_compute = True

        self._nc_prev_input_even = [tensor.clone() for tensor in raw_input]

    residual_cache = self._nc_cache_even if is_cond_forward else self._nc_cache_odd
    if not self._nc_should_compute and residual_cache is not None:
        self._nc_skip_count += 1
        self._nc_forward_count += 1
        return [
            (tensor + residual).float()
            for tensor, residual in zip(raw_input, residual_cache)
        ]

    # The official Wan2.2 model interface is (..., seq_len, y=None). In
    # particular, do not forward Wan2.1's incompatible clip_fea argument.
    output = self._navicache_native_forward(x, t, context, seq_len, y=y)
    self._nc_compute_count += 1

    if is_cond_forward:
        if (
            self._nc_prev_output_even is not None
            and self._nc_prev_prev_input_even is not None
        ):
            output_change = _mean_abs_cat(
                [
                    current - previous
                    for current, previous in zip(output, self._nc_prev_output_even)
                ]
            )
            input_change = _mean_abs_cat(
                [
                    current - previous
                    for current, previous in zip(
                        self._nc_prev_input_even,
                        self._nc_prev_prev_input_even,
                    )
                ]
            )
            observation = output_change / (input_change + 1e-8)
            if self._nc_kalman_k is None:
                self._nc_kalman_k = observation
                self._nc_kalman_p = 1.0
            else:
                self._nc_kalman_p += config.kalman_q
                gain = self._nc_kalman_p / (
                    self._nc_kalman_p + config.kalman_r + 1e-8
                )
                self._nc_kalman_k = self._nc_kalman_k + gain * (
                    observation - self._nc_kalman_k
                )
                self._nc_kalman_p = (1.0 - gain) * self._nc_kalman_p
            self._nc_k = self._nc_kalman_k

        self._nc_prev_prev_input_even = [
            tensor.clone() for tensor in self._nc_prev_input_even
        ]
        self._nc_prev_output_even = [tensor.clone() for tensor in output]
        self._nc_cache_even = [
            current - raw
            for current, raw in zip(output, raw_input)
        ]
    else:
        self._nc_prev_output_odd = [tensor.clone() for tensor in output]
        self._nc_cache_odd = [
            current - raw
            for current, raw in zip(output, raw_input)
        ]

    self._nc_forward_count += 1
    return output


def install_navicache(model, config: NaviCacheConfig) -> None:
    """Install an instance-local NaviCache hook on an official Wan2.2 model."""
    if not hasattr(model, "_navicache_native_forward"):
        model._navicache_native_forward = model.forward
        model.forward = types.MethodType(_navicache_forward, model)
    model._navicache_config = config
    reset_navicache_state(model)


def navicache_stats(model) -> dict[str, object]:
    """Return lightweight generation statistics without writing trace files."""
    return {
        "compute_forwards": model._nc_compute_count,
        "skip_forwards": model._nc_skip_count,
        "kalman_k_final": (
            float(model._nc_k.detach().cpu())
            if torch.is_tensor(model._nc_k)
            else model._nc_k
        ),
    }


def _str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a video with NaviCache for Wan2.2-TI2V-5B."
    )
    parser.add_argument("--task", default="ti2v-5B", choices=["ti2v-5B"])
    parser.add_argument("--size", default="1280*704", choices=["1280*704", "704*1280"])
    parser.add_argument("--frame_num", type=int, default=121)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--offload_model", type=_str2bool, default=None)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--save_file", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--image", default=None)
    parser.add_argument("--sample_solver", default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--convert_model_dtype", action="store_true", default=False)
    parser.add_argument(
        "--navicache_thresh",
        type=float,
        default=0.05,
        help="Accumulated normalized predicted-error threshold.",
    )
    parser.add_argument(
        "--navicache_ret_steps",
        type=int,
        default=10,
        help="Number of initial denoising steps forced to compute.",
    )
    parser.add_argument(
        "--navicache_kalman_q",
        type=float,
        default=0.05,
        help="Kalman process-noise covariance Q.",
    )
    parser.add_argument(
        "--navicache_kalman_r",
        type=float,
        default=0.05,
        help="Kalman measurement-noise covariance R.",
    )
    return parser


def _validate_args(args) -> None:
    if args.frame_num <= 0 or (args.frame_num - 1) % 4 != 0:
        raise ValueError("frame_num must be 4n+1")
    if args.sample_steps <= 0:
        raise ValueError("sample_steps must be positive")
    if args.base_seed < 0:
        args.base_seed = random.randint(0, sys.maxsize)
    if args.offload_model is None:
        args.offload_model = False if int(os.getenv("WORLD_SIZE", "1")) > 1 else True


def generate(args) -> None:
    """Run the official WanTI2V pipeline with the NaviCache model hook."""
    _validate_args(args)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size != 1:
        raise NotImplementedError(
            "This public Wan2.2 NaviCache entry is validated for single-GPU inference only."
        )

    # Delayed imports keep module import and --help usable before Wan2.2 is
    # installed. Generation itself must run from an official Wan2.2 checkout.
    try:
        import wan
        from PIL import Image
        from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
        from wan.utils.utils import save_video
    except ImportError as exc:
        raise RuntimeError(
            "Official Wan2.2 is not importable. Run this script from the root "
            "of an installed Wan2.2 checkout."
        ) from exc

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )
    cfg = WAN_CONFIGS[args.task]
    image = Image.open(args.image).convert("RGB") if args.image else None
    logging.info("Generation job args: %s", args)
    logging.info("Creating WanTI2V pipeline.")
    pipeline = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )
    install_navicache(
        pipeline.model,
        NaviCacheConfig(
            threshold=args.navicache_thresh,
            ret_steps=args.navicache_ret_steps,
            kalman_q=args.navicache_kalman_q,
            kalman_r=args.navicache_kalman_r,
            sample_steps=args.sample_steps,
        ),
    )

    logging.info("Generating video with NaviCache.")
    video = pipeline.generate(
        args.prompt,
        img=image,
        size=SIZE_CONFIGS[args.size],
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model,
    )
    expected_forwards = args.sample_steps * 2
    if pipeline.model._nc_forward_count != expected_forwards:
        raise RuntimeError(
            f"unexpected CFG forward count: {pipeline.model._nc_forward_count} "
            f"(expected {expected_forwards})"
        )

    if args.save_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
        args.save_file = (
            f"{args.task}_{args.size}_{safe_prompt}_navicache_{timestamp}.mp4"
        )
    logging.info("Saving generated video to %s", args.save_file)
    save_video(
        tensor=video[None],
        save_file=args.save_file,
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    logging.info("NaviCache stats: %s", navicache_stats(pipeline.model))
    del video
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    logging.info("Finished.")


def main() -> None:
    args = build_parser().parse_args()
    generate(args)


if __name__ == "__main__":
    main()
