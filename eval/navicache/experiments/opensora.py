import sys
import argparse
from functools import partial
from pathlib import Path

import torch
from einops import rearrange

ROOT_DIR = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils import generate_func, read_prompt_list
from videosys import OpenSoraConfig, VideoSysEngine
from videosys.core.comm import gather_sequence, get_pad, set_pad, split_sequence
from videosys.models.transformers.open_sora_transformer_3d import auto_grad_checkpoint
from videosys.utils.utils import batch_func


def navicache_forward(
    self,
    x,
    timestep,
    all_timesteps,
    y,
    mask=None,
    x_mask=None,
    fps=None,
    height=None,
    width=None,
    **kwargs,
):
    # === Split batch ===
    if self.parallel_manager.cp_size > 1:
        x, timestep, y, x_mask, mask = batch_func(
            partial(split_sequence, process_group=self.parallel_manager.cp_group, dim=0),
            x,
            timestep,
            y,
            x_mask,
            mask,
        )

    dtype = self.x_embedder.proj.weight.dtype
    B = x.size(0)
    x = x.to(dtype)
    timestep = timestep.to(dtype)
    y = y.to(dtype)

    # === get pos embed ===
    _, _, Tx, Hx, Wx = x.size()
    T, H, W = self.get_dynamic_size(x)
    S = H * W
    base_size = round(S**0.5)
    resolution_sq = (height[0].item() * width[0].item()) ** 0.5
    scale = resolution_sq / self.input_sq_size
    pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

    # === get timestep embed ===
    t = self.t_embedder(timestep, dtype=x.dtype)
    fps = self.fps_embedder(fps.unsqueeze(1), B)
    t = t + fps
    t_mlp = self.t_block(t)
    t0 = t0_mlp = None
    if x_mask is not None:
        t0_timestep = torch.zeros_like(timestep)
        t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
        t0 = t0 + fps
        t0_mlp = self.t_block(t0)

    # === get y embed ===
    if self.config.skip_y_embedder:
        y_lens = mask
        if isinstance(y_lens, torch.Tensor):
            y_lens = y_lens.long().tolist()
    else:
        y, y_lens = self.encode_text(y, mask)

    # === get x embed ===
    x = self.x_embedder(x)
    x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
    x = x + pos_emb

    if self.parallel_manager.sp_size > 1:
        set_pad("temporal", T, self.parallel_manager.sp_group)
        set_pad("spatial", S, self.parallel_manager.sp_group)
        x = split_sequence(x, self.parallel_manager.sp_group, dim=1, grad_scale="down", pad=get_pad("temporal"))
        T = x.shape[1]
        x_mask_org = x_mask
        x_mask = split_sequence(
            x_mask,
            self.parallel_manager.sp_group,
            dim=1,
            grad_scale="down",
            pad=get_pad("temporal"),
        )

    x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)
    raw_input = x.clone()

    if self.navicache_step < self.navicache_align_steps or self.navicache_step >= self.navicache_num_steps - 1:
        should_compute = True
        self.navicache_accumulated_error = 0
    else:
        if (
            hasattr(self, "navicache_previous_raw_input")
            and hasattr(self, "navicache_previous_output")
            and self.navicache_previous_raw_input is not None
            and self.navicache_previous_output is not None
        ):
            raw_input_change = (raw_input - self.navicache_previous_raw_input).abs().mean()

            if hasattr(self, "navicache_state_ratio") and self.navicache_state_ratio is not None:
                self.navicache_prediction_ratio = self.navicache_state_ratio

            if hasattr(self, "navicache_prediction_ratio") and self.navicache_prediction_ratio is not None:
                output_norm = self.navicache_previous_output.abs().mean()
                pred_change = self.navicache_prediction_ratio * (raw_input_change / (output_norm + 1e-8))
                self.navicache_accumulated_error += pred_change

                if self.navicache_accumulated_error < self.navicache_thresh:
                    should_compute = False
                else:
                    should_compute = True
                    self.navicache_accumulated_error = 0
            else:
                should_compute = True
        else:
            should_compute = True

    self.navicache_previous_raw_input = raw_input.clone()

    if not should_compute and self.navicache_residual is not None:
        x = raw_input + self.navicache_residual
        self.navicache_skipped_steps += 1
    else:
        for spatial_block, temporal_block in zip(self.spatial_blocks, self.temporal_blocks):
            x = auto_grad_checkpoint(
                spatial_block,
                x,
                y,
                t_mlp,
                y_lens,
                x_mask,
                t0_mlp,
                T,
                S,
                timestep,
                all_timesteps=all_timesteps,
            )

            x = auto_grad_checkpoint(
                temporal_block,
                x,
                y,
                t_mlp,
                y_lens,
                x_mask,
                t0_mlp,
                T,
                S,
                timestep,
                all_timesteps=all_timesteps,
            )

        self.navicache_residual = x - raw_input
        if self.navicache_previous_output is not None:
            output_change = (x - self.navicache_previous_output).abs().mean()
            if self.navicache_prior_raw_input is not None:
                input_change = (self.navicache_previous_raw_input - self.navicache_prior_raw_input).abs().mean()
                z = output_change / (input_change + 1e-8)
                is_warmup = self.navicache_step < self.navicache_align_steps

                if self.navicache_state_ratio is None or is_warmup:
                    self.navicache_state_ratio = z
                    self.navicache_uncertainty = 1.0
                else:
                    self.navicache_uncertainty = self.navicache_uncertainty + self.navicache_process_noise
                    calibrated_fusion_factor = self.navicache_uncertainty / (
                        self.navicache_uncertainty + self.navicache_measurement_noise + 1e-8
                    )
                    self.navicache_state_ratio = self.navicache_state_ratio + calibrated_fusion_factor * (
                        z - self.navicache_state_ratio
                    )
                    self.navicache_uncertainty = (1 - calibrated_fusion_factor) * self.navicache_uncertainty

                self.navicache_prediction_ratio = self.navicache_state_ratio

        self.navicache_prior_raw_input = self.navicache_previous_raw_input
        self.navicache_previous_output = x.clone()

    self.navicache_step += 1
    if self.navicache_step >= self.navicache_num_steps:
        self.navicache_step = 0
        self.navicache_accumulated_error = 0
        self.navicache_previous_raw_input = None
        self.navicache_previous_output = None
        self.navicache_prior_raw_input = None
        self.navicache_residual = None

    if self.parallel_manager.sp_size > 1:
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = gather_sequence(x, self.parallel_manager.sp_group, dim=1, grad_scale="up", pad=get_pad("temporal"))
        T, S = x.shape[1], x.shape[2]
        x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)
        x_mask = x_mask_org

    # === final layer ===
    x = self.final_layer(x, t, x_mask, t0, T, S)
    x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)

    x = x.to(torch.float32)

    # === Gather Output ===
    if self.parallel_manager.cp_size > 1:
        x = gather_sequence(x, self.parallel_manager.cp_group, dim=0)

    return x


def configure_navicache(
    transformer,
    num_sampling_steps,
    navicache_thresh=0.35,
    navicache_align_steps=5,
    navicache_process_noise=0.05,
    navicache_measurement_noise=0.05,
):
    transformer_class = transformer.__class__
    transformer_class.forward = navicache_forward
    transformer_class.enable_navicache = True
    transformer_class.navicache_step = 0
    transformer_class.navicache_num_steps = num_sampling_steps
    transformer_class.navicache_thresh = navicache_thresh
    transformer_class.navicache_align_steps = navicache_align_steps
    transformer_class.navicache_prediction_ratio = None
    transformer_class.navicache_state_ratio = None
    transformer_class.navicache_uncertainty = 1.0
    transformer_class.navicache_process_noise = navicache_process_noise
    transformer_class.navicache_measurement_noise = navicache_measurement_noise
    transformer_class.navicache_accumulated_error = 0
    transformer_class.navicache_previous_raw_input = None
    transformer_class.navicache_previous_output = None
    transformer_class.navicache_prior_raw_input = None
    transformer_class.navicache_residual = None
    transformer_class.navicache_skipped_steps = 0


def build_opensora_config(args):
    return OpenSoraConfig(args.transformer_path, args.vae_path, args.text_encoder_path)


def eval_base(prompt_list, args):
    config = build_opensora_config(args)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, args.base_output_dir, loop=args.loop)


def eval_ours(prompt_list, args):
    config = build_opensora_config(args)
    engine = VideoSysEngine(config)
    configure_navicache(
        engine.driver_worker.transformer,
        num_sampling_steps=config.num_sampling_steps,
        navicache_thresh=args.navicache_thresh,
        navicache_align_steps=args.navicache_align_steps,
        navicache_process_noise=args.navicache_process_noise,
        navicache_measurement_noise=args.navicache_measurement_noise,
    )
    generate_func(engine, prompt_list, args.navicache_output_dir, loop=args.loop)


def eval_navicache(prompt_list, args):
    eval_ours(prompt_list, args)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Open-Sora videos with NaviCache.")
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=str(EVAL_DIR / "vbench" / "VBench_full_info.json"),
        help="Path to a VBench-style prompt JSON file.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Optional first prompt index, useful for debugging or sharded generation.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=-1,
        help="Optional exclusive end prompt index. Use -1 to run to the end.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "base", "navicache"],
        default="all",
        help="Which generation pass to run.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(EVAL_DIR / "samples"),
        help="Directory for generated videos.",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=5,
        help="Number of random generations per prompt. VBench evaluation uses five independent runs for each prompt by default.",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default="hpcai-tech/OpenSora-STDiT-v3",
        help="Open-Sora STDiT checkpoint path or Hugging Face repository id.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="hpcai-tech/OpenSora-VAE-v1.2",
        help="Open-Sora VAE checkpoint path or Hugging Face repository id.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="DeepFloyd/t5-v1_1-xxl",
        help="T5 text encoder path or Hugging Face repository id.",
    )
    parser.add_argument(
        "--navicache_thresh",
        type=float,
        default=0.35,
        help="Accumulated predicted-error threshold. Larger values increase cache reuse.",
    )
    parser.add_argument(
        "--navicache_align_steps",
        type=int,
        default=5,
        help="Number of initial diffusion steps computed for alignment.",
    )
    parser.add_argument(
        "--navicache_process_noise",
        type=float,
        default=0.05,
        help="Process-noise covariance used by NaviCache state estimation.",
    )
    parser.add_argument(
        "--navicache_measurement_noise",
        type=float,
        default=0.05,
        help="Measurement-noise covariance used by NaviCache state estimation.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    args.base_output_dir = str(output_dir / "opensora_base")
    args.navicache_output_dir = str(output_dir / "opensora_navicache")
    return args


if __name__ == "__main__":
    args = parse_args()

    prompt_list = read_prompt_list(args.prompt_path)
    if args.end_index != -1:
        prompt_list = prompt_list[args.start_index : args.end_index]
    else:
        prompt_list = prompt_list[args.start_index :]

    if args.mode in ("all", "base"):
        eval_base(prompt_list, args)
    if args.mode in ("all", "navicache"):
        eval_ours(prompt_list, args)
