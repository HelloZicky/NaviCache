

import os
import sys
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

from hyvideo.modules.attenion import get_cu_seqlens
from typing import Optional, Union, Dict
import torch
import json

def navicache_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        text_states_2: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,
        return_dict: bool = True,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    torch.cuda.synchronize()
    start_time = time.time()

    out = {}
    raw_input = x.clone()
    img = x
    txt = text_states
    _, _, ot, oh, ow = x.shape
    tt, th, tw = (
        ot // self.patch_size[0],
        oh // self.patch_size[1],
        ow // self.patch_size[2],
    )

    vec = self.time_in(t)
    vec = vec + self.vector_in(text_states_2)

    if self.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + self.guidance_in(guidance)

    if self.navicache_step < self.navicache_align_steps or self.navicache_step >= self.navicache_num_steps - 1:
        should_compute = True
        self.navicache_accumulated_error = 0
    else:

        if hasattr(self, 'navicache_previous_raw_input') and hasattr(self, 'navicache_previous_output')\
                and self.navicache_previous_raw_input is not None and self.navicache_previous_output is not None:

            raw_input_change = (raw_input - self.navicache_previous_raw_input).abs().mean()

            if hasattr(self, 'navicache_state_ratio') and self.navicache_state_ratio is not None:
                self.navicache_prediction_ratio = self.navicache_state_ratio

            if hasattr(self, 'navicache_prediction_ratio') and self.navicache_prediction_ratio is not None:
                output_norm = self.navicache_previous_output.abs().mean()
                pred_change = self.navicache_prediction_ratio * (raw_input_change / output_norm)
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
        result = raw_input + self.navicache_residual
        self.navicache_step += 1

        if self.navicache_step >= self.navicache_num_steps:
            self.navicache_step = 0

        torch.cuda.synchronize()
        end_time = time.time()
        self.navicache_total_time += (end_time - start_time)

        if return_dict:
            out["x"] = result
            return out
        return result

    img = self.img_in(img)
    if self.text_projection == "linear":
        txt = self.txt_in(txt)
    elif self.text_projection == "single_refiner":
        txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
    else:
        raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

    txt_seq_len = txt.shape[1]
    img_seq_len = img.shape[1]

    cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
    cu_seqlens_kv = cu_seqlens_q
    max_seqlen_q = img_seq_len + txt_seq_len
    max_seqlen_kv = max_seqlen_q

    freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

    for _, block in enumerate(self.double_blocks):
        double_block_args = [img, txt, vec, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, freqs_cis]
        img, txt = block(*double_block_args)

    x = torch.cat((img, txt), 1)
    if len(self.single_blocks) > 0:
        for _, block in enumerate(self.single_blocks):
            single_block_args = [x, vec, txt_seq_len, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, (freqs_cos, freqs_sin)]
            x = block(*single_block_args)

    img = x[:, :img_seq_len, ...]
    img = self.final_layer(img, vec)
    result = self.unpatchify(img, tt, th, tw)

    self.navicache_residual = result - raw_input
    if hasattr(self, 'navicache_previous_output') and self.navicache_previous_output is not None:
        output_change = (result - self.navicache_previous_output).abs().mean()
        if hasattr(self, 'navicache_prior_raw_input') and self.navicache_prior_raw_input is not None:
            input_change = (self.navicache_previous_raw_input - self.navicache_prior_raw_input).abs().mean()

            z = output_change / (input_change + 1e-8)

            is_warmup = self.navicache_step < self.navicache_align_steps

            if not hasattr(self, 'navicache_state_ratio') or self.navicache_state_ratio is None or is_warmup:

                self.navicache_state_ratio = z
                self.navicache_uncertainty = 1.0
            else:

                self.navicache_uncertainty = self.navicache_uncertainty + self.navicache_process_noise

                calibrated_fusion_factor = self.navicache_uncertainty / (self.navicache_uncertainty + self.navicache_measurement_noise + 1e-8)
                self.navicache_state_ratio = self.navicache_state_ratio + calibrated_fusion_factor * (z - self.navicache_state_ratio)
                self.navicache_uncertainty = (1 - calibrated_fusion_factor) * self.navicache_uncertainty

            self.navicache_prediction_ratio = self.navicache_state_ratio

    self.navicache_prior_raw_input = getattr(self, 'navicache_previous_raw_input', None)
    self.navicache_previous_output = result.clone()

    self.navicache_step += 1
    if self.navicache_step >= self.navicache_num_steps:
        self.navicache_step = 0

    torch.cuda.synchronize()
    end_time = time.time()
    self.navicache_total_time += (end_time - start_time)

    if return_dict:
        out["x"] = result
        return out
    return result

def _extract_navicache_args():
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--navicache_thresh", type=float, default=0.025)
    parser.add_argument("--navicache_align_steps", type=int, default=5)
    parser.add_argument("--navicache_process_noise", type=float, default=0.05)
    parser.add_argument("--navicache_measurement_noise", type=float, default=0.05)
    navicache_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return navicache_args


def _configure_navicache(
        transformer,
        infer_steps,
        navicache_thresh=0.025,
        navicache_align_steps=5,
        navicache_process_noise=0.05,
        navicache_measurement_noise=0.05,
):
    transformer_class = transformer.__class__
    transformer_class.forward = navicache_forward
    transformer_class.navicache_step = 0
    transformer_class.navicache_num_steps = infer_steps
    transformer_class.navicache_thresh = navicache_thresh
    transformer_class.navicache_align_steps = navicache_align_steps
    transformer_class.navicache_prediction_ratio = None
    transformer_class.navicache_total_time = 0.0
    transformer_class.navicache_state_ratio = 0.0
    transformer_class.navicache_uncertainty = 1.0
    transformer_class.navicache_process_noise = navicache_process_noise
    transformer_class.navicache_measurement_noise = navicache_measurement_noise
    transformer_class.navicache_accumulated_error = 0
    transformer_class.navicache_previous_raw_input = None
    transformer_class.navicache_previous_output = None
    transformer_class.navicache_prior_raw_input = None
    transformer_class.navicache_residual = None

def main():
    navicache_args = _extract_navicache_args()
    args = parse_args()
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    os.makedirs(args.save_path, exist_ok=True)

    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
        models_root_path, args=args)
    args = hunyuan_video_sampler.args
    transformer = hunyuan_video_sampler.pipeline.transformer
    _configure_navicache(
        transformer,
        args.infer_steps,
        navicache_thresh=navicache_args.navicache_thresh,
        navicache_align_steps=navicache_args.navicache_align_steps,
        navicache_process_noise=navicache_args.navicache_process_noise,
        navicache_measurement_noise=navicache_args.navicache_measurement_noise,
    )

    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt,
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale
    )

    generation_time = transformer.navicache_total_time
    samples = outputs['samples']

    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            save_path = f"{args.save_path}/navicache_{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/', '')}.mp4"
            save_videos_grid(sample, save_path, fps=24)
            logger.info(f'Sample save to: {save_path}')

        time_cost = {
            "GPU_Device": torch.cuda.get_device_name(0),
            "number_prompt": 1,
            "avg_cost_time": generation_time
        }
        logger.info(
            f"GPU_Device: {time_cost['GPU_Device']}, number_prompt: {time_cost['number_prompt']}, avg_cost_time: {time_cost['avg_cost_time']}")

        time_cost_path = os.path.join(args.save_path, "time_cost.json")
        try:
            with open(time_cost_path, "r", encoding="utf-8") as time_cost_file:
                time_cost_records = json.load(time_cost_file)
        except (FileNotFoundError, json.JSONDecodeError):
            time_cost_records = []

        time_cost_records.append(time_cost)
        with open(time_cost_path, "w", encoding="utf-8") as time_cost_file:
            json.dump(time_cost_records, time_cost_file, indent=4)

if __name__ == "__main__":
    main()
