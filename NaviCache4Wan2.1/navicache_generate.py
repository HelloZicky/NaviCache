# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Copyright 2026 The NaviCache Authors. All rights reserved.

import argparse
from datetime import datetime
import json
import logging
import os
import sys

import warnings
from time import time

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

import gc
from contextlib import contextmanager
import torchvision.transforms.functional as TF
import torch.cuda.amp as amp
import numpy as np
import math
from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                  get_sampling_sigmas, retrieve_timesteps)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from tqdm import tqdm

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "A simple and elegant beauty",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}

def t2v_generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
    r"""
    Generates video frames from text prompt using diffusion process.

    Args:
        input_prompt (`str`):
            Text prompt for content generation
        size (tupele[`int`], *optional*, defaults to (1280,720)):
            Controls video resolution, (width,height).
        frame_num (`int`, *optional*, defaults to 81):
            How many frames to sample from a video. The number should be 4n+1
        shift (`float`, *optional*, defaults to 5.0):
            Noise schedule shift parameter. Affects temporal dynamics
        sample_solver (`str`, *optional*, defaults to 'unipc'):
            Solver used to sample the video.
        sampling_steps (`int`, *optional*, defaults to 40):
            Number of diffusion sampling steps. Higher values improve quality but slow generation
        guide_scale (`float`, *optional*, defaults 5.0):
            Classifier-free guidance scale. Controls prompt adherence vs. creativity
        n_prompt (`str`, *optional*, defaults to ""):
            Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
        seed (`int`, *optional*, defaults to -1):
            Random seed for noise generation. If -1, use random seed.
        offload_model (`bool`, *optional*, defaults to True):
            If True, offloads models to CPU during generation to save VRAM

    Returns:
        torch.Tensor:
            Generated video frames tensor. Dimensions: (C, N H, W) where:
            - C: Color channels (3 for RGB)
            - N: Number of frames (81)
            - H: Frame height (from size)
            - W: Frame width from size)
    """

    F = frame_num
    target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                    size[1] // self.vae_stride[1],
                    size[0] // self.vae_stride[2])

    seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (self.patch_size[1] * self.patch_size[2]) *
                        target_shape[1] / self.sp_size) * self.sp_size

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)

    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device('cpu'))
        context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        context = [t.to(self.device) for t in context]
        context_null = [t.to(self.device) for t in context_null]

    noise = [
        torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g)
    ]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, 'no_sync', noop_no_sync)

    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        latents = noise

        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}

        for _, t in enumerate(tqdm(timesteps)):
            torch.cuda.synchronize()
            start_time = time()
            latent_model_input = latents
            timestep = [t]

            timestep = torch.stack(timestep)

            self.model.to(self.device)
            noise_pred_cond = self.model(
                latent_model_input, t=timestep, **arg_c)[0]
            noise_pred_uncond = self.model(
                latent_model_input, t=timestep, **arg_null)[0]

            noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

            torch.cuda.synchronize()
            self.cost_time += (time() - start_time)

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g)[0]
            latents = [temp_x0.squeeze(0)]

        x0 = latents
        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()
        if self.rank == 0:
            videos = self.vae.decode(x0)

    del noise, latents
    del sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return videos[0] if self.rank == 0 else None

def i2v_generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
    r"""
    Generates video frames from input image and text prompt using diffusion process.

    Args:
        input_prompt (`str`):
            Text prompt for content generation.
        img (PIL.Image.Image):
            Input image tensor. Shape: [3, H, W]
        max_area (`int`, *optional*, defaults to 720*1280):
            Maximum pixel area for latent space calculation. Controls video resolution scaling
        frame_num (`int`, *optional*, defaults to 81):
            How many frames to sample from a video. The number should be 4n+1
        shift (`float`, *optional*, defaults to 5.0):
            Noise schedule shift parameter. Affects temporal dynamics
            [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
        sample_solver (`str`, *optional*, defaults to 'unipc'):
            Solver used to sample the video.
        sampling_steps (`int`, *optional*, defaults to 40):
            Number of diffusion sampling steps. Higher values improve quality but slow generation
        guide_scale (`float`, *optional*, defaults 5.0):
            Classifier-free guidance scale. Controls prompt adherence vs. creativity
        n_prompt (`str`, *optional*, defaults to ""):
            Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
        seed (`int`, *optional*, defaults to -1):
            Random seed for noise generation. If -1, use random seed
        offload_model (`bool`, *optional*, defaults to True):
            If True, offloads models to CPU during generation to save VRAM

    Returns:
        torch.Tensor:
            Generated video frames tensor. Dimensions: (C, N H, W) where:
            - C: Color channels (3 for RGB)
            - N: Number of frames (81)
            - H: Frame height (from max_area)
            - W: Frame width from max_area)
    """
    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

    F = frame_num
    h, w = img.shape[1:]
    aspect_ratio = h / w
    lat_h = round(
        np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
        self.patch_size[1] * self.patch_size[1])
    lat_w = round(
        np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
        self.patch_size[2] * self.patch_size[2])
    h = lat_h * self.vae_stride[1]
    w = lat_w * self.vae_stride[2]

    max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
    max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)
    noise = torch.randn(
        self.vae.model.z_dim,
        (F - 1) // self.vae_stride[0] + 1,
        lat_h,
        lat_w,
        dtype=torch.float32,
        generator=seed_g,
        device=self.device)

    msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
    msk[:, 1:] = 0
    msk = torch.concat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
    ],
        dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt

    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device('cpu'))
        context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        context = [t.to(self.device) for t in context]
        context_null = [t.to(self.device) for t in context_null]

    self.clip.model.to(self.device)
    clip_context = self.clip.visual([img[:, None, :, :]])
    if offload_model:
        self.clip.model.cpu()

    y = self.vae.encode([
        torch.concat([
            torch.nn.functional.interpolate(
                img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                0, 1),
            torch.zeros(3, F - 1, h, w)
        ],
            dim=1).to(self.device)
    ])[0]
    y = torch.concat([msk, y])

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, 'no_sync', noop_no_sync)

    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        latent = noise

        arg_c = {
            'context': [context[0]],
            'clip_fea': clip_context,
            'seq_len': max_seq_len,
            'y': [y],

        }

        arg_null = {
            'context': context_null,
            'clip_fea': clip_context,
            'seq_len': max_seq_len,
            'y': [y],

        }

        if offload_model:
            torch.cuda.empty_cache()

        self.model.to(self.device)
        for _, t in enumerate(tqdm(timesteps)):
            torch.cuda.synchronize()
            start_time = time()
            latent_model_input = [latent.to(self.device)]
            timestep = [t]

            timestep = torch.stack(timestep).to(self.device)

            noise_pred_cond = self.model(
                latent_model_input, t=timestep, **arg_c)[0].to(
                torch.device('cpu') if offload_model else self.device)
            if offload_model:
                torch.cuda.empty_cache()
            noise_pred_uncond = self.model(
                latent_model_input, t=timestep, **arg_null)[0].to(
                torch.device('cpu') if offload_model else self.device)
            if offload_model:
                torch.cuda.empty_cache()

            noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

            latent = latent.to(
                torch.device('cpu') if offload_model else self.device)

            torch.cuda.synchronize()
            self.cost_time += (time() - start_time)

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latent.unsqueeze(0),
                return_dict=False,
                generator=seed_g)[0]
            latent = temp_x0.squeeze(0)

            x0 = [latent.to(self.device)]
            del latent_model_input, timestep

        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()

        if self.rank == 0:
            videos = self.vae.decode(x0)

    del noise, latent
    del sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return videos[0] if self.rank == 0 else None

def navicache_forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
):
    """
    NaviCache forward pass for the Wan diffusion model.

    Args:
        x (List[Tensor]): Input video latent tensors with shape [C_in, F, H, W].
        t (Tensor): Diffusion timesteps with shape [B].
        context (List[Tensor]): Text embeddings with shape [L, C].
        seq_len (int): Maximum sequence length for positional encoding.
        clip_fea (Tensor, optional): CLIP image features for image-to-video mode.
        y (List[Tensor], optional): Conditional video inputs for image-to-video mode.

    Returns:
        List[Tensor]: Denoised video tensors.
    """
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None

    # Preserve the raw latent input for residual reuse on skipped forwards.
    raw_input = [u.clone() for u in x]

    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # CFG invokes conditional and unconditional forwards as pairs. NaviCache
    # decides on the conditional forward and reuses that decision for the pair.
    self.navicache_is_cond_forward = (self.navicache_forward_count % 2 == 0)

    if self.navicache_is_cond_forward:

        if self.navicache_forward_count < self.navicache_align_forwards or self.navicache_forward_count >= self.navicache_cutoff_forwards:
            self.navicache_should_compute_pair = True
            self.navicache_accumulated_error = 0
        else:

            if hasattr(self, 'navicache_previous_raw_cond_input') and hasattr(self, 'navicache_previous_raw_cond_output') and\
                    self.navicache_previous_raw_cond_input is not None and self.navicache_previous_raw_cond_output is not None:

                raw_input_change = torch.cat([
                    (u - v).flatten() for u, v in zip(raw_input, self.navicache_previous_raw_cond_input)
                ]).abs().mean()

                if hasattr(self, 'navicache_state_ratio') and self.navicache_state_ratio is not None:

                    self.navicache_prediction_ratio = self.navicache_state_ratio

                if hasattr(self, 'navicache_prediction_ratio') and self.navicache_prediction_ratio is not None:

                    output_norm = torch.cat([
                        u.flatten() for u in self.navicache_previous_raw_cond_output
                    ]).abs().mean()

                    pred_change = self.navicache_prediction_ratio * (raw_input_change / output_norm)
                    combined_pred_change = pred_change

                    if not hasattr(self, 'navicache_accumulated_error'):
                        self.navicache_accumulated_error = 0
                    self.navicache_accumulated_error += combined_pred_change

                    if self.navicache_accumulated_error < self.navicache_thresh:
                        self.navicache_should_compute_pair = False
                    else:
                        self.navicache_should_compute_pair = True
                        self.navicache_accumulated_error = 0
                else:

                    self.navicache_should_compute_pair = True
            else:

                self.navicache_should_compute_pair = True

        self.navicache_previous_raw_cond_input = [u.clone() for u in raw_input]

    if self.navicache_is_cond_forward and not self.navicache_should_compute_pair and\
            hasattr(self, 'navicache_previous_raw_cond_output') and self.navicache_previous_raw_cond_output is not None:

        self.navicache_forward_count += 1
        if self.navicache_forward_count >= self.navicache_num_forwards: self.navicache_forward_count = 0

        return [(u + v).float() for u, v in zip(raw_input, self.navicache_cond_residual)]

    elif not self.navicache_is_cond_forward and not self.navicache_should_compute_pair and\
            hasattr(self, 'navicache_previous_raw_uncond_output') and self.navicache_previous_raw_uncond_output is not None:

        self.navicache_forward_count += 1
        if self.navicache_forward_count >= self.navicache_num_forwards: self.navicache_forward_count = 0

        return [(u + v).float() for u, v in zip(raw_input, self.navicache_uncond_residual)]

    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                  dim=1) for u in x
    ])

    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)

    for block in self.blocks:
        x = block(x, **kwargs)

    x = self.head(x, e)
    output = self.unpatchify(x, grid_sizes)

    if self.navicache_is_cond_forward:

        if hasattr(self, 'navicache_previous_raw_cond_output') and self.navicache_previous_raw_cond_output is not None:

            output_change = torch.cat([
                (u - v).flatten() for u, v in zip(output, self.navicache_previous_raw_cond_output)
            ]).abs().mean()

            if hasattr(self, 'navicache_prior_raw_cond_input') and self.navicache_prior_raw_cond_input is not None:

                input_change = torch.cat([
                    (u - v).flatten() for u, v in zip(
                        self.navicache_previous_raw_cond_input, self.navicache_prior_raw_cond_input
                    )
                ]).abs().mean()

                z = output_change / (input_change + 1e-8)

                is_warmup = self.navicache_forward_count < self.navicache_align_forwards

                if not hasattr(self, 'navicache_state_ratio') or self.navicache_state_ratio is None or is_warmup:
                    self.navicache_state_ratio = z
                    self.navicache_uncertainty = 1.0
                else:

                    self.navicache_uncertainty = self.navicache_uncertainty + self.navicache_process_noise

                    calibrated_fusion_factor = self.navicache_uncertainty / (self.navicache_uncertainty + self.navicache_measurement_noise + 1e-8)

                    self.navicache_state_ratio = self.navicache_state_ratio + calibrated_fusion_factor * (z - self.navicache_state_ratio)

                    self.navicache_uncertainty = (1 - calibrated_fusion_factor) * self.navicache_uncertainty

                self.navicache_prediction_ratio = self.navicache_state_ratio

        self.navicache_prior_raw_cond_input = getattr(self, 'navicache_previous_raw_cond_input', None)

        self.navicache_previous_raw_cond_output = [u.clone() for u in output]

        self.navicache_cond_residual = [u - v for u, v in zip(output, raw_input)]

    else:

        self.navicache_previous_raw_uncond_output = [u.clone() for u in output]
        self.navicache_uncond_residual = [u - v for u, v in zip(output, raw_input)]

    self.navicache_forward_count += 1
    if self.navicache_forward_count >= self.navicache_num_forwards:
        self.navicache_forward_count = 0
        self.navicache_skipped_cond_steps = []
        self.navicache_skipped_uncond_steps = []

    return [u.float() for u in output]

def _configure_navicache(model, args):
    """Attach NaviCache runtime state to a Wan diffusion model."""
    model_class = model.__class__
    model_class.forward = navicache_forward
    model_class.navicache_forward_count = 0
    model_class.navicache_num_forwards = args.sample_steps * 2
    model_class.navicache_num_steps = args.sample_steps
    model_class.navicache_thresh = args.navicache_thresh
    model_class.navicache_align_steps = args.navicache_align_steps
    model_class.navicache_align_forwards = args.navicache_align_steps * 2
    model_class.navicache_cutoff_forwards = args.sample_steps * 2 - 2
    model_class.navicache_accumulated_error = 0
    model_class.navicache_should_compute_pair = True
    model_class.navicache_prediction_ratio = None
    model_class.navicache_state_ratio = 0.0
    model_class.navicache_uncertainty = 1.0
    model_class.navicache_process_noise = args.navicache_process_noise
    model_class.navicache_measurement_noise = args.navicache_measurement_noise
    model_class.navicache_previous_raw_cond_input = None
    model_class.navicache_previous_raw_cond_output = None
    model_class.navicache_previous_raw_uncond_output = None
    model_class.navicache_prior_raw_cond_input = None
    model_class.navicache_cond_residual = None
    model_class.navicache_uncond_residual = None
    model_class.navicache_skipped_cond_steps = []
    model_class.navicache_skipped_uncond_steps = []

def _validate_args(args):

    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)

    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an image or video with NaviCache for Wan2.1"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-1.3B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="832*480",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )

    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="ch",
        choices=["ch", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--navicache_thresh",
        type=float,
        default=0.05,
        help="Accumulated predicted-error threshold. Larger values increase cache reuse.")
    parser.add_argument(
        "--navicache_align_steps",
        default=10,
        type=int,
        help="Number of initial diffusion steps computed for alignment.")
    parser.add_argument(
        "--navicache_process_noise",
        default=0.05,
        type=float,
        help="Process-noise covariance used by NaviCache state estimation.")
    parser.add_argument(
        "--navicache_measurement_noise",
        default=0.05,
        type=float,
        help="Measurement-noise covariance used by NaviCache state estimation.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./output",
        help="Directory used for time_cost.json.")

    args = parser.parse_args()

    _validate_args(args)

    return args

def _init_logging(rank):

    if rank == 0:

        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
                args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
                args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel,
                                             init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model, is_vl="i2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if "t2v" in args.task or "t2i" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        logging.info(f"Input prompt: {args.prompt}")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        generation_time = []
        time_cost = {"GPU_Device": torch.cuda.get_device_name(0), "number_prompt": None, "avg_cost_time": None}
        wan_t2v.__class__.generate = t2v_generate
        _configure_navicache(wan_t2v.model, args)
        wan_t2v.cost_time = 0

        print(
            f"Generating {'image' if 't2i' in args.task else 'video'} ...")

        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
        generation_time.append(wan_t2v.cost_time)
        if rank == 0:
            if args.save_file is None:
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
                suffix = '.png' if "t2i" in args.task else '.mp4'
                args.save_file = os.path.join("outputs", f"{args.task}_navicache_thresh{args.navicache_thresh}_step{args.sample_steps}_{formatted_prompt}_{formatted_time}" + suffix)
                os.makedirs(os.path.dirname(args.save_file), exist_ok=True)

            if "t2i" in args.task:
                logging.info(f"Saving generated image to {args.save_file}")
                cache_image(
                    tensor=video.squeeze(1)[None],
                    save_file=args.save_file,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
            else:
                logging.info(f"Saving generated video to {args.save_file}")
                cache_video(
                    tensor=video[None],
                    save_file=args.save_file,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
        logging.info("Finished.")

    else:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None:
            args.image = EXAMPLE_PROMPT[args.task]["image"]
        print(f"Input prompt: {args.prompt}")
        print(f"Input image: {args.image}")

        img = Image.open(args.image).convert("RGB")
        if args.use_prompt_extend:
            print("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    image=img,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    print(
                        f"Extending prompt failed: {prompt_output.message}")
                    print("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            print(f"Extended prompt: {args.prompt}")

        print("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )
        generation_time = []
        time_cost = {"GPU_Device": torch.cuda.get_device_name(0), "number_prompt": None, "avg_cost_time": None}
        wan_i2v.__class__.generate = i2v_generate
        _configure_navicache(wan_i2v.model, args)
        wan_i2v.cost_time = 0

        print("Generating video ...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
        generation_time.append(wan_i2v.cost_time)

        if rank == 0:
            if args.save_file is None:
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
                suffix = '.png' if "t2i" in args.task else '.mp4'
                args.save_file = os.path.join("outputs", f"{args.task}_navicache_thresh{args.navicache_thresh}_step{args.sample_steps}_{formatted_prompt}_{formatted_time}" + suffix)
                os.makedirs(os.path.dirname(args.save_file), exist_ok=True)

            if "t2i" in args.task:
                logging.info(f"Saving generated image to {args.save_file}")
                cache_image(
                    tensor=video.squeeze(1)[None],
                    save_file=args.save_file,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
            else:
                logging.info(f"Saving generated video to {args.save_file}")
                cache_video(
                    tensor=video[None],
                    save_file=args.save_file,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
        logging.info("Finished.")

    time_cost["number_prompt"] = len(generation_time)
    time_cost["avg_cost_time"] = sum(generation_time) / (len(generation_time)) if len(generation_time) > 0 else 0

    print(
        f"GPU_Device:{time_cost['GPU_Device']}, number_prompt: {time_cost['number_prompt']}, avg_cost_time: {time_cost['avg_cost_time']}")

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        time_cost_path = os.path.join(args.out_dir, "time_cost.json")
        try:
            with open(time_cost_path, "r", encoding="utf-8") as time_cost_file:
                time_cost_records = json.load(time_cost_file)
        except (FileNotFoundError, json.JSONDecodeError):
            time_cost_records = []

        time_cost_records.append(time_cost)
        with open(time_cost_path, "w", encoding="utf-8") as time_cost_file:
            json.dump(time_cost_records, time_cost_file, indent=4)

if __name__ == "__main__":
    args = _parse_args()
    generate(args)
