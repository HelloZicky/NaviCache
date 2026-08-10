# NaviCache for Wan2.2-TI2V-5B

NaviCache adds training-free, test-time self-calibrated DiT residual caching to the official Wan2.2-TI2V-5B inference pipeline.

## Usage

The integration targets the official Wan2.2 code at commit `42bf4cfaa384bc21833865abc2f9e6c0e67233dc` and is validated on one GPU.

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
git clone https://github.com/HelloZicky/NaviCache.git

cd Wan2.2
pip install -r requirements.txt
cp ../NaviCache/NaviCache4Wan2.2/navicache_generate.py .
mkdir -p outputs
```

Download the official `Wan2.2-TI2V-5B` checkpoint, then run the balanced configuration:

```bash
python navicache_generate.py \
  --task ti2v-5B \
  --size '1280*704' \
  --frame_num 121 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --sample_solver unipc \
  --sample_steps 50 \
  --sample_shift 5.0 \
  --sample_guide_scale 5.0 \
  --base_seed 42 \
  --offload_model False \
  --convert_model_dtype \
  --navicache_thresh 0.05 \
  --navicache_ret_steps 10 \
  --navicache_kalman_q 0.05 \
  --navicache_kalman_r 0.05 \
  --save_file ./outputs/wan22_ti2v_navicache.mp4
```

Add `--image /path/to/input.png` to use the image-conditioned path of the TI2V checkpoint. Without `--image`, the official pipeline uses its text-to-video path.

| NaviCache option | Default | Description |
|---|---:|---|
| `--navicache_thresh` | `0.05` | Accumulated normalized predicted-error threshold; larger values generally skip more DiT forwards. |
| `--navicache_ret_steps` | `10` | Initial denoising steps that always compute both CFG branches. |
| `--navicache_kalman_q` | `0.05` | Process-noise covariance for online ratio estimation. |
| `--navicache_kalman_r` | `0.05` | Measurement-noise covariance for online ratio estimation. |

The public entry point keeps the official scheduler, CFG combination, seed handling, VAE decode, and video writer unchanged. Its cache hook matches Wan2.2's native `forward(..., y=None)` interface and deliberately does not forward Wan2.1's incompatible `clip_fea` argument. Prompt-dependent cache and Kalman state is reset before each generation.

## Results

All published values below were recomputed from complete raw records for the same 92 prompts and settings: seed 42, 1280×704, 121 frames, 24 fps, 50 UniPC steps, shift 5.0, and guidance scale 5.0. Both runs produced 92 unique successes with no missing or duplicate prompt rows, and all 92 MP4 files decoded successfully.

### Inference Latency Comparison

| Method | Prompts | DiT/CFG latency (s/video) ↓ | Speedup ↑ |
|---|---:|---:|---:|
| Native Wan2.2-TI2V-5B | 92 | 332.66 | 1.00× |
| NaviCache | 92 | **142.83** | **2.33×** |

Latency is the CUDA-event time inside the sampling callback: two model forwards and the CFG combination at each step. It excludes text encoding, scheduler updates, VAE decoding, and file writing. Speedup is `native_latency / method_latency` using unrounded means (`332.658710562452 / 142.82547465633886`).

### Visual Quality Comparison

The NaviCache outputs were compared against the matching native outputs over the same complete 92-prompt set.

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| NaviCache | 28.4491 | 0.9285 | 0.0451 |

Only this fully validated balanced configuration is reported. Partial fast/slow sweeps are intentionally excluded.

## Limitations

- The public entry point is validated for single-GPU inference. It rejects distributed execution rather than silently changing behavior.
- Results above apply to the exact model, resolution, sampling settings, hardware environment, and timing boundary described here; they are not end-to-end wall-clock measurements.
- Cache decisions are prompt-local. Call `reset_navicache_state(model)` before reusing a manually installed hook for another generation.

## Acknowledgements

This integration is adapted from the official [Wan2.2](https://github.com/Wan-Video/Wan2.2) TI2V pipeline under the Apache License 2.0. We also thank the authors of [EasyCache](https://github.com/H-EmbodVis/EasyCache), TeaCache, and MagCache for their open-source caching research. NaviCache's public hook was independently rewritten from a provenance-tracked internal runner and retains the repository's Apache License 2.0 terms.
