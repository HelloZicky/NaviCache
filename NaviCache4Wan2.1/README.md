# NaviCache for Wan2.1

NaviCache is a training-free test-time self-calibration caching method for accelerating video diffusion models. This directory provides the NaviCache generation script for [Wan2.1](https://github.com/Wan-Video/Wan2.1).

## Usage

Follow [Wan2.1](https://github.com/Wan-Video/Wan2.1) to clone the repo, finish the installation, and download the required model weights. Then copy `navicache_generate.py` in this repo to the Wan2.1 repo.

```bash
git clone https://github.com/Wan-Video/Wan2.1.git
git clone https://github.com/HelloZicky/NaviCache.git

cp NaviCache/NaviCache4Wan2.1/navicache_generate.py Wan2.1/
cd Wan2.1
```

Please make sure that `--ckpt_dir` points to the matching Wan2.1 checkpoint directory. The T2V-14B and T2V-1.3B models use different weights. The I2V 480P and I2V 720P models also use different weights. 

## Text-to-Video

For T2V with the 1.3B model:

```bash
python navicache_generate.py \
    --task t2v-1.3B \
    --size 832*480 \
    --ckpt_dir ./Wan2.1-T2V-1.3B \
    --offload_model True \
    --t5_cpu \
    --navicache_thresh 0.05 \
    --navicache_align_steps 10 \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

For T2V with the 14B model:

```bash
python navicache_generate.py \
    --task t2v-14B \
    --size 1280*720 \
    --ckpt_dir ./Wan2.1-T2V-14B \
    --offload_model True \
    --t5_cpu \
    --navicache_thresh 0.05 \
    --navicache_align_steps 10 \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

## Image-to-Video

For I2V with the 480P model:

```bash
python navicache_generate.py \
    --task i2v-14B \
    --size 832*480 \
    --ckpt_dir ./Wan2.1-I2V-14B-480P \
    --image examples/i2v_input.JPG \
    --offload_model True \
    --t5_cpu \
    --navicache_thresh 0.05 \
    --navicache_align_steps 10 \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds."
```

For I2V with the 720P model:

```bash
python navicache_generate.py \
    --task i2v-14B \
    --size 1280*720 \
    --ckpt_dir ./Wan2.1-I2V-14B-720P \
    --image examples/i2v_input.JPG \
    --offload_model True \
    --t5_cpu \
    --navicache_thresh 0.05 \
    --navicache_align_steps 10 \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds."
```

The I2V commands use the sample image included in Wan2.1 at `examples/i2v_input.JPG`; replace it with your own input image as needed.Generated files are written to `outputs/` unless `--save_file` is specified. Generation timing records are appended to `output/time_cost.json` by default. Use `--out_dir` to choose another timing-output directory.

## Acknowledgements

We would like to thank the contributors to [Wan2.1](https://github.com/Wan-Video/Wan2.1).
