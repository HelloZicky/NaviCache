# NaviCache for Wan2.2

NaviCache is a training-free test-time self-calibration caching method for accelerating video diffusion models. This directory provides the NaviCache generation script for [Wan2.2](https://github.com/Wan-Video/Wan2.2).

## Usage

Follow [Wan2.2](https://github.com/Wan-Video/Wan2.2) to clone the repository, finish the installation, and download the Wan2.2-TI2V-5B model weights. Then copy `navicache_generate.py` from this repository to the Wan2.2 repository.

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
git clone https://github.com/HelloZicky/NaviCache.git

cp NaviCache/NaviCache4Wan2.2/navicache_generate.py Wan2.2/
cd Wan2.2
mkdir -p outputs
```

Make sure that `--ckpt_dir` points to the downloaded Wan2.2-TI2V-5B checkpoint directory.

## Text-to-Video

Run Wan2.2-TI2V-5B without an input image:

```bash
python navicache_generate.py \
    --task ti2v-5B \
    --size 1280*704 \
    --frame_num 121 \
    --ckpt_dir ./Wan2.2-TI2V-5B \
    --offload_model False \
    --convert_model_dtype \
    --navicache_thresh 0.05 \
    --navicache_align_steps 10 \
    --navicache_process_noise 0.05 \
    --navicache_measurement_noise 0.05 \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
    --save_file ./outputs/wan22_t2v_navicache.mp4
```

## Image-to-Video

Pass an input image to the same TI2V checkpoint with `--image`:

```bash
python navicache_generate.py \
    --task ti2v-5B \
    --size 1280*704 \
    --frame_num 121 \
    --ckpt_dir ./Wan2.2-TI2V-5B \
    --image examples/i2v_input.JPG \
    --offload_model False \
    --convert_model_dtype \
    --navicache_thresh 0.05 \
    --navicache_align_steps 10 \
    --navicache_process_noise 0.05 \
    --navicache_measurement_noise 0.05 \
    --prompt "A small wooden sailboat glides across a calm lake at sunrise, cinematic and detailed." \
    --save_file ./outputs/wan22_i2v_navicache.mp4
```

The command uses the sample image included in Wan2.2 at `examples/i2v_input.JPG`; replace it with your own input image as needed. Generated videos are written to the path provided by `--save_file`.

## Acknowledgements

We would like to thank the contributors to [Wan2.2](https://github.com/Wan-Video/Wan2.2).
