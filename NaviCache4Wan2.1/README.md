# NaviCache for Wan2.1

NaviCache is a training-free, test-time self-calibration caching method for accelerating [Wan2.1](https://github.com/Wan-Video/Wan2.1) text-to-video and image-to-video generation.

## Usage

Install Wan2.1 following the [official repository](https://github.com/Wan-Video/Wan2.1), then copy the NaviCache integration script into the Wan2.1 repository:

```bash
git clone https://github.com/Wan-Video/Wan2.1.git
git clone https://github.com/HelloZicky/NaviCache.git

cp NaviCache/NaviCache4Wan2.1/navicache_generate.py Wan2.1/
cd Wan2.1
```

### T2V 1.3B, 832x480

```bash
python navicache_generate.py \
    --task t2v-1.3B \
    --size 832*480 \
    --ckpt_dir ./Wan2.1-T2V-1.3B \
    --offload_model True \
    --t5_cpu \
    --base_seed 42 \
    --navicache_thresh 0.05 \
    --navicache_align_steps 10 \
    --prompt "A boat sailing leisurely along the Seine River with the Eiffel Tower in background, zoom out"
```

### T2V 14B, 1280x720

```bash
python navicache_generate.py \
    --task t2v-14B \
    --size 1280*720 \
    --ckpt_dir ./Wan2.1-T2V-14B \
    --offload_model True \
    --t5_cpu \
    --base_seed 42 \
    --navicache_thresh 0.05 \
    --navicache_align_steps 10 \
    --prompt "A person is skateboarding"
```

### I2V 14B, 832x480

```bash
python navicache_generate.py \
    --task i2v-14B \
    --size 832*480 \
    --ckpt_dir ./Wan2.1-I2V-14B-480P \
    --image examples/i2v_input.JPG \
    --offload_model True \
    --t5_cpu \
    --base_seed 42 \
    --navicache_thresh 0.05 \
    --navicache_align_steps 10 \
    --prompt "A beautiful coastal beach in spring, waves lapping on sand by Hokusai, in the style of Ukiyo"
```

### I2V 14B, 1280x720

```bash
python navicache_generate.py \
    --task i2v-14B \
    --size 1280*720 \
    --ckpt_dir ./Wan2.1-I2V-14B-720P \
    --image examples/i2v_input.JPG \
    --frame_num 61 \
    --offload_model True \
    --t5_cpu \
    --base_seed 42 \
    --navicache_thresh 0.05 \
    --navicache_align_steps 10 \
    --prompt "an orange cat"
```

## Results

### Inference Latency Comparison

| Wan2.1 configuration (92 prompts) | Resolution | Wan2.1 | NaviCache | Speedup |
|:----------------------------------|:----------:|:------:|:---------:|:-------:|
| T2V 1.3B | 832x480 | 488.0 s | 160.1 s | 3.05x |
| T2V 14B | 1280x720 | 3528.1 s | 1361.1 s | 2.59x |
| I2V 14B 480P | 832x480 | 825.9 s | 481.4 s | 1.72x |
| I2V 14B 720P | 1280x720 | 1887.2 s | 1007.9 s | 1.87x |

### Visual Quality Comparison

#### T2V 1.3B, 832x480

<div align="center">
  <video src="../assets/wan21_comparison/t2v13b480_boat_comparison.mp4" width="936" controls muted loop></video>
</div>

| Wan2.1 | TeaCache | NaviCache |
|:------:|:--------:|:---------:|
| 488.0 s (1.00x) | Visual reference | **160.1 s (3.05x)** |

<details>
<summary>Prompt: A boat sailing leisurely along the Seine River with the Eiffel Tower in background, zoom out</summary>

A boat sailing leisurely along the Seine River with the Eiffel Tower in background, zoom out

</details>

#### T2V 14B, 1280x720

<div align="center">
  <video src="../assets/wan21_comparison/t2v14b720_skateboarding_comparison.mp4" width="960" controls muted loop></video>
</div>

| Wan2.1 | TeaCache | NaviCache |
|:------:|:--------:|:---------:|
| 3528.1 s (1.00x) | Visual reference | **1361.1 s (2.59x)** |

<details>
<summary>Prompt: A person is skateboarding</summary>

A person is skateboarding

</details>

#### I2V 14B, 832x480

<div align="center">
  <video src="../assets/wan21_comparison/i2v14b480_ribbon_dancer_comparison.mp4" width="960" controls muted loop></video>
</div>

| Method | Showcase latency | Speedup vs TeaCache |
|:-------|:----------------:|:-------------------:|
| TeaCache | 678.4 s | 1.00x |
| NaviCache | **578.2 s** | **1.17x** |

<details>
<summary>Prompt</summary>

The ribbon dancer spins rapidly across the courtyard, both red silk ribbons tracing wide fluid arcs as her layered costume billows; the camera tracks sideways with strong natural background parallax.

</details>

#### I2V 14B, 1280x720

<div align="center">
  <video src="../assets/wan21_comparison/i2v14b720_clockwork_hummingbird_comparison.mp4" width="960" controls muted loop></video>
</div>

| Method | Showcase latency | Speedup vs TeaCache |
|:-------|:----------------:|:-------------------:|
| TeaCache | 1343.5 s | 1.00x |
| NaviCache | **1167.8 s** | **1.15x** |

<details>
<summary>Prompt</summary>

The clockwork hummingbird beats both articulated wings rapidly and flies from one red flower to the next; tiny gears turn visibly while the camera arcs around it through the greenhouse with deep parallax.

</details>

## Acknowledgements

We would like to thank the contributors to [Wan2.1](https://github.com/Wan-Video/Wan2.1), [TeaCache](https://github.com/ali-vilab/TeaCache), [EasyCache](https://github.com/H-EmbodVis/EasyCache), and [MagCache](https://github.com/Zehong-Ma/MagCache).
