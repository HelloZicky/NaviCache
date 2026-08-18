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

### Visual Quality Comparison

#### T2V 1.3B, 832x480

| Method | Latency | Speedup | Hardware |
|:-------|--------:|--------:|:---------|
| Wan2.1 | 214.93 s | 1.00x | RTX 4090 |
| TeaCache | 121.57 s | 1.77x | RTX 4090 |
| **NaviCache** (`threshold=0.05`, `align_steps=10`) | **106.97 s** | **2.01x** | RTX 4090 |

<div align="center">
  <video src="https://github.com/user-attachments/assets/d5d888eb-a045-43bb-bce5-da73c6ad130f" width="936" controls muted loop></video>
  <br />
  <img src="../assets/wan21_comparison/speedup/t2v13b480_speedup.png" width="936" alt="Wan2.1 T2V 1.3B latency and speedup comparison" />
</div>

<details>
<summary>Prompt: A boat sailing leisurely along the Seine River with the Eiffel Tower in background, zoom out</summary>

A boat sailing leisurely along the Seine River with the Eiffel Tower in background, zoom out

#### T2V 14B, 1280x720

| Method | Latency | Speedup | Hardware |
|:-------|--------:|--------:|:---------|
| Wan2.1 | 3483.95 s | 1.00x | NVIDIA A100-SXM4-80GB |
| TeaCache (`threshold=0.20`) | 1736.61 s | 2.01x | NVIDIA A100-SXM4-80GB |
| **NaviCache** (`threshold=0.05`, `align_steps=10`) | **1286.33 s** | **2.71x** | NVIDIA A100-SXM4-80GB |

<div align="center">
  <video src="https://github.com/user-attachments/assets/9fc996d2-1aa1-4f2d-bc0c-0431f6f5ec7a" width="960" controls muted loop></video>
  <br />
  <img src="../assets/wan21_comparison/speedup/t2v14b720_speedup.png" width="960" alt="Wan2.1 T2V 14B latency and speedup comparison" />
</div>

<details>
<summary>Prompt: A person is skateboarding</summary>

A person is skateboarding

#### I2V 14B, 832x480

| Method | Latency | Speedup | Hardware |
|:-------|--------:|--------:|:---------|
| Wan2.1 | 744.00 s | 1.00x | NVIDIA A100-SXM4-80GB |
| TeaCache (`threshold=0.10`) | 532.27 s | 1.40x | NVIDIA A100-SXM4-80GB |
| **NaviCache** (`threshold=0.05`, `align_steps=10`) | **386.70 s** | **1.92x** | NVIDIA A100-SXM4-80GB |

<div align="center">
  <video src="https://github.com/user-attachments/assets/d4ae7e82-3094-4e9c-975e-19529a88ec0e" width="936" controls muted loop></video>
  <br />
  <img src="../assets/wan21_comparison/speedup/i2v14b480_speedup.png" width="936" alt="Wan2.1 I2V 14B 480p latency and speedup comparison" />
</div>

<details>
<summary>Prompt</summary>

The ribbon dancer spins rapidly across the courtyard, both red silk ribbons tracing wide fluid arcs as her layered costume billows; the camera tracks sideways with strong natural background parallax.

#### I2V 14B, 1280x720

| Method | Latency | Speedup | Hardware |
|:-------|--------:|--------:|:---------|
| Wan2.1 | 1817.79 s | 1.00x | NVIDIA A100-SXM4-80GB |
| TeaCache (`threshold=0.15`) | 1213.66 s | 1.50x | NVIDIA A100-SXM4-80GB |
| **NaviCache** (`threshold=0.05`, `align_steps=10`) | **937.67 s** | **1.94x** | NVIDIA A100-SXM4-80GB |

<div align="center">
  <video src="https://github.com/user-attachments/assets/43bf9834-5910-48b6-baf2-0324ee409608" width="960" controls muted loop></video>
  <br />
  <img src="../assets/wan21_comparison/speedup/i2v14b720_speedup.png" width="960" alt="Wan2.1 I2V 14B 720p latency and speedup comparison" />
</div>

<details>
<summary>Prompt</summary>

The clockwork hummingbird beats both articulated wings rapidly and flies from one red flower to the next; tiny gears turn visibly while the camera arcs around it through the greenhouse with deep parallax.

## Acknowledgements

We would like to thank the contributors to [Wan2.1](https://github.com/Wan-Video/Wan2.1), [TeaCache](https://github.com/ali-vilab/TeaCache), [EasyCache](https://github.com/H-EmbodVis/EasyCache), and [MagCache](https://github.com/Zehong-Ma/MagCache).
