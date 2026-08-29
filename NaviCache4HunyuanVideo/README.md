# NaviCache for HunyuanVideo

NaviCache is a training-free, test-time self-calibration caching method for accelerating [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) generation.

## Usage

Install HunyuanVideo following the official repository, then copy the NaviCache sampling script into it:

```bash
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo.git
git clone https://github.com/HelloZicky/NaviCache.git

cp NaviCache/NaviCache4HunyuanVideo/navicache_sample_video.py HunyuanVideo/
cd HunyuanVideo
```

Run the paper configuration at 129 frames, 960x544, and 50 sampling steps:

```bash
python3 navicache_sample_video.py \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --use-cpu-offload \
    --navicache_thresh 0.040 \
    --navicache_align_steps 5 \
    --save-path ./navicache_results
```

## Results

### Visual Quality Comparison

#### HunyuanVideo, 129 frames, 960x544, 50 steps

| Method | Latency | Speedup | Hardware |
|:-------|--------:|--------:|:---------|
| HunyuanVideo | 2363.83 s | 1.00x | NVIDIA H20 |
| TeaCache | 1070.14 s | 2.21x | NVIDIA H20 |
| **NaviCache** (`threshold=0.040`, `align_steps=5`) | **928.45 s** | **2.55x** | NVIDIA H20 |

<div align="center">
  <video src="https://github.com/user-attachments/assets/f06df5c8-ca89-49fc-a49c-82f8e6a9bcfb" width="960" controls muted loop></video>
  <br />
  <img src="../assets/hunyuanvideo_comparison/speedup/hunyuanvideo_speedup.png" width="960" alt="HunyuanVideo latency and speedup comparison" />
</div>

**Prompt:** `a book`<br />
**Analysis:** TeaCache drifts to an unrelated subject, while NaviCache stays close to the Native result.

## Acknowledgements

We would like to thank the contributors to [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) and [TeaCache](https://github.com/ali-vilab/TeaCache).
