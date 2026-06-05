# NaviCache for HunyuanVideo

NaviCache is a training-free test-time self-calibration caching method for accelerating video generation. This directory provides the NaviCache sampling script for [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo).

## Usage

Follow [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) to clone the repo, finish the installation, and download the required model weights. Then copy `navicache_sample_video.py` in this repo to the HunyuanVideo repo.

```bash
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo.git
git clone https://github.com/HelloZicky/NaviCache.git

cp NaviCache/NaviCache4HunyuanVideo/navicache_sample_video.py HunyuanVideo/
cd HunyuanVideo
```

For 544 x 960 generation, set `--video-size` to `544 960`:

```bash
python3 navicache_sample_video.py \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --use-cpu-offload \
    --navicache_thresh 0.025 \
    --navicache_align_steps 5 \
    --save-path ./navicache_results
```

For 720 x 1280 generation, set `--video-size` to `720 1280`:

```bash
python3 navicache_sample_video.py \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --use-cpu-offload \
    --navicache_thresh 0.025 \
    --navicache_align_steps 5 \
    --save-path ./navicache_results
```

The generated video and `time_cost.json` are saved under `--save-path`.

## Acknowledgements

We would like to thank the contributors to [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo).
