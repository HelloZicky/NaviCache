# NaviCache for Open-Sora 1.2

NaviCache is a training-free, test-time self-calibration caching method for accelerating [Open-Sora 1.2](https://github.com/hpcaitech/Open-Sora) text-to-video generation.

## Usage

Clone NaviCache, install its dependencies, and download the Open-Sora 1.2 checkpoints:

```bash
git clone https://github.com/HelloZicky/NaviCache.git
cd NaviCache
pip install -r requirements.txt

huggingface-cli download hpcai-tech/OpenSora-STDiT-v3 \
    --local-dir checkpoints/opensora/OpenSora-STDiT-v3
huggingface-cli download hpcai-tech/OpenSora-VAE-v1.2 \
    --local-dir checkpoints/opensora/OpenSora-VAE-v1.2
huggingface-cli download DeepFloyd/t5-v1_1-xxl \
    --local-dir checkpoints/opensora/t5-v1_1-xxl
```

Run the paper configuration at 51 frames, 848x480, and 30 sampling steps:

```bash
python NaviCache4OpenSora/eval/navicache/experiments/opensora.py \
    --transformer_path checkpoints/opensora/OpenSora-STDiT-v3 \
    --vae_path checkpoints/opensora/OpenSora-VAE-v1.2 \
    --text_encoder_path checkpoints/opensora/t5-v1_1-xxl \
    --prompt_path NaviCache4OpenSora/eval/navicache/vbench/VBench_full_info.json \
    --navicache_thresh 0.35 \
    --navicache_align_steps 5
```

The integration defaults select Open-Sora 1.2, 51 frames, 480p at a 9:16 aspect ratio (848x480), and 30 sampling steps.

## Results

### Visual Quality Comparison

#### Open-Sora 1.2, 51 frames, 848x480, 30 steps

| Method | Latency | Speedup | Hardware |
|:-------|--------:|--------:|:---------|
| Open-Sora 1.2 | 56.48 s | 1.00x | RTX 4090 |
| TeaCache | 41.38 s | 1.36x | RTX 4090 |
| **NaviCache** (`threshold=0.35`, `align_steps=5`) | **35.29 s** | **1.60x** | RTX 4090 |

<p align="center">
  <video src="https://github.com/user-attachments/assets/36676974-bf52-4d08-bc63-a27704687702" width="960" controls muted loop></video>
</p>

<p align="center">
  <img src="../assets/opensora_comparison/speedup/opensora12_speedup.png" width="960" alt="Open-Sora 1.2 speedup comparison" />
</p>

**Prompt:** `a train accelerating to gain speed`<br />
**Analysis:** TeaCache turns the tall green train into a low blue-and-red streak with a mismatched car body, whereas NaviCache retains the Native train, track-side composition, and left-to-right trajectory.

## Acknowledgements

We would like to thank the contributors to [Open-Sora](https://github.com/hpcaitech/Open-Sora), [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys), and [TeaCache](https://github.com/ali-vilab/TeaCache).
