# NaviCache for Open-Sora

This directory provides the Open-Sora NaviCache evaluation code. It contains:

- `videosys/`: the VideoSys-based Open-Sora inference framework used by the evaluation script.
- `eval/`: generation, VBench, PSNR, LPIPS, and SSIM evaluation utilities.

## Installation

Prerequisites:

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.6

We recommend using Anaconda to create a clean environment:

```shell
conda create -n navicache python=3.10 -y
conda activate navicache
```

Install NaviCache from the repository root:

```shell
git clone https://github.com/HelloZicky/NaviCache
cd NaviCache
pip install -r requirements.txt
```

## Open-Sora Weights

The Open-Sora evaluation script uses the following Hugging Face model ids by default:

```text
hpcai-tech/OpenSora-STDiT-v3
hpcai-tech/OpenSora-VAE-v1.2
DeepFloyd/t5-v1_1-xxl
```

For offline or more reproducible evaluation, download the weights before running:

```shell
mkdir -p checkpoints/opensora

huggingface-cli download hpcai-tech/OpenSora-STDiT-v3 \
    --local-dir checkpoints/opensora/OpenSora-STDiT-v3

huggingface-cli download hpcai-tech/OpenSora-VAE-v1.2 \
    --local-dir checkpoints/opensora/OpenSora-VAE-v1.2

huggingface-cli download DeepFloyd/t5-v1_1-xxl \
    --local-dir checkpoints/opensora/t5-v1_1-xxl
```

If Hugging Face access is slow in your region, set a mirror endpoint before running the same commands:

```shell
export HF_ENDPOINT=https://hf-mirror.com
```

## Evaluation

The evaluation workflow first generates videos from VBench prompts, then computes VBench, PSNR, LPIPS, and SSIM scores.

### Generate Videos

Run from the repository root:

```shell
python navicache4opensora/eval/navicache/experiments/opensora.py \
    --transformer_path checkpoints/opensora/OpenSora-STDiT-v3 \
    --vae_path checkpoints/opensora/OpenSora-VAE-v1.2 \
    --text_encoder_path checkpoints/opensora/t5-v1_1-xxl \
    --prompt_path navicache4opensora/eval/navicache/vbench/VBench_full_info.json \
    --navicache_thresh 0.35 \
    --navicache_align_steps 5
```

By default, `opensora.py` uses `--loop 5` and generates five random videos for each prompt, following the VBench evaluation protocol.

The script reads prompts from:

```text
navicache4opensora/eval/navicache/vbench/VBench_full_info.json
```

and saves videos to:

```text
navicache4opensora/eval/navicache/samples/opensora_base
navicache4opensora/eval/navicache/samples/opensora_navicache
```

To run only one generation branch:

```shell
python navicache4opensora/eval/navicache/experiments/opensora.py --mode base
python navicache4opensora/eval/navicache/experiments/opensora.py --mode navicache
```

NaviCache hyperparameters can be changed directly from the command line.

### Calculate VBench

```shell
python navicache4opensora/eval/navicache/vbench/run_vbench.py \
    --video_path navicache4opensora/eval/navicache/samples/opensora_navicache \
    --save_path navicache4opensora/eval/navicache/vbench_results/navicache

python navicache4opensora/eval/navicache/vbench/cal_vbench.py \
    --score_dir navicache4opensora/eval/navicache/vbench_results/navicache
```

### Calculate PSNR, LPIPS, and SSIM

```shell
python navicache4opensora/eval/navicache/common_metrics/eval.py \
    --gt_video_dir navicache4opensora/eval/navicache/samples/opensora_base \
    --generated_video_dir navicache4opensora/eval/navicache/samples/opensora_navicache
```

## Acknowledgements

We would like to thank the contributors to [Open-Sora](https://github.com/hpcaitech/Open-Sora) and [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys).
