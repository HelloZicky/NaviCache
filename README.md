# 🚀 NaviCache: Test-Time Self-Calibration Caching for Video Generation (ICML 2026)

[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-lightgrey?logo=arxiv)](#)
[![Code](https://img.shields.io/badge/Code-Available-green?logo=github)](#)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

This is the official repository for **NaviCache: Test-Time Self-Calibration Caching for Video Generation**, accepted to **ICML 2026**.

**NaviCache** is a training-free, offline calibration-free, test-time self-calibration caching method for accelerating video diffusion models. It calibrates itself during inference, tracks feature evolution with a lightweight state-space estimator, and adaptively decides whether to skip or compute DiT blocks.

## ✨ Highlights

- **No offline calibration**: no calibration datasets, pre-processing, or per-model fitting.
- **Test-time self-calibration**: the cache updates its feature-change estimator during inference.
- **Plug-and-play acceleration**: lightweight integration for Wan2.1, HunyuanVideo, and Open-Sora.
- **Adaptive computation allocation**: skip/update decisions are controlled by an uncertainty-aware gate.
- **Strong speed-quality trade-off**: multiple presets are provided for fast, mid, and slow modes.

## 🔔 News

- **[May 2026]** NaviCache was accepted to ICML 2026.
- **[Coming Soon]** arXiv preprint.
- **[Coming Soon]** Project page and demo videos.

## 📖 Citation

If you find NaviCache useful, please consider citing:

```bibtex
@inproceedings{lv2026navicache,
  author    = {Zheqi Lv and Zhibo Zhu and Jinke Wang and Qi Tian and Shengyu Zhang and Zhengyu Chen and Chengxi Zang and Zhou Zhao and Fei Wu},
  title     = {NaviCache: Test-Time Self-Calibration Caching for Video Generation},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
  publisher = {PMLR}
}
```

## 🖼️ Figures

The following paper figures are included as PDF files under [`assets/`](assets/). No PNG previews are required.

| Figure | Description | Link |
|---|---|---|
| Figure 1 | Prediction accuracy comparison and offline calibration-free/test-time self-calibration characteristics | [PDF](assets/introduction.pdf) |
| Figure 2 | Overview of the NaviCache framework | [PDF](assets/method.pdf) |
| Figure 3 | Video generation case study | [PDF](assets/exp_video_case_brief.pdf) |
| Figure 4 | Skip frequency and compute ratio across timesteps | [PDF](assets/exp_request_freq.pdf) |

## 🎬 Video Case Study

We provide the generated videos from the Wan2.1 case study for direct comparison.

| Method | Video |
|---|---|
| Wan2.1 | [wan.mp4](https://github.com/HelloZicky/NaviCache/blob/main/assets/case/wan.mp4) |
| TeaCache | [wan_teacache.mp4](https://github.com/HelloZicky/NaviCache/blob/main/assets/case/wan_teacache.mp4) |
| MagCache | [wan_magcache.mp4](https://github.com/HelloZicky/NaviCache/blob/main/assets/case/wan_magcache.mp4) |
| EasyCache | [wan_easycache.mp4](https://github.com/HelloZicky/NaviCache/blob/main/assets/case/wan_easycache.mp4) |
| NaviCache | [wan_navicache.mp4](https://github.com/HelloZicky/NaviCache/blob/main/assets/case/wan_navicache.mp4) |


For easier preview and quick visual comparison, we also provide clickable GIF previews below. Click each GIF to open the corresponding full video.

<table>
  <tr>
    <td align="center"><b>Wan2.1</b></td>
    <td align="center"><b>TeaCache</b></td>
    <td align="center"><b>MagCache</b></td>
    <td align="center"><b>EasyCache</b></td>
    <td align="center"><b>NaviCache</b></td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/HelloZicky/NaviCache/blob/main/assets/case/wan.mp4">
        <img src="assets/case/wan.gif" width="160">
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/HelloZicky/NaviCache/blob/main/assets/case/wan_teacache.mp4">
        <img src="assets/case/wan_teacache.gif" width="160">
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/HelloZicky/NaviCache/blob/main/assets/case/wan_magcache.mp4">
        <img src="assets/case/wan_magcache.gif" width="160">
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/HelloZicky/NaviCache/blob/main/assets/case/wan_easycache.mp4">
        <img src="assets/case/wan_easycache.gif" width="160">
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/HelloZicky/NaviCache/blob/main/assets/case/wan_navicache.mp4">
        <img src="assets/case/wan_navicache.gif" width="160">
      </a>
    </td>
  </tr>
</table>



## 🧩 Supported Models

| Model | Task | NaviCache entry point | Example scripts |
|---|---|---|---|
| Wan2.1 | Text-to-Video / Image-to-Video | `NaviCache4Wan2.1/navicache_generate.py` | `scripts/wan/` |
| HunyuanVideo | Text-to-Video | `NaviCache4HunyuanVideo/navicache_sample_video.py` | `scripts/hunyuan/` |
| Open-Sora 1.2 | Text-to-Video / Evaluation | `eval/navicache/experiments/opensora.py` | `scripts/opensora/` |

## ⚙️ Installation

```bash
git clone https://github.com/HelloZicky/NaviCache.git
cd NaviCache
pip install -r requirements.txt
```

For Wan2.1 and HunyuanVideo, please first install the corresponding official repository and download the required model weights. The helper scripts below copy the NaviCache entry script into the official repository automatically when executed from the official repository directory.

## 🚀 Quick Start with Scripts

The `scripts/` directory contains three top-level launchers. Each top-level launcher calls model-specific run scripts in its corresponding subdirectory.

```text
scripts/
├── navicache_wan.sh
├── navicache_hunyuan.sh
├── navicache_opensora.sh
├── wan/
│   ├── run_wan_t2v_1.3b.sh
│   └── run_wan_i2v_480p.sh
├── hunyuan/
│   ├── run_hunyuan_544p.sh
│   └── run_hunyuan_720p.sh
└── opensora/
    ├── run_opensora_generate.sh
    ├── run_opensora_vbench.sh
    └── run_opensora_common_metrics.sh
```

### Wan2.1

Run from the official Wan2.1 repository directory:

```bash
git clone https://github.com/Wan-Video/Wan2.1.git
git clone https://github.com/HelloZicky/NaviCache.git

cd Wan2.1
bash ../NaviCache/scripts/navicache_wan.sh
```

To run a single Wan example instead of the top-level launcher:

```bash
cd Wan2.1
bash ../NaviCache/scripts/wan/run_wan_t2v_1.3b.sh
bash ../NaviCache/scripts/wan/run_wan_i2v_480p.sh
```

### HunyuanVideo

Run from the official HunyuanVideo repository directory:

```bash
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo.git
git clone https://github.com/HelloZicky/NaviCache.git

cd HunyuanVideo
bash ../NaviCache/scripts/navicache_hunyuan.sh
```

To run a single HunyuanVideo example instead of the top-level launcher:

```bash
cd HunyuanVideo
bash ../NaviCache/scripts/hunyuan/run_hunyuan_544p.sh
bash ../NaviCache/scripts/hunyuan/run_hunyuan_720p.sh
```

### Open-Sora Evaluation

Run from the NaviCache repository root:

```bash
cd NaviCache
bash scripts/navicache_opensora.sh
```

To run one Open-Sora stage at a time:

```bash
cd NaviCache
bash scripts/opensora/run_opensora_generate.sh
bash scripts/opensora/run_opensora_vbench.sh
bash scripts/opensora/run_opensora_common_metrics.sh
```

## 🔧 Recommended Hyperparameters

| Model | Mode | `NAVICACHE_THRESH` | `NAVICACHE_ALIGN_STEPS` |
|---|---|---:|---:|
| Wan2.1 | fast | `0.07` | `10` |
| Wan2.1 | mid | `0.05` | `10` |
| Wan2.1 | slow | `0.04` | `10` |
| HunyuanVideo | fast | `0.040` | `5` |
| HunyuanVideo | mid | `0.035` | `5` |
| HunyuanVideo | slow | `0.025` | `5` |
| Open-Sora 1.2 | fast | `0.55` | `5` |
| Open-Sora 1.2 | mid | `0.35` | `5` |
| Open-Sora 1.2 | slow | `0.15` | `5` |

All shell scripts expose common options through environment variables. For example:

```bash
cd Wan2.1
NAVICACHE_THRESH=0.07 NAVICACHE_ALIGN_STEPS=10 bash ../NaviCache/scripts/wan/run_wan_t2v_1.3b.sh
```

```bash
cd HunyuanVideo
PROMPT="A cinematic shot of a corgi running through a snowy forest." bash ../NaviCache/scripts/hunyuan/run_hunyuan_544p.sh
```

## 📊 Results

NaviCache provides a strong speed-quality trade-off across multiple video diffusion backbones.

| Model | Setting | Latency | Speedup | Notes |
|---|---|---:|---:|---|
| Wan2.1-1.3B | NaviCache-fast | 96.40 s | 2.23× | Fastest Wan2.1 setting |
| Wan2.1-1.3B | NaviCache-mid | 106.97 s | 2.01× | Balanced speed and fidelity |
| Wan2.1-1.3B | NaviCache-slow | 115.86 s | 1.86× | Highest visual retention among Wan presets |
| HunyuanVideo | NaviCache-fast | 928.45 s | 2.55× | Strong acceleration on long videos |
| HunyuanVideo | NaviCache-mid | 1089.43 s | 2.17× | Better PSNR/LPIPS than EasyCache at similar latency |
| Open-Sora 1.2 | NaviCache-mid | 35.29 s | 1.60× | Balanced Open-Sora setting |

Please refer to the paper for the full comparison with PAB, TeaCache, MagCache, and EasyCache.

## 🧠 Method Overview

NaviCache reformulates feature caching in video diffusion models as a test-time state estimation problem.

1. **Initial Alignment**: run several full-computation steps to initialize the feature-change ratio and uncertainty.
2. **Test-Time Self-Calibration**: track the relationship between input feature changes and output feature changes during inference.
3. **Uncertainty-Aware Skipping**: skip computation when the accumulated predicted error is below a fidelity threshold, and perform a full update when the threshold is exceeded.

## 📁 Repository Structure

```text
NaviCache/
├── NaviCache4Wan2.1/              # NaviCache script for Wan2.1
├── NaviCache4HunyuanVideo/        # NaviCache script for HunyuanVideo
├── assets/                        # Paper figure PDFs
├── eval/navicache/                # Open-Sora evaluation and metrics
├── scripts/                       # Runnable helper scripts
├── videosys/                      # VideoSys-related modules for Open-Sora evaluation
├── requirements.txt
├── LICENSE
└── README.md
```

## 📖 Citation

If you find NaviCache useful, please consider citing:

```bibtex
@inproceedings{lv2026navicache,
  author    = {Zheqi Lv and Zhibo Zhu and Jinke Wang and Qi Tian and Shengyu Zhang and Zhengyu Chen and Chengxi Zang and Zhou Zhao and Fei Wu},
  title     = {NaviCache: Test-Time Self-Calibration Caching for Video Generation},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
  publisher = {PMLR}
}
```

## 🙏 Acknowledgements

We thank the contributors of Wan2.1, HunyuanVideo, Open-Sora, VideoSys, TeaCache, EasyCache, MagCache, and PAB for their excellent open-source work and inspiring research.

## 📄 License

This project is released under the Apache License 2.0.

## ⚠️ Responsible Use

NaviCache accelerates video generation and may lower the cost of producing synthetic media. We encourage responsible usage and support the development of detection, watermarking, and provenance-tracking tools.
