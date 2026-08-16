# NaviCache README Latency Source Policy Design

## Goal

Define a reusable release policy for NaviCache model-integration READMEs that reuses latency values from the NaviCache paper when the evaluated configuration matches exactly, while limiting new measurements for uncovered configurations to subset8. Preserve the existing showcase videos and relax dataset restrictions for visual sample selection.

## Source Precedence

1. Use the NaviCache paper's latency values when the README configuration exactly matches a configuration evaluated in the paper.
2. For configurations not exactly covered by the paper, run a real subset8 benchmark.
3. Never infer, interpolate, scale, or manually alter latency values.

The paper's `Inference Latency` is treated as `model_forward_latency_sec`. Paper-derived and locally measured values may therefore appear in the same latency table, provided every row identifies its hardware and source.

## Exact-Match Rule

A paper result is reusable only when all relevant fields match:

- model family and version;
- parameter scale;
- generation task, including T2V versus I2V;
- spatial resolution;
- frame count;
- sampling-step count;
- acceleration method and the method configuration represented by the published mode.

If any field differs, the configuration is treated as uncovered and requires subset8 measurement.

## Paper-Covered Configurations

The paper reports these primary latency configurations:

| Model | Configuration |
|---|---|
| Wan2.1-1.3B | 81 frames, 832x480, 50 steps |
| HunyuanVideo | 129 frames, 960x544, 50 steps |
| Open-Sora 1.2 | 51 frames, 848x480, 30 steps |

Paper-derived rows must be labeled `RTX 4090, reported in paper`. The local provenance record must retain the paper table number, complete configuration, selected mode, and copied values.

The policy applies to every model integration README, including Wan2.1, HunyuanVideo, Open-Sora, Wan2.2, and future integrations. For exact matches, the reusable Table 1 values are:

| Model/configuration | Method | Latency | Speedup |
|---|---|---:|---:|
| Wan2.1-1.3B, 81f, 832x480, 50 steps | Native | 214.93 s | 1.00x |
| Wan2.1-1.3B, 81f, 832x480, 50 steps | TeaCache | 121.57 s | 1.77x |
| Wan2.1-1.3B, 81f, 832x480, 50 steps | NaviCache fast/mid/slow | 96.40 / 106.97 / 115.86 s | 2.23x / 2.01x / 1.86x |
| HunyuanVideo, 129f, 960x544, 50 steps | Native | 2363.83 s | 1.00x |
| HunyuanVideo, 129f, 960x544, 50 steps | TeaCache / MagCache / EasyCache | 1070.14 / 882.76 / 1100.30 s | 2.21x / 2.68x / 2.15x |
| HunyuanVideo, 129f, 960x544, 50 steps | NaviCache fast/mid/slow | 928.45 / 1089.43 / 1150.87 s | 2.55x / 2.17x / 2.05x |
| Open-Sora 1.2, 51f, 848x480, 30 steps | Native | 56.48 s | 1.00x |
| Open-Sora 1.2, 51f, 848x480, 30 steps | TeaCache / MagCache / EasyCache | 41.38 / 26.07 / 34.55 s | 1.36x / 2.17x / 1.63x |
| Open-Sora 1.2, 51f, 848x480, 30 steps | NaviCache fast/mid/slow | 31.80 / 35.29 / 40.98 s | 1.78x / 1.60x / 1.38x |

Published NaviCache mode parameters are: Wan2.1 `Nalign=10`, thresholds fast/mid/slow `0.07/0.05/0.04`; HunyuanVideo `Nalign=5`, thresholds `0.040/0.035/0.025`; Open-Sora 1.2 `Nalign=5`, thresholds `0.55/0.35/0.15`. A README may use a published value only when its command selects the corresponding mode.

## Subset8 Rule

Every configuration not exactly covered by the paper uses subset8, regardless of model family. Within one configuration:

- Native, competitor, and NaviCache use the same eight prompts;
- all methods use `model_forward_latency_sec`;
- seed, sampling steps, resolution, frame count, and other generation settings match;
- method parameters are recorded;
- success count, missing prompts, duplicate rows, and prompt IDs are validated before aggregation;
- the README records the actual GPU model used for that result.

Each summary must remain reproducible from its raw JSONL. Speedup is always `native_latency / method_latency`.

## README Presentation

The main latency table adds a `Hardware / Source` column. Paper rows and subset8 rows may share the table because their timing field is the same, but their different hardware must be visible. The visual-quality mini-table reuses the corresponding latency row and does not introduce separate runtime measurements.

Current integration routing includes:

| Configuration | Source |
|---|---|
| T2V 1.3B, 81 frames, 832x480, 50 steps | Paper, RTX 4090 |
| T2V 14B, 1280x720 | subset8, actual GPU labeled |
| I2V 14B 480P | subset8, actual GPU labeled |
| I2V 14B 720P | subset8, actual GPU labeled |
| HunyuanVideo, 129 frames, 960x544, 50 steps | Paper, RTX 4090 |
| Open-Sora 1.2, 51 frames, 848x480, 30 steps | Paper, RTX 4090 |
| Wan2.2 and any other unmatched task/configuration | subset8, actual GPU labeled |

## Visual Showcase Policy

- Keep existing video files unchanged unless the user separately requests replacement.
- Do not require VBench or any other fixed dataset for showcase selection.
- Select samples where the competitor is visibly worse and NaviCache remains closer to Native.
- Preserve the method order `Native | Competitor | NaviCache`.
- Preserve MP4 format, correct panel geometry, and the existing README layout rules.

## Validation

Before release:

- verify every paper-derived value against the cited paper table and exact configuration;
- verify every subset8 value against raw JSONL and its actual hardware;
- recompute displayed speedups from displayed rounded latencies;
- ensure every latency row identifies hardware and source;
- ensure mini-tables match the main table;
- ensure showcase video paths still resolve and no video file was modified;
- scan the README for stale sample-count claims, superseded latency values, placeholders, and contradictory hardware statements.

## Scope

Implementation changes are limited to the shared release standard and the affected model README latency text or tables. Existing videos and unrelated repository files remain untouched.
