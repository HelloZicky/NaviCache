# NaviCache README Latency Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the shared NaviCache README release standard and provide a new-session goal prompt that applies the approved paper-first latency policy across every model integration without changing existing videos.

**Architecture:** Keep policy in the workspace-root `AGENTS.md`, where all model integrations inherit it. Treat exact paper configurations and locally benchmarked subset8 configurations as two provenance paths that converge on the same `model_forward_latency_sec` table format, with hardware and source shown per row.

**Tech Stack:** Markdown, Git, PowerShell validation commands

## Global Constraints

- Paper `Inference Latency` is `model_forward_latency_sec`.
- Reuse paper values only for an exact configuration match.
- Label paper values `RTX 4090, reported in paper`.
- Run uncovered configurations on subset8 and label the actual GPU.
- Keep existing showcase video files unchanged.
- Showcase selection is dataset-agnostic and prioritizes samples where the competitor is visibly worse while NaviCache remains close to Native.
- Preserve unrelated user changes in all integration READMEs, including the existing change in `NaviCache4Wan2.1/README.md`.

---

### Task 1: Update Shared Release Policy

**Files:**
- Modify: `../AGENTS.md`

**Interfaces:**
- Consumes: Approved design in `docs/superpowers/specs/2026-08-16-navicache-readme-latency-source-policy-design.md`.
- Produces: A single shared Markdown policy used by future NaviCache model README tasks.

- [x] **Step 1: Replace the latency provenance restriction**

Allow either an exact matching NaviCache paper table or completed local `summary.json`, `summary.csv`, or `per_prompt_latency.jsonl`. Require paper table number, exact configuration, method mode, hardware, and values in the local provenance record.

- [x] **Step 2: Add exact-match fields**

Require model family/version, parameter scale, T2V/I2V task, resolution, frame count, sampling steps, acceleration method, and method configuration to match before reusing paper values.

- [x] **Step 3: Replace the old benchmark-size rule**

Use paper values for exact matches. Use subset8 for all uncovered or parameter-mismatched configurations. Require Native, competitor, and NaviCache to share the same eight prompts and generation settings.

- [x] **Step 4: Add hardware/source presentation rules**

Require a `Hardware / Source` field or equivalent explicit annotation for every latency row. Mark paper rows `RTX 4090, reported in paper`; mark subset8 rows with the actual GPU.

- [x] **Step 5: Update visual showcase policy**

State that existing video files remain unchanged unless replacement is explicitly requested, no fixed dataset is required, and samples should expose clear competitor degradation while NaviCache remains close to Native.

- [x] **Step 6: Validate the standard**

Run:

```powershell
rg -n "model_forward_latency_sec|RTX 4090|subset8|Hardware / Source|dataset|视频文件" ..\AGENTS.md
rg -n "超过 5 分钟|subset92|每一个 latency 都必须来自真实跑完" ..\AGENTS.md
```

Expected: the first command finds every new policy concept; the second command finds no superseded benchmark-size or provenance rule.

### Task 2: Verify Scope and Prepare New-Session Goal

**Files:**
- Verify only: `../AGENTS.md`
- Verify only: `NaviCache4Wan2.1/README.md`
- Verify only: `NaviCache4HunyuanVideo/README.md`
- Verify only: `NaviCache4OpenSora/README.md`
- Verify only: `docs/superpowers/specs/2026-08-16-navicache-readme-latency-source-policy-design.md`

**Interfaces:**
- Consumes: Updated shared policy from Task 1.
- Produces: A self-contained Chinese goal prompt for a new Codex session.

- [x] **Step 1: Check repository scope**

Run:

```powershell
git status --short
git diff -- NaviCache4Wan2.1/README.md NaviCache4HunyuanVideo/README.md NaviCache4OpenSora/README.md
```

Expected: no video file is modified; pre-existing README changes remain preserved and distinguishable from this policy work.

- [x] **Step 2: Check policy consistency**

Confirm the new policy does not simultaneously require paper-derived latency to have local JSONL, does not retain the old five-minute subset rule, and does not require a fixed showcase dataset.

- [x] **Step 3: Write the handoff goal**

The goal prompt must tell the new session to read `AGENTS.md`, the approved design, the paper, the Wan2.1, HunyuanVideo, and Open-Sora integration READMEs, and current git status before editing. It must audit only those three current model integrations; reuse exact Table 1 values for matching Wan2.1-1.3B, HunyuanVideo, and Open-Sora 1.2 configurations; use validated subset8 results for every uncovered or mismatched configuration only after confirming actual GPU and raw provenance; add hardware/source labeling; keep videos unchanged; synchronize mini-tables; and run fresh verification.

- [x] **Step 4: Commit only the plan document if requested**

Do not include the user's existing README modification in a policy or plan commit.
