# Wan2.2 README and CLI Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Wan2.2 model README match the concise GitHub model-README style and make its public NaviCache CLI names exactly match the repository's established names.

**Architecture:** Change only the argparse public surface and its mapping into the existing `NaviCacheConfig`; the cache algorithm and internal configuration fields remain unchanged. Rewrite the model README as a compact usage document, and enforce the public naming contract with parser tests that require the standard names and reject the removed Wan2.2-only names.

**Tech Stack:** Python 3.11, argparse, pytest, Markdown, Git, PowerShell.

## Global Constraints

- Implementation changes are limited to `NaviCache4Wan2.2/navicache_generate.py`, `tests/test_wan22_navicache.py`, and `NaviCache4Wan2.2/README.md`.
- The public CLI must expose `--navicache_thresh`, `--navicache_align_steps`, `--navicache_process_noise`, and `--navicache_measurement_noise`.
- The public CLI must reject `--navicache_ret_steps`, `--navicache_kalman_q`, and `--navicache_kalman_r`; do not add aliases.
- Keep defaults exactly `0.05`, `10`, `0.05`, and `0.05`, respectively.
- Do not change `NaviCacheConfig`, cache decisions, Kalman equations, Wan2.2 native-forward arguments, or generation behavior.
- The model README must use the concise GitHub style and must not contain Results, parameter tables, Limitations, internal provenance, or benchmark-infrastructure language.
- Do not modify, stage, or commit the existing `NaviCache4Wan2.1/README.md` user change.
- Do not modify the repository root README or move external benchmark/smoke evidence into the release repository.

## File Responsibilities

- `NaviCache4Wan2.2/navicache_generate.py`: owns the public argparse names and maps them into the existing algorithm configuration.
- `tests/test_wan22_navicache.py`: owns the parser naming/default/mapping/rejection contract and existing algorithm regression coverage.
- `NaviCache4Wan2.2/README.md`: owns concise installation and direct T2V/I2V commands using only public option names.

---

### Task 1: Hard-rename the Wan2.2 public CLI with TDD

**Files:**
- Modify: `tests/test_wan22_navicache.py:213-239`
- Modify: `NaviCache4Wan2.2/navicache_generate.py:260-284`
- Modify: `NaviCache4Wan2.2/navicache_generate.py:340-348`

**Interfaces:**
- Consumes: `build_parser() -> argparse.ArgumentParser` and `NaviCacheConfig(threshold, ret_steps, kalman_q, kalman_r, sample_steps)`.
- Produces: parsed attributes `navicache_thresh: float`, `navicache_align_steps: int`, `navicache_process_noise: float`, and `navicache_measurement_noise: float`; the existing internal config still receives `ret_steps`, `kalman_q`, and `kalman_r`.

- [ ] **Step 1: Write failing public-name tests**

Replace the four old assertions in `test_public_cli_defaults_match_verified_configuration` and append explicit mapping and rejection tests:

```python
def test_public_cli_defaults_match_verified_configuration():
    module = load_public_module()
    parser = module.build_parser()
    args = parser.parse_args(["--ckpt_dir", "weights"])

    assert args.task == "ti2v-5B"
    assert args.size == "1280*704"
    assert args.frame_num == 121
    assert args.sample_steps == 50
    assert args.sample_shift == 5.0
    assert args.sample_guide_scale == 5.0
    assert args.base_seed == 42
    assert args.navicache_thresh == 0.05
    assert args.navicache_align_steps == 10
    assert args.navicache_process_noise == 0.05
    assert args.navicache_measurement_noise == 0.05


def test_public_cli_maps_repository_standard_navicache_names():
    module = load_public_module()
    args = module.build_parser().parse_args(
        [
            "--ckpt_dir",
            "weights",
            "--navicache_thresh",
            "0.07",
            "--navicache_align_steps",
            "8",
            "--navicache_process_noise",
            "0.03",
            "--navicache_measurement_noise",
            "0.04",
        ]
    )

    assert args.navicache_thresh == 0.07
    assert args.navicache_align_steps == 8
    assert args.navicache_process_noise == 0.03
    assert args.navicache_measurement_noise == 0.04


@pytest.mark.parametrize(
    "removed_name",
    [
        "--navicache_ret_steps",
        "--navicache_kalman_q",
        "--navicache_kalman_r",
    ],
)
def test_public_cli_rejects_removed_wan22_only_names(removed_name):
    module = load_public_module()
    parser = module.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--ckpt_dir", "weights", removed_name, "1"])
```

- [ ] **Step 2: Run the parser tests and verify RED**

Run:

```powershell
python -m pytest tests/test_wan22_navicache.py -q
```

Expected: the new default and explicit mapping tests fail because `navicache_align_steps`, `navicache_process_noise`, and `navicache_measurement_noise` do not yet exist; existing algorithm tests remain green.

- [ ] **Step 3: Implement the minimal public hard rename**

Replace the three Wan2.2-only parser arguments with:

```python
    parser.add_argument(
        "--navicache_align_steps",
        type=int,
        default=10,
        help="Number of initial diffusion steps computed for alignment.",
    )
    parser.add_argument(
        "--navicache_process_noise",
        type=float,
        default=0.05,
        help="Process-noise covariance used by NaviCache state estimation.",
    )
    parser.add_argument(
        "--navicache_measurement_noise",
        type=float,
        default=0.05,
        help="Measurement-noise covariance used by NaviCache state estimation.",
    )
```

Map the public names into the unchanged internal configuration:

```python
        NaviCacheConfig(
            threshold=args.navicache_thresh,
            ret_steps=args.navicache_align_steps,
            kalman_q=args.navicache_process_noise,
            kalman_r=args.navicache_measurement_noise,
            sample_steps=args.sample_steps,
        ),
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_wan22_navicache.py -q
```

Expected: 14 tests pass, including three parameterized rejection cases.

- [ ] **Step 5: Verify the public help surface**

Run:

```powershell
$help = python NaviCache4Wan2.2/navicache_generate.py --help 2>&1 | Out-String
$required = @('--navicache_thresh','--navicache_align_steps','--navicache_process_noise','--navicache_measurement_noise')
$removed = @('--navicache_ret_steps','--navicache_kalman_q','--navicache_kalman_r')
foreach ($name in $required) { if (-not $help.Contains($name)) { throw "Missing CLI name: $name" } }
foreach ($name in $removed) { if ($help.Contains($name)) { throw "Removed CLI name remains: $name" } }
```

Expected: command exits successfully without throwing.

- [ ] **Step 6: Commit the tested CLI contract**

```powershell
git add -- NaviCache4Wan2.2/navicache_generate.py tests/test_wan22_navicache.py
git diff --cached --check
git commit -m "fix: align Wan2.2 NaviCache CLI names"
```

Expected: the commit contains exactly the public entry point and its tests; `NaviCache4Wan2.1/README.md` remains unstaged.

---

### Task 2: Rewrite the Wan2.2 README in the GitHub model style

**Files:**
- Modify: `NaviCache4Wan2.2/README.md:1-84`

**Interfaces:**
- Consumes: the standard public CLI names produced by Task 1.
- Produces: one concise model README with directly runnable T2V and image-conditioned commands.

- [ ] **Step 1: Replace the README with the approved compact content**

Use this complete content:

````markdown
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
````

- [ ] **Step 2: Verify structure and removed content**

Run:

```powershell
$readme = Get-Content -Raw NaviCache4Wan2.2/README.md
$expectedHeadings = @('# NaviCache for Wan2.2','## Usage','## Text-to-Video','## Image-to-Video','## Acknowledgements')
foreach ($heading in $expectedHeadings) { if (-not $readme.Contains($heading)) { throw "Missing heading: $heading" } }
$removedSections = @('## Results','Inference Latency Comparison','Visual Quality Comparison','## Limitations','--navicache_ret_steps','--navicache_kalman_q','--navicache_kalman_r')
foreach ($text in $removedSections) { if ($readme.Contains($text)) { throw "Removed README content remains: $text" } }
```

Expected: command exits successfully without throwing.

- [ ] **Step 3: Verify every documented flag exists in argparse**

Run:

```powershell
$readme = Get-Content -Raw NaviCache4Wan2.2/README.md
$help = python NaviCache4Wan2.2/navicache_generate.py --help 2>&1 | Out-String
$documented = [regex]::Matches($readme, '--[a-zA-Z0-9_]+') | ForEach-Object Value | Sort-Object -Unique
foreach ($flag in $documented) { if (-not $help.Contains($flag)) { throw "README flag absent from CLI: $flag" } }
```

Expected: every documented flag is present in help output.

- [ ] **Step 4: Scan release text and commit the README**

```powershell
$bad = Select-String -Path NaviCache4Wan2.2/README.md -Pattern 'TBD|TODO|/data/zhuzhibo|/var/tmp/zhuzhibo|js1.blockelite|driver_launch|prompt_stats_shard' -CaseSensitive:$false
if ($bad) { $bad; exit 2 }
git diff --check -- NaviCache4Wan2.2/README.md
git add -- NaviCache4Wan2.2/README.md
git diff --cached --check
git commit -m "docs: align Wan2.2 model README style"
```

Expected: the commit contains only `NaviCache4Wan2.2/README.md`; the Wan2.1 user modification remains unstaged.

---

### Task 3: Final regression, scope, and push verification

**Files:**
- Verify: `NaviCache4Wan2.2/navicache_generate.py`
- Verify: `tests/test_wan22_navicache.py`
- Verify: `NaviCache4Wan2.2/README.md`

**Interfaces:**
- Consumes: the committed CLI contract and compact README from Tasks 1 and 2.
- Produces: a pushed branch whose public docs and parser agree and whose only remaining worktree modification is the protected Wan2.1 file.

- [ ] **Step 1: Run complete local verification**

```powershell
python -m pytest -q
python -m py_compile NaviCache4Wan2.2/navicache_generate.py
python NaviCache4Wan2.2/navicache_generate.py --help | Select-Object -First 20
git diff --check
```

Expected: all tests pass, syntax compilation succeeds, help prints the standard names, and diff checking reports no task-file errors.

- [ ] **Step 2: Verify naming and protected scope**

```powershell
$publicFiles = @('NaviCache4Wan2.2/navicache_generate.py','NaviCache4Wan2.2/README.md')
$old = Select-String -Path $publicFiles -Pattern '--navicache_ret_steps|--navicache_kalman_q|--navicache_kalman_r'
if ($old) { $old; exit 2 }
$status = @(git status --short)
if ($status.Count -ne 1 -or $status[0] -notmatch 'NaviCache4Wan2\.1/README\.md') { $status; throw 'Unexpected final worktree status' }
```

Expected: no removed public flag remains, and only `NaviCache4Wan2.1/README.md` is modified in the worktree.

- [ ] **Step 3: Verify remote relation and push**

```powershell
git fetch origin codex/tidy-release-directory-names
$relation = @(git rev-list --left-right --count HEAD...origin/codex/tidy-release-directory-names)
if (($relation -join ' ') -notmatch '^\d+\s+0$') { throw "Remote branch moved or relation is unsafe: $relation" }
git push origin codex/tidy-release-directory-names
git fetch origin codex/tidy-release-directory-names
if ((git rev-parse HEAD) -ne (git rev-parse origin/codex/tidy-release-directory-names)) { throw 'HEAD and origin differ after push' }
```

Expected: push succeeds and local HEAD equals the origin branch SHA.
