# Wan2.2 Model README and CLI Alignment Design

## Objective

Rewrite `NaviCache4Wan2.2/README.md` so it follows the concise model-integration README style currently used by `NaviCache4Wan2.1/README.md` and `NaviCache4HunyuanVideo/README.md` on GitHub `main`. Make the Wan2.2 NaviCache command-line option names exactly match the established public names used by the other integrations.

## Scope

Implementation changes are limited to `NaviCache4Wan2.2/README.md`, `NaviCache4Wan2.2/navicache_generate.py`, and `tests/test_wan22_navicache.py`. The repository root README, algorithm behavior, benchmark evidence, and the existing unstaged `NaviCache4Wan2.1/README.md` user modification will remain unchanged.

## Document Structure

The rewritten README will use this order:

1. `# NaviCache for Wan2.2`
2. A two-sentence introduction describing NaviCache and linking to official Wan2.2.
3. `## Usage`, with clone, installation, checkpoint-download, script-copy, and working-directory instructions.
4. `## Text-to-Video`, with one directly runnable Wan2.2-TI2V-5B command using the validated balanced NaviCache configuration.
5. `## Image-to-Video`, with one directly runnable command that adds an example input image path.
6. A short output-path note.
7. `## Acknowledgements`, thanking and linking to official Wan2.2.

## Command-Line Naming

The Wan2.2 public CLI will expose only the established repository names:

| Meaning | Required public name | Removed Wan2.2-only name |
|---|---|---|
| Accumulated error threshold | `--navicache_thresh` | None; already consistent |
| Initial alignment steps | `--navicache_align_steps` | `--navicache_ret_steps` |
| Kalman process-noise covariance | `--navicache_process_noise` | `--navicache_kalman_q` |
| Kalman measurement-noise covariance | `--navicache_measurement_noise` | `--navicache_kalman_r` |

The removed names will not be retained as aliases. The parser will reject them. The renamed arguments will map to the same internal configuration values, so defaults and caching behavior remain unchanged.

## Content Rules

- Match the tone and compactness of the existing GitHub model READMEs.
- Keep commands consistent with the actual public CLI and its validated defaults.
- Use `./Wan2.2-TI2V-5B` as the checkpoint example and the official `examples/i2v_input.JPG` sample image.
- Retain the balanced NaviCache values: threshold `0.05`, alignment steps `10`, process noise `0.05`, and measurement noise `0.05`.
- State that users should replace the example image with their own file.
- Remove the parameter table, benchmark Results section, latency explanation, quality metrics, Limitations section, internal runner/provenance language, and detailed implementation commentary.
- Do not move benchmark or smoke evidence into the release repository; existing external audit records remain the source of verification.
- Do not document or expose the removed Wan2.2-only command-line names.

## Verification

- Update tests before implementation so they require the established public names, verify their default and explicit-value mapping, and verify that the removed names are rejected.
- Run the complete Wan2.2 unit-test suite after implementation.
- Run Python syntax compilation and the public CLI `--help` command.
- Confirm every documented option appears in CLI help and that the three removed names do not appear.
- Scan the rewritten README for placeholders, internal server paths, benchmark infrastructure terms, and contradictory output descriptions.
- Confirm the implementation diff contains only the Wan2.2 README, public entry point, and tests, apart from the already committed design document.
- Explicitly stage only those three implementation files and preserve the unstaged Wan2.1 user modification.
