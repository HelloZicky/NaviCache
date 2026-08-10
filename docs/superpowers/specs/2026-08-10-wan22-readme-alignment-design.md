# Wan2.2 Model README Alignment Design

## Objective

Rewrite `NaviCache4Wan2.2/README.md` so it follows the concise model-integration README style currently used by `NaviCache4Wan2.1/README.md` and `NaviCache4HunyuanVideo/README.md` on GitHub `main`.

## Scope

Only `NaviCache4Wan2.2/README.md` will change during implementation. The public Wan2.2 Python entry point, its tests, the repository root README, benchmark evidence, and the existing unstaged `NaviCache4Wan2.1/README.md` user modification will remain unchanged.

## Document Structure

The rewritten README will use this order:

1. `# NaviCache for Wan2.2`
2. A two-sentence introduction describing NaviCache and linking to official Wan2.2.
3. `## Usage`, with clone, installation, checkpoint-download, script-copy, and working-directory instructions.
4. `## Text-to-Video`, with one directly runnable Wan2.2-TI2V-5B command using the validated balanced NaviCache configuration.
5. `## Image-to-Video`, with one directly runnable command that adds an example input image path.
6. A short output-path note.
7. `## Acknowledgements`, thanking and linking to official Wan2.2.

## Content Rules

- Match the tone and compactness of the existing GitHub model READMEs.
- Keep commands consistent with the actual public CLI and its validated defaults.
- Use `./Wan2.2-TI2V-5B` as the checkpoint example and `examples/input.png` as the replaceable image example.
- Retain the balanced NaviCache values: threshold `0.05`, retained steps `10`, Kalman Q `0.05`, and Kalman R `0.05`.
- State that users should replace the example image with their own file.
- Remove the parameter table, benchmark Results section, latency explanation, quality metrics, Limitations section, internal runner/provenance language, and detailed implementation commentary.
- Do not move benchmark or smoke evidence into the release repository; existing external audit records remain the source of verification.

## Verification

- Run the existing Wan2.2 unit tests.
- Run Python syntax compilation and the public CLI `--help` command.
- Confirm every documented option appears in the CLI help output.
- Scan the rewritten README for placeholders, internal server paths, benchmark infrastructure terms, and contradictory output descriptions.
- Confirm the final diff contains only `NaviCache4Wan2.2/README.md`, apart from the already committed design document.
- Explicitly stage only the Wan2.2 README and preserve the unstaged Wan2.1 user modification.
