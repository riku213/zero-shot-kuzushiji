# Repository Guidance

## Project Context
- This repository builds zero-shot Kuzushiji recognition models using IDS structure and image-derived FaRE codes.
- Read `docs/experiment-history.md` when choosing or comparing experiment artifacts, checkpoints, or codebooks. It records provenance and known caveats without loading the full experiment history for every task.
- Treat files under `outputs/` as generated artifacts, not as interchangeable sources of truth. Prefer the codebook explicitly named by the experiment being continued, and verify its metadata before use.

## Working Rules
- Preserve existing user changes and do not overwrite or regenerate large artifacts unless explicitly requested.
- Reuse the IDS parser and codebook logic in `src/structured_codebook.py` and `src/fare_pipeline.py` rather than implementing parallel parsing behavior.
- Keep experiment outputs in a dedicated `outputs/<experiment-id>_<description>/` directory. Record the command, inputs, model weights, codebook path, and key result in `docs/experiment-history.md` when an experiment is added or its provenance is established.
- The repository ignores `outputs/`; documentation and source changes should remain reproducible without committing generated model/data artifacts.
- Use the workspace `.venv` for Python checks when available. Run focused tests after code changes.

## Codebook Notes
- The 64-dimensional radical/glyph dictionary is an input to a final class codebook; it is not itself the final class codebook.
- A missing IDS definition does not inherently mean a character cannot be represented: it can be represented as a terminal glyph if its 64-dimensional code exists.
- Binary-code duplicate fallback is deterministic for fixed inputs, record ordering, and random seed; changing the thresholding scheme can change the dictionary.
- See `docs/experiment-history.md` for verified artifact paths and uncertainty notes.