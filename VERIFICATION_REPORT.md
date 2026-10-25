# Website verification — 2026-10-25

## Scope

Refreshed the core source explorer from the current local `core/` sources.
FEM and Multiscale example snapshots and application/detail pages were preserved.
No numerical examples or sensitivity sweeps were rerun for this update.

## Source and documentation checks

- Regenerated 9 core directory pages and 708 function detail pages using fresh Doxygen XML.
- All 708 displayed implementations match their current source ranges, allowing only trailing whitespace used to render blank lines.
- Function signatures and descriptions in directory listings match the extracted function records and detail pages.
- Reviewed the changed output-folder parser and path configuration against their implementation and the shared application call sequence.
- Updated their function descriptions and the Input / Configuration and Postprocessing pages for `Create Folder Name`, `folder_name`, `auto`, folder sanitization, `model_output_subdir`, and the actual output directory names.
- 85 function records have source-derived or curated descriptions; 623 use short name-based fallback descriptions. This check does not establish a detailed semantic explanation for every function.
- Compared 221 tracked example files and application/detail pages against the previously published revision: all unchanged.

## Validation

- Doxygen extraction regression test: passed.
- MathJax configuration verification: passed.
- Local file-link check across 790 HTML pages: no missing targets.

## Preserved example status

FEM and Multiscale examples remain at their existing snapshots as requested.
Earlier freshness checks reported some input/source differences and outdated sensitivity verification hashes. This core-documentation update does not refresh those examples or certify their numerical results against the latest solver sources.

