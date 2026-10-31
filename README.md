# HiPaCs Website - split static version

This package converts the previous single-file `index.html` website into a multi-page GitHub Pages structure.

## Structure

- `index.html`: landing page only
- `main.html`: central documentation hub
- `framework.html`: framework concept page
- `architecture.html`: implementation flow and real core-directory entry page
- `applications.html`: application overview
- `install.html`, `gallery.html`, `contact.html`: independent top-level pages
- `pages/`: application/detail/framework subpages
- `code/`: generated directory → file → function explorers
- `functions/`: temporary compatibility redirects marked `TODO(legacy-removal)`
- `assets/css/styles.css`: shared stylesheet
- `assets/js/main.js`: shared tab/menu/preview behavior
- `assets/js/mathjax-config.js`: shared MathJax delimiters and SVG output configuration
- `assets/js/code-explorer.js`: generated source explorer behavior
- `tools/Doxyfile`: Doxygen XML configuration
- `tools/generate_code_docs.py`: static explorer generator
- `examples/`: runnable input/model snapshots synchronized from validated source cases
- `tools/sync_examples.py`: `sync` copies canonical case files; `check` detects stale website copies
- `tools/generate_homogenization_page.py`: refreshes the Elastic/Inelastic coupling excerpts
- `tools/generate_multiscale_example_pages.py`: regenerates four example pages and their run scripts
- `tools/generate_rve_verification.py`: renders measured Single RVE error plots and component tables

## Updated standalone application examples

The eight standalone application pages use the curated snapshots in `examples/`
(hyperelastic-3d, elastoplastic, viscoelastic, newmark, vem-linear, vem-quadratic,
fsi, incompressible-flow). Preserve their supplied inputs and descriptive model
filenames when refreshing documentation. The Linear VEM snapshot uses
`VEM_ORDER 1`. Each page runs its snapshot by passing input paths directly to
`FEM/bin/fem`, without replacing the active FEM input files.

Newmark, FSI, and incompressible flow have browser-ready H.264 MP4 versions alongside the original
AVI recordings. Newmark also shows its displacement-versus-time SVG.
`generate_fsi_page.py` reads the published FSI snapshot and refreshes only its
code excerpts and reference-mesh illustration.

The legacy standalone mappings in `sync_examples.py` predate these snapshots;
do not run an unrestricted sync over the curated standalone examples. Use the
explicit multiscale/Single RVE case selections below for those branches.

## Synchronize runnable examples

```sh
python3 Web/tools/generate_multiscale_inputs.py
python3 Web/tools/sync_examples.py sync --case multiscale-fem-hyperelastic --case multiscale-vem
python3 Web/tools/sync_examples.py check --case multiscale-fem-hyperelastic --case multiscale-vem
python3 Web/tools/generate_homogenization_page.py
python3 Web/tools/generate_multiscale_example_pages.py
python3 Web/tools/generate_fsi_page.py
python3 Web/tools/generate_fsi_page.py --check
python3 Web/tools/verify_mathjax.py
```

For the synchronized multiscale/Single RVE cases, update the canonical model/input file,
run `sync`, and commit both the source and synchronized example snapshot.

The unified FSI page is `pages/app-fsi.html`, with Theory and Code tabs and a 2D
FEM example. `generate_fsi_page.py` refreshes its marked C++ excerpts, actual Q9
mesh SVG, and the mesh illustration directly from `examples/fsi/`.
Run it after changing the FSI source or inputs; `--check` verifies freshness
without writing. Mesh-motion equations are short options within the same page.
The old BH and HE page URLs redirect to the unified page.

## Computational homogenization

`pages/detail-multiscale-analysis.html` has Elastic and Inelastic tabs. Both
focus on the micro stress/tangent contract, periodic-equilibrium sensitivity,
volume normalization, and macro AD injection. Material theory links back to
Solid. Inelastic adds plastic/viscous local-state derivatives and trial/commit
history ownership.

The two representative elastic examples are FE² (macro Q4 / micro Q9) and VE²
(macro k=1 / micro k=2). Macro and micro DOFs are separate inputs, selected with
`--dofs` and `--micro-dofs`. The Q9 mesh is deterministically elevated from the
existing Q4 RVE; `generate_multiscale_inputs.py --check` verifies freshness.
The inelastic examples retain the existing FEM elastoplastic/viscoelastic
configuration. The old quadratic-macro VEM page redirects to the representative
VE² page; its existing downloads remain available.

To sync only these snapshots, repeat `--case`, for example:

```sh
python3 Web/tools/sync_examples.py sync --case multiscale-fem-hyperelastic --case multiscale-vem
python3 Web/tools/generate_homogenization_page.py --check
python3 Web/tools/generate_multiscale_example_pages.py --check
```

The elastic verification records contain the measured two-step macro solve and
a four-direction finite-difference check of `C / volume` against converged
average stress. Records include input and implementation hashes; regeneration
rejects a verification claim when those sources change. To reproduce the FE²
check after activating the HiPaCs environment, run from the repository root:

```sh
hom_case=multiscale-fem-hyperelastic
hom_macro=models/weak_form_macro_fem.cpp
hom_micro=../FEM/models/Elastic_Models/weak_form_hyper_elastic_ad.cpp
hom_inputs="Web/examples/$hom_case"
bash "$hom_inputs/run.sh" > /tmp/hom-solve.log
make -C Multiscale APP_SRCS=../Web/tools/check_rve_sensitivity.cpp \
  MACRO_MODEL_SRC="$hom_macro" MICRO_MODEL_SRC="$hom_micro" TARGET=bin/check_web_sensitivity
OMP_NUM_THREADS=2 ./Multiscale/bin/check_web_sensitivity \
  "$hom_inputs/input/rve/Micro_Mesh.txt" "$hom_inputs/input/rve/DOFS_info.txt" \
  "$hom_inputs/input/solver/Material.txt" "$hom_inputs/input/solver/Solver.txt" \
  "$hom_inputs/input/macro/BC.txt" > /tmp/hom-sensitivity.log
python3 Web/tools/record_multiscale_validation.py --case "$hom_case" \
  --solve-log /tmp/hom-solve.log --sensitivity-log /tmp/hom-sensitivity.log
python3 Web/tools/generate_multiscale_example_pages.py
```

For VE², set `hom_case=multiscale-vem`,
`hom_macro=models/weak_form_macro_vem.cpp`, and
`hom_micro=../FEM/models/VEM/weak_form_vem_quadratic.cpp` before those commands.
The sensitivity check uses one elastic increment at the deformation printed in
the verification record. It does not validate a history-dependent substep path.

## Single RVE sensitivity verification

The Single RVE overview and its FEM, linear VEM, and quadratic VEM example pages
show measured finite-difference error curves and component tables. All three use
the integrated-Piola derivative returned by `RveSensitivity.C`. Eight central
perturbations from `1e-2` through `1e-9` require 65 independently equilibrated RVE
solves per case. The same preselected `1e-6` perturbation supplies each table.
The main table contains 16 in-plane components; a disclosure contains the other
20 plane-strain entries. JSON and CSV downloads retain all components and steps.

`generate_rve_verification.py --check` validates source/input hashes, plot hashes,
error norms, generated table values, and the full component CSV without requiring
Matplotlib. To reproduce a sweep from an activated HiPaCs environment:

```sh
rve_case=single-rve-fem
rve_model=../FEM/models/Elastic_Models/weak_form_hyper_elastic_ad.cpp
rve_inputs="Web/examples/$rve_case/input"
make -C Multiscale APP_SRCS=../Web/tools/sweep_rve_sensitivity.cpp \
  MACRO_MODEL_SRC=models/weak_form_macro_fem.cpp MICRO_MODEL_SRC="$rve_model" \
  TARGET=bin/sweep_web_rve
OMP_NUM_THREADS=2 ./Multiscale/bin/sweep_web_rve \
  "$rve_inputs/ProblemMesh.txt" "$rve_inputs/DOFS_info.txt" \
  "$rve_inputs/Material.txt" "$rve_inputs/Solver.txt" "$rve_inputs/BC.txt" \
  /tmp/rve-sensitivity-sweep.json
"$CONDA_PREFIX/bin/python" Web/tools/generate_rve_verification.py \
  --collect "$rve_case=/tmp/rve-sensitivity-sweep.json"
python3 Web/tools/generate_rve_verification.py --check
```

For `single-rve-vem`, select `../FEM/models/VEM/weak_form_vem.cpp`; for
`single-rve-vem-quadratic`, select
`../FEM/models/VEM/weak_form_vem_quadratic.cpp`. The rendering step uses Matplotlib
and produces standalone SVG/PNG graphs. Numerical sweeps are rerun only when the
underlying inputs, checker, or model/core implementation changes.

## Regenerate code documentation

Add English Doxygen comments to public and core functions, then run:

```sh
python3 Web/tools/generate_code_docs.py
```

Generation clears stale Doxygen XML and rebuilds the explorer from the current source.
Run the extraction regression test with:

```sh
python3 -m unittest discover -s Web/tools -p test_generate_code_docs.py
```

Every `.h` and `.cpp` file and every named function is listed. Existing Doxygen
descriptions are preferred; deterministic summaries are generated for undocumented
internal helpers and trivial special members. The GitHub Actions
workflow regenerates the same pages and deploys `Web/` to
`YongbinChoi91.github.io`; configure the `WEBSITE_DEPLOY_TOKEN` repository secret
before enabling deployment.
