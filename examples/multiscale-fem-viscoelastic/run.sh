#!/usr/bin/env bash
set -euo pipefail
case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$case_dir/../../.." && pwd)"
# Select the canonical models matching the downloadable source snapshots.
make -C "$repo_dir/Multiscale" \
  MACRO_MODEL_SRC=models/weak_form_macro_fem.cpp \
  MICRO_MODEL_SRC=../FEM/models/Inelastic_Models/weak_form_visco_elastic.cpp \
  TARGET=bin/web_multiscale-fem-viscoelastic
run_dir="$repo_dir/Multiscale/output/web/multiscale-fem-viscoelastic"
mkdir -p "$run_dir"
post_file="$run_dir/Postprocessing.txt"
cat "$case_dir/input/solver/Postprocessing.txt" > "$post_file"
printf '\nplot_window off\nplot_each_step off\nmultiscale_plot_show off\n' >> "$post_file"
cd "$run_dir"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
exec "$repo_dir/Multiscale/bin/web_multiscale-fem-viscoelastic" \
  --macro-mesh "$case_dir/input/macro/Macro_Mesh.txt" \
  --micro-mesh "$case_dir/input/rve/Micro_Mesh.txt" \
  --dofs "$case_dir/input/macro/DOFS_info.txt" \
  --micro-dofs "$case_dir/input/rve/DOFS_info.txt" \
  --macro-bc "$case_dir/input/macro/BC.txt" \
  --material "$case_dir/input/solver/Material.txt" \
  --solver "$case_dir/input/solver/Solver.txt" \
  --post "$post_file"
