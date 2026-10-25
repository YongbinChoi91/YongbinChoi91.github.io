#!/usr/bin/env python3
"""Synchronize runnable website examples with their validated source files."""

from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "Web" / "examples"

COMMON = {
    "BC.txt": "input/BC.txt",
    "DOFS_info.txt": "input/DOFS_info.txt",
    "Material.txt": "input/Material.txt",
    "Postprocessing.txt": "input/Postprocessing.txt",
    "ProblemMesh.txt": "input/ProblemMesh.txt",
    "Solver.txt": "input/Solver.txt",
}

CASES: dict[str, dict[str, str]] = {
    "hyperelastic-3d": {
        "FEM/models/Elastic_Models/weak_form_hyper_elastic_ad.cpp": "weak_form.cpp",
        "FEM/models/Elastic_Models/Mesh/Cook3D_16.txt": COMMON["ProblemMesh.txt"],
        "FEM/models/Elastic_Models/input/BC_3D.txt": COMMON["BC.txt"],
        "FEM/models/Elastic_Models/input/Info_Solid.txt": COMMON["DOFS_info.txt"],
        "FEM/models/Elastic_Models/input/Material.txt": COMMON["Material.txt"],
        "FEM/models/Elastic_Models/input/Solver.txt": COMMON["Solver.txt"],
        "FEM/models/Elastic_Models/input/Postprocessing.txt": COMMON["Postprocessing.txt"],
    },
    "navier-stokes-3d": {
        "FEM/models/Fluid_Models/weak_form_NSE.cpp": "weak_form.cpp",
        "FEM/models/Fluid_Models/Mesh/cylinder3D_r0.txt": COMMON["ProblemMesh.txt"],
        "FEM/models/Fluid_Models/input/BC_3D.txt": COMMON["BC.txt"],
        "FEM/models/Fluid_Models/input/Info_Fluid.txt": COMMON["DOFS_info.txt"],
        "FEM/models/Fluid_Models/input/Material.txt": COMMON["Material.txt"],
        "FEM/models/Fluid_Models/input/Solver.txt": COMMON["Solver.txt"],
        "FEM/models/Fluid_Models/input/Postprocessing.txt": COMMON["Postprocessing.txt"],
    },
    "newmark": {
        "FEM/models/Dynamic_Models/weak_form_Newmark.cpp": "weak_form.cpp",
        "FEM/models/Dynamic_Models/Mesh/Cook16.txt": COMMON["ProblemMesh.txt"],
        "FEM/models/Dynamic_Models/input/BC.txt": COMMON["BC.txt"],
        "FEM/models/Dynamic_Models/input/Info_Newmark.txt": COMMON["DOFS_info.txt"],
        "FEM/models/Dynamic_Models/input/Material.txt": COMMON["Material.txt"],
        "FEM/models/Dynamic_Models/input/Solver.txt": COMMON["Solver.txt"],
        "FEM/models/Dynamic_Models/input/Postprocessing.txt": COMMON["Postprocessing.txt"],
    },
    "elastoplastic": {
        "FEM/models/Inelastic_Models/weak_form_elasto_plastic.cpp": "weak_form.cpp",
        "FEM/models/Inelastic_Models/Mesh/Lshape80.txt": COMMON["ProblemMesh.txt"],
        "FEM/models/Inelastic_Models/input/BC.txt": COMMON["BC.txt"],
        "FEM/models/Inelastic_Models/input/Info_EP.txt": COMMON["DOFS_info.txt"],
        "FEM/models/Inelastic_Models/input/Material_EP.txt": COMMON["Material.txt"],
        "FEM/models/Inelastic_Models/input/Solver.txt": COMMON["Solver.txt"],
        "FEM/models/Inelastic_Models/input/Postprocessing.txt": COMMON["Postprocessing.txt"],
    },
    "viscoelastic": {
        "FEM/models/Inelastic_Models/weak_form_visco_elastic.cpp": "weak_form.cpp",
        "FEM/models/Inelastic_Models/Mesh/Lshape80.txt": COMMON["ProblemMesh.txt"],
        "FEM/models/Inelastic_Models/input/BC.txt": COMMON["BC.txt"],
        "FEM/models/Inelastic_Models/input/Info_VE.txt": COMMON["DOFS_info.txt"],
        "FEM/models/Inelastic_Models/input/Material_VE.txt": COMMON["Material.txt"],
        "FEM/models/Inelastic_Models/input/Solver.txt": COMMON["Solver.txt"],
        "FEM/models/Inelastic_Models/input/Postprocessing.txt": COMMON["Postprocessing.txt"],
    },
    "vem-linear": {
        "FEM/models/VEM/weak_form_vem.cpp": "weak_form.cpp",
        "FEM/models/VEM/Mesh/Voronoi.txt": COMMON["ProblemMesh.txt"],
        "FEM/models/VEM/input/BC.txt": COMMON["BC.txt"],
        "FEM/models/VEM/input/Info_VEM.txt": COMMON["DOFS_info.txt"],
        "FEM/models/VEM/input/Material.txt": COMMON["Material.txt"],
        "FEM/models/VEM/input/Solver.txt": COMMON["Solver.txt"],
        "FEM/models/VEM/input/Postprocessing.txt": COMMON["Postprocessing.txt"],
    },
    "vem-quadratic": {
        "FEM/models/VEM/weak_form_vem_quadratic.cpp": "weak_form.cpp",
        "FEM/models/VEM/Mesh/Voronoi_2.txt": COMMON["ProblemMesh.txt"],
        "FEM/models/VEM/input/BC.txt": COMMON["BC.txt"],
        "FEM/models/VEM/input/Info_VEM2.txt": COMMON["DOFS_info.txt"],
        "FEM/models/VEM/input/Material.txt": COMMON["Material.txt"],
        "FEM/models/VEM/input/Solver.txt": COMMON["Solver.txt"],
        "FEM/models/VEM/input/Postprocessing.txt": COMMON["Postprocessing.txt"],
    },
    "fsi-he": {
        "FEM/models/FSI_Models/weak_form_fsi.cpp": "weak_form.cpp",
        "FEM/models/FSI_Models/Mesh/fsi_2d.txt": COMMON["ProblemMesh.txt"],
        "FEM/models/FSI_Models/input/BC.txt": COMMON["BC.txt"],
        "FEM/models/FSI_Models/input/Info_FSI.txt": COMMON["DOFS_info.txt"],
        "FEM/models/FSI_Models/input/Material.txt": COMMON["Material.txt"],
        "FEM/models/FSI_Models/input/Solver.txt": COMMON["Solver.txt"],
        "FEM/models/FSI_Models/input/Postprocessing.txt": COMMON["Postprocessing.txt"],
    },
    "fsi-bh": {
        "FEM/models/FSI_Models/weak_form_fsi.cpp": "weak_form.cpp",
        "FEM/models/FSI_Models/Mesh/fsi_2d.txt": COMMON["ProblemMesh.txt"],
        "FEM/models/FSI_Models/input/BC.txt": COMMON["BC.txt"],
        "FEM/models/FSI_Models/input/Info_FSI.txt": COMMON["DOFS_info.txt"],
        "FEM/models/FSI_Models/input/Material_BH.txt": COMMON["Material.txt"],
        "FEM/models/FSI_Models/input/Solver.txt": COMMON["Solver.txt"],
        "FEM/models/FSI_Models/input/Postprocessing.txt": COMMON["Postprocessing.txt"],
    },
    "single-rve-fem": {
        "FEM/models/Elastic_Models/weak_form_hyper_elastic_ad.cpp": "weak_form.cpp",
        "RVEtest/input/Mesh/Micro_Mesh_FEM.txt": COMMON["ProblemMesh.txt"],
        "RVEtest/input/BC.txt": COMMON["BC.txt"],
        "RVEtest/input/DOFS_info.txt": COMMON["DOFS_info.txt"],
        "RVEtest/input/Material.txt": COMMON["Material.txt"],
        "RVEtest/input/Solver.txt": COMMON["Solver.txt"],
        "RVEtest/input/Postprocessing.txt": COMMON["Postprocessing.txt"],
    },
    "single-rve-vem": {
        "FEM/models/VEM/weak_form_vem.cpp": "weak_form.cpp",
        "RVEtest/input/Mesh/Micro_Mesh_VEM.txt": COMMON["ProblemMesh.txt"],
        "FEM/models/VEM/input/BC.txt": COMMON["BC.txt"],
        "RVEtest/input/Mesh/DOFS_info_VEM.txt": COMMON["DOFS_info.txt"],
        "FEM/models/VEM/input/Material.txt": COMMON["Material.txt"],
        "FEM/models/VEM/input/Solver.txt": COMMON["Solver.txt"],
        "FEM/models/VEM/input/Postprocessing.txt": COMMON["Postprocessing.txt"],
    },
    "single-rve-vem-quadratic": {
        "FEM/models/VEM/weak_form_vem_quadratic.cpp": "weak_form.cpp",
        "RVEtest/input/Mesh/Micro_Mesh_VEM2.txt": COMMON["ProblemMesh.txt"],
        "FEM/models/VEM/input/BC.txt": COMMON["BC.txt"],
        "RVEtest/input/Mesh/DOFS_info_VEM2.txt": COMMON["DOFS_info.txt"],
        "FEM/models/VEM/input/Material.txt": COMMON["Material.txt"],
        "FEM/models/VEM/input/Solver.txt": COMMON["Solver.txt"],
        "FEM/models/VEM/input/Postprocessing.txt": COMMON["Postprocessing.txt"],
    },
    "multiscale-fem-hyperelastic": {
        "Multiscale/models/weak_form_macro_fem.cpp": "weak_form_macro.cpp",
        "FEM/models/Elastic_Models/weak_form_hyper_elastic_ad.cpp": "constitutive_model.cpp",
        "Multiscale/input/Mesh/Macro_Mesh_FEM.txt": "input/macro/Macro_Mesh.txt",
        "Multiscale/input/Examples/Macro_FEM_DOFs.txt": "input/macro/DOFS_info.txt",
        "Multiscale/input/BC/BC.txt": "input/macro/BC.txt",
        "Multiscale/input/Mesh/Micro_Mesh_FEM_Q9.txt": "input/rve/Micro_Mesh.txt",
        "Multiscale/input/Mesh/DOFS_info_FEM.txt": "input/rve/DOFS_info.txt",
        "Multiscale/input/Parameters/Material_HE.txt": "input/solver/Material.txt",
        "Multiscale/input/Examples/Elastic_Solver.txt": "input/solver/Solver.txt",
        "Multiscale/input/Examples/Elastic_Postprocessing.txt": "input/solver/Postprocessing.txt",
    },
    "multiscale-fem-elastoplastic": {
        "Multiscale/models/weak_form_macro_fem.cpp": "weak_form_macro.cpp",
        "FEM/models/Inelastic_Models/weak_form_elasto_plastic.cpp": "constitutive_model.cpp",
        "Multiscale/input/Mesh/Lshape20.txt": "input/macro/Macro_Mesh.txt",
        "Multiscale/input/Examples/Macro_FEM_DOFs.txt": "input/macro/DOFS_info.txt",
        "Multiscale/input/BC/Boundary_EP.txt": "input/macro/BC.txt",
        "Multiscale/input/Mesh/Micro_Mesh_FEM.txt": "input/rve/Micro_Mesh.txt",
        "Multiscale/input/Mesh/DOFS_FEM_ElastoPlastic.txt": "input/rve/DOFS_info.txt",
        "Multiscale/input/Parameters/Material_EP.txt": "input/solver/Material.txt",
        "Multiscale/input/Solver.txt": "input/solver/Solver.txt",
        "Multiscale/input/Postprocessing/Postprocessing_EP.txt": "input/solver/Postprocessing.txt",
    },
    "multiscale-fem-viscoelastic": {
        "Multiscale/models/weak_form_macro_fem.cpp": "weak_form_macro.cpp",
        "FEM/models/Inelastic_Models/weak_form_visco_elastic.cpp": "constitutive_model.cpp",
        "Multiscale/input/Mesh/Lshape20.txt": "input/macro/Macro_Mesh.txt",
        "Multiscale/input/Examples/Macro_FEM_DOFs.txt": "input/macro/DOFS_info.txt",
        "Multiscale/input/BC/Boundary_EP.txt": "input/macro/BC.txt",
        "Multiscale/input/Mesh/Micro_Mesh_FEM.txt": "input/rve/Micro_Mesh.txt",
        "Multiscale/input/Mesh/Info_VE.txt": "input/rve/DOFS_info.txt",
        "Multiscale/input/Parameters/Material_VE.txt": "input/solver/Material.txt",
        "Multiscale/input/Solver.txt": "input/solver/Solver.txt",
        "Multiscale/input/Postprocessing/Postprocessing_EP.txt": "input/solver/Postprocessing.txt",
    },
    "multiscale-vem": {
        "Multiscale/models/weak_form_macro_vem.cpp": "weak_form_macro.cpp",
        "FEM/models/VEM/weak_form_vem_quadratic.cpp": "constitutive_model.cpp",
        "Multiscale/input/Mesh/Macro_Mesh_VEM.txt": "input/macro/Macro_Mesh.txt",
        "Multiscale/input/Mesh/DOFS_info_VEM.txt": "input/macro/DOFS_info.txt",
        "Multiscale/input/BC/BC.txt": "input/macro/BC.txt",
        "Multiscale/input/Mesh/Micro_Mesh_VEM2.txt": "input/rve/Micro_Mesh.txt",
        "Multiscale/input/Mesh/DOFS_info_VEM2.txt": "input/rve/DOFS_info.txt",
        "Multiscale/input/Parameters/Material_HE.txt": "input/solver/Material.txt",
        "Multiscale/input/Examples/Elastic_Solver.txt": "input/solver/Solver.txt",
        "Multiscale/input/Examples/Elastic_Postprocessing.txt": "input/solver/Postprocessing.txt",
    },
    "multiscale-vem-quadratic": {
        "Multiscale/models/weak_form_macro_vem_quad.cpp": "weak_form_macro.cpp",
        "FEM/models/VEM/weak_form_vem_quadratic.cpp": "constitutive_model.cpp",
        "Multiscale/input/Mesh/Macro_Mesh_VEM2.txt": "input/macro/Macro_Mesh.txt",
        "Multiscale/input/Examples/Macro_VEM2_DOFs.txt": "input/macro/DOFS_info.txt",
        "Multiscale/input/BC/BC.txt": "input/macro/BC.txt",
        "Multiscale/input/Mesh/Micro_Mesh_VEM2.txt": "input/rve/Micro_Mesh.txt",
        "Multiscale/input/Mesh/DOFS_info_VEM2.txt": "input/rve/DOFS_info.txt",
        "Multiscale/input/Parameters/Material_HE.txt": "input/solver/Material.txt",
        "Multiscale/input/Solver.txt": "input/solver/Solver.txt",
        "Multiscale/input/Postprocessing/Postprocessing.txt": "input/solver/Postprocessing.txt",
    },
}


def sync(cases=None) -> int:
    selected = CASES if cases is None else {case: CASES[case] for case in cases}
    for case, files in selected.items():
        for source, target in files.items():
            destination = EXAMPLES / case / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / source, destination)
    return 0


def check(cases=None) -> int:
    selected = CASES if cases is None else {case: CASES[case] for case in cases}
    stale: list[str] = []
    for case, files in selected.items():
        for source, target in files.items():
            destination = EXAMPLES / case / target
            if not destination.is_file() or not filecmp.cmp(ROOT / source, destination, shallow=False):
                stale.append(f"{case}/{target} <- {source}")
    if stale:
        print("Website example files are missing or stale:")
        print("\n".join(f"  {item}" for item in stale))
        print("Run: python3 Web/tools/sync_examples.py sync")
        return 1
    print(f"Checked {sum(map(len, selected.values()))} synchronized example files.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("sync", "check"))
    parser.add_argument("--case", action="append", choices=CASES, help="Select a case; repeat for multiple cases")
    args = parser.parse_args()
    return sync(args.case) if args.mode == "sync" else check(args.case)


if __name__ == "__main__":
    raise SystemExit(main())
