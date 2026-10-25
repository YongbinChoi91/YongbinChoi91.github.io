#!/usr/bin/env python3
"""Generate two elastic representative examples and two existing inelastic cases."""

import argparse
import hashlib
import html
import json
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
PAGES = WEB / "pages"
EXAMPLES = WEB / "examples"
CASES = [
    {"case": "multiscale-fem-hyperelastic", "title": "FE² · Linear Macro / Quadratic Micro", "behavior": "Elastic", "order": "Macro Q4 / Micro Q9", "macro": "models/weak_form_macro_fem.cpp", "micro": "../FEM/models/Elastic_Models/weak_form_hyper_elastic_ad.cpp", "material_page": "detail-hyperelastic.html", "description": "A 2D linear macro FE mesh coupled to a quadratic perforated RVE. The micro material is hyperelastic; linear and quadratic describe interpolation order."},
    {"case": "multiscale-vem", "title": "VE² · Linear Macro / Quadratic Micro", "behavior": "Elastic", "order": "Macro k=1 / Micro k=2", "macro": "models/weak_form_macro_vem.cpp", "micro": "../FEM/models/VEM/weak_form_vem_quadratic.cpp", "material_page": "detail-hyperelastic.html", "description": "A 2D linear macro VEM model coupled to the actual quadratic VEM micro kernel. Separate macro and micro DOF files select orders one and two."},
    {"case": "multiscale-fem-elastoplastic", "title": "FE² · Elastoplastic Micro Response", "behavior": "Inelastic", "order": "Existing FEM macro / micro meshes", "macro": "models/weak_form_macro_fem.cpp", "micro": "../FEM/models/Inelastic_Models/weak_form_elasto_plastic.cpp", "material_page": "detail-plasticity.html", "description": "The existing FEM elastoplastic case, with local plastic-state sensitivity and material history committed after an accepted macro step."},
    {"case": "multiscale-fem-viscoelastic", "title": "FE² · Viscoelastic Micro Response", "behavior": "Inelastic", "order": "Existing FEM macro / micro meshes", "macro": "models/weak_form_macro_fem.cpp", "micro": "../FEM/models/Inelastic_Models/weak_form_visco_elastic.cpp", "material_page": "detail-viscoelastic.html", "description": "The existing FEM viscoelastic case, with the macro time increment forwarded to the local viscous update and its algorithmic tangent."},
]
DOWNLOADS = [
    ("weak_form_macro.cpp", "Macro model"), ("constitutive_model.cpp", "Micro model"),
    ("input/macro/Macro_Mesh.txt", "Macro mesh"), ("input/macro/DOFS_info.txt", "Macro DOFs"),
    ("input/macro/BC.txt", "Macro boundary"), ("input/rve/Micro_Mesh.txt", "Micro mesh"),
    ("input/rve/DOFS_info.txt", "Micro DOFs"), ("input/solver/Material.txt", "Material"),
    ("input/solver/Solver.txt", "Solver"), ("input/solver/Postprocessing.txt", "Output"),
]


def hashes(case):
    return {path: hashlib.sha256((EXAMPLES / case / path).read_bytes()).hexdigest() for path, _ in DOWNLOADS}


def implementation_hashes(config):
    root = WEB.parent
    paths = set(root.joinpath("core").rglob("*.cpp")) | set(root.joinpath("core").rglob("*.h"))
    paths.update([
        root / "apps/multiscale_main.cpp",
        root / "Multiscale/Makefile",
        (root / "Multiscale" / config["macro"]).resolve(),
        (root / "Multiscale" / config["micro"]).resolve(),
        WEB / "tools/check_rve_sensitivity.cpp",
    ])
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def run_script(config):
    return f'''#!/usr/bin/env bash
set -euo pipefail
case_dir="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
repo_dir="$(cd "$case_dir/../../.." && pwd)"
# Select the canonical models matching the downloadable source snapshots.
make -C "$repo_dir/Multiscale" \\
  MACRO_MODEL_SRC={config['macro']} \\
  MICRO_MODEL_SRC={config['micro']} \\
  TARGET=bin/web_{config['case']}
run_dir="$repo_dir/Multiscale/output/web/{config['case']}"
mkdir -p "$run_dir"
post_file="$run_dir/Postprocessing.txt"
cat "$case_dir/input/solver/Postprocessing.txt" > "$post_file"
printf '\\nplot_window off\\nplot_each_step off\\nmultiscale_plot_show off\\n' >> "$post_file"
cd "$run_dir"
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-2}}"
exec "$repo_dir/Multiscale/bin/web_{config['case']}" \\
  --macro-mesh "$case_dir/input/macro/Macro_Mesh.txt" \\
  --micro-mesh "$case_dir/input/rve/Micro_Mesh.txt" \\
  --dofs "$case_dir/input/macro/DOFS_info.txt" \\
  --micro-dofs "$case_dir/input/rve/DOFS_info.txt" \\
  --macro-bc "$case_dir/input/macro/BC.txt" \\
  --material "$case_dir/input/solver/Material.txt" \\
  --solver "$case_dir/input/solver/Solver.txt" \\
  --post "$post_file"
'''


def validation_section(config):
    case = config["case"]
    record = EXAMPLES / case / "validation.json"
    if config["behavior"] != "Elastic" or not record.exists():
        return '<section class="doc-section section-wrap cook-page"><p class="eyebrow">Existing case</p><h2>Material-history example</h2><p>This case retains its existing loading and material configuration. The new linear-macro/quadratic-micro validation is provided by the two Elastic examples.</p></section>'
    report = json.loads(record.read_text())
    if report["input_sha256"] != hashes(case):
        raise ValueError(f"{case}: validation belongs to different sources or inputs; rerun the check")
    if report["implementation_sha256"] != implementation_hashes(config):
        raise ValueError(f"{case}: coupling implementation has changed since validation; rerun the check")
    values = report["solve"]
    tangent = report["sensitivity"]
    if not values["converged"] or not tangent["passed"]:
        raise ValueError(f"{case}: validation did not pass")
    return f'''<section class="doc-section section-wrap cook-page"><p class="eyebrow">Verification</p><h2>Two load increments and four sensitivity directions.</h2><div class="equation-code-table-wrap"><table class="equation-code-table"><thead><tr><th>Check</th><th>Observed result</th></tr></thead><tbody>
<tr><td>Macro solve</td><td>{values['accepted_steps']} accepted steps, final load time {values['final_time']}, {values['total_newton_iterations']} total Newton iterations</td></tr>
<tr><td>Final macro residual</td><td>{values['final_residual']:.3e}</td></tr>
<tr><td>Central-difference sensitivity check</td><td>4 macro deformation-gradient directions; relative Frobenius error {tangent['relative_error']:.3e}, perturbation {tangent['epsilon']:.0e}</td></tr>
</tbody></table></div><p>The sensitivity check compares <code>C / volume</code> with derivatives of the converged average stress at a prescribed finite deformation. <a href="../examples/{case}/validation.json" download>Download the verification record</a>.</p></section>'''


def page(config):
    case = config["case"]
    title = html.escape(config["title"])
    description = html.escape(config["description"])
    cards = "".join(f'<a class="download-card" download href="../examples/{case}/{path}"><span class="tag">{label}</span><h3>{Path(path).name}</h3></a>' for path, label in DOWNLOADS)
    cards += f'<a class="download-card" download href="../examples/{case}/run.sh"><span class="tag">Run</span><h3>run.sh</h3></a>'
    elastic = config["behavior"] == "Elastic"
    inputs = ('Two load increments: <code>dt 0.1</code>, <code>total_time 0.2</code>; UMFPACK; RVE substepping off. '
              'The macro boundary file fixes the left edge and applies <code>NEUMANN RIGHT uy 1.0*t</code>.' if elastic else
              'The existing macro loading, material, and RVE history configuration are retained. Macro and micro DOFs are passed separately; material history is declared in the micro DOF file.')
    return rf'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="{description}"><title>{title} · HiPaCs</title>
<link href="../assets/css/styles.css" rel="stylesheet"><script src="../assets/js/mathjax-config.js"></script><script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script></head>
<body><header class="topbar"><a class="brand" href="../index.html"><span class="brand-mark">HC</span><span>HiPaCs</span></a><button class="menu-button" type="button" aria-expanded="false" aria-label="Open navigation"><span></span><span></span><span></span></button><nav class="top-links" aria-label="Primary navigation"><a href="../index.html">Home</a><a href="../architecture.html">Code Structure</a><a href="../applications.html">Applications</a><a href="detail-multiscale-analysis.html">Computational Homogenization</a></nav></header>
<main class="homogenization-example"><section class="subpage-hero section-wrap cook-page"><a class="back-button route-card" href="detail-multiscale-analysis.html">&larr; Back to Computational Homogenization</a><p class="eyebrow">{config['behavior']} / {config['order']}</p><h1>{title}</h1><p class="lead">{description}</p></section>
<section class="doc-section section-wrap cook-page"><div class="section-heading"><p class="eyebrow">Response transfer</p><h2>One equilibrated RVE response per macro evaluation point.</h2><p>See <a href="detail-multiscale-analysis.html">Computational Homogenization</a> for the sensitivity derivation and <a href="{config['material_page']}">Solid</a> for the micro material theory.</p></div>
<div class="math-explain-grid theory-grid"><article class="math-card wide"><span class="tag">Stress and tangent</span><div class="formula-block">\[\bar{{\mathbf P}}=\frac{{\mathrm{{intP}}}}{{V_\mu}},\qquad\bar{{\mathbb A}}=\frac{{\mathrm{{C}}}}{{V_\mu}},\qquad\mathrm{{C}}=\frac{{\partial\mathrm{{intP}}}}{{\partial\mathbf u_\mu}}\frac{{\partial\mathbf u_\mu}}{{\partial\bar{{\mathbf F}}}}.\]</div><p>The micro model provides <code>volume</code>, <code>intP</code>, and <code>dIntPdu</code>. The periodic solver computes <code>du_dF</code>, and the macro kernel uses the normalized sensitivity in its AD stress expression.</p></article></div>
<div class="section-heading compact-heading"><p class="eyebrow">Input</p><h2>{config['order']}</h2><p>{inputs}</p></div>
<div class="equation-code-table-wrap"><table class="equation-code-table"><thead><tr><th>Component</th><th>Source / configuration</th></tr></thead><tbody>
<tr><td>Macro model</td><td><code>Multiscale/{config['macro']}</code></td></tr><tr><td>Micro model</td><td><code>{html.escape(config['micro'].removeprefix('../'))}</code></td></tr>
<tr><td>Macro field layout</td><td><code>input/macro/DOFS_info.txt</code></td></tr><tr><td>Micro field and history layout</td><td><code>input/rve/DOFS_info.txt</code>, selected with <code>--micro-dofs</code></td></tr>
</tbody></table></div>
<div class="section-heading compact-heading"><p class="eyebrow">Build and run</p><h2>Run from a HiPaCs checkout.</h2><p>Activate the environment described in <a href="../install.html">Installation</a>, then run this command from the repository root. The script selects the two model sources and passes input paths directly; results are written under <code>Multiscale/output/web/{case}/</code>. Interactive plotting is disabled for this command.</p></div>
<div class="code-window"><pre><code>bash Web/examples/{case}/run.sh</code></pre></div>
<details class="code-details"><summary>Complete build and input command</summary><div class="code-window"><pre><code>{html.escape(run_script(config))}</code></pre></div></details>
<div class="section-heading compact-heading"><p class="eyebrow">Downloads</p><h2>Macro, micro, and solver files</h2></div><div class="download-grid">{cards}</div></section>
{validation_section(config)}</main><script defer src="../assets/js/main.js"></script></body></html>
'''


def generated_outputs():
    outputs = {}
    for config in CASES:
        outputs[PAGES / f"detail-{config['case']}.html"] = page(config)
        outputs[EXAMPLES / config["case"] / "run.sh"] = run_script(config)
    outputs[PAGES / "detail-multiscale-vem-quadratic.html"] = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="0; url=detail-multiscale-vem.html"><link rel="canonical" href="https://yongbinchoi91.github.io/pages/detail-multiscale-vem.html"><meta name="robots" content="noindex"><title>VE² example moved · HiPaCs</title></head><body><main><h1>VE² · Linear Macro / Quadratic Micro</h1><p><a href="detail-multiscale-vem.html">Open the representative VE² example →</a></p></main></body></html>\n'''
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = generated_outputs()
    stale = [path for path, text in outputs.items() if not path.exists() or path.read_text() != text]
    if args.check:
        if stale:
            raise SystemExit("Stale multiscale pages or commands:\n" + "\n".join(str(p.relative_to(WEB)) for p in stale))
        print("Four multiscale example pages, commands, and the legacy redirect are current.")
    else:
        for path, text in outputs.items():
            path.write_text(text)
        print("Generated four example pages, four run commands, and the legacy redirect.")


if __name__ == "__main__":
    main()
