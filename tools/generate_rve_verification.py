#!/usr/bin/env python3
"""Render actual Single RVE finite-difference sweeps as plots and component tables."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

from sync_examples import CASES as SOURCE_CASES

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "Web"
SCRIPT = Path(__file__).resolve()
CHECKER = WEB / "tools/sweep_rve_sensitivity.cpp"
CASES = {
    "single-rve-fem": ("FEM", "detail-single-rve-fem.html", "#276b92"),
    "single-rve-vem": ("VEM k=1", "detail-single-rve-vem.html", "#bd6c39"),
    "single-rve-vem-quadratic": ("VEM k=2", "detail-single-rve-vem-quadratic.html", "#388475"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(case):
    return {target: digest(WEB / "examples" / case / target) for target in SOURCE_CASES[case].values()}


def implementation(case):
    model = next(source for source, target in SOURCE_CASES[case].items() if target == "weak_form.cpp")
    files = set((ROOT / "core").rglob("*.cpp")) | set((ROOT / "core").rglob("*.h"))
    files.update([ROOT / model, CHECKER, ROOT / "Multiscale/Makefile"])
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(files)}


def validate(report):
    assert report["passed"] and report["quantity"] == "integrated_piola"
    assert report["solve_count"] == 65
    assert len(report["analytic"]) == 9 and all(len(row) == 4 for row in report["analytic"])
    assert len(report["sweep"]) == 8
    assert all(len(m["finite_difference"]) == 9 and all(len(row) == 4 for row in m["finite_difference"]) for m in report["sweep"])
    for m in report["sweep"]:
        fd = m["finite_difference"]
        square_error = sum((report["analytic"][p][q] - fd[p][q]) ** 2 for p in range(9) for q in range(4))
        square_reference = sum(fd[p][q] ** 2 for p in range(9) for q in range(4))
        expected = math.sqrt(square_error / max(square_reference, 1e-30))
        assert math.isfinite(expected) and expected > 0
        assert math.isclose(expected, m["relative_error"], rel_tol=1e-10)
    assert representative(report)["relative_error"] < report["acceptance_relative_error"]


def representative(report):
    return next(m for m in report["sweep"] if m["epsilon"] == report["representative_epsilon"])


def component_csv(report):
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["epsilon", "Piola_component", "F_component", "integrated_sensitivity", "central_difference", "absolute_error"])
    for measurement in report["sweep"]:
        for p in range(9):
            for q in range(4):
                value = report["analytic"][p][q]
                fd = measurement["finite_difference"][p][q]
                writer.writerow([format(measurement["epsilon"], ".17g"), f"P{p // 3 + 1}{p % 3 + 1}",
                                 f"F{q // 2 + 1}{q % 2 + 1}", format(value, ".17g"), format(fd, ".17g"), format(abs(value - fd), ".17g")])
    return out.getvalue()


def component_table(report, in_plane=True):
    measurement = representative(report)
    rows = []
    for p in range(9):
        if (p in (0, 1, 3, 4)) != in_plane:
            continue
        for q in range(4):
            analytic = report["analytic"][p][q]
            fd = measurement["finite_difference"][p][q]
            rows.append(f'<tr><td>P{p // 3 + 1}{p % 3 + 1}</td><td>F{q // 2 + 1}{q % 2 + 1}</td>'
                        f'<td>{analytic:.11e}</td><td>{fd:.11e}</td><td>{abs(analytic - fd):.3e}</td></tr>')
    return ('<div class="equation-code-table-wrap"><table class="equation-code-table rve-component-table">'
            '<thead><tr><th scope="col">Stress</th><th scope="col">Perturbation</th><th scope="col">Sensitivity</th>'
            '<th scope="col">Central difference</th><th scope="col">Absolute error</th></tr></thead><tbody>'
            + "".join(rows) + '</tbody></table></div>')


METHOD = r'''<div class="math-explain-grid theory-grid"><article class="math-card wide"><span class="tag">Central-difference verification</span><h3>Re-equilibrate the RVE at every perturbation.</h3>
<div class="formula-block">\[\mathbf I_P(\bar{\mathbf F})=\int_{\Omega_\mu^0}\mathbf P_\mu\,dV,\qquad C_{iAjB}=\frac{d I_{P,iA}}{d\bar F_{jB}}.\]</div>
<div class="formula-block">\[C^{\mathrm{FD}}_{iAjB}(\varepsilon)=\frac{I_{P,iA}(\bar{\mathbf F}+\varepsilon\mathbf E^{jB})-I_{P,iA}(\bar{\mathbf F}-\varepsilon\mathbf E^{jB})}{2\varepsilon},\qquad e_{\mathrm{rel}}=\frac{\|\mathbf C-\mathbf C^{\mathrm{FD}}\|_F}{\|\mathbf C^{\mathrm{FD}}\|_F}.\]</div>
<p>Both sides use the integrated Piola convention of <code>result.sensitivity.C</code>. All 9 stress entries and 4 in-plane deformation-gradient directions enter the error norm. Dividing both sensitivities by the same fixed reference volume gives the averaged-stress tangent comparison.</p></article></div>'''


def child_section(case, report):
    label = CASES[case][0]
    m = representative(report)
    return rf'''<section id="sensitivity-verification" class="doc-section section-wrap cook-page rve-verification">
<div class="section-heading"><p class="eyebrow">Sensitivity verification / {label}</p><h2>Finite-difference agreement across perturbation sizes.</h2><p>Measured on this downloadable elastic RVE at \(\bar{{\mathbf F}}=\begin{{bmatrix}}1.015&amp;0.02\\0.005&amp;1.01\end{{bmatrix}}\). Each of the 64 perturbed problems is solved independently, plus one base solve. UMFPACK is used with RVE tolerance \(10^{{-12}}\); adaptive micro substepping is off.</p></div>
{METHOD}
<figure class="rve-error-figure"><img src="../examples/{case}/sensitivity-error.svg" width="1100" height="620" loading="lazy" alt="{label}: log-log graph of measured integrated-stress sensitivity relative error versus central-difference perturbation size."/><figcaption>Large perturbations introduce truncation error; very small perturbations amplify numerical errors. The dashed vertical line marks the same preselected \(\varepsilon=10^{{-6}}\) used in every component table.</figcaption></figure>
<div class="rve-check-summary"><article><span>Relative Frobenius error</span><strong>{m['relative_error']:.3e}</strong></article><article><span>Maximum component absolute error</span><strong>{m['max_absolute_error']:.3e}</strong></article><article><span>Largest converged RVE residual</span><strong>{report['max_rve_residual']:.3e}</strong></article></div>
<div class="section-heading compact-heading"><p class="eyebrow">Component comparison</p><h2>In-plane components at \(\varepsilon=10^{{-6}}\).</h2><p>Each row compares the implemented integrated sensitivity with the independently re-equilibrated central difference. Absolute errors remain meaningful for zero and near-zero components.</p></div>
{component_table(report)}
<details class="code-details"><summary>Additional plane-strain stress components (20 entries)</summary>{component_table(report, False)}</details>
<p class="rve-download-links"><a href="../examples/{case}/sensitivity.json" download>Full measurement record (JSON)</a> &middot; <a href="../examples/{case}/sensitivity-components.csv" download>All components and perturbations (CSV)</a> &middot; <a href="../examples/{case}/sensitivity-error.svg" download>Graph (SVG)</a> &middot; <a href="../examples/{case}/sensitivity-error.png" download>Graph (PNG)</a></p>
<p>The measured discrepancies are finite numerical errors. This check verifies the homogenized integrated-stress tangent at the stated deformation; it does not measure spatial-discretization error or independently test each intermediate sensitivity.</p>
</section>'''


def overview(reports):
    rows = []
    for case, report in reports.items():
        label, page, _ = CASES[case]
        m = representative(report)
        rows.append(f'<tr><th scope="row"><a href="{page}#sensitivity-verification">{label}</a></th>'
                    f'<td>{m["relative_error"]:.3e}</td><td>{m["max_absolute_error"]:.3e}</td>'
                    f'<td><a href="{page}#sensitivity-verification">Graph and component table →</a></td></tr>')
    return r'''<section id="rve-sensitivity-overview" class="doc-section section-wrap cook-page rve-verification">
<div class="section-heading"><p class="eyebrow">Finite-difference verification</p><h2>Check the transferred sensitivity against solved RVE responses.</h2><p>All three elastic examples use the same prescribed deformation and perturbation sweep. Every perturbed RVE is re-equilibrated before taking a central difference of its integrated Piola stress.</p></div>
<figure class="rve-error-figure"><img src="../assets/images/rve-sensitivity-comparison.svg" width="1100" height="620" loading="lazy" alt="Measured sensitivity relative-error curves for FEM, linear VEM, and quadratic VEM over eight central-difference perturbation sizes."/><figcaption>The curves measure agreement of each implementation with its own finite-difference response. They are not a comparison of FE and VE discretization accuracy.</figcaption></figure>
<div class="section-heading compact-heading"><h3>Same representative perturbation: \(\varepsilon=10^{-6}\)</h3></div>
<div class="equation-code-table-wrap"><table class="equation-code-table"><thead><tr><th>RVE</th><th>Relative Frobenius error</th><th>Maximum absolute component error</th><th>Details</th></tr></thead><tbody>''' + "".join(rows) + r'''</tbody></table></div>
<p>\(\bar{\mathbf F}=\begin{bmatrix}1.015&amp;0.02\\0.005&amp;1.01\end{bmatrix}\), RVE tolerance \(10^{-12}\), 65 equilibrium solves per example. Small nonzero errors reflect finite-difference truncation, solver convergence, and floating-point precision.</p>
<p><a href="../assets/images/rve-sensitivity-comparison.svg" download>Download comparison graph (SVG)</a></p></section>'''


def replace_section(source, marker, content):
    pattern = rf"(<!-- GENERATED:{marker}:START -->).*?(<!-- GENERATED:{marker}:END -->)"
    result, count = re.subn(pattern, lambda m: m[1] + "\n" + content + "\n" + m[2], source, flags=re.S)
    if count != 1:
        raise ValueError(f"Expected exactly one {marker} section")
    return result


def render_plot(reports, output_stem, comparison=False):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hipacs-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt": "hipacs-single-rve-sensitivity", "font.family": "DejaVu Sans", "font.size": 12})
    fig, ax = plt.subplots(figsize=(11, 6.2), layout="constrained")
    for case, report in reports.items():
        label, _, color = CASES[case]
        ax.loglog([m["epsilon"] for m in report["sweep"]], [m["relative_error"] for m in report["sweep"]],
                  "o-", color=color, lw=2.2, ms=6, label=label if comparison else "Measured relative error")
    ax.axvline(1e-6, color="#8695a2", ls="--", lw=1.2, label="Component table: ε = 10⁻⁶")
    if not comparison:
        report = next(iter(reports.values()))
        reference = next(m for m in report["sweep"] if m["epsilon"] == 1e-3)
        x = [1e-4, 1e-3, 1e-2]
        ax.loglog(x, [reference["relative_error"] * (v / 1e-3) ** 2 for v in x], ":", color="#8797a0", lw=1.5, label="ε² reference slope")
    ax.set_xlabel("Central-difference perturbation ε")
    ax.set_ylabel("Relative Frobenius error of integrated sensitivity")
    ax.set_title("Single RVE sensitivity · finite-difference verification", loc="left", pad=18, fontweight="bold")
    ax.grid(which="major", color="#dce5eb", lw=.8)
    ax.grid(which="minor", color="#edf2f6", lw=.4)
    ax.legend(loc="upper left", frameon=True, framealpha=.95, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(5e-10, 2e-2)
    for suffix in (".svg", ".png"):
        path = output_stem.with_suffix(suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, metadata={"Date": None} if suffix == ".svg" else {})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="append", default=[], metavar="CASE=OUTPUT.json")
    parser.add_argument("--check", action="store_true", help="Check measured-data freshness and generated pages without plotting")
    args = parser.parse_args()
    if args.check and args.collect:
        parser.error("--check cannot collect new measurements")
    for item in args.collect:
        case, path = item.split("=", 1)
        if case not in CASES:
            parser.error(f"Unknown case: {case}")
        report = json.loads(Path(path).read_text())
        validate(report)
        report.update({"case": case, "validated_at_utc": datetime.now(timezone.utc).isoformat(),
                       "input_sha256": inputs(case), "implementation_sha256": implementation(case)})
        (WEB / "examples" / case / "sensitivity.json").write_text(json.dumps(report, indent=2) + "\n")
    reports = {}
    for case in CASES:
        report = json.loads((WEB / "examples" / case / "sensitivity.json").read_text())
        validate(report)
        if report["input_sha256"] != inputs(case) or report["implementation_sha256"] != implementation(case):
            raise SystemExit(f"{case}: measured sensitivities belong to different sources or inputs; rerun the sweep")
        reports[case] = report
    combined_stem = WEB / "assets/images/rve-sensitivity-comparison"
    if not args.check:
        for case, report in reports.items():
            render_plot({case: report}, WEB / "examples" / case / "sensitivity-error")
        render_plot(reports, combined_stem, comparison=True)
        for case, report in reports.items():
            artifacts = [WEB / "examples" / case / f"sensitivity-error{suffix}" for suffix in (".svg", ".png")]
            artifacts += [combined_stem.with_suffix(suffix) for suffix in (".svg", ".png")]
            report["plot_generator_sha256"] = digest(SCRIPT)
            report["artifacts_sha256"] = {str(path.relative_to(WEB)): digest(path) for path in artifacts}
            (WEB / "examples" / case / "sensitivity.json").write_text(json.dumps(report, indent=2) + "\n")
    outputs = {}
    for case, report in reports.items():
        if args.check:
            assert report["plot_generator_sha256"] == digest(SCRIPT), "Plot generator changed; regenerate plots"
            for relative, expected in report["artifacts_sha256"].items():
                assert digest(WEB / relative) == expected, f"Plot changed: {relative}"
        page = WEB / "pages" / CASES[case][1]
        outputs[page] = replace_section(page.read_text(), "RVE_VERIFICATION", child_section(case, report))
        outputs[WEB / "examples" / case / "sensitivity-components.csv"] = component_csv(report)
    page = WEB / "pages/detail-single-rve.html"
    outputs[page] = replace_section(page.read_text(), "RVE_OVERVIEW", overview(reports))
    if args.check:
        stale = [str(path.relative_to(WEB)) for path, text in outputs.items() if not path.exists() or path.read_text() != text]
        if stale:
            raise SystemExit("Stale verification pages or component data:\n" + "\n".join(stale))
        print("Three Single RVE sweeps, plots, component tables, and source hashes are current.")
    else:
        for path, text in outputs.items():
            path.write_text(text)
        print("Rendered Single RVE error plots and component tables for all three measured examples.")


if __name__ == "__main__":
    main()
