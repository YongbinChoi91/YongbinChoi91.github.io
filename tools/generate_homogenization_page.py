#!/usr/bin/env python3
"""Refresh the homogenization page's excerpts from the actual coupling code."""

import argparse
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "Web/pages/detail-multiscale-analysis.html"


def snippet(title, path, start, end, opened=False):
    source = (ROOT / path).read_text()
    first = source.index(start)
    last = source.index(end, first + len(start))
    code = source[first:last].rstrip()
    line = source[:first].count("\n") + 1
    url = f"https://github.com/YongbinChoi91/HiPaCs/blob/main/{path}#L{line}"
    return (f'<details class="code-details"{" open" if opened else ""}><summary>{html.escape(title)}</summary>'
            f'<p class="homogenization-source-reference"><a href="{url}" target="_blank" rel="noopener">'
            f'{html.escape(path)}:{line}</a></p>'
            f'<div class="code-window"><pre><code>{html.escape(code)}</code></pre></div></details>')


def generated_page():
    elastic = [
        snippet("1. Four periodic sensitivity solves", "core/multiscale/rve_solver.cpp",
                "static std::vector<std::vector<double>> compute_rve_displacement_sensitivity(",
                "struct RveRuntimeCache", True),
        snippet("2. Integrated-stress chain rule", "core/multiscale/rve_homogenization.cpp",
                "RveSensitivity compute_rve_macro_sensitivity_from_piola_du(",
                "RveSensitivity compute_rve_macro_sensitivity("),
        snippet("3. Normalize and expose the tangent to macro AD", "Multiscale/models/weak_form_macro_fem.cpp",
                "                Scalar MacroP = Scalar(rve.macro.Pbar[i][A]);",
                "                G += MacroP"),
    ]
    inelastic = [
        snippet("1. Elastoplastic local-state sensitivity", "FEM/models/Inelastic_Models/weak_form_elasto_plastic.cpp",
                "    const auto Q_h_lu = hpx::factorize_local_dense_lu(Q_h,",
                "static Matrix implicit_dh_dF_hpx(", True),
        snippet("2. Viscoelastic local-state sensitivity", "FEM/models/Inelastic_Models/weak_form_visco_elastic.cpp",
                "    const auto Q_h_lu = hpx::factorize_local_dense_lu(Q_h,",
                "static Matrix implicit_dh_dF_hpx("),
        snippet("3. Forward the time data and accepted micro history", "core/multiscale/macro_rve_common.h",
                "    RveBatchTask task;", "inline std::array<std::array<double,2>,2> macro_F_from_grad("),
        snippet("4. Separate accepted history from trial warm starts", "core/multiscale/multiscale_context.cpp",
                "RveWarmStartState get_multiscale_rve_warm_start(",
                "void store_multiscale_rve_warm_start("),
        snippet("5. Commit or discard at the macro-step boundary", "core/multiscale/multiscale_context.cpp",
                "void commit_multiscale_rve_history_cache()", "void observe_multiscale_rve_diagnostics("),
    ]
    source = PAGE.read_text()
    for key, sections in (("ELASTIC", elastic), ("INELASTIC", inelastic)):
        pattern = rf"(<!-- GENERATED:HOM_{key}_CODE:START -->).*?(<!-- GENERATED:HOM_{key}_CODE:END -->)"
        source, count = re.subn(pattern, lambda m: m[1] + "\n" + "\n".join(sections) + "\n" + m[2], source, flags=re.S)
        if count != 1:
            raise ValueError(f"Expected exactly one {key} source-excerpt section")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = generated_page()
    if args.check:
        if PAGE.read_text() != source:
            raise SystemExit("Homogenization excerpts are stale; run python3 Web/tools/generate_homogenization_page.py")
        print("Homogenization excerpts match the current coupling and material-update code.")
    else:
        PAGE.write_text(source)
        print("Updated elastic and inelastic coupling excerpts.")


if __name__ == "__main__":
    main()
