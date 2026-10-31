#!/usr/bin/env python3
"""Refresh the FSI page from its published model and input snapshot."""

import argparse
import html
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "Web"
MODEL = Path("Web/examples/fsi/weak_form_fsi.cpp")
MESH = ROOT / "Web/examples/fsi/input/ProblemMesh.txt"


def excerpt(path, start, end):
    source = (ROOT / path).read_text()
    first = source.index(start)
    last = source.index(end, first + len(start))
    code = source[first:last].rstrip()
    line = source[:first].count("\n") + 1
    return code, line


def snippets():
    sections = [
        ("1. Local fields and shifted time weights", MODEL,
         '    const DomainLocalLayout L = make_domain_layout(mesh, elem_id, state.domain_type);',
         '    for (const auto& gp : gauss_rule'),
        ("2. Fluid momentum and incompressibility", MODEL,
         '        // Momentum weak scalar.', '        // Flow-domain mesh motion.'),
        ("3. Solid momentum, kinematics, and pressure equation", MODEL,
         '        // Momentum balance against delta_v.', '    return G;'),
        ("4. Domain dispatcher", MODEL,
         'static Scalar weak_scalar_domain(', '// ============================================================================\n// 5.'),
        ("5. AD residual and tangent", MODEL,
         'ElementSystem compute_constitutive_element_system(', 'static bool fsi_postprocess_requested'),
        ("6. Neo-Hookean Piola helper", Path("core/space/kinematics.h"),
         'inline std::vector<std::vector<Scalar>> neo_hookean_first_piola_lame(',
         'template <class Scalar>\ninline std::vector<std::vector<Scalar>> neo_hookean_first_piola_young'),
        ("7. Shifted Crank–Nicolson helper", Path("core/hpx/time_integrator.h"),
         'inline TimeSchemeStencils Shift_Crank_Nicolson(', 'inline LinearMultistepCoefficients bdf_coefficients'),
    ]
    output = []
    for index, (title, path, start, end) in enumerate(sections):
        code, line = excerpt(path, start, end)
        link = f"https://github.com/YongbinChoi91/HiPaCs/blob/main/{path}#L{line}"
        output.append(
            f'<details class="code-details"{" open" if index == 0 else ""}>'
            f'<summary>{html.escape(title)}</summary>'
            f'<p class="fsi-source-reference"><a href="{link}" target="_blank" rel="noopener">'
            f'{html.escape(str(path))}:{line}</a></p>'
            f'<div class="code-window"><pre><code>{html.escape(code)}</code></pre></div></details>'
        )
    return "\n".join(output)


def mesh_svg():
    nodes = {}
    elements = []
    section = domain = ""
    for raw in MESH.read_text().splitlines():
        line = raw.strip()
        if line.startswith("DOMAIN "):
            domain = line.split()[1]
            section = ""
        elif line == "END_DOMAIN":
            domain = section = ""
        elif line == "VELOCITY_NODES":
            section = "nodes"
        elif line == "MIXED_ELEMENTS" and domain:
            section = "elements"
        elif re.match(r"^\d+\s", line):
            if section == "nodes":
                node, x, y = line.split()[:3]
                nodes[int(node)] = (float(x), -float(y))
            elif section == "elements":
                ids = [int(value) for value in line.split("|")[0].split()[1:]]
                if len(ids) != 9:
                    raise ValueError("The FSI preview expects Q9 velocity elements")
                elements.append((domain, ids))
        elif line and not line.startswith("#"):
            section = ""
    assert nodes and {domain for domain, _ in elements} == {"fluid", "solid"}
    counts = Counter(domain for domain, _ in elements)
    groups = []
    for domain, color, stroke in (("fluid", "#e7f3f9", "#76a9c3"), ("solid", "#f4b183", "#a9562a")):
        paths = []
        for element_domain, ids in elements:
            if element_domain != domain:
                continue
            points = [nodes[node] for node in ids]
            path = f'M {points[0][0]:.7g},{points[0][1]:.7g}'
            for corner in range(4):
                a, b, mid = points[corner], points[(corner + 1) % 4], points[corner + 4]
                control = (2 * mid[0] - (a[0] + b[0]) / 2, 2 * mid[1] - (a[1] + b[1]) / 2)
                path += f' Q {control[0]:.7g},{control[1]:.7g} {b[0]:.7g},{b[1]:.7g}'
            paths.append(f'<path d="{path} Z" vector-effect="non-scaling-stroke"/>')
        groups.append(f'<g fill="{color}" stroke="{stroke}" stroke-width="0.45">' + "".join(paths) + '</g>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="650" viewBox="0 0 1120 650" role="img" aria-labelledby="title description">
<title id="title">2D FEM FSI reference mesh</title>
<desc id="description">Actual quadratic element edges from fsi_2d.txt: {counts['fluid']} fluid elements and {counts['solid']} solid elements. Full channel above; cylinder and flag detail below.</desc>
<defs><g id="mesh">{''.join(groups)}</g></defs>
<rect width="1120" height="650" rx="20" fill="#f8fbfd"/>
<g fill="#17384a" font-family="system-ui, sans-serif"><text x="34" y="40" font-size="23" font-weight="700">2D reference mesh</text>
<text x="34" y="70" font-size="16">{counts['fluid']:,} fluid elements · {counts['solid']:,} solid elements · Q9 / Q9 / Q4</text>
<text x="34" y="108" font-size="16">Full channel</text><text x="34" y="340" font-size="16">Cylinder and flexible flag · interface detail</text></g>
<svg x="34" y="125" width="1052" height="184" viewBox="-0.01 -0.42 2.52 0.44" preserveAspectRatio="xMidYMid meet"><use href="#mesh"/></svg>
<svg x="34" y="360" width="1052" height="235" viewBox="0.09 -0.32 0.56 0.24" preserveAspectRatio="xMidYMid meet"><use href="#mesh"/></svg>
<g font-family="system-ui, sans-serif" font-size="16" fill="#17384a"><rect x="34" y="617" width="18" height="12" fill="#e7f3f9" stroke="#76a9c3"/><text x="62" y="629">Fluid</text><rect x="140" y="617" width="18" height="12" fill="#f4b183" stroke="#a9562a"/><text x="168" y="629">Solid</text><text x="270" y="629">Undeformed geometry · not a simulation result</text></g>
</svg>
'''


def generated_outputs():
    page = WEB / "pages/app-fsi.html"
    source = page.read_text()
    pattern = r"(<!-- GENERATED:FSI_SNIPPETS:START -->).*?(<!-- GENERATED:FSI_SNIPPETS:END -->)"
    rendered, count = re.subn(pattern, lambda match: match[1] + "\n" + snippets() + "\n" + match[2], source, flags=re.S)
    if count != 1:
        raise ValueError("Expected exactly one FSI source-excerpt section")
    outputs = {page: rendered.encode(), WEB / "assets/images/fsi-2d-mesh.svg": mesh_svg().encode()}
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check freshness without writing files")
    args = parser.parse_args()
    outputs = generated_outputs()
    stale = [path for path, content in outputs.items() if not path.exists() or path.read_bytes() != content]
    if args.check:
        if stale:
            print("Stale FSI artifacts:\n" + "\n".join(str(path.relative_to(ROOT)) for path in stale))
            return 1
        print("FSI excerpts and 2D mesh match the published example snapshot.")
        return 0
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    print(f"Updated {len(outputs)} FSI artifacts from the published 2D example.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
