#!/usr/bin/env python3
"""Elevate the existing Q4 RVE to Q9 for the linear-macro/quadratic-micro example."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Multiscale/input/Mesh/Micro_Mesh_FEM.txt"
TARGET = ROOT / "Multiscale/input/Mesh/Micro_Mesh_FEM_Q9.txt"


def quadratic_mesh():
    nodes, elements, boundaries = {}, [], {}
    section = ""
    for raw in SOURCE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "DISPLACEMENT_NODES" or line == "MIXED_ELEMENTS" or line.startswith("DISPLACEMENT_BOUNDARY_"):
            section = line
            if line.startswith("DISPLACEMENT_BOUNDARY_"):
                boundaries[line] = []
        elif line[0].isdigit():
            values = line.split()
            if section == "DISPLACEMENT_NODES":
                nodes[int(values[0])] = tuple(map(float, values[1:]))
            elif section == "MIXED_ELEMENTS":
                assert len(values) == 5, "Expected displacement-only Q4 input"
                elements.append(tuple(map(int, values)))
            elif section in boundaries:
                assert len(values) == 3, "Expected two-node Q4 boundary edges"
                boundaries[section].append(tuple(map(int, values)))
        else:
            section = ""
    assert nodes and elements
    mids = {}
    def edge_mid(a, b):
        key = tuple(sorted((a, b)))
        if key not in mids:
            node = max(nodes) + 1
            nodes[node] = tuple((x + y) / 2 for x, y in zip(nodes[a], nodes[b]))
            mids[key] = node
        return mids[key]
    elevated = []
    for element, *corners in elements:
        edges = [edge_mid(corners[i], corners[(i + 1) % 4]) for i in range(4)]
        center = max(nodes) + 1
        nodes[center] = tuple(sum(nodes[n][axis] for n in corners) / 4 for axis in range(len(nodes[corners[0]])))
        elevated.append([element, *corners, *edges, center])
    out = ["DIM 2", "MESH_TYPE GENERAL_MIXED", "FIELDS displacement", "",
           "# Q9 elevation of Micro_Mesh_FEM.txt; shared edge midpoints and bilinear centers.",
           "DISPLACEMENT_NODES"]
    out += [str(node) + " " + " ".join(f"{value:.17g}" for value in coordinates) for node, coordinates in sorted(nodes.items())]
    out += ["", "MIXED_ELEMENTS"]
    out += [" ".join(map(str, row)) for row in elevated]
    for name, rows in boundaries.items():
        out += ["", name]
        for edge, a, b in rows:
            out.append(f"{edge} {a} {b} {mids[tuple(sorted((a, b)))]}")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = quadratic_mesh()
    if args.check:
        if not TARGET.exists() or TARGET.read_text() != expected:
            raise SystemExit("Q9 RVE is stale; run python3 Web/tools/generate_multiscale_inputs.py")
        print("Q9 RVE matches its Q4 source.")
    else:
        TARGET.write_text(expected)
        print(f"Generated {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
