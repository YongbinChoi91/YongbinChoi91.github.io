#!/usr/bin/env python3
"""Record actual macro-solve and RVE finite-difference check output for an elastic example."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from generate_multiscale_example_pages import CASES, EXAMPLES, hashes, implementation_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=[case["case"] for case in CASES[:2]])
    parser.add_argument("--solve-log", required=True, type=Path)
    parser.add_argument("--sensitivity-log", required=True, type=Path)
    args = parser.parse_args()
    config = next(case for case in CASES if case["case"] == args.case)
    log = args.solve_log.read_text()
    if "Multiscale solve converged: yes" not in log:
        raise SystemExit("The macro solve did not report convergence")
    accepted = re.findall(r"Time step (\d+) accepted: t = ([\d.]+)", log)
    residuals = re.findall(r"Newton Iter \d+: Residual = ([\deE.+-]+)", log)
    iterations = re.findall(r"total Newton iters\s*:\s*(\d+)", log)
    if not accepted or accepted[-1][0] != "2" or float(accepted[-1][1]) != 0.2 or not residuals or not iterations:
        raise SystemExit("Expected the supplied two-step elastic case to finish at t=0.2")
    candidates = []
    for line in args.sensitivity_log.read_text().splitlines():
        if line.startswith('{"passed":'):
            candidates.append(json.loads(line))
    if not candidates:
        raise SystemExit("No sensitivity-check JSON was found")
    sensitivity = candidates[-1]
    error = sensitivity["relative_error"]
    if not sensitivity["passed"] or not math.isfinite(error) or error > 1e-5 or sensitivity["directions"] != 4:
        raise SystemExit("The four-direction sensitivity check did not pass")
    report = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": args.case,
        "input_sha256": hashes(args.case),
        "implementation_sha256": implementation_hashes(config),
        "solve": {
            "converged": True, "accepted_steps": 2, "final_time": 0.2,
            "total_newton_iterations": int(iterations[-1]), "final_residual": float(residuals[-1]),
            "backend": "umfpack",
        },
        "sensitivity": {
            **sensitivity, "deformation_gradient": [[1.015, 0.02], [0.005, 1.01]], "history": "elastic",
            "method": "central difference of converged average stress; all 3x3 stress entries, four in-plane F directions",
        },
    }
    output = EXAMPLES / args.case / "validation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Recorded passing checks for {args.case}; regenerate the example pages next.")


if __name__ == "__main__":
    main()
