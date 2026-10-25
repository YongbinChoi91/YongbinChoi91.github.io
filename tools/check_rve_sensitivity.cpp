// Integration check: differentiate the actual converged periodic RVE response.
// Build with Multiscale/Makefile and the same micro model as the example.
#include "mesh.h"
#include "field_layout.h"
#include "multiscale.h"
#include "parameter_reader.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 6) {
            throw std::runtime_error("Usage: check_rve_sensitivity mesh dofs material solver bc");
        }
        const DofsInfo dofs = read_dofs_info(argv[2]);
        FieldMesh mesh = read_mixed_mesh(argv[1]);
        apply_dofs_info(mesh, dofs);
        Problem problem;
        problem.dim = mesh.dim;
        problem.shared_dofs = false;
        problem.domains.resize(1);
        problem.domains[0].name = "micro_rve";
        problem.domains[0].type = DomainType::Solid;
        problem.domains[0].mesh = mesh;
        apply_dofs_info(problem, dofs);
        ProblemParameters params = read_problem_parameters(argv[3], argv[5], argv[4]);
        params.field_dirichlet_bcs.clear();
        params.neumann_bcs.clear();
        params.solver.linear.backend = "umfpack";
        RveSolveOptions options;
        options.Fmacro = {{{{1.015, 0.02}}, {{0.005, 1.01}}}};
        options.dt = options.dt_previous = 0.1;
        options.tol = 1e-11;
        options.max_iter = 30;
        options.adaptive_substepping = false;
        options.vtk_output = options.csv_output = "none";
        const auto base = solve_rve_pbc_augmented(problem, params, options);
        if (!base.converged) throw std::runtime_error("Base RVE did not converge");
        const double epsilon = 1e-6;
        double error_squared = 0.0, reference_squared = 0.0;
        for (int q = 0; q < 4; ++q) {
            auto plus_options = options, minus_options = options;
            plus_options.Fmacro[q / 2][q % 2] += epsilon;
            minus_options.Fmacro[q / 2][q % 2] -= epsilon;
            const auto plus = solve_rve_pbc_augmented(problem, params, plus_options);
            const auto minus = solve_rve_pbc_augmented(problem, params, minus_options);
            if (!plus.converged || !minus.converged) {
                throw std::runtime_error("Perturbed RVE did not converge");
            }
            for (int i = 0; i < 3; ++i) {
                for (int A = 0; A < 3; ++A) {
                    const double finite_difference = (plus.macro.Pbar[i][A] - minus.macro.Pbar[i][A]) / (2 * epsilon);
                    const double analytic = base.sensitivity.C[i][A][q / 2][q % 2] / base.macro.volume;
                    error_squared += std::pow(analytic - finite_difference, 2);
                    reference_squared += finite_difference * finite_difference;
                }
            }
        }
        const double relative_error = std::sqrt(error_squared / std::max(reference_squared, 1e-30));
        if (!std::isfinite(relative_error) || relative_error > 1e-5) {
            std::cerr << "RVE tangent relative error: " << relative_error << "\n";
            return 1;
        }
        std::cout << std::setprecision(12)
                  << "{\"passed\":true,\"relative_error\":" << relative_error
                  << ",\"epsilon\":" << epsilon
                  << ",\"directions\":4,\"volume\":" << base.macro.volume << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
