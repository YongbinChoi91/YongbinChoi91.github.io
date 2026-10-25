// Standalone integration check of the converged RVE's integrated-Piola tangent.
// Each finite-difference perturbation is solved independently to equilibrium.
#include "mesh.h"
#include "field_layout.h"
#include "multiscale.h"
#include "parameter_reader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using Tangent = std::array<std::array<double, 4>, 9>;

static void write_tangent(std::ostream& out, const Tangent& value) {
    out << '[';
    for (int p = 0; p < 9; ++p) {
        if (p) out << ',';
        out << '[';
        for (int q = 0; q < 4; ++q) {
            if (q) out << ',';
            out << value[p][q];
        }
        out << ']';
    }
    out << ']';
}

static double residual_norm(const RveSolveResult& result) {
    double maximum = 0;
    for (double value : result.residual) maximum = std::max(maximum, std::abs(value));
    return maximum;
}

struct Measurement {
    double epsilon = 0;
    Tangent finite_difference{};
    double relative_error = 0;
    double absolute_error = 0;
    double maximum_error = 0;
    std::array<double, 4> direction_errors{};
};

int main(int argc, char** argv) {
    try {
        if (argc != 7) {
            throw std::runtime_error("Usage: sweep_rve_sensitivity mesh dofs material solver bc output.json");
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
        options.tol = 1e-12;
        options.max_iter = 30;
        options.verbose = false;
        options.use_warm_start = false;
        options.adaptive_substepping = false;
        options.vtk_output = options.csv_output = "none";
        const auto base = solve_rve_pbc_augmented(problem, params, options);
        if (!base.converged) throw std::runtime_error("Base RVE did not converge");
        Tangent analytic{};
        for (int p = 0; p < 9; ++p) {
            for (int q = 0; q < 4; ++q) {
                analytic[p][q] = base.sensitivity.C[p / 3][p % 3][q / 2][q % 2];
            }
        }
        double max_residual = residual_norm(base);
        std::vector<Measurement> measurements;
        const std::array<double, 8> epsilons{{1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9}};
        for (double epsilon : epsilons) {
            Measurement measurement;
            measurement.epsilon = epsilon;
            double error_squared = 0, reference_squared = 0;
            for (int q = 0; q < 4; ++q) {
                auto plus_options = options, minus_options = options;
                plus_options.Fmacro[q / 2][q % 2] += epsilon;
                minus_options.Fmacro[q / 2][q % 2] -= epsilon;
                const auto plus = solve_rve_pbc_augmented(problem, params, plus_options);
                const auto minus = solve_rve_pbc_augmented(problem, params, minus_options);
                if (!plus.converged || !minus.converged) {
                    throw std::runtime_error("Perturbed RVE did not converge");
                }
                max_residual = std::max({max_residual, residual_norm(plus), residual_norm(minus)});
                double column_error = 0, column_reference = 0;
                for (int p = 0; p < 9; ++p) {
                    const double fd = (plus.macro.intP[p / 3][p % 3] - minus.macro.intP[p / 3][p % 3]) / (2 * epsilon);
                    if (!std::isfinite(fd) || !std::isfinite(analytic[p][q])) {
                        throw std::runtime_error("Non-finite sensitivity component");
                    }
                    measurement.finite_difference[p][q] = fd;
                    const double error = analytic[p][q] - fd;
                    column_error += error * error;
                    column_reference += fd * fd;
                    measurement.maximum_error = std::max(measurement.maximum_error, std::abs(error));
                }
                error_squared += column_error;
                reference_squared += column_reference;
                measurement.direction_errors[q] = std::sqrt(column_error / std::max(column_reference, 1e-30));
            }
            measurement.absolute_error = std::sqrt(error_squared);
            measurement.relative_error = std::sqrt(error_squared / std::max(reference_squared, 1e-30));
            std::cerr << "epsilon=" << epsilon << " relative_error=" << measurement.relative_error << '\n';
            measurements.push_back(measurement);
        }
        // The same preselected perturbation is used in all component tables.
        const bool passed = measurements[4].relative_error < 1e-5;
        if (!passed) throw std::runtime_error("Representative epsilon=1e-6 did not meet the sensitivity tolerance");
        std::ofstream out(argv[6]);
        if (!out) throw std::runtime_error("Cannot open sweep output");
        out << std::setprecision(17)
            << "{\"passed\":true,\"quantity\":\"integrated_piola\","
            << "\"deformation_gradient\":[[1.015,0.02],[0.005,1.01]],"
            << "\"solver_tolerance\":" << options.tol
            << ",\"max_rve_residual\":" << max_residual
            << ",\"volume\":" << base.macro.volume
            << ",\"representative_epsilon\":1e-6,\"acceptance_relative_error\":1e-5,\"solve_count\":65,"
            << "\"analytic\":";
        write_tangent(out, analytic);
        out << ",\"sweep\":[";
        for (std::size_t index = 0; index < measurements.size(); ++index) {
            const auto& m = measurements[index];
            if (index) out << ',';
            out << "{\"epsilon\":" << m.epsilon
                << ",\"relative_error\":" << m.relative_error
                << ",\"absolute_frobenius_error\":" << m.absolute_error
                << ",\"max_absolute_error\":" << m.maximum_error
                << ",\"direction_relative_errors\":[";
            for (int q = 0; q < 4; ++q) {
                if (q) out << ',';
                out << m.direction_errors[q];
            }
            out << "],\"finite_difference\":";
            write_tangent(out, m.finite_difference);
            out << '}';
        }
        out << "]}\n";
        if (!out) throw std::runtime_error("Could not write complete sweep output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
