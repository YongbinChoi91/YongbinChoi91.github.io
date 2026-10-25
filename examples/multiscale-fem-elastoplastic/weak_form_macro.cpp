#include "multiscale.h"
#include "kinematics.h"
#include "macro_rve_common.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENMP
#include <omp.h>
#endif

const char* active_macro_constitutive_model_name() {
    return "Homogenization Macro FEM";
}
namespace {

struct MacroFemGpState {
    std::vector<std::array<double,3>> dNu;
    double dV_macro = 0.0;
    RveSolveResult rve;
};

template <class Scalar>
static Scalar weak_scalar_macro_fem(
    const FieldMesh& mesh,
    int elem_id,
    const std::vector<Scalar>& x_elem,
    const std::vector<Scalar>& test_elem,
    const ProblemParameters& params,
    const ElementState& state,
    const std::vector<MacroFemGpState>& gp_states
) {
    (void)params;
    (void)state;

    const LocalFieldView L = make_local_field_view(mesh, elem_id, {"displacement"});
    if (static_cast<int>(x_elem.size()) != L.n_local()) {
        throw std::runtime_error("Bad current element vector size in macro FEM weak form.");
    }
    if (test_elem.size() != x_elem.size()) {
        throw std::runtime_error("Bad test element vector size in macro FEM weak form.");
    }

    Scalar G(0.0);

    for (const MacroFemGpState& gp_state : gp_states) {
        const auto F = hpx::deformation_gradient_as<Scalar>(L, x_elem, gp_state.dNu);
        const auto deltaF = hpx::field_grad_as<Scalar>(L, "displacement", test_elem, gp_state.dNu);
        const RveSolveResult& rve = gp_state.rve;
        const double V = std::max(rve.macro.volume, 1.0e-30);

        for (int i = 0; i < L.dim; ++i) {
            for (int A = 0; A < L.dim; ++A) {
                Scalar MacroP = Scalar(rve.macro.Pbar[i][A]);
                // Residual uses the RVE homogenized MacroP directly.  The
                // sensitivity term is zero at the solved point F == rve.Fmacro,
                // but exposes dMacroP/dMacroF to AD for the consistent tangent.
                for (int j = 0; j < L.dim; ++j) {
                    for (int B = 0; B < L.dim; ++B) {
                        const double dMacroP_dMacroF = rve.sensitivity.C[i][A][j][B] / V;
                        MacroP += Scalar(dMacroP_dMacroF) * (F[j][B] - Scalar(rve.Fmacro[j][B]));
                    }
                }
                G += MacroP * deltaF[i][A] * Scalar(gp_state.dV_macro);
            }
        }
    }

    return G;
}

} // namespace

// -----------------------------------------------------------------------------
// Macro multiscale element system.
// Residual: int P_hom : Grad(delta u) dV_macro
// Tangent : int A_hom : Grad(u_inc) : Grad(delta u) dV_macro
// -----------------------------------------------------------------------------
ElementSystem compute_macro_constitutive_element_system(
    const FieldMesh& mesh,
    int elem_id,
    const Vector& x_elem,
    const ProblemParameters& params,
    const ElementState& state
) {
    (void)state;
    if (!multiscale_rve_context_configured()) {
        throw std::runtime_error("weak_form_macro.cpp was used but no RVE context was configured. Call configure_multiscale_rve_context(...) in multiscale_main.cpp.");
    }

    const LocalFieldView L = make_local_field_view(mesh, elem_id, {"displacement"});
    const auto& U = L.field("displacement");
    const int n = L.n_local();
    if (static_cast<int>(x_elem.size()) != n) throw std::runtime_error("Bad macro element vector size in multiscale weak form.");

    const ElementGeometry geom = make_displacement_element_geometry(mesh, elem_id);
    const auto gps = gauss_rule(U.element_type, params.solver.gauss_order);

    std::string first_error;
    struct MacroGpData {
        std::vector<std::array<double,3>> dNu;
        double dV_macro = 0.0;
    };

    std::vector<MacroGpData> gp_data(gps.size());
    std::vector<RveBatchTask> rve_tasks(gps.size());

    for (int gp_index = 0; gp_index < static_cast<int>(gps.size()); ++gp_index) {
        try {
            const auto& gp = gps[gp_index];
            const ShapeData su = compute_shape(U.element_type, gp.xi, gp.eta, gp.zeta);
            const JacobianData Jgeo = compute_jacobian_data(geom, su);
            if (Jgeo.detJ <= 0.0) throw std::runtime_error("Negative macro reference detJ in multiscale element.");
            const auto dNu = compute_dN_dx(su, Jgeo);
            const double dV_macro = Jgeo.detJ * gp.weight;

            const auto F2 = hpx::deformation_gradient_as<double>(L, x_elem, dNu);

            gp_data[gp_index].dNu = dNu;
            gp_data[gp_index].dV_macro = dV_macro;

            rve_tasks[gp_index] = make_macro_rve_task(
                elem_id,
                gp_index,
                {{{{F2[0][0], F2[0][1]}}, {{F2[1][0], F2[1][1]}}}},
                state.dt,
                state.dt_previous,
                state.step
            );
        } catch (const std::exception& e) {
            if (first_error.empty()) first_error = e.what();
        }
    }
    if (!first_error.empty()) throw std::runtime_error(first_error);

    RveSolveOptions base_opt = make_multiscale_rve_solve_options();
    base_opt.verbose = false;
    base_opt.vtk_output = "none";
    base_opt.csv_output = "none";
    base_opt.initial_displacement = nullptr;
    base_opt.initial_lambda = nullptr;

    // The outer macro element loop is already parallel for "macro"/"hybrid".
    // Use the RVE batch threads only when the user selects the RVE-only policy;
    // otherwise each macro worker solves this element's GP batch serially and
    // the RVE's own element assembly policy remains unchanged.
    const int batch_threads =
        (params.solver.multiscale_parallel_level == "rve")
            ? params.solver.multiscale_rve_threads
            : 1;
    std::vector<RveBatchResult> rve_results =
        solve_rve_pbc_augmented_batch(multiscale_rve_micro_problem(), multiscale_rve_micro_params(), base_opt, rve_tasks, batch_threads);

    std::vector<MacroFemGpState> gp_states;
    gp_states.reserve(rve_results.size());

    for (int gp_index = 0; gp_index < static_cast<int>(rve_results.size()); ++gp_index) {
        try {
            if (!rve_results[gp_index].error.empty()) {
                throw std::runtime_error(rve_results[gp_index].error);
            }

            const RveSolveResult& rve = rve_results[gp_index].rve;
            const double rve_res = multiscale_vector_inf_norm(rve.residual);
            observe_multiscale_rve_diagnostics(rve_res, rve.iterations, rve.converged, &rve.history_diagnostics);
            if (!rve.converged) {
                std::ostringstream os;
                os << "RVE PBC solve did not converge at macro element " << elem_id
                   << ", Gauss point " << gp_index
                   << ": |R|_inf=" << rve_res
                   << ", iterations=" << rve.iterations
                   << ", max_iter=" << base_opt.max_iter;
                throw std::runtime_error(os.str());
            }

            store_multiscale_rve_warm_start(elem_id, gp_index, rve);
            if (gp_index == 0) {
                write_multiscale_rve_samples_for_element(mesh, elem_id, params, state.step, rve.displacement);
            }

            MacroFemGpState gp_state;
            gp_state.dNu = gp_data[gp_index].dNu;
            gp_state.dV_macro = gp_data[gp_index].dV_macro;
            gp_state.rve = rve;
            gp_states.push_back(std::move(gp_state));
        } catch (const std::exception& e) {
            if (first_error.empty()) first_error = e.what();
        }
    }

    if (!first_error.empty()) throw std::runtime_error(first_error);
    return hpx::scalar_weak_form_element_system(
        x_elem,
        [&](const auto& x_ad, const auto& test_ad) {
            return weak_scalar_macro_fem(mesh, elem_id, x_ad, test_ad, params, state, gp_states);
        }
    );
}

bool try_compute_multiscale_batch_domain_element_systems(
    const FieldMesh& mesh,
    DomainType domain_type,
    int domain_id,
    const ProblemParameters& params,
    const Vector& current_dofs,
    const PastDofs& past_dofs,
    double dt,
    double dt_previous,
    int step,
    const std::vector<std::vector<int>>& dof_map,
    std::vector<ElementSystem>& local_systems,
    std::vector<std::string>& element_errors
) {
    (void)domain_type;
    (void)domain_id;
    (void)past_dofs;
    (void)dt;
    (void)dt_previous;

    if (!multiscale_rve_context_configured()) return false;
    if (params.solver.multiscale_rve_assembly_context) return false;
    if (!has_displacement_field(mesh)) return false;
    if (dof_map.size() != mesh.displacement_elements.size()) return false;

    struct MacroGpData {
        int elem_id = -1;
        int gp_index = -1;
        LocalFieldView L;
        std::vector<std::array<double,3>> dNu;
        double dV_macro = 0.0;
    };

    std::vector<MacroGpData> gp_data;
    std::vector<RveBatchTask> rve_tasks;
    gp_data.reserve(dof_map.size());
    rve_tasks.reserve(dof_map.size());

    local_systems.resize(dof_map.size());
    element_errors.assign(dof_map.size(), std::string{});

    for (int elem_id = 0; elem_id < static_cast<int>(dof_map.size()); ++elem_id) {
        try {
            const Vector x_elem = gather_local_vector(current_dofs, dof_map[elem_id]);
            const LocalFieldView L = make_local_field_view(mesh, elem_id, {"displacement"});
            const auto& U = L.field("displacement");
            const int n = L.n_local();
            if (static_cast<int>(x_elem.size()) != n) {
                throw std::runtime_error("Bad macro element vector size in multiscale batch weak form.");
            }

            local_systems[elem_id].residual.assign(n, 0.0);
            local_systems[elem_id].tangent.assign(n*n, 0.0);

            const ElementGeometry geom = make_displacement_element_geometry(mesh, elem_id);
            const auto gps = gauss_rule(U.element_type, params.solver.gauss_order);
            for (int gp_index = 0; gp_index < static_cast<int>(gps.size()); ++gp_index) {
                const auto& gp = gps[gp_index];
                const ShapeData su = compute_shape(U.element_type, gp.xi, gp.eta, gp.zeta);
                const JacobianData Jgeo = compute_jacobian_data(geom, su);
                if (Jgeo.detJ <= 0.0) {
                    throw std::runtime_error("Negative macro reference detJ in multiscale batch element.");
                }
                const auto dNu = compute_dN_dx(su, Jgeo);
                const auto F2 = hpx::deformation_gradient_as<double>(L, x_elem, dNu);

                MacroGpData data;
                data.elem_id = elem_id;
                data.gp_index = gp_index;
                data.L = L;
                data.dNu = dNu;
                data.dV_macro = Jgeo.detJ * gp.weight;
                gp_data.push_back(std::move(data));

                rve_tasks.push_back(make_macro_rve_task(
                    elem_id,
                    gp_index,
                    {{{{F2[0][0], F2[0][1]}}, {{F2[1][0], F2[1][1]}}}},
                    dt,
                    dt_previous,
                    step
                ));
            }
        } catch (const std::exception& e) {
            element_errors[elem_id] = e.what();
        } catch (...) {
            element_errors[elem_id] = "Unknown exception while collecting multiscale batch macro GP tasks.";
        }
    }

    for (const std::string& err : element_errors) {
        if (!err.empty()) return true;
    }

    RveSolveOptions base_opt = make_multiscale_rve_solve_options();
    base_opt.verbose = false;
    base_opt.vtk_output = "none";
    base_opt.csv_output = "none";
    base_opt.initial_displacement = nullptr;
    base_opt.initial_lambda = nullptr;

    ProblemParameters micro_params = multiscale_rve_micro_params();
    // The global macro-GP batch is now the outer parallel region.  Keep each
    // individual RVE assembly serial to avoid nested OpenMP oversubscription.
    micro_params.solver.multiscale_parallel_level = "macro";
    const int batch_threads = params.solver.multiscale_macro_threads;
    std::vector<RveBatchResult> rve_results =
        solve_rve_pbc_augmented_batch(multiscale_rve_micro_problem(), micro_params, base_opt, rve_tasks, batch_threads);

    for (int q = 0; q < static_cast<int>(rve_results.size()); ++q) {
        const MacroGpData& data = gp_data[q];
        try {
            if (!rve_results[q].error.empty()) {
                throw std::runtime_error(rve_results[q].error);
            }

            const RveSolveResult& rve = rve_results[q].rve;
            const double rve_res = multiscale_vector_inf_norm(rve.residual);
            observe_multiscale_rve_diagnostics(rve_res, rve.iterations, rve.converged, &rve.history_diagnostics);
            if (!rve.converged) {
                std::ostringstream os;
                os << "RVE PBC solve did not converge at macro element " << data.elem_id
                   << ", Gauss point " << data.gp_index
                   << ": |R|_inf=" << rve_res
                   << ", iterations=" << rve.iterations
                   << ", max_iter=" << base_opt.max_iter;
                throw std::runtime_error(os.str());
            }

            store_multiscale_rve_warm_start(data.elem_id, data.gp_index, rve);
            if (data.gp_index == 0) {
                write_multiscale_rve_samples_for_element(mesh, data.elem_id, params, step, rve.displacement);
            }

            ElementSystem& sys = local_systems[data.elem_id];
            const Vector x_elem = gather_local_vector(current_dofs, dof_map[data.elem_id]);
            MacroFemGpState gp_state;
            gp_state.dNu = data.dNu;
            gp_state.dV_macro = data.dV_macro;
            gp_state.rve = rve;
            const std::vector<MacroFemGpState> gp_states{std::move(gp_state)};
            add_element_system_into(
                sys,
                hpx::scalar_weak_form_element_system(
                    x_elem,
                    [&](const auto& x_ad, const auto& test_ad) {
                        ElementState batch_state;
                        batch_state.step = step;
                        return weak_scalar_macro_fem(mesh, data.elem_id, x_ad, test_ad, params, batch_state, gp_states);
                    }
                )
            );
        } catch (const std::exception& e) {
            element_errors[data.elem_id] = e.what();
        } catch (...) {
            element_errors[data.elem_id] = "Unknown exception while assembling multiscale batch macro element.";
        }
    }

    return true;
}
