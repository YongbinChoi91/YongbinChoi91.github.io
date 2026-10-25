#include "multiscale.h"
#include "kinematics.h"
#include "macro_rve_common.h"
#include "vem.h"

#include <algorithm>
#include <array>
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
    return "Homogenization Macro Second Order VEM";
}

namespace {

struct MacroVemQuadState {
    int qp_index = -1;
    double weight = 0.0;
    bool center_stabilization = false;
    double Gamma = 0.0;
    RveSolveResult rve;
};

template <class Scalar>
static Scalar weak_scalar_macro_vem_quadratic(
    const FieldMesh& mesh,
    int elem_id,
    const std::vector<Scalar>& x_elem,
    const std::vector<Scalar>& test_elem,
    const ProblemParameters& params,
    const ElementState& state,
    const std::vector<MacroVemQuadState>& states
) {
    (void)state;

    const int q_order = std::max(3, params.solver.gauss_order);
    const vem::QuadraticElementCache& cache = vem::get_cached_quadratic_element(mesh, elem_id, q_order);
    const LocalFieldView& L = cache.layout;
    const auto& U = L.field("displacement");
    if (static_cast<int>(x_elem.size()) != L.n_local()) {
        throw std::runtime_error("Bad current element vector size in quadratic VEM macro weak form.");
    }
    if (test_elem.size() != x_elem.size()) {
        throw std::runtime_error("Bad test element vector size in quadratic VEM macro weak form.");
    }

    const auto u_pi = vem::quadratic_projection_coefficients(cache, x_elem);
    const auto du_pi = vem::quadratic_projection_coefficients(cache, test_elem);
    Scalar G(0.0);

    for (const MacroVemQuadState& s : states) {
        if (s.center_stabilization) {
            if (s.Gamma <= 0.0) continue;

            // DOFI stabilization weak form:
            //   G_stab = Gamma * sum_s (u_s - Pi u(X_s))
            //                       dot (delta u_s - Pi delta u(X_s)).
            for (int slot = 0; slot < U.n_nodes; ++slot) {
                const auto Pi_u_at_slot = vem::quadratic_projection_value_basis(u_pi, cache.slot_basis[slot]);
                const auto Pi_delta_u_at_slot = vem::quadratic_projection_value_basis(du_pi, cache.slot_basis[slot]);
                for (int i = 0; i < 2; ++i) {
                    const Scalar u_s = x_elem[hpx::local_dof(U, slot, i)];
                    const Scalar delta_u_s = test_elem[hpx::local_dof(U, slot, i)];
                    const Scalar u_s_minus_Pi_u = u_s - Pi_u_at_slot[i];
                    const Scalar delta_u_s_minus_Pi_delta_u = delta_u_s - Pi_delta_u_at_slot[i];

                    G += Scalar(s.Gamma) * u_s_minus_Pi_u * delta_u_s_minus_Pi_delta_u;
                }
            }
            continue;
        }

        // Projected constitutive weak form at a quadratic VEM internal quadrature point:
        //   G_mat = MacroP(F_pi) : Grad(delta u_pi) dV.
        const auto& qp = cache.internal_qps[s.qp_index];
        const auto grad_u = vem::quadratic_projection_gradient_basis(u_pi, qp.grad_basis);
        const auto grad_du = vem::quadratic_projection_gradient_basis(du_pi, qp.grad_basis);
        const RveSolveResult& rve = s.rve;
        const double V = std::max(rve.macro.volume, 1.0e-30);

        for (int i = 0; i < 2; ++i) {
            for (int A = 0; A < 2; ++A) {
                Scalar MacroP = Scalar(rve.macro.Pbar[i][A]);
                // Residual uses the RVE homogenized MacroP directly.  The
                // sensitivity term is zero at the solved point F == rve.Fmacro,
                // but exposes dMacroP/dMacroF to AD for the consistent tangent.
                for (int j = 0; j < 2; ++j) {
                    for (int B = 0; B < 2; ++B) {
                        const double dMacroP_dMacroF = rve.sensitivity.C[i][A][j][B] / V;
                        const Scalar F = (j == B ? Scalar(1.0) : Scalar(0.0)) + grad_u[j][B];
                        MacroP += Scalar(dMacroP_dMacroF) * (F - Scalar(rve.Fmacro[j][B]));
                    }
                }
                G += MacroP * grad_du[i][A] * Scalar(s.weight);
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
// P_hom and A_hom come from the ACEFEM-style augmented RVE PBC solve.
// -----------------------------------------------------------------------------
ElementSystem compute_macro_constitutive_element_system(
    const FieldMesh& mesh,
    int elem_id,
    const Vector& x_elem,
    const ProblemParameters& params,
    const ElementState& state
) {
    if (!multiscale_rve_context_configured()) {
        throw std::runtime_error("weak_form_macro.cpp was used but no RVE context was configured. Call configure_multiscale_rve_context(...) in multiscale_main.cpp.");
    }

    // Quadratic VEM needs the second-order projection data and the internal T6
    // quadrature stored by core/vem.*.  This mirrors weak_form_vem_quadratic.cpp.
    const int q_order = std::max(3, params.solver.gauss_order);
    const vem::QuadraticElementCache& cache = vem::get_cached_quadratic_element(mesh, elem_id, q_order);
    const LocalFieldView& L = cache.layout;
    const int n = L.n_local();

    if (mesh.dim != 2 || L.dim != 2) {
        throw std::runtime_error("Quadratic VEM macro multiscale currently supports only 2D displacement polygonal meshes.");
    }
    if (static_cast<int>(x_elem.size()) != n) {
        throw std::runtime_error("Bad macro element vector size in quadratic VEM multiscale weak form.");
    }

    const auto u_pi = vem::quadratic_projection_coefficients(cache, x_elem);

    auto solve_rve_for_gradient = [&](const vem::Mat2T<double>& grad_u,
                                      int gp_index,
                                      const std::string& contribution_name,
                                      bool write_samples) -> RveSolveResult {
        RveSolveOptions opt = make_multiscale_rve_solve_options();
        opt.Fmacro[0][0] = 1.0 + grad_u[0][0];
        opt.Fmacro[0][1] =       grad_u[0][1];
        opt.Fmacro[1][0] =       grad_u[1][0];
        opt.Fmacro[1][1] = 1.0 + grad_u[1][1];
        opt.dt = state.dt;
        opt.dt_previous = state.dt_previous;
        opt.step = state.step;
        opt.verbose = false;
        opt.vtk_output = "none";
        opt.csv_output = "none";

        Vector warm_u;
        Vector warm_lambda;
        const RveWarmStartState warm = get_multiscale_rve_warm_start(elem_id, gp_index);
        if (warm.valid && opt.use_warm_start) {
            warm_u = warm.displacement;
            warm_lambda = warm.lambda;
            opt.initial_displacement = &warm_u;
            opt.initial_lambda = &warm_lambda;
            opt.initial_Fmacro = &warm.Fmacro;
            if (warm.has_history) opt.initial_history = &warm.history;
        }

        RveSolveResult rve = solve_rve_pbc_augmented(multiscale_rve_micro_problem(), multiscale_rve_micro_params(), opt);

        store_multiscale_rve_warm_start(elem_id, gp_index, rve);

        if (write_samples && rve.converged) {
            write_multiscale_rve_samples_for_element(mesh, elem_id, params, state.step, rve.displacement);
        }

        const double rve_res = multiscale_vector_inf_norm(rve.residual);
        observe_multiscale_rve_diagnostics(rve_res, rve.iterations, rve.converged, &rve.history_diagnostics);
        if (!rve.converged) {
            std::ostringstream os;
            os << "RVE PBC solve did not converge at quadratic VEM macro element " << elem_id
               << " (" << contribution_name << ")"
               << ": |R|_inf=" << rve_res
               << ", iterations=" << rve.iterations
               << ", max_iter=" << opt.max_iter
               << ", Fmacro=[[" << opt.Fmacro[0][0] << ", " << opt.Fmacro[0][1]
               << "], [" << opt.Fmacro[1][0] << ", " << opt.Fmacro[1][1] << "]]";
            throw std::runtime_error(os.str());
        }
        return rve;
    };

    std::vector<MacroVemQuadState> states;

    // Quadratic projected-consistent part: evaluate the homogenized RVE material
    // at every internal T6 quadrature point of the quadratic VEM projection.
    for (int q = 0; q < static_cast<int>(cache.internal_qps.size()); ++q) {
        const auto& qp = cache.internal_qps[q];
        const auto grad_u = vem::quadratic_projection_gradient_basis(u_pi, qp.grad_basis);

        std::ostringstream name;
        name << "quadratic-projection-qp " << q;
        const RveSolveResult rve = solve_rve_for_gradient(grad_u, q, name.str(), q == 0);
        states.push_back(MacroVemQuadState{q, qp.dV, false, 0.0, rve});
    }

    // DOFI-like stabilization for the macro quadratic VEM.  The uploaded
    // single-scale quadratic weak form stabilizes the non-polynomial remainder.
    // Here Gamma is taken from the homogenized center tangent, so the residual
    // and tangent remain consistent with the current RVE material stiffness.
    const auto grad_center = vem::quadratic_projection_gradient(u_pi, {{0.0, 0.0}});
    const int center_gp_index = static_cast<int>(cache.internal_qps.size());
    const RveSolveResult rve_center = solve_rve_for_gradient(grad_center,
                                                             center_gp_index,
                                                             "dofi-center-stabilization",
                                                             cache.internal_qps.empty());
    const double Vc = std::max(rve_center.macro.volume, 1.0e-30);
    const double A0000 = rve_center.sensitivity.C[0][0][0][0] / Vc;
    const double A1111 = rve_center.sensitivity.C[1][1][1][1] / Vc;
    const double A0101 = rve_center.sensitivity.C[0][1][0][1] / Vc;
    const double A1010 = rve_center.sensitivity.C[1][0][1][0] / Vc;
    const double gamma_raw = (A0000 + A1111 + 0.5 * (A0101 + A1010)) / 16.0;
    const double Gamma = params.material.stabilization_parameter * std::max(gamma_raw, 0.0);

    states.push_back(MacroVemQuadState{center_gp_index, 0.0, true, Gamma, rve_center});

    return hpx::scalar_weak_form_element_system(
        x_elem,
        [&](const auto& x_ad, const auto& test_ad) {
            return weak_scalar_macro_vem_quadratic(mesh, elem_id, x_ad, test_ad, params, state, states);
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
    if (mesh.vem_order != 2) return false;
    if (dof_map.size() != mesh.displacement_elements.size()) return false;

    struct Contribution {
        int elem_id = -1;
        int gp_index = -1;
        LocalFieldInfo U;
        std::vector<std::array<double,2>> coeff;
        double weight = 0.0;
        bool write_samples = false;
        bool center_stabilization = false;
        std::vector<std::vector<double>> H;
        Vector x_elem;
        std::string name;
    };

    std::vector<Contribution> contributions;
    std::vector<RveBatchTask> rve_tasks;
    contributions.reserve(dof_map.size());
    rve_tasks.reserve(dof_map.size());

    local_systems.resize(dof_map.size());
    element_errors.assign(dof_map.size(), std::string{});

    for (int elem_id = 0; elem_id < static_cast<int>(dof_map.size()); ++elem_id) {
        try {
            const Vector x_elem = gather_local_vector(current_dofs, dof_map[elem_id]);
            const int q_order = std::max(3, params.solver.gauss_order);
            const vem::QuadraticElementCache& cache = vem::get_cached_quadratic_element(mesh, elem_id, q_order);
            const LocalFieldView& L = cache.layout;
            const auto& U = L.field("displacement");
            const int n = L.n_local();

            if (mesh.dim != 2 || L.dim != 2) {
                throw std::runtime_error("Quadratic VEM macro multiscale currently supports only 2D displacement polygonal meshes.");
            }
            if (static_cast<int>(x_elem.size()) != n) {
                throw std::runtime_error("Bad macro element vector size in quadratic VEM multiscale batch weak form.");
            }
            local_systems[elem_id].residual.assign(n, 0.0);
            local_systems[elem_id].tangent.assign(n*n, 0.0);

            const auto u_pi = vem::quadratic_projection_coefficients(cache, x_elem);

            for (int q = 0; q < static_cast<int>(cache.internal_qps.size()); ++q) {
                const auto& qp = cache.internal_qps[q];
                const auto grad_u = vem::quadratic_projection_gradient_basis(u_pi, qp.grad_basis);

                Contribution c;
                c.elem_id = elem_id;
                c.gp_index = q;
                c.U = U;
                c.coeff = cache.internal_gradient_coefficients[q];
                c.weight = qp.dV;
                c.write_samples = (q == 0);
                c.x_elem = x_elem;
                c.name = "quadratic-projection-qp " + std::to_string(q);
                contributions.push_back(std::move(c));

                rve_tasks.push_back(make_macro_rve_task(elem_id, q, macro_F_from_grad(grad_u), dt, dt_previous, step));
            }

            const auto grad_center = vem::quadratic_projection_gradient(u_pi, {{0.0, 0.0}});
            const int center_gp_index = static_cast<int>(cache.internal_qps.size());
            Contribution center;
            center.elem_id = elem_id;
            center.gp_index = center_gp_index;
            center.U = U;
            center.write_samples = cache.internal_qps.empty();
            center.center_stabilization = true;
            center.H = cache.value_residual_matrix;
            center.x_elem = x_elem;
            center.name = "dofi-center-stabilization";
            contributions.push_back(std::move(center));

            RveBatchTask task = make_macro_rve_task(elem_id, center_gp_index, macro_F_from_grad(grad_center), dt, dt_previous, step);
            rve_tasks.push_back(std::move(task));
        } catch (const std::exception& e) {
            element_errors[elem_id] = e.what();
        } catch (...) {
            element_errors[elem_id] = "Unknown exception while collecting quadratic VEM multiscale batch tasks.";
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
    micro_params.solver.multiscale_parallel_level = "macro";
    const int batch_threads = params.solver.multiscale_macro_threads;
    std::vector<RveBatchResult> rve_results =
        solve_rve_pbc_augmented_batch(multiscale_rve_micro_problem(), micro_params, base_opt, rve_tasks, batch_threads);

    for (int q = 0; q < static_cast<int>(rve_results.size()); ++q) {
        const Contribution& c = contributions[q];
        try {
            if (!rve_results[q].error.empty()) throw std::runtime_error(rve_results[q].error);
            const RveSolveResult& rve = rve_results[q].rve;
            const double rve_res = multiscale_vector_inf_norm(rve.residual);
            observe_multiscale_rve_diagnostics(rve_res, rve.iterations, rve.converged, &rve.history_diagnostics);
            if (!rve.converged) {
                std::ostringstream os;
                os << "RVE PBC solve did not converge at quadratic VEM macro element " << c.elem_id
                   << " (" << c.name << ")"
                   << ": |R|_inf=" << rve_res
                   << ", iterations=" << rve.iterations
                   << ", max_iter=" << base_opt.max_iter;
                throw std::runtime_error(os.str());
            }

            store_multiscale_rve_warm_start(c.elem_id, c.gp_index, rve);
            if (c.write_samples) {
                write_multiscale_rve_samples_for_element(mesh, c.elem_id, params, step, rve.displacement);
            }

            ElementSystem& sys = local_systems[c.elem_id];
            const Vector x_elem = gather_local_vector(current_dofs, dof_map[c.elem_id]);
            ElementState batch_state;
            batch_state.step = step;

            if (c.center_stabilization) {
                const double Vc = std::max(rve.macro.volume, 1.0e-30);
                const double A0000 = rve.sensitivity.C[0][0][0][0] / Vc;
                const double A1111 = rve.sensitivity.C[1][1][1][1] / Vc;
                const double A0101 = rve.sensitivity.C[0][1][0][1] / Vc;
                const double A1010 = rve.sensitivity.C[1][0][1][0] / Vc;
                const double gamma_raw = (A0000 + A1111 + 0.5 * (A0101 + A1010)) / 16.0;
                const double Gamma = params.material.stabilization_parameter * std::max(gamma_raw, 0.0);

                MacroVemQuadState state;
                state.qp_index = c.gp_index;
                state.center_stabilization = true;
                state.Gamma = Gamma;
                state.rve = rve;
                const std::vector<MacroVemQuadState> states{std::move(state)};
                add_element_system_into(
                    sys,
                    hpx::scalar_weak_form_element_system(
                        x_elem,
                        [&](const auto& x_ad, const auto& test_ad) {
                            return weak_scalar_macro_vem_quadratic(mesh, c.elem_id, x_ad, test_ad, params, batch_state, states);
                        }
                    )
                );
                continue;
            }

            if (c.weight == 0.0) continue;
            MacroVemQuadState state;
            state.qp_index = c.gp_index;
            state.weight = c.weight;
            state.rve = rve;
            const std::vector<MacroVemQuadState> states{std::move(state)};
            add_element_system_into(
                sys,
                hpx::scalar_weak_form_element_system(
                    x_elem,
                    [&](const auto& x_ad, const auto& test_ad) {
                        return weak_scalar_macro_vem_quadratic(mesh, c.elem_id, x_ad, test_ad, params, batch_state, states);
                    }
                )
            );
        } catch (const std::exception& e) {
            element_errors[c.elem_id] = e.what();
        } catch (...) {
            element_errors[c.elem_id] = "Unknown exception while assembling quadratic VEM multiscale batch contribution.";
        }
    }

    return true;
}
