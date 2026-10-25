#include "multiscale.h"
#include "kinematics.h"
#include "macro_rve_common.h"
#include "vem.h"

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
    return "Homogenization Macro Linear VEM";
}

namespace {

struct MacroVemState {
    int contribution_index = -1;
    int triangle_index = -1;
    double weight = 0.0;
    RveSolveResult rve;
};

template <class Scalar>
static Scalar weak_scalar_macro_vem(
    const FieldMesh& mesh,
    int elem_id,
    const std::vector<Scalar>& x_elem,
    const std::vector<Scalar>& test_elem,
    const ProblemParameters& params,
    const ElementState& state,
    const std::vector<MacroVemState>& states
) {
    (void)state;

    const int q_order = (params.solver.gauss_order > 0) ? params.solver.gauss_order : 1;
    const vem::LinearElementCache& cache = vem::get_cached_linear_element(mesh, elem_id, q_order);
    const LocalFieldView& L = cache.layout;
    if (static_cast<int>(x_elem.size()) != L.n_local()) {
        throw std::runtime_error("Bad current element vector size in VEM macro weak form.");
    }
    if (test_elem.size() != x_elem.size()) {
        throw std::runtime_error("Bad test element vector size in VEM macro weak form.");
    }

    Scalar G(0.0);
    for (const MacroVemState& s : states) {
        vem::Mat2T<Scalar> grad_u{};
        vem::Mat2T<Scalar> grad_du{};
        if (s.triangle_index >= 0) {
            const auto& tri = cache.triangles[s.triangle_index];
            grad_u = vem::linear_triangle_gradient(cache, tri, x_elem);
            grad_du = vem::linear_triangle_gradient(cache, tri, test_elem);
        } else {
            grad_u = vem::linear_projected_gradient(cache, x_elem);
            grad_du = vem::linear_projected_gradient(cache, test_elem);
        }

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
        throw std::runtime_error("weak_form_macro_vem.cpp was used but no RVE context was configured. Call configure_multiscale_rve_context(...) in multiscale_main.cpp.");
    }

    const int q_order = (params.solver.gauss_order > 0) ? params.solver.gauss_order : 1;
    const vem::LinearElementCache& cache = vem::get_cached_linear_element(mesh, elem_id, q_order);
    const LocalFieldView& L = cache.layout;
    const auto& U = L.field("displacement");
    const int n = L.n_local();

    if (mesh.dim != 2 || L.dim != 2) {
        throw std::runtime_error("VEM macro multiscale currently supports 2D displacement polygonal VEM only.");
    }
    if (static_cast<int>(x_elem.size()) != n) {
        throw std::runtime_error("Bad macro element vector size in VEM multiscale weak form.");
    }

    // -------------------------------------------------------------------------
    // Method A: make the macro VE2 weak form follow the single-scale linear VEM
    // file exactly in structure:
    //
    //   G_macro = alpha * sum_T |T| Pbar(F_T) : Grad_T(delta u)
    //           + (1-alpha) * |fan| Pbar(F_pi) : Grad_pi(delta u)
    //
    // and the consistent macro tangent uses the RVE sensitivity dMacroP/dMacroF
    // evaluated at the same deformation gradient used by each contribution.
    // This intentionally calls the RVE once per internal fan triangle and once
    // for the projected polygon contribution.
    // -------------------------------------------------------------------------
    const double alpha = 0.1;

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
            os << "RVE PBC solve did not converge at VEM macro element " << elem_id
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

    std::vector<MacroVemState> states;

    // 1) Internal fan triangle part: one RVE solve for every triangle gradient.
    double fan_measure_sum = 0.0;
    for (int t = 0; t < static_cast<int>(cache.triangles.size()); ++t) {
        const auto& tri = cache.triangles[t];
        fan_measure_sum += tri.measure;

        const auto grad_u_T = vem::linear_triangle_gradient(cache, tri, x_elem);

        for (int la = 0; la < 3; ++la) {
            const int slot = tri.slots[la];
            if (slot < 0 || slot >= U.n_nodes) {
                throw std::runtime_error("Bad VEM internal triangle slot in macro multiscale weak form.");
            }
        }

        std::ostringstream name;
        name << "fan-triangle " << t;
        const int gp_index = t;
        const RveSolveResult rve_T = solve_rve_for_gradient(grad_u_T, gp_index, name.str(), false);
        states.push_back(MacroVemState{gp_index, t, alpha * tri.measure, rve_T});
    }

    // 2) Projected polygon part: one RVE solve for the projected VEM gradient.
    const auto grad_u_pi = vem::linear_projected_gradient(cache, x_elem);

    const int projected_gp_index = static_cast<int>(cache.triangles.size());
    const RveSolveResult rve_pi = solve_rve_for_gradient(grad_u_pi,
                                                         projected_gp_index,
                                                         "projected-gradient",
                                                         true);
    states.push_back(MacroVemState{projected_gp_index, -1, (1.0 - alpha) * fan_measure_sum, rve_pi});

    return hpx::scalar_weak_form_element_system(
        x_elem,
        [&](const auto& x_ad, const auto& test_ad) {
            return weak_scalar_macro_vem(mesh, elem_id, x_ad, test_ad, params, state, states);
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
    if (mesh.vem_order != 1 && mesh.displacement_type != ElementType::Polygon) return false;
    if (dof_map.size() != mesh.displacement_elements.size()) return false;

    struct Contribution {
        int elem_id = -1;
        int gp_index = -1;
        int triangle_index = -1;
        double weight = 0.0;
        bool write_samples = false;
        std::string name;
    };

    std::vector<Contribution> contributions;
    std::vector<RveBatchTask> rve_tasks;
    contributions.reserve(dof_map.size());
    rve_tasks.reserve(dof_map.size());

    local_systems.resize(dof_map.size());
    element_errors.assign(dof_map.size(), std::string{});
    const double alpha = 0.1;

    for (int elem_id = 0; elem_id < static_cast<int>(dof_map.size()); ++elem_id) {
        try {
            const Vector x_elem = gather_local_vector(current_dofs, dof_map[elem_id]);
            const int q_order = (params.solver.gauss_order > 0) ? params.solver.gauss_order : 1;
            const vem::LinearElementCache& cache = vem::get_cached_linear_element(mesh, elem_id, q_order);
            const LocalFieldView& L = cache.layout;
            const auto& U = L.field("displacement");
            const int n = L.n_local();

            if (mesh.dim != 2 || L.dim != 2) {
                throw std::runtime_error("VEM macro multiscale currently supports 2D displacement polygonal VEM only.");
            }
            if (static_cast<int>(x_elem.size()) != n) {
                throw std::runtime_error("Bad macro element vector size in VEM multiscale batch weak form.");
            }
            local_systems[elem_id].residual.assign(n, 0.0);
            local_systems[elem_id].tangent.assign(n*n, 0.0);

            double fan_measure_sum = 0.0;
            for (int t = 0; t < static_cast<int>(cache.triangles.size()); ++t) {
                const auto& tri = cache.triangles[t];
                fan_measure_sum += tri.measure;
                const auto grad_u_T = vem::linear_triangle_gradient(cache, tri, x_elem);

                Contribution c;
                c.elem_id = elem_id;
                c.gp_index = t;
                c.triangle_index = t;
                c.weight = alpha * tri.measure;
                c.name = "fan-triangle " + std::to_string(t);
                for (int la = 0; la < 3; ++la) {
                    const int slot = tri.slots[la];
                    if (slot < 0 || slot >= U.n_nodes) {
                        throw std::runtime_error("Bad VEM internal triangle slot in macro multiscale batch weak form.");
                    }
                }
                contributions.push_back(std::move(c));

                rve_tasks.push_back(make_macro_rve_task(elem_id, t, macro_F_from_grad(grad_u_T), dt, dt_previous, step));
            }

            const auto grad_u_pi = vem::linear_projected_gradient(cache, x_elem);
            Contribution c;
            c.elem_id = elem_id;
            c.gp_index = static_cast<int>(cache.triangles.size());
            c.triangle_index = -1;
            c.weight = (1.0 - alpha) * fan_measure_sum;
            c.write_samples = true;
            c.name = "projected-gradient";
            contributions.push_back(std::move(c));

            const int projected_gp_index = static_cast<int>(cache.triangles.size());
            rve_tasks.push_back(make_macro_rve_task(elem_id, projected_gp_index, macro_F_from_grad(grad_u_pi), dt, dt_previous, step));
        } catch (const std::exception& e) {
            element_errors[elem_id] = e.what();
        } catch (...) {
            element_errors[elem_id] = "Unknown exception while collecting VEM multiscale batch tasks.";
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
                os << "RVE PBC solve did not converge at VEM macro element " << c.elem_id
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

            if (c.weight == 0.0) continue;
            ElementSystem& sys = local_systems[c.elem_id];
            const Vector x_elem = gather_local_vector(current_dofs, dof_map[c.elem_id]);
            MacroVemState state;
            state.contribution_index = c.gp_index;
            state.triangle_index = c.triangle_index;
            state.weight = c.weight;
            state.rve = rve;
            const std::vector<MacroVemState> states{std::move(state)};
            ElementState batch_state;
            batch_state.step = step;
            add_element_system_into(
                sys,
                hpx::scalar_weak_form_element_system(
                    x_elem,
                    [&](const auto& x_ad, const auto& test_ad) {
                        return weak_scalar_macro_vem(mesh, c.elem_id, x_ad, test_ad, params, batch_state, states);
                    }
                )
            );
        } catch (const std::exception& e) {
            element_errors[c.elem_id] = e.what();
        } catch (...) {
            element_errors[c.elem_id] = "Unknown exception while assembling VEM multiscale batch contribution.";
        }
    }

    return true;
}
