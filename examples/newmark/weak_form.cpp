#include "weak_form.h"
#include "fem_utils.h"
#include "kinematics.h"

#include <Sacado_Fad_DFad.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// =============================================================================
// MODEL MAP: weak_form_Newmark.cpp
// =============================================================================
//
// Purpose
// -------
// Displacement-only dynamic hyper-elastic solid formulation.  The global solve
// contains only displacement DOFs, while velocity and acceleration are stored as
// Gauss-point history variables and updated by the Newmark method.
//
// Required DOFS_info.txt
// ----------------------
//   2D:
//     field_key displacement
//     history newmark 4     // [v_x, v_y, a_x, a_y] at every Gauss point
//
//   3D:
//     field_key displacement
//     history newmark 6     // [v_x, v_y, v_z, a_x, a_y, a_z]
//
// Execution flow
// --------------
//   assembly.cpp
//     -> active_constitutive_field_names(mesh)
//          returns {"displacement"}; only displacement element DOFs are assembled.
//
//     -> compute_constitutive_element_system(...)
//          1. wraps current local displacement x_elem as outer AD variables
//          2. wraps virtual displacement test_elem as inner AD variables
//          3. calls weak_scalar_hyperelastic_newmark_solid(...) to build scalar G
//          4. extracts residual/tangent from nested AD:
//               R_i  = dG / d(delta_u_i)
//               K_ij = dR_i / d(u_j)
//
//   weak_scalar_hyperelastic_newmark_solid(...)
//     -> for each Gauss point:
//          1. read previous history h_n = [v_n, a_n]
//          2. interpolate u_{n+1}, u_n, and delta_u at the Gauss point
//          3. compute elastic part: P(F(u)) : Grad(delta_u)
//          4. reconstruct Newmark acceleration a_{n+1}(u)
//          5. add inertia part: rho * a_{n+1} · delta_u
//          6. write candidate history h_trial = [v_{n+1}, a_{n+1}]
//
// History convention
// ------------------
//   h_n[0       ... dim-1]  = v_n^gp
//   h_n[dim     ... 2dim-1] = a_n^gp
//
// During Newton iteration, the current acceleration is reconstructed from the
// current displacement iterate and previous accepted Gauss-point history:
//
//   a_{n+1} = 1/(beta dt^2) * (u_{n+1} - u_n - dt v_n)
//             - (1/(2 beta) - 1) * a_n
//
// The corresponding velocity candidate is
//
//   v_{n+1} = v_n + dt * ((1-gamma) a_n + gamma a_{n+1})
//
// Important solver convention
// ---------------------------
// h_trial is only a candidate state during Newton iterations.  The solver
// commits h_trial -> h_n only after the global time step is accepted.  Therefore
// rejected time steps do not corrupt the previous accepted history.
//
// Scalar weak form
// ----------------
//   G(u,delta_u) = int_Omega P(F(u)) : Grad(delta_u) dV
//                + int_Omega rho a_{n+1}(u) · delta_u dV
//
// Code-change rule for this annotated version
// -------------------------------------------
// Only comments were added/reorganized.  The executable code and formulas are
// kept identical to the uploaded version.
//
// =============================================================================

// -----------------------------------------------------------------------------
// Model identity queried by the solver/logger.
// -----------------------------------------------------------------------------
const char* active_constitutive_model_name() {
    return "weak_form_hyper_elastic_neohookean_newmark_gp_history_displacement_only";
}

// -----------------------------------------------------------------------------
// Field selection for generic assembly.
// This dynamic solid intentionally assembles only displacement DOFs.
// -----------------------------------------------------------------------------
std::vector<std::string> active_constitutive_field_names(const FieldMesh& mesh) {
    (void)mesh;
    return {"displacement"};
}

// Name-based local field view.  It avoids hard-coded local offsets and keeps the
// weak form readable even if the global framework supports many fields.
using NewmarkLocalLayout = LocalFieldView;

static NewmarkLocalLayout make_newmark_layout(const FieldMesh& mesh, int elem_id) {
    return make_local_field_view(mesh, elem_id, {"displacement"});
}

// -----------------------------------------------------------------------------
// Weak-scalar AD extraction types
// -----------------------------------------------------------------------------
using WeakInnerFad = Sacado::Fad::DFad<double>;
using WeakOuterFad = Sacado::Fad::DFad<WeakInnerFad>;

static double weak_inner_dx(const WeakInnerFad& x, int i) {
    return (i >= 0 && i < x.size()) ? x.dx(i) : 0.0;
}

// Extract a double value from either double, inner Fad, or nested Fad.
// This is used ONLY for writing h_trial.  The residual/tangent still see the
// full AD expression through a_newmark below.
static double scalar_value(double x) {
    return x;
}

static double scalar_value(const WeakInnerFad& x) {
    return x.val();
}

static double scalar_value(const WeakOuterFad& x) {
    return x.val().val();
}

// Newmark parameters.  For now they are local constants.  Later, these can be
// moved to SolverSettings if you want to control them from Material/Solver txt.
struct NewmarkParameters {
    double newmark_beta = 0.25;  // average acceleration method
    double newmark_gamma = 0.50;
};

// Read Newmark beta/gamma from ProblemParameters.  This keeps the time integrator
// parameters outside the weak form formula itself.
static NewmarkParameters make_newmark_parameters(const ProblemParameters& params) {
    NewmarkParameters nm;
    nm.newmark_beta = params.material.newmark_beta;
    nm.newmark_gamma = params.material.newmark_gamma;
    return nm;
}

void print_active_constitutive_parameters(std::ostream& os, const ProblemParameters& params) {
    const NewmarkParameters nm = make_newmark_parameters(params);
    os << "  newmark_beta  : " << nm.newmark_beta << "\n"
       << "  newmark_gamma : " << nm.newmark_gamma << "\n";
}

// Validate that the assembler/solver supplied a Gauss-point history object.
// Size is checked later at each Gauss point because each point owns h_n/h_trial.
static void require_newmark_history(const ElementState& state, int dim) {
    if (!state.material_history) {
        throw std::runtime_error(
            "Newmark hyper-elastic weak form requires Gauss-point history. "
            "Add 'history newmark 2*dim' to DOFS_info.txt, e.g. 'history newmark 4' for 2D."
        );
    }
    const int required = 2 * dim;
    if (required <= 0) {
        throw std::runtime_error("Invalid dimension for Newmark history.");
    }
}

// -----------------------------------------------------------------------------
// Scalar weak form:
//   elastic part : int P(F(u)) : Grad(delta_u) dV
//   inertia part : int rho * a_newmark_gp(u,u_n,v_n,a_n) . delta_u dV
// -----------------------------------------------------------------------------
template <class Scalar>
static Scalar weak_scalar_hyperelastic_newmark_solid(
    const FieldMesh& mesh,
    int elem_id,
    const std::vector<Scalar>& x_elem,     // current Newton iterate u_{n+1}
    const std::vector<Scalar>& test_elem,  // independent virtual displacement delta_u
    const ProblemParameters& params,
    const ElementState& state
) {
    const NewmarkLocalLayout L = make_newmark_layout(mesh, elem_id);
    const auto& U = L.field("displacement");

    if (static_cast<int>(x_elem.size()) != L.n_local()) {
        throw std::runtime_error("Bad current element vector size in Newmark hyper-elastic weak form.");
    }
    if (static_cast<int>(state.x_n_elem.size()) != L.n_local()) {
        throw std::runtime_error("Bad previous element vector size in Newmark hyper-elastic weak form.");
    }
    if (test_elem.size() != x_elem.size()) {
        throw std::runtime_error("Bad test element vector size in Newmark hyper-elastic weak form.");
    }

    require_newmark_history(state, L.dim);

    const double dt = state.dt;
    if (dt <= 0.0) {
        throw std::runtime_error("Newmark hyper-elastic weak form requires positive dt.");
    }

    const double rho = params.material.structure_rho; // solid density
    const double E   = params.material.youngs_modulus;
    const double nu  = params.material.poisson_ratio;

    const NewmarkParameters nm = make_newmark_parameters(params);
    if (nm.newmark_beta <= 0.0) {
        throw std::runtime_error("Newmark beta must be positive.");
    }

    Scalar G(0.0);

    // Geometry is displacement-based because this is a displacement-only solid.
    const ElementGeometry geom = make_displacement_geometry_element(mesh, elem_id);

    const auto gps = gauss_rule(U.element_type, params.solver.gauss_order);
    if (state.material_history->gp.size() != gps.size()) {
        // This should already be initialized by initialize_material_history_store.
        // Resize here only as a defensive fallback.
        state.material_history->gp.resize(gps.size());
    }

    for (int q = 0; q < static_cast<int>(gps.size()); ++q) {
        const auto& gp = gps[q];

        MaterialPointHistory& hist = state.material_history->gp[q];

        // ---------------------------------------------------------------------
        // History layout at this Gauss point:
        //   h_n[0..dim-1]       : v_n^gp
        //   h_n[dim..2dim-1]    : a_n^gp
        //   h_trial receives v_{n+1}^gp and a_{n+1}^gp during Newton evaluation.
        // ---------------------------------------------------------------------
        const int required_history = 2 * L.dim;
        if (static_cast<int>(hist.h_n.size()) != required_history) {
            throw std::runtime_error(
                "Bad Newmark history size. Use 'history newmark 2*dim' in DOFS_info.txt "
                "(2D -> 4, 3D -> 6)."
            );
        }
        if (static_cast<int>(hist.h_trial.size()) != required_history) {
            hist.h_trial.assign(required_history, 0.0);
        }

        const ShapeData su = compute_shape(U.element_type, gp.xi, gp.eta, gp.zeta);
        const JacobianData Jgeo = compute_jacobian_data(geom, su);
        if (Jgeo.detJ <= 0.0) {
            throw std::runtime_error("Negative reference detJ in Newmark hyper-elastic solid element.");
        }

        const auto dNu = compute_dN_dx(su, Jgeo);
        const double dV = Jgeo.detJ * gp.weight;

        // ---------------------------------------------------------------------
        // Current and previous accepted displacement at the Gauss point.
        //
        // u_gp   = sum_a N_a u_{a,n+1}
        // u_n_gp = sum_a N_a u_{a,n}
        //
        // state.x_n_elem is filled by the assembler from the previous accepted
        // global solution x_n in the same local field order as x_elem.
        // ---------------------------------------------------------------------
        const auto u_gp =
            kinematic_field_value_as<Scalar>(L, "displacement", x_elem, su.N);

        const auto u_n_gp =
            kinematic_field_value_as<Scalar>(L, "displacement", state.x_n_elem, su.N);

        // Virtual displacement and virtual deformation gradient.
        const auto delta_u_gp =
            kinematic_field_value_as<Scalar>(L, "displacement", test_elem, su.N);

        const auto deltaF =
            kinematic_field_gradient_as<Scalar>(L, "displacement", test_elem, dNu);

        // ---------------------------------------------------------------------
        // Elastic part:
        //   F = I + Grad(u)
        //   P = Neo-Hookean first Piola stress
        //   G_elastic += int P : deltaF dV
        //
        // In 2D, F is embedded into 3D as plane strain before evaluating P.
        // ---------------------------------------------------------------------
        const auto F = kinematic_deformation_gradient_as<Scalar>(L, x_elem, dNu);
        const auto F3 = kinematic_embed_plane_strain_3d(F, L.dim);
        const auto P3 = kinematic_neo_hookean_first_piola_young(F3, E, nu);

        for (int i = 0; i < L.dim; ++i) {
            for (int A = 0; A < L.dim; ++A) {
                G += P3[i][A] * deltaF[i][A] * Scalar(dV);
            }
        }

        // ---------------------------------------------------------------------
        // Newmark inertia part:
        //
        // h_n stores the previous Gauss-point velocity/acceleration:
        //   v_n^gp = h_n[0..dim-1]
        //   a_n^gp = h_n[dim..2dim-1]
        //
        // Current acceleration is reconstructed locally from current u:
        //   a_{n+1}^gp =
        //       (u_{n+1}^gp - u_n^gp - dt v_n^gp)/(beta dt^2)
        //       - (1/(2 beta) - 1) a_n^gp
        //
        // Then we add:
        //   G_inertia += int rho a_{n+1}^gp . delta_u^gp dV
        //
        // Because a_{n+1} is written with Scalar, AD automatically contributes
        // the consistent mass-like tangent term:
        //   d(rho a_{n+1})/du = rho/(beta dt^2) * N_a N_b.
        // ---------------------------------------------------------------------
        std::vector<Scalar> a_np1_gp(L.dim, Scalar(0.0));
        std::vector<Scalar> v_np1_gp(L.dim, Scalar(0.0));

        const double inv_beta_dt2 = 1.0 / (nm.newmark_beta * dt * dt);
        const double accel_old_factor = (1.0 / (2.0 * nm.newmark_beta)) - 1.0;

        for (int i = 0; i < L.dim; ++i) {
            const double v_n_gp = hist.h_n[i];
            const double a_n_gp = hist.h_n[L.dim + i];

            a_np1_gp[i] =
                Scalar(inv_beta_dt2) *
                (u_gp[i] - u_n_gp[i] - Scalar(dt * v_n_gp))
                - Scalar(accel_old_factor * a_n_gp);

            v_np1_gp[i] =
                Scalar(v_n_gp)
                + Scalar(dt) *
                  (Scalar(1.0 - nm.newmark_gamma) * Scalar(a_n_gp)
                   + Scalar(nm.newmark_gamma) * a_np1_gp[i]);

            G += Scalar(rho) * a_np1_gp[i] * delta_u_gp[i] * Scalar(dV);
        }

        // ---------------------------------------------------------------------
        // Trial history update.
        //
        // This is called inside Newton iterations, so it must be treated as
        // "candidate state".  The solver's commit_material_history_store()
        // performs the actual update h_trial -> h_n only after the time step
        // is accepted.  If the time step is rejected, h_n remains unchanged.
        // ---------------------------------------------------------------------
        for (int i = 0; i < L.dim; ++i) {
            hist.h_trial[i]         = scalar_value(v_np1_gp[i]);
            hist.h_trial[L.dim + i] = scalar_value(a_np1_gp[i]);
        }
        hist.state_trial = 0.0;
    }

    return G;
}

// -----------------------------------------------------------------------------
// Public element interface used by assembly.cpp
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Assembly-facing API.
// assembly.cpp calls exactly this function for each active element.
// -----------------------------------------------------------------------------
ElementSystem compute_constitutive_element_system(
    const FieldMesh& mesh,
    int elem_id,
    const Vector& x_elem,
    const ProblemParameters& params,
    const ElementState& state
) {
    const int n = static_cast<int>(x_elem.size());

    // x_ad are current displacement unknowns u_{n+1}.
    // Outer AD derivatives are with respect to current DOFs.
    std::vector<WeakOuterFad> x_ad(n);

    // test_ad are independent virtual displacement coefficients delta_u.
    // Inner AD derivatives are with respect to test DOFs.
    std::vector<WeakOuterFad> test_ad(n);

    for (int j = 0; j < n; ++j) {
        const WeakInnerFad x_value(x_elem[j]);
        x_ad[j] = WeakOuterFad(n, j, x_value);
    }

    for (int i = 0; i < n; ++i) {
        test_ad[i] = WeakOuterFad(WeakInnerFad(n, i, 0.0));
    }

    const WeakOuterFad G =
        weak_scalar_hyperelastic_newmark_solid(mesh, elem_id, x_ad, test_ad, params, state);

    ElementSystem sys;
    sys.residual.assign(n, 0.0);
    sys.tangent.assign(n * n, 0.0);

    // Residual extraction:
    //   R_i = dG / d(delta_u_i)
    const WeakInnerFad G_value = G.val();

    for (int i = 0; i < n; ++i) {
        sys.residual[i] = weak_inner_dx(G_value, i);

        // Tangent extraction:
        //   K_ij = dR_i / d(u_j)
        //
        // Because x_ad uses outer derivatives and test_ad uses inner derivatives,
        // G.dx(j) gives dG/du_j as an inner-Fad object.  Its inner derivative
        // with respect to test_i gives d^2G/(d test_i d u_j).
        const int row = i * n;
        for (int j = 0; j < n; ++j) {
            const WeakInnerFad dG_duj =
                (j >= 0 && j < G.size()) ? G.dx(j) : WeakInnerFad(0.0);

            sys.tangent[row + j] = weak_inner_dx(dG_duj, i);
        }
    }

    return sys;
}
