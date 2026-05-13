/*
================================================================================
MODEL MAP: weak_form_hyper_elastic.cpp
================================================================================
Purpose
  Pure displacement-only hyperelastic solid model using a Neo-Hookean material.

How this file is called
  solver.cpp
    -> assembly.cpp
      -> compute_constitutive_element_system(...)
        -> weak_scalar_hyperelastic_solid(...)
        -> Sacado AD residual/tangent extraction

Active unknowns
  displacement only. Even if DOFS_info.txt contains velocity or pressure for
  another model, this constitutive kernel explicitly asks assembly for only the
  displacement block.

Weak-form idea
  G(u; delta_u) = integral P(F(u)) : Grad(delta_u) dV.
  Residual R_i  = dG / d(delta_u_i).
  Tangent  K_ij = dR_i / du_j.

Element-level flow
  active_constitutive_field_names(...)
    -> returns {"displacement"}.
  make_hyperelastic_layout(...)
    -> local displacement view.
  weak_scalar_hyperelastic_solid(...)
    -> Gauss loop, deformation gradient, plane-strain embedding, Piola stress.
  compute_constitutive_element_system(...)
    -> nested AD extraction of local R and K.

2D convention
  In 2D, the deformation gradient is embedded into 3D plane strain before
  evaluating the Neo-Hookean first Piola stress. Only in-plane test gradients
  contribute to G.

Code-change policy in this annotated copy
  Only comments and section markers were added. Executable statements are left
  in the same order and with the same expressions.
================================================================================
*/
#include "weak_form.h"
#include "fem_utils.h"
#include "kinematics.h"

#include <Sacado_Fad_DFad.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// Model registration
// -----------------------------------------------------------------------------
// The executable links exactly one weak_form_*.cpp at a time.  The following
// function gives the linked model a human-readable name for solver diagnostics.
const char* active_constitutive_model_name() {
    return "weak_form_hyper_elastic_neohookean_displacement_only_ad_weak_scalar_direct";
}

// -----------------------------------------------------------------------------
// Active field declaration
// -----------------------------------------------------------------------------
// Assembly asks this function which fields should appear in the element-local
// vector.  The returned order controls the local layout used below.
std::vector<std::string> active_constitutive_field_names(const FieldMesh& mesh) {
    (void)mesh;
    // This constitutive kernel is a pure solid displacement formulation.
    // Even if DOFS_info.txt still lists velocity/pressure for compatibility,
    // only displacement is assembled by this weak form.
    return {"displacement"};
}

void print_active_constitutive_parameters(std::ostream& os, const ProblemParameters& params) {
    os << "  E  : " << params.material.youngs_modulus << "\n"
       << "  nu : " << params.material.poisson_ratio << "\n";
}

using HyperElasticLocalLayout = LocalFieldView;

// Local displacement-only layout.  The assembler may know many fields globally,
// but this model requests only displacement DOFs for its element system.
static HyperElasticLocalLayout make_hyperelastic_layout(const FieldMesh& mesh, int elem_id) {
    return make_local_field_view(mesh, elem_id, {"displacement"});
}

// -----------------------------------------------------------------------------
// Direct scalar weak form + AD extraction
// -----------------------------------------------------------------------------
// This file intentionally defines the scalar weak form directly as
//     G(u, delta_u) = int_Omega P(F(u)) : Grad(delta_u) dV.
// The element residual and tangent are then extracted as
//     R_i  = dG / d(delta_u_i),
//     K_ij = dR_i / du_j.
// No intermediate residual vector is built inside the weak form.
// -----------------------------------------------------------------------------
// Nested automatic differentiation types
// -----------------------------------------------------------------------------
// WeakInnerFad differentiates with respect to virtual/test DOFs.
// WeakOuterFad wraps WeakInnerFad and differentiates with respect to solution
// DOFs.  This nested structure gives both residual and tangent from one scalar G.
using WeakInnerFad = Sacado::Fad::DFad<double>;
using WeakOuterFad = Sacado::Fad::DFad<WeakInnerFad>;

// Safely read d()/d(test_i).  Sacado objects can sometimes have zero derivative
// slots when a value is constant; this helper returns 0 in that case.
static double weak_inner_dx(const WeakInnerFad& x, int i) {
    return (i >= 0 && i < x.size()) ? x.dx(i) : 0.0;
}

template <class Scalar>
// -----------------------------------------------------------------------------
// Scalar weak form for the Neo-Hookean solid
// -----------------------------------------------------------------------------
// G = int P(F(u)) : Grad(delta_u) dV.  Because G is scalar, Sacado can derive
// both local residual and exact AD tangent without a hand-coded material tangent.
static Scalar weak_scalar_hyperelastic_solid(
    const FieldMesh& mesh,
    int elem_id,
    const std::vector<Scalar>& x_elem,
    const std::vector<Scalar>& test_elem,
    const ProblemParameters& params,
    const ElementState& state
) {
    (void)state;

    const HyperElasticLocalLayout L = make_hyperelastic_layout(mesh, elem_id);
    const auto& U = L.field("displacement");

    if (static_cast<int>(x_elem.size()) != L.n_local()) {
        throw std::runtime_error("Bad current element vector size in hyper-elastic solid weak form.");
    }
    if (test_elem.size() != x_elem.size()) {
        throw std::runtime_error("Bad test element vector size in hyper-elastic solid weak form.");
    }

    Scalar G(0.0);

    // Geometry and interpolation are taken from the displacement field.
    const ElementGeometry geom = make_displacement_geometry_element(mesh, elem_id);

    const double E  = params.material.youngs_modulus;
    const double nu = params.material.poisson_ratio;

    for (const auto& gp : gauss_rule(U.element_type, params.solver.gauss_order)) {
        const ShapeData su = compute_shape(U.element_type, gp.xi, gp.eta, gp.zeta);
        const JacobianData Jgeo = compute_jacobian_data(geom, su);
        if (Jgeo.detJ <= 0.0) {
            throw std::runtime_error("Negative reference detJ in hyper-elastic solid element.");
        }

        const auto dNu = compute_dN_dx(su, Jgeo);
        const double dV = Jgeo.detJ * gp.weight;

        // Kinematics at this Gauss point.  F is built from the displacement field; in
        // 2D it is embedded into 3D plane strain before material evaluation.
        const auto F = kinematic_deformation_gradient_as<Scalar>(L, x_elem, dNu);
        const auto deltaF = kinematic_field_gradient_as<Scalar>(L, "displacement", test_elem, dNu);
        const auto F3 = kinematic_embed_plane_strain_3d(F, L.dim);
        const auto P3 = kinematic_neo_hookean_first_piola_young(F3, E, nu);

        // G = int P : deltaF dV.  In 2D, P is evaluated by plane-strain
        // embedding, while only in-plane virtual gradients contribute.
        for (int i = 0; i < L.dim; ++i) {
            for (int A = 0; A < L.dim; ++A) {
                G += P3[i][A] * deltaF[i][A] * Scalar(dV);
            }
        }
    }

    return G;
}

// -----------------------------------------------------------------------------
// Public element API used by assembly.cpp
// -----------------------------------------------------------------------------
// Input : one element-local vector x_elem and time/history state.
// Output: ElementSystem containing
//         residual[a]      = dG / d(test_a),
//         tangent[a*n + b] = d residual[a] / d x_b.
ElementSystem compute_constitutive_element_system(
    const FieldMesh& mesh,
    int elem_id,
    const Vector& x_elem,
    const ProblemParameters& params,
    const ElementState& state
) {
    const int n = static_cast<int>(x_elem.size());

    std::vector<WeakOuterFad> x_ad(n);
    std::vector<WeakOuterFad> test_ad(n);
    for (int j = 0; j < n; ++j) {
        const WeakInnerFad x_value(x_elem[j]);
        x_ad[j] = WeakOuterFad(n, j, x_value);
    }
    for (int i = 0; i < n; ++i) {
        test_ad[i] = WeakOuterFad(WeakInnerFad(n, i, 0.0));
    }

    // Evaluate G with nested AD variables and then read out R and K below.
    const WeakOuterFad G = weak_scalar_hyperelastic_solid(mesh, elem_id, x_ad, test_ad, params, state);

    ElementSystem sys;
    sys.residual.assign(n, 0.0);
    sys.tangent.assign(n * n, 0.0);

    const WeakInnerFad G_value = G.val();
    for (int i = 0; i < n; ++i) {
        sys.residual[i] = weak_inner_dx(G_value, i);
        const int row = i * n;
        for (int j = 0; j < n; ++j) {
            const WeakInnerFad dG_dxj = (j >= 0 && j < G.size()) ? G.dx(j) : WeakInnerFad(0.0);
            sys.tangent[row + j] = weak_inner_dx(dG_dxj, i);
        }
    }

    return sys;
}


// Non-multiscale diagnostic stubs used by solver.cpp.
MultiscaleRveDiagnostics consume_multiscale_rve_diagnostics() {
    return MultiscaleRveDiagnostics{};
}

bool multiscale_rve_context_configured() {
    return false;
}
