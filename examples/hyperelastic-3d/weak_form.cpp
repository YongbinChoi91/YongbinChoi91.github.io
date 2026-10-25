/*
================================================================================
MODEL MAP: weak_form_hyper_elastic_ad.cpp
================================================================================
Purpose
  Pure displacement-only hyperelastic solid model using a local hpx::grad constitutive Neo-Hookean material path.

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
#include "kinematics.h"
#include "autodiff.h"
#include "local_element_system_ad.h"

#include <algorithm>
#include <array>
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
    return "HyperElastic_Neohook_HPX_AD";
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

std::vector<ConstitutivePostprocessOutputSpec> active_constitutive_postprocess_outputs() {
    return {{"von_mises", 1}, {"J", 1}};
}

using HyperElasticLocalLayout = LocalFieldView;

// Local displacement-only layout.  The assembler may know many fields globally,
// but this model requests only displacement DOFs for its element system.
static HyperElasticLocalLayout make_hyperelastic_layout(const FieldMesh& mesh, int elem_id) {
    return make_local_field_view(mesh, elem_id, {"displacement"});
}

static bool output_requested(const std::vector<std::string>& requested,const std::string& name){
    return std::find(requested.begin(),requested.end(),name)!=requested.end();
}

template <class Scalar, std::size_t R, std::size_t C>
static std::array<std::array<Scalar, C>, R> to_fixed_tensor2(
    const std::vector<std::vector<Scalar>>& A,
    const char* name
) {
    if (A.size() != R) {
        throw std::runtime_error(std::string(name) + " has an unexpected row count.");
    }
    std::array<std::array<Scalar, C>, R> out{};
    for (std::size_t i = 0; i < R; ++i) {
        if (A[i].size() != C) {
            throw std::runtime_error(std::string(name) + " has an unexpected column count.");
        }
        for (std::size_t j = 0; j < C; ++j) {
            out[i][j] = A[i][j];
        }
    }
    return out;
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
    const ElementGeometry geom = make_displacement_element_geometry(mesh, elem_id);

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
        const auto F = hpx::deformation_gradient_as<Scalar>(L, x_elem, dNu);
        const auto deltaF = hpx::field_grad_as<Scalar>(L, "displacement", test_elem, dNu);
        // G = int P : deltaF dV.  In 2D plane strain, only the 2x2 in-plane
        // block contributes, so seed four constitutive AD variables instead of
        // a full embedded 3x3 tensor.
        if (L.dim == 2) {
            const auto F2 = to_fixed_tensor2<Scalar, 2, 2>(F, "F2");
            const auto P2 = hpx::grad(
                [&](const auto& F_ad) {
                    return hpx::neo_hookean_energy_young(F_ad, E, nu);
                },
                F2
            );
            for (int i = 0; i < 2; ++i) {
                for (int A = 0; A < 2; ++A) {
                    G += P2[i][A] * deltaF[i][A] * Scalar(dV);
                }
            }
        } else {
            const auto F3_vec = hpx::embed_plane_strain_3d(F, L.dim);
            const auto F3 = to_fixed_tensor2<Scalar, 3, 3>(F3_vec, "F3");
            const auto P3 = hpx::grad(
                [&](const auto& F_ad) {
                    return hpx::neo_hookean_energy_young(F_ad, E, nu);
                },
                F3
            );
            for (int i = 0; i < L.dim; ++i) {
                for (int A = 0; A < L.dim; ++A) {
                    G += P3[i][A] * deltaF[i][A] * Scalar(dV);
                }
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
    return hpx::scalar_weak_form_element_system(
        x_elem,
        [&](const auto& x_ad, const auto& test_ad) {
            return weak_scalar_hyperelastic_solid(mesh, elem_id, x_ad, test_ad, params, state);
        }
    );
}

void compute_constitutive_postprocess(
    const FieldMesh& mesh,
    int elem_id,
    const Vector& x_elem,
    const ProblemParameters& params,
    const ElementState& state,
    const std::vector<std::string>& requested,
    std::vector<ConstitutivePostprocessOutputValues>& outputs
) {
    (void)state;
    if(!output_requested(requested,"von_mises") && !output_requested(requested,"j")) return;

    const auto L = make_hyperelastic_layout(mesh, elem_id);
    const auto& U = L.field("displacement");
    const ElementGeometry geom = make_displacement_element_geometry(mesh, elem_id);
    const auto gps = gauss_rule(U.element_type, params.solver.gauss_order);
    const double E = params.material.youngs_modulus;
    const double nu = params.material.poisson_ratio;

    ConstitutivePostprocessOutputValues vm{"von_mises",1,{}};
    ConstitutivePostprocessOutputValues jout{"j",1,{}};
    for(const auto& gp:gps){
        const ShapeData su = compute_shape(U.element_type,gp.xi,gp.eta,gp.zeta);
        const JacobianData Jgeo = compute_jacobian_data(geom,su);
        if(Jgeo.detJ<=0.0) throw std::runtime_error("Negative detJ while exporting hyperelastic Gauss data.");
        const auto dNu = compute_dN_dx(su,Jgeo);
        const auto F = hpx::deformation_gradient_as<double>(L,x_elem,dNu);
        const auto F3 = hpx::embed_plane_strain_3d(F,L.dim);
        const double J = hpx::det(F3);
        if(output_requested(requested,"von_mises")) vm.values.push_back(hpx::neo_hookean_von_mises(F3,E,nu));
        if(output_requested(requested,"j")) jout.values.push_back(J);
    }
    if(output_requested(requested,"von_mises")) outputs.push_back(std::move(vm));
    if(output_requested(requested,"j")) outputs.push_back(std::move(jout));
}

namespace {
template <class Scalar>
static void accumulate_hyperelastic_multiscale_response(
    const FieldMesh& mesh,
    const std::vector<Scalar>& displacement,
    const ProblemParameters& params,
    std::array<std::array<Scalar,3>,3>& intP,
    double& volume
) {
    const double E  = params.material.youngs_modulus;
    const double nu = params.material.poisson_ratio;
    const DofLayout& layout = get_cached_dof_layout(mesh);

    for (int e = 0; e < static_cast<int>(mesh.displacement_elements.size()); ++e) {
        const HyperElasticLocalLayout L = make_hyperelastic_layout(mesh, e);
        const auto& U = L.field("displacement");
        const auto dofs = make_element_dofs_for_fields(layout, e, {"displacement"});
        std::vector<Scalar> ue(dofs.size());
        for (int a = 0; a < static_cast<int>(dofs.size()); ++a) ue[a] = displacement[dofs[a]];

        const ElementGeometry geom = make_displacement_element_geometry(mesh, e);
        for (const auto& gp : gauss_rule(U.element_type, params.solver.gauss_order)) {
            const ShapeData su = compute_shape(U.element_type, gp.xi, gp.eta, gp.zeta);
            const JacobianData Jgeo = compute_jacobian_data(geom, su);
            if (Jgeo.detJ <= 0.0) {
                throw std::runtime_error("Negative reference detJ in hyperelastic multiscale response.");
            }
            const auto dNu = compute_dN_dx(su, Jgeo);
            const double dV = Jgeo.detJ * gp.weight;
            const auto F = hpx::deformation_gradient_as<Scalar>(L, ue, dNu);
            const auto F3 = hpx::embed_plane_strain_3d(F, L.dim);
            const auto P3 = hpx::neo_hookean_first_piola_young(F3, E, nu);

            volume += dV;
            for (int i = 0; i < 3; ++i) {
                for (int A = 0; A < 3; ++A) {
                    intP[i][A] += P3[i][A] * Scalar(dV);
                }
            }
        }
    }
}
}

ConstitutiveMultiscaleResponse compute_constitutive_multiscale_response(
    const FieldMesh& mesh,
    const Vector& displacement,
    const ProblemParameters& params,
    const MaterialHistoryStore* history,
    const ElementState& response_state
) {
    (void)history;
    (void)response_state;
    using Fad = Sacado::Fad::DFad<double>;

    const int ndofs = total_dof_count(mesh);
    if (static_cast<int>(displacement.size()) < ndofs) {
        throw std::runtime_error("Hyperelastic multiscale response received a displacement vector shorter than mesh DOFs.");
    }

    std::vector<Fad> u_ad(ndofs);
    for (int a = 0; a < ndofs; ++a) u_ad[a] = Fad(ndofs, a, displacement[a]);

    std::array<std::array<Fad,3>,3> intP_ad{};
    for (auto& row : intP_ad) {
        for (auto& v : row) v = Fad(ndofs, 0.0);
    }
    double volume = 0.0;
    accumulate_hyperelastic_multiscale_response(mesh, u_ad, params, intP_ad, volume);
    if (volume <= 0.0) {
        throw std::runtime_error("Hyperelastic multiscale response volume is non-positive.");
    }

    ConstitutiveMultiscaleResponse out;
    out.volume = volume;
    out.dIntPdu.assign(9, std::vector<double>(ndofs, 0.0));
    for (int i = 0; i < 3; ++i) {
        for (int A = 0; A < 3; ++A) {
            const int row = 3 * i + A;
            const Fad Pint = intP_ad[i][A];
            out.intP[i][A] = Pint.val();
            for (int a = 0; a < ndofs; ++a) {
                out.dIntPdu[row][a] = (a < Pint.size()) ? Pint.dx(a) : 0.0;
            }
        }
    }
    return out;
}
