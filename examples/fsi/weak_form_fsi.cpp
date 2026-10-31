/*
================================================================================
MODEL MAP: multi-domain weak form
================================================================================
Purpose
  Multi-domain element model. The same local interface is used for the current
  continuum domains, while the domain type decides which scalar weak form is used.

How this file is called
  solver.cpp
    -> assembly.cpp over domains/elements
      -> compute_constitutive_element_system(...)
        -> weak_scalar_domain(...)
          -> weak_scalar_fluid_domain(...) for fluid domain
          -> weak_scalar_solid_domain(...) for solid domain
        -> Sacado AD residual/tangent extraction

Main field roles
  velocity      : fluid velocity unknown, used in fluid domain
  displacement  : ALE mesh displacement in fluid, physical displacement in solid
  pressure      : fluid incompressibility multiplier

Fluid-domain G contains
  1. ALE mass/time term,
  2. convective term in mapped coordinates,
  3. mesh-motion correction,
  4. viscous and pressure stresses pulled through F^{-1},
  5. ALE incompressibility,
  6. mesh-motion PDE for displacement, selected by mesh_motion_type = BH or HE.

Solid-domain G contains
  1. displacement inertia term,
  2. Neo-Hookean first Piola stress term,
  3. optional plane-strain 3D embedding through kinematics helpers.

Code-change policy in this annotated copy
  Only comments and section markers were added. Executable statements are left
  in the same order and with the same expressions.
================================================================================
*/
#include "weak_form.h"
#include "kinematics.h"
#include "local_element_system_ad.h"
#include "hpx/time_integrator.h"

#include <Sacado_Fad_DFad.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

// ============================================================================
// 0. Model registration
// ============================================================================
// The executable links exactly one weak_form_*.cpp at a time.  The following
// function gives the linked model a human-readable name for solver diagnostics.
const char* active_constitutive_model_name() {
    return "FSI_Neohook";
}


// ============================================================================
// 1. Active fields and local layout
// ============================================================================
// Assembly asks this function which fields should appear in the element-local
// vector.  The returned order controls the local layout used below.
std::vector<std::string> active_constitutive_field_names(const FieldMesh& mesh) {
    if (!mesh.dof_field_order.empty()) {
        return mesh.dof_field_order;
    }
    return active_dof_field_names(mesh);
}

void print_active_constitutive_parameters(std::ostream& os, const ProblemParameters& params) {
    os << "  rho                       : " << params.material.rho << "\n"
       << "  mu                        : " << params.material.mu << "\n"
       << "  u_bar                     : " << params.material.u_bar << "\n"
       << "  structure_rho             : " << params.material.structure_rho << "\n"
       << "  E                         : " << params.material.youngs_modulus << "\n"
       << "  nu                        : " << params.material.poisson_ratio << "\n"
       << "  mesh_motion_type          : " << params.material.mesh_motion_type << "\n"
       << "  mesh_motion_coefficient   : " << params.material.mesh_motion_coefficient << "\n"
       << "  mesh_motion_poisson_ratio : " << params.material.mesh_motion_poisson_ratio << "\n";
}

std::vector<ConstitutivePostprocessOutputSpec> active_constitutive_postprocess_outputs() {
    return {
        {"sigma", 9},
        {"minJ", 1},
        {"traction", 3, ConstitutivePostprocessLocation::Boundary}
    };
}

using DomainLocalLayout = LocalFieldView;

// Keep the domain-to-field decision in one place.
// For this compatibility step both branches still use the active field order
// expected by the existing element vector.  Later, this function is the only
// place that should be specialized for displacement-only solids, dynamic
// solids, thermal domains, or other domain-local field sets.
static std::vector<std::string> active_domain_field_names(
    const FieldMesh& mesh,
    DomainType domain_type
) {
    switch (domain_type) {
        case DomainType::Fluid:
        case DomainType::Solid:
            return active_constitutive_field_names(mesh);
        default:
            throw std::runtime_error("Unsupported domain type while selecting local fields.");
    }
}

// Name-based local layout for the selected domain branch.
// New constitutive kernels must use LocalFieldView instead of the legacy
// FieldElementLayout adapter.
static DomainLocalLayout make_domain_layout(
    const FieldMesh& mesh,
    int elem_id,
    DomainType domain_type
) {
    return make_local_field_view(mesh, elem_id, active_domain_field_names(mesh, domain_type));
}


// ============================================================================
// AD types and scalar weak forms
// ============================================================================
// The domain kernels below define G directly from virtual fields
// (delta_v, delta_u, delta_p) and their gradients.  No intermediate residual
// vector is built and re-multiplied by test coefficients.
// Residual: R_i = dG/d(delta_x_i)
// Tangent : K_ij = dR_i/dx_j = d^2G/(d(delta_x_i) dx_j)
template <class Scalar>
// ============================================================================
// 2. Fluid-domain weak scalar
// ============================================================================
// Uses ALE kinematics.  Displacement acts as the mesh motion field; velocity and
// pressure define the incompressible fluid residual.
static Scalar weak_scalar_fluid_domain(
    const FieldMesh& mesh,
    int elem_id,
    const std::vector<Scalar>& x_elem,
    const std::vector<Scalar>& test_elem,
    const ProblemParameters& params,
    const ElementState& state
) {
    const double dt = state.dt;
    if (dt <= 0.0) throw std::runtime_error("Fluid-domain state dt must be positive.");

    const DomainLocalLayout L = make_domain_layout(mesh, elem_id, state.domain_type);
    const auto& V = L.field("velocity");
    const auto& U = L.field("displacement");
    const auto& Pf = L.field("pressure");
    if (static_cast<int>(x_elem.size()) != L.n_local()) throw std::runtime_error("Bad current element vector size in fluid domain.");
    if (state.past_dofs.size() < 1 || static_cast<int>(state.past_dofs[0].size()) != L.n_local()) throw std::runtime_error("Bad previous element vector size in fluid domain.");
    if (test_elem.size() != x_elem.size()) throw std::runtime_error("Bad test element vector size in fluid-domain weak form.");

    Scalar G(0.0);
    const ElementGeometry geom = make_velocity_element_geometry(mesh, elem_id);

    const double rho = params.material.rho;
    const double mu = params.material.mu;
    const hpx::TimeSchemeStencils time_scheme = hpx::Shift_Crank_Nicolson(params.solver.theta, dt);
    const double theta_current = time_scheme.value.current();
    const double theta_past = time_scheme.value.past(0);
    const double mesh_coeff = params.material.mesh_motion_coefficient;

    for (const auto& gp : gauss_rule(V.element_type, params.solver.gauss_order)) {
        const ShapeData sv = compute_shape(V.element_type, gp.xi, gp.eta, gp.zeta);
        const ShapeData su = compute_shape(U.element_type, gp.xi, gp.eta, gp.zeta);
        const ShapeData sp = compute_shape(Pf.element_type, gp.xi, gp.eta, gp.zeta);
        const JacobianData Jgeo = compute_jacobian_data(geom, sv);
        if (Jgeo.detJ <= 0.0) throw std::runtime_error("Negative detJ in fluid-domain element.");

        const auto dNv = compute_dN_dx(sv, Jgeo);
        const auto dNu = compute_dN_dx(su, Jgeo);
        const double dV = Jgeo.detJ * gp.weight;

        const auto v = hpx::field_value_as<Scalar>(L, "velocity", x_elem, sv.N);
        const auto vn = hpx::field_value_as<Scalar>(L, "velocity", state.past_dofs[0], sv.N);
        const auto u = hpx::field_value_as<Scalar>(L, "displacement", x_elem, su.N);
        const auto un = hpx::field_value_as<Scalar>(L, "displacement", state.past_dofs[0], su.N);
        const auto grad_v = hpx::field_grad_as<Scalar>(L, "velocity", x_elem, dNv);
        const auto grad_vn = hpx::field_grad_as<Scalar>(L, "velocity", state.past_dofs[0], dNv);
        const auto H = hpx::field_grad_as<Scalar>(L, "displacement", x_elem, dNu);

        const auto delta_v = hpx::field_value_as<Scalar>(L, "velocity", test_elem, sv.N);
        const auto grad_delta_v = hpx::field_grad_as<Scalar>(L, "velocity", test_elem, dNv);
        const auto grad_delta_u = hpx::field_grad_as<Scalar>(L, "displacement", test_elem, dNu);
        const Scalar delta_p = hpx::scalar_field_value_as<Scalar>(L, "pressure", test_elem, sp.N);

        const Scalar p = hpx::scalar_field_value_as<Scalar>(L, "pressure", x_elem, sp.N);
        // Python multi-domain reference currently sets P_n from the current ansatz,
        // therefore P_time is effectively the current pressure. Keep this for
        // one-to-one C++/Python comparison.
        const Scalar p_time = p;

        const auto F = hpx::deformation_gradient_as<Scalar>(L, x_elem, dNu);
        const auto Fn = hpx::deformation_gradient_as<Scalar>(L, state.past_dofs[0], dNu);
        const Scalar J = hpx::det(F);
        const Scalar Jn = hpx::det(Fn);
        const auto Finv = hpx::inv(F);

        std::vector<std::vector<Scalar>> Ftime(L.dim, std::vector<Scalar>(L.dim, Scalar(0.0)));
        for (int i = 0; i < L.dim; ++i) {
            for (int j = 0; j < L.dim; ++j) {
                Ftime[i][j] = Scalar(theta_current) * F[i][j] + Scalar(theta_past) * Fn[i][j];
            }
        }
        const auto Finv_time = hpx::inv(Ftime);
        const Scalar J_time = Scalar(theta_current) * J + Scalar(theta_past) * Jn;

        std::vector<Scalar> v_time(L.dim, Scalar(0.0));
        std::vector<std::vector<Scalar>> grad_v_time(L.dim, std::vector<Scalar>(L.dim, Scalar(0.0)));
        for (int i = 0; i < L.dim; ++i) {
            v_time[i] = Scalar(theta_current) * v[i] + Scalar(theta_past) * vn[i];
            for (int j = 0; j < L.dim; ++j) {
                grad_v_time[i][j] = Scalar(theta_current) * grad_v[i][j] + Scalar(theta_past) * grad_vn[i][j];
            }
        }

        // conv = rho * J_theta * (grad_v_theta * F_theta^{-1} * v_theta)
        std::vector<Scalar> Finv_v(L.dim, Scalar(0.0));
        for (int i = 0; i < L.dim; ++i) {
            for (int j = 0; j < L.dim; ++j) Finv_v[i] += Finv_time[i][j] * v_time[j];
        }
        std::vector<Scalar> conv(L.dim, Scalar(0.0));
        for (int i = 0; i < L.dim; ++i) {
            for (int j = 0; j < L.dim; ++j) conv[i] += grad_v_time[i][j] * Finv_v[j];
            conv[i] *= Scalar(rho) * J_time;
        }

        // mesh velocity term: rho * J * grad_v * F^{-1} * (u-u_n)
        std::vector<Scalar> du(L.dim, Scalar(0.0));
        for (int i = 0; i < L.dim; ++i) du[i] = u[i] - un[i];
        std::vector<Scalar> Finv_du(L.dim, Scalar(0.0));
        for (int i = 0; i < L.dim; ++i)
            for (int j = 0; j < L.dim; ++j) Finv_du[i] += Finv[i][j] * du[j];
        std::vector<Scalar> meshmove(L.dim, Scalar(0.0));
        for (int i = 0; i < L.dim; ++i) {
            for (int j = 0; j < L.dim; ++j) meshmove[i] += grad_v[i][j] * Finv_du[j];
            meshmove[i] *= Scalar(rho) * J;
        }

        // stress_fluid = J_theta * rho*mu*(grad_v_theta F_inv_theta + F_inv_theta^T grad_v_theta^T) * F_inv_theta^T
        std::vector<std::vector<Scalar>> A(L.dim, std::vector<Scalar>(L.dim, Scalar(0.0)));
        for (int i = 0; i < L.dim; ++i) {
            for (int j = 0; j < L.dim; ++j) {
                for (int k = 0; k < L.dim; ++k) {
                    A[i][j] += grad_v_time[i][k] * Finv_time[k][j]
                             + Finv_time[k][i] * grad_v_time[j][k];
                }
                A[i][j] *= Scalar(rho * mu);
            }
        }

        std::vector<std::vector<Scalar>> stress(L.dim, std::vector<Scalar>(L.dim, Scalar(0.0)));
        std::vector<std::vector<Scalar>> pressure_stress(L.dim, std::vector<Scalar>(L.dim, Scalar(0.0)));
        for (int i = 0; i < L.dim; ++i) {
            for (int j = 0; j < L.dim; ++j) {
                for (int k = 0; k < L.dim; ++k) {
                    stress[i][j] += J_time * A[i][k] * Finv_time[j][k];
                }
                pressure_stress[i][j] = -J_time * p_time * Finv_time[j][i];
            }
        }

        // Momentum weak scalar.
        for (int i = 0; i < L.dim; ++i) {
            const Scalar mass_conv_mesh = Scalar(rho) * Scalar(0.5) * (J + Jn) * (v[i] - vn[i])
                                        + Scalar(dt) * conv[i]
                                        - meshmove[i];
            G += mass_conv_mesh * delta_v[i] * Scalar(dV);
            for (int j = 0; j < L.dim; ++j) {
                G += Scalar(dt) * (stress[i][j] + pressure_stress[i][j])
                   * grad_delta_v[i][j] * Scalar(dV);
            }
        }

        // Incompressibility in ALE form: (J_theta F_theta^{-T}) : grad(v) delta_p.
        Scalar div_ale(0.0);
        for (int i = 0; i < L.dim; ++i)
            for (int j = 0; j < L.dim; ++j) div_ale += J_time * Finv_time[j][i] * grad_v[i][j];
        G += div_ale * delta_p * Scalar(dV);

        // Flow-domain mesh motion.
        const std::string mmpde_type = params.material.mesh_motion_type;
        if (mmpde_type == "HE") {
            const double nu_m = params.material.mesh_motion_poisson_ratio;
            const double lambda_m = nu_m / ((1.0 + nu_m) * (1.0 - 2.0 * nu_m));
            const double mu_m = 1.0 / (2.0 * (1.0 + nu_m));

            std::vector<std::vector<Scalar>> Pm(L.dim, std::vector<Scalar>(L.dim, Scalar(0.0)));
            const Scalar coeff = Scalar(0.5 * lambda_m) * (J * J - Scalar(1.0));
            for (int i = 0; i < L.dim; ++i) {
                for (int j = 0; j < L.dim; ++j) {
                    const Scalar FinvT = Finv[j][i];
                    Pm[i][j] = Scalar(mu_m) * (F[i][j] - FinvT) + coeff * FinvT;
                }
            }
            for (int i = 0; i < L.dim; ++i)
                for (int j = 0; j < L.dim; ++j)
                    G += Scalar(mesh_coeff) * Pm[i][j] * grad_delta_u[i][j] * Scalar(dV);
        } else if (mmpde_type == "BH") {
            for (int i = 0; i < L.dim; ++i)
                for (int j = 0; j < L.dim; ++j)
                    G += Scalar(mesh_coeff) * H[i][j] * grad_delta_u[i][j] / J * Scalar(dV);
        } else {
            throw std::runtime_error("Unknown mesh_motion_type in fluid domain. Use BH or HE.");
        }
    }

    return G;
}

template <class Scalar>
// ============================================================================
// 3. Solid-domain weak scalar
// ============================================================================
// Uses displacement as the physical solid unknown.  The contribution is inertial
// plus hyperelastic internal virtual work.
static Scalar weak_scalar_solid_domain(
    const FieldMesh& mesh,
    int elem_id,
    const std::vector<Scalar>& x_elem,
    const std::vector<Scalar>& test_elem,
    const ProblemParameters& params,
    const ElementState& state
) {
    const double dt = state.dt;
    if (dt <= 0.0) throw std::runtime_error("Solid-domain state dt must be positive.");

    const DomainLocalLayout L = make_domain_layout(mesh, elem_id, state.domain_type);
    const auto& V = L.field("velocity");
    const auto& U = L.field("displacement");
    const auto& Pf = L.field("pressure");
    if (static_cast<int>(x_elem.size()) != L.n_local()) throw std::runtime_error("Bad current element vector size in solid domain.");
    if (state.past_dofs.size() < 1 || static_cast<int>(state.past_dofs[0].size()) != L.n_local()) throw std::runtime_error("Bad previous element vector size in solid domain.");
    if (test_elem.size() != x_elem.size()) throw std::runtime_error("Bad test element vector size in solid-domain weak form.");

    Scalar G(0.0);
    const ElementGeometry geom = make_velocity_element_geometry(mesh, elem_id);

    const double rho_s = params.material.structure_rho;
    const double E = params.material.youngs_modulus;
    const double nu = params.material.poisson_ratio;
    const hpx::TimeSchemeStencils time_scheme = hpx::Shift_Crank_Nicolson(params.solver.theta, dt);
    const double theta_current = time_scheme.value.current();
    const double theta_past = time_scheme.value.past(0);

    const double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const double mu = E / (2.0 * (1.0 + nu));

    for (const auto& gp : gauss_rule(V.element_type, params.solver.gauss_order)) {
        const ShapeData sv = compute_shape(V.element_type, gp.xi, gp.eta, gp.zeta);
        const ShapeData su = compute_shape(U.element_type, gp.xi, gp.eta, gp.zeta);
        const ShapeData sp = compute_shape(Pf.element_type, gp.xi, gp.eta, gp.zeta);
        const JacobianData Jgeo = compute_jacobian_data(geom, sv);
        if (Jgeo.detJ <= 0.0) throw std::runtime_error("Negative detJ in solid-domain element.");

        const auto dNv = compute_dN_dx(sv, Jgeo);
        const auto dNu = compute_dN_dx(su, Jgeo);
        const double dV = Jgeo.detJ * gp.weight;

        const auto v = hpx::field_value_as<Scalar>(L, "velocity", x_elem, sv.N);
        const auto vn = hpx::field_value_as<Scalar>(L, "velocity", state.past_dofs[0], sv.N);
        const auto u = hpx::field_value_as<Scalar>(L, "displacement", x_elem, su.N);
        const auto un = hpx::field_value_as<Scalar>(L, "displacement", state.past_dofs[0], su.N);

        const auto delta_v = hpx::field_value_as<Scalar>(L, "velocity", test_elem, sv.N);
        const auto grad_delta_v = hpx::field_grad_as<Scalar>(L, "velocity", test_elem, dNv);
        const auto delta_u = hpx::field_value_as<Scalar>(L, "displacement", test_elem, su.N);
        const Scalar delta_p = hpx::scalar_field_value_as<Scalar>(L, "pressure", test_elem, sp.N);

        const Scalar p = hpx::scalar_field_value_as<Scalar>(L, "pressure", x_elem, sp.N);

        std::vector<Scalar> v_time(L.dim, Scalar(0.0));
        for (int i = 0; i < L.dim; ++i) {
            v_time[i] = Scalar(theta_current) * v[i] + Scalar(theta_past) * vn[i];
        }

        const auto F = hpx::deformation_gradient_as<Scalar>(L, x_elem, dNu);
        const auto Fn = hpx::deformation_gradient_as<Scalar>(L, state.past_dofs[0], dNu);
        const auto P = hpx::neo_hookean_first_piola_lame(hpx::embed_3d(F, L.dim), lambda, mu);
        const auto Pn = hpx::neo_hookean_first_piola_lame(hpx::embed_3d(Fn, L.dim), lambda, mu);

        // Momentum balance against delta_v.
        for (int i = 0; i < L.dim; ++i) {
            G += Scalar(rho_s) * (v[i] - vn[i]) * delta_v[i] * Scalar(dV);
            for (int j = 0; j < L.dim; ++j) {
                const Scalar Ptime = Scalar(theta_current) * P[i][j] + Scalar(theta_past) * Pn[i][j];
                G += Scalar(dt) * Ptime * grad_delta_v[i][j] * Scalar(dV);
            }
        }

        // Kinematic equation against delta_u.
        for (int i = 0; i < L.dim; ++i) {
            G += Scalar(rho_s) * (u[i] - un[i] - Scalar(dt) * v_time[i]) * delta_u[i] * Scalar(dV);
        }

        // Pressure mass equation, matching the solid-domain Python reference: p * delta_p.
        G += p * delta_p * Scalar(dV);
    }

    return G;
}

template <class Scalar>
// ============================================================================
// 4. Domain weak-scalar dispatcher
// ============================================================================
// Prefer the mesh domain name for readability in FSI input files.  Keep
// DomainType as the fallback so single-domain and older TYPE-based inputs keep
// the same behavior.
static Scalar weak_scalar_domain(
    const FieldMesh& mesh,
    int elem_id,
    const std::vector<Scalar>& x_elem,
    const std::vector<Scalar>& test_elem,
    const ProblemParameters& params,
    const ElementState& state
) {
    if (state.domain_name == "fluid") {
        return weak_scalar_fluid_domain(mesh, elem_id, x_elem, test_elem, params, state);
    }
    if (state.domain_name == "solid") {
        return weak_scalar_solid_domain(mesh, elem_id, x_elem, test_elem, params, state);
    }

    switch (state.domain_type) {
        case DomainType::Fluid:
            return weak_scalar_fluid_domain(mesh, elem_id, x_elem, test_elem, params, state);
        case DomainType::Solid:
            return weak_scalar_solid_domain(mesh, elem_id, x_elem, test_elem, params, state);
        default:
            throw std::runtime_error("Unsupported domain type in weak form.");
    }
}

// ============================================================================
// 5. AD residual/tangent extraction
// ============================================================================
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
    // FSI mixed Q9/Q9/Q4 elements have about 40 local DOFs.  Nested SLFad<64>
    // is slower than DFad here because every scalar operation carries a large
    // fixed derivative array through the full velocity/displacement/pressure
    // weak form.
    return hpx::scalar_weak_form_element_system(
    // return hpx::scalar_weak_form_element_system(    
        x_elem,
        [&](const auto& x_ad, const auto& test_ad) {
            return weak_scalar_domain(mesh, elem_id, x_ad, test_ad, params, state);
        }
    );
}

static bool fsi_postprocess_requested(const std::vector<std::string>& requested,const std::string& name){
    return std::find(requested.begin(),requested.end(),name)!=requested.end();
}

static std::vector<double> fsi_sigma_at(
    const DomainLocalLayout& L,
    const FieldMesh& mesh,
    int elem_id,
    const Vector& x_elem,
    const ProblemParameters& params,
    const ElementState& state,
    double xi,
    double eta,
    double zeta
){
    const auto& V = L.field("velocity");
    const auto& U = L.field("displacement");
    const auto& Pf = L.field("pressure");
    const ElementGeometry geom = make_velocity_element_geometry(mesh, elem_id);
    const ShapeData sv = compute_shape(V.element_type,xi,eta,zeta);
    const ShapeData su = compute_shape(U.element_type,xi,eta,zeta);
    const ShapeData sp = compute_shape(Pf.element_type,xi,eta,zeta);
    const JacobianData Jgeo = compute_jacobian_data(geom,sv);
    if(Jgeo.detJ<=0.0) throw std::runtime_error("Negative detJ while exporting FSI postprocess data.");
    const auto dNv = compute_dN_dx(sv,Jgeo);
    const auto dNu = compute_dN_dx(su,Jgeo);

    std::vector<double> sigma(9,0.0);
    const bool solid = (state.domain_name == "solid") ||
                       (state.domain_name != "fluid" && state.domain_type == DomainType::Solid);
    if(solid){
        const double E = params.material.youngs_modulus;
        const double nu = params.material.poisson_ratio;
        const double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        const double mu = E / (2.0 * (1.0 + nu));
        const auto F = hpx::deformation_gradient_as<double>(L,x_elem,dNu);
        const auto F3 = hpx::embed_3d(F,L.dim);
        const auto P = hpx::neo_hookean_first_piola_lame(F3,lambda,mu);
        const double J = hpx::det(F3);
        if(J<=0.0) throw std::runtime_error("Non-positive J while exporting FSI solid sigma.");
        for(int i=0;i<3;++i){
            for(int j=0;j<3;++j){
                double sij=0.0;
                for(int A=0;A<3;++A) sij += P[i][A]*F3[j][A]/J;
                sigma[3*i+j]=sij;
            }
        }
        return sigma;
    }

    const double mu = params.material.mu;
    const auto grad_v = hpx::field_grad_as<double>(L,"velocity",x_elem,dNv);
    const double p = hpx::scalar_field_value_as<double>(L,"pressure",x_elem,sp.N);
    for(int i=0;i<3;++i){
        for(int j=0;j<3;++j){
            if(i<L.dim && j<L.dim){
                sigma[3*i+j] = (i==j ? -p : 0.0) + mu*(grad_v[i][j] + grad_v[j][i]);
            }
        }
    }
    return sigma;
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
    const bool need_sigma = fsi_postprocess_requested(requested,"sigma");
    const bool need_minj = fsi_postprocess_requested(requested,"minj");
    const bool need_traction = fsi_postprocess_requested(requested,"traction");
    if(!need_sigma && !need_minj && !need_traction) return;

    const DomainLocalLayout L = make_domain_layout(mesh,elem_id,state.domain_type);

    if(state.postprocess_location == ConstitutivePostprocessLocation::Boundary && need_traction){
        const auto sigma = fsi_sigma_at(
            L,mesh,elem_id,x_elem,params,state,
            state.postprocess_reference_point[0],
            state.postprocess_reference_point[1],
            state.postprocess_reference_point[2]
        );
        ConstitutivePostprocessOutputValues traction{
            "traction",3,ConstitutivePostprocessLocation::Boundary,{}
        };
        for(int i=0;i<3;++i){
            double ti=0.0;
            for(int j=0;j<3;++j) ti += sigma[3*i+j] * state.postprocess_normal_measure[j];
            traction.values.push_back(ti);
        }
        outputs.push_back(std::move(traction));
        return;
    }

    if(state.postprocess_location != ConstitutivePostprocessLocation::GaussPoint) return;

    const auto& U = L.field("displacement");
    const ElementGeometry geom = make_velocity_element_geometry(mesh,elem_id);
    const auto gps = gauss_rule(U.element_type,params.solver.gauss_order);

    ConstitutivePostprocessOutputValues sigma_out{"sigma",9,{}};
    ConstitutivePostprocessOutputValues minj_out{"minJ",1,{}};
    for(const auto& gp:gps){
        if(need_sigma){
            const auto sigma = fsi_sigma_at(L,mesh,elem_id,x_elem,params,state,gp.xi,gp.eta,gp.zeta);
            sigma_out.values.insert(sigma_out.values.end(),sigma.begin(),sigma.end());
        }
        if(!need_minj) continue;
        const ShapeData su = compute_shape(U.element_type,gp.xi,gp.eta,gp.zeta);
        const JacobianData Jgeo = compute_jacobian_data(geom,su);
        if(Jgeo.detJ<=0.0) throw std::runtime_error("Negative detJ while exporting FSI minJ.");
        const auto dNu = compute_dN_dx(su,Jgeo);
        const auto F = hpx::deformation_gradient_as<double>(L,x_elem,dNu);
        const auto F3 = hpx::embed_3d(F,L.dim);
        minj_out.values.push_back(hpx::det(F3));
    }
    if(need_sigma) outputs.push_back(std::move(sigma_out));
    if(need_minj) outputs.push_back(std::move(minj_out));
}
