/*
================================================================================
MODEL MAP: weak_form_classical.cpp
================================================================================
Purpose
  Classical incompressible fluid element model.
  Active unknowns: velocity + pressure. If the global DOF layout also contains
  displacement, this model keeps that block in the local vector for assembler
  compatibility, but it does not add displacement residual terms.

How this file is called
  solver.cpp
    -> assembly.cpp assembles each element
      -> compute_constitutive_element_system(...)
        -> build AD variables x_ad and virtual/test variables test_ad
        -> weak_scalar_classical_fluid(...)
        -> extract residual and tangent from Sacado derivatives

Weak-form idea
  1. Define one scalar virtual-work functional G(v, p; delta_v, delta_p).
  2. Let Sacado differentiate G with respect to test DOFs to obtain residual R.
  3. Let Sacado differentiate R with respect to solution DOFs to obtain tangent K.

Element-level flow
  active_constitutive_model_name()
    -> only identifies the model in console/log output.
  active_constitutive_field_names(mesh)
    -> tells assembly which fields must be included in the local element vector.
  make_classical_layout(mesh, elem_id)
    -> creates name-based local field access: L.dof("velocity", a, i), etc.
  weak_scalar_classical_fluid(...)
    -> Gauss loop, interpolation, BDF acceleration, convection, stress,
       incompressibility, optional SUPG/PSPG stabilization.
  compute_constitutive_element_system(...)
    -> wraps the scalar weak form in nested AD and fills ElementSystem.

Important convention
  Residual sign is internal-minus-external. Natural boundary loads are added in
  assembly.cpp as external terms, so this file only defines the volume terms.

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
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// Model registration
// -----------------------------------------------------------------------------
// The executable links exactly one weak_form_*.cpp at a time.  The following
// function gives the linked model a human-readable name for solver diagnostics.
const char* active_constitutive_model_name() {
    return "Navier_Stokes_Eq";
}


// -----------------------------------------------------------------------------
// Active field declaration
// -----------------------------------------------------------------------------
// Assembly asks this function which fields should appear in the element-local
// vector.  The returned order controls the local layout used below.
std::vector<std::string> active_constitutive_field_names(const FieldMesh& mesh) {
    if (!mesh.dof_field_order.empty()) {
        return mesh.dof_field_order;
    }
    return active_dof_field_names(mesh);
}

void print_active_constitutive_parameters(std::ostream& os, const ProblemParameters& params) {
    os << "  rho   : " << params.material.rho << "\n"
       << "  mu    : " << params.material.mu << "\n"
       << "  u_bar : " << params.material.u_bar << "\n";
}

std::vector<ConstitutivePostprocessOutputSpec> active_constitutive_postprocess_outputs() {
    return {
        {"sigma", 9},
        {"traction", 3, ConstitutivePostprocessLocation::Boundary}
    };
}

using ClassicalLocalLayout = LocalFieldView;

// Name-based local layout builder.  After this point all reads/writes should use
// L.field(...), L.dof(...), or kinematic_* helpers instead of manual offsets.
static ClassicalLocalLayout make_classical_layout(
    const FieldMesh& mesh,
    int elem_id
) {
    // Keep the residual as one large function, but access fields by name.
    // Include displacement when present so x_elem/R/K stay in the assembler's
    // local order.  Classical fluid simply does not write a displacement block.
    return make_local_field_view(mesh, elem_id, active_constitutive_field_names(mesh));
}

static double reference_measure_for_stabilization(ElementType type) {
    switch(type){
        case ElementType::T3:
        case ElementType::T6:
            return 0.5;
        case ElementType::Q4:
        case ElementType::Q9:
            return 4.0;
        case ElementType::Tet4:
        case ElementType::Tet10:
            return 1.0 / 6.0;
        case ElementType::H8:
        case ElementType::H20:
        case ElementType::H27:
            return 8.0;
        default:
            throw std::runtime_error("Unsupported element type in fluid stabilization length scale.");
    }
}

static double stabilization_element_length(
    ElementType type,
    int dim,
    const JacobianData& J
) {
    const double measure = std::abs(J.detJ) * reference_measure_for_stabilization(type);
    return std::pow(std::max(measure, 1.0e-24), 1.0 / static_cast<double>(dim));
}


// -----------------------------------------------------------------------------
// Direct scalar weak form + AD extraction
// -----------------------------------------------------------------------------
// The scalar weak form is assembled directly from physical test fields
// (delta_v, delta_p), not from a pre-built residual vector.
// Residual: R_i = dG/d(delta_x_i)
// Tangent : K_ij = dR_i/dx_j = d^2G/(d(delta_x_i) dx_j)
template <class Scalar>
// -----------------------------------------------------------------------------
// Scalar weak form for the classical fluid
// -----------------------------------------------------------------------------
// This function returns only the scalar G.  It deliberately does not fill R/K
// directly; AD extraction at the bottom of the file does that consistently.
static Scalar weak_scalar_classical_fluid(
    const FieldMesh& mesh,
    int elem_id,
    const std::vector<Scalar>& x_elem,
    const std::vector<Scalar>& test_elem,
    const ProblemParameters& params,
    const ElementState& state
) {
    using std::sqrt;

    const double dt = state.dt;
    if (dt <= 0.0) {
        throw std::runtime_error("Constitutive state dt must be positive.");
    }

    const ClassicalLocalLayout L = make_classical_layout(mesh, elem_id);
    const auto& V = L.field("velocity");
    const auto& P = L.field("pressure");

    if (static_cast<int>(x_elem.size()) != L.n_local()) {
        throw std::runtime_error("Bad current element vector size in classical fluid constitutive model.");
    }
    if (state.past_dofs.size() < 1 || static_cast<int>(state.past_dofs[0].size()) != L.n_local()) {
        throw std::runtime_error("Bad previous element vector size in classical fluid constitutive model.");
    }
    if (state.past_dofs.size() < 2 || static_cast<int>(state.past_dofs[1].size()) != L.n_local()) {
        throw std::runtime_error("Bad older element vector size in classical fluid constitutive model.");
    }
    if (test_elem.size() != x_elem.size()) {
        throw std::runtime_error("Bad test element vector size in classical fluid weak form.");
    }

    const hpx::TimeDofStencil bdf = hpx::BDF(2, dt, state.dt_previous, state.step);

    Scalar G(0.0);
    const ElementGeometry geom = make_velocity_element_geometry(mesh, elem_id);

    const double rho = params.material.rho;
    const double mu  = params.material.mu;

    const double nu = mu / rho;
    const double sigma_BDF = params.solver.stabilization_time_order;
    const double C_k = params.solver.stabilization_Ck;

    // Direct weak form:
    //   G_m = int [rho*a_dt · delta_v
    //            + dt*rho*((v.grad)v) · delta_v
    //            + dt*mu*(grad(v)+grad(v)^T) : grad(delta_v)
    //            - dt*p*div(delta_v)] dOmega
    //   G_c = int div(v) * delta_p dOmega
    // Stabilization is also written directly with grad(delta_v), div(delta_v),
    // and grad(delta_p).
    for (const auto& gp : gauss_rule(V.element_type, params.solver.gauss_order)) {
        const ShapeData sv = compute_shape(V.element_type, gp.xi, gp.eta, gp.zeta);
        const ShapeData sp = compute_shape(P.element_type, gp.xi, gp.eta, gp.zeta);
        const JacobianData J = compute_jacobian_data(geom, sv);
        if (J.detJ <= 0.0) {
            throw std::runtime_error("Negative detJ in classical fluid element. Check connectivity ordering.");
        }

        const auto dNv = compute_dN_dx(sv, J);
        const auto d2Nv = compute_d2N_dx2(geom, sv, J, gp.xi, gp.eta, gp.zeta);
        const auto dNp = compute_dN_dx(sp, J);
        const double dV = J.detJ * gp.weight;

        const auto v = hpx::field_value_as<Scalar>(L, "velocity", x_elem, sv.N);
        const auto v_n = hpx::field_value_as<Scalar>(L, "velocity", state.past_dofs[0], sv.N);
        const auto v_nm1 = hpx::field_value_as<Scalar>(L, "velocity", state.past_dofs[1], sv.N);
        const auto grad_v = hpx::field_grad_as<Scalar>(L, "velocity", x_elem, dNv);

        const auto delta_v = hpx::field_value_as<Scalar>(L, "velocity", test_elem, sv.N);
        const auto grad_delta_v = hpx::field_grad_as<Scalar>(L, "velocity", test_elem, dNv);
        const Scalar delta_p = hpx::scalar_field_value_as<Scalar>(L, "pressure", test_elem, sp.N);
        const auto grad_delta_p_all = hpx::field_grad_as<Scalar>(L, "pressure", test_elem, dNp);
        const auto& grad_delta_p = grad_delta_p_all[0];

        std::vector<Scalar> dt_times_accel(L.dim, Scalar(0.0));
        std::vector<Scalar> conv(L.dim, Scalar(0.0));
        for (int i = 0; i < L.dim; ++i) {
            dt_times_accel[i] =
                Scalar(bdf.current()) * v[i]
              + Scalar(bdf.past(0)) * v_n[i]
              + Scalar(bdf.past(1)) * v_nm1[i];

            for (int j = 0; j < L.dim; ++j) {
                conv[i] += v[j] * grad_v[i][j];
            }
        }

        const Scalar p = hpx::scalar_field_value_as<Scalar>(L, "pressure", x_elem, sp.N);
        const auto grad_p_all = hpx::field_grad_as<Scalar>(L, "pressure", x_elem, dNp);
        const auto& grad_p = grad_p_all[0];

        Scalar div_v(0.0);
        Scalar div_delta_v(0.0);
        for (int i = 0; i < L.dim; ++i) {
            div_v += grad_v[i][i];
            div_delta_v += grad_delta_v[i][i];
        }

        for (int i = 0; i < L.dim; ++i) {
            G += Scalar(rho) * dt_times_accel[i] * delta_v[i] * Scalar(dV);
            G += Scalar(dt) * Scalar(rho) * conv[i] * delta_v[i] * Scalar(dV);
        }
        for (int i = 0; i < L.dim; ++i) {
            for (int j = 0; j < L.dim; ++j) {
                G += Scalar(dt) * Scalar(mu)
                   * (grad_v[i][j] + grad_v[j][i])
                   * grad_delta_v[i][j]
                   * Scalar(dV);
            }
        }
        G -= Scalar(dt) * p * div_delta_v * Scalar(dV);
        G += div_v * delta_p * Scalar(dV);

        // Optional residual-based stabilization.
        // Tau_M and Tau_C are computed at the current Gauss point from time scale,
        // viscous scale, element size, and velocity magnitude.
        if (params.solver.stabilization) {
            Scalar v_norm2(0.0);
            for (int i = 0; i < L.dim; ++i) v_norm2 += v[i] * v[i];

            const double h_T = std::max(stabilization_element_length(V.element_type, L.dim, J), 1.0e-12);
            const double inv_h2 = 1.0 / (h_T * h_T);
            const double inv_h4 = inv_h2 * inv_h2;
            const double time_part = (sigma_BDF * sigma_BDF) / (dt * dt);
            const double visc_part = C_k * nu * nu * inv_h4;
            const Scalar denom = Scalar(time_part + visc_part) + v_norm2 * Scalar(inv_h2);
            const Scalar Tau_M = Scalar(1.0) / sqrt(denom);
            const Scalar Tau_C = Scalar(h_T * h_T) / Tau_M;
            const Scalar penalty = Scalar(0.5 * nu);

            std::vector<std::vector<std::vector<Scalar>>> hess_v(
                L.dim,
                std::vector<std::vector<Scalar>>(L.dim, std::vector<Scalar>(L.dim, Scalar(0.0)))
            );
            for (int a = 0; a < V.n_nodes; ++a) {
                for (int i = 0; i < L.dim; ++i) {
                    const int id = L.dof("velocity", a, i);
                    const Scalar va = x_elem[id];
                    for (int j = 0; j < L.dim; ++j) {
                        for (int k = 0; k < L.dim; ++k) {
                            hess_v[i][j][k] += va * Scalar(d2Nv[a][j][k]);
                        }
                    }
                }
            }

            std::vector<Scalar> div_sigma(L.dim, Scalar(0.0));
            for (int i = 0; i < L.dim; ++i) {
                for (int j = 0; j < L.dim; ++j) {
                    div_sigma[i] += Scalar(nu) * (hess_v[i][j][j] + hess_v[j][i][j]);
                }
            }

            std::vector<Scalar> strong_form_stress(L.dim, Scalar(0.0));
            std::vector<Scalar> strong_res(L.dim, Scalar(0.0));
            for (int i = 0; i < L.dim; ++i) {
                strong_form_stress[i] = grad_p[i] - div_sigma[i];
                strong_res[i] = dt_times_accel[i]
                              + Scalar(dt) * conv[i]
                              + Scalar(dt) * strong_form_stress[i];
            }

            std::vector<Scalar> grad_delta_v_times_v(L.dim, Scalar(0.0));
            for (int i = 0; i < L.dim; ++i) {
                for (int j = 0; j < L.dim; ++j) {
                    grad_delta_v_times_v[i] += grad_delta_v[i][j] * v[j];
                }
            }
            for (int i = 0; i < L.dim; ++i) {
                G += Tau_M * strong_res[i] * grad_delta_v_times_v[i] * Scalar(dV);
                G += Tau_M * strong_res[i] * grad_delta_p[i] * Scalar(dV);
            }
            G += penalty * Tau_C * div_v * div_delta_v * Scalar(dV);
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
            return weak_scalar_classical_fluid(mesh, elem_id, x_ad, test_ad, params, state);
        }
    );
}

static bool classical_postprocess_requested(const std::vector<std::string>& requested,const std::string& name){
    return std::find(requested.begin(),requested.end(),name)!=requested.end();
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
    const bool need_sigma = classical_postprocess_requested(requested,"sigma");
    const bool need_traction = classical_postprocess_requested(requested,"traction");
    if(!need_sigma && !need_traction) return;

    const ClassicalLocalLayout L = make_classical_layout(mesh, elem_id);
    const auto& V = L.field("velocity");
    const auto& P = L.field("pressure");
    if(static_cast<int>(x_elem.size()) != L.n_local()){
        throw std::runtime_error("Bad current element vector size in classical fluid postprocess.");
    }

    const ElementGeometry geom = make_velocity_element_geometry(mesh, elem_id);
    const auto gps = gauss_rule(V.element_type, params.solver.gauss_order);
    const double mu = params.material.mu;

    auto append_sigma_at = [&](double xi,double eta,double zeta,std::vector<double>& values){
        const ShapeData sv = compute_shape(V.element_type,xi,eta,zeta);
        const ShapeData sp = compute_shape(P.element_type,xi,eta,zeta);
        const JacobianData J = compute_jacobian_data(geom,sv);
        if(J.detJ<=0.0) throw std::runtime_error("Negative detJ while exporting classical fluid sigma.");
        const auto dNv = compute_dN_dx(sv,J);
        const auto grad_v = hpx::field_grad_as<double>(L,"velocity",x_elem,dNv);
        const double p = hpx::scalar_field_value_as<double>(L,"pressure",x_elem,sp.N);

        for(int i=0;i<3;++i){
            for(int j=0;j<3;++j){
                double sij = 0.0;
                if(i<L.dim && j<L.dim){
                    sij = (i==j ? -p : 0.0) + mu*(grad_v[i][j] + grad_v[j][i]);
                }
                values.push_back(sij);
            }
        }
    };

    if(state.postprocess_location == ConstitutivePostprocessLocation::Boundary && need_traction){
        std::vector<double> sigma_values;
        append_sigma_at(state.postprocess_reference_point[0],
                        state.postprocess_reference_point[1],
                        state.postprocess_reference_point[2],
                        sigma_values);
        ConstitutivePostprocessOutputValues traction{
            "traction",3,ConstitutivePostprocessLocation::Boundary,{}
        };
        for(int i=0;i<3;++i){
            double ti=0.0;
            for(int j=0;j<3;++j) ti += sigma_values[3*i+j] * state.postprocess_normal_measure[j];
            traction.values.push_back(ti);
        }
        outputs.push_back(std::move(traction));
        return;
    }

    if(!need_sigma) return;
    ConstitutivePostprocessOutputValues sigma{"sigma",9,{}};
    for(const auto& gp:gps){
        append_sigma_at(gp.xi,gp.eta,gp.zeta,sigma.values);
    }
    outputs.push_back(std::move(sigma));
}
