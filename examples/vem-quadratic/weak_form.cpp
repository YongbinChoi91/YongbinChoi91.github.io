/*
================================================================================
MODEL MAP: weak_form_vem_quadratic.cpp
================================================================================
Purpose
  Displacement-only second-order polygonal VEM weak form.

VEM-specific geometry, moments, projection matrices, and element caches live in
core/vem.*.  This file keeps only the material weak form and AD extraction.
================================================================================
*/

#include "weak_form.h"
#include "kinematics.h"
#include "vem.h"
#include "local_element_system_ad.h"

#include <Sacado_Fad_DFad.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

const char* active_constitutive_model_name() {
    return "Second Order VEM";
}

std::vector<std::string> active_constitutive_field_names(const FieldMesh& mesh) {
    (void)mesh;
    return {"displacement"};
}

void print_active_constitutive_parameters(std::ostream& os, const ProblemParameters& params) {
    os << "  VEM type      : second-order polygonal VEM, DOFI stabilization\n"
       << "  VEM library   : core/vem.* geometry, moments, projection cache\n"
       << "  internal map  : centroid fan, T6 geometry/quadrature from space.cpp\n"
       << "  stabilization : Neo-Hookean C-tangent gamma at projected center\n"
       << "  E             : " << params.material.youngs_modulus << "\n"
       << "  nu            : " << params.material.poisson_ratio << "\n"
       << "  DOFI factor   : " << params.material.stabilization_parameter << "\n";
}

std::vector<ConstitutivePostprocessOutputSpec> active_constitutive_postprocess_outputs() {
    return {{"f", 9}, {"c", 9}, {"j", 1}, {"von_mises", 1}};
}

namespace {

static bool vem_output_requested(const std::vector<std::string>& requested,const std::string& name){
    for(std::string item:requested){
        std::replace(item.begin(),item.end(),'-','_');
        std::transform(item.begin(),item.end(),item.begin(),[](unsigned char ch){ return static_cast<char>(std::tolower(ch)); });
        if(item == name) return true;
    }
    return false;
}

static void append_matrix3(
    const std::vector<std::vector<double>>& A,
    std::vector<double>& values
){
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) values.push_back(A[i][j]);
}

static void append_vem_postprocess_values(
    const vem::Mat2T<double>& grad_u,
    const ProblemParameters& params,
    bool need_f,
    bool need_c,
    bool need_j,
    bool need_vm,
    ConstitutivePostprocessOutputValues& fout,
    ConstitutivePostprocessOutputValues& cout,
    ConstitutivePostprocessOutputValues& jout,
    ConstitutivePostprocessOutputValues& vmout
){
    const auto F = hpx::deformation_gradient_2d(grad_u);
    if(need_f) append_matrix3(F,fout.values);
    if(need_c) append_matrix3(hpx::right_cauchy_green(F),cout.values);
    if(need_j) jout.values.push_back(hpx::det(F));
    if(need_vm) vmout.values.push_back(
        hpx::neo_hookean_von_mises(F,params.material.youngs_modulus,params.material.poisson_ratio)
    );
}

template <class Scalar>
static Scalar neo_hookean_plane_strain_density(
    const vem::Mat2T<Scalar>& grad_u,
    const vem::Mat2T<Scalar>& grad_du,
    const ProblemParameters& params
) {
    const double E = params.material.youngs_modulus;
    const double nu = params.material.poisson_ratio;
    const double mu = E / (2.0 * (1.0 + nu));
    const double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

    const Scalar a = Scalar(1.0) + grad_u[0][0];
    const Scalar c = grad_u[0][1];
    const Scalar b = grad_u[1][0];
    const Scalar d = Scalar(1.0) + grad_u[1][1];
    const Scalar J = a * d - b * c;
    const Scalar coeff = Scalar(0.5 * lambda) * (J * J - Scalar(1.0));

    const Scalar invT00 = d / J;
    const Scalar invT01 = -b / J;
    const Scalar invT10 = -c / J;
    const Scalar invT11 = a / J;

    const Scalar P00 = Scalar(mu) * (a - invT00) + coeff * invT00;
    const Scalar P01 = Scalar(mu) * (c - invT01) + coeff * invT01;
    const Scalar P10 = Scalar(mu) * (b - invT10) + coeff * invT10;
    const Scalar P11 = Scalar(mu) * (d - invT11) + coeff * invT11;

    return P00 * grad_du[0][0]
         + P01 * grad_du[0][1]
         + P10 * grad_du[1][0]
         + P11 * grad_du[1][1];
}

template <class Scalar>
static Scalar dofi_gamma_from_neo_hookean_c_tangent(
    const vem::Mat2T<Scalar>& grad_u,
    const ProblemParameters& params
) {
    // Python parity: Solid_models.VEM_Models.Dofi_stabilization_virt builds
    // Gamma from jax.jacfwd(jax.jacrev(neo_hooke_C))(C) and then contracts
    // the in-plane 2x2x2x2 block as
    //   0.25 * (C_abcd delta_ac delta_bd + C_abcd delta_ad delta_bc).
    // Here C = F^T F is the plane-strain right Cauchy-Green tensor.
    const double nu = params.material.poisson_ratio;
    const double E  = params.material.youngs_modulus;
    const double lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const double mu  = E / (2.0 * (1.0 + nu));

    std::vector<std::vector<Scalar>> F(3, std::vector<Scalar>(3, Scalar(0.0)));
    F[0][0] = Scalar(1.0) + grad_u[0][0];
    F[0][1] = grad_u[0][1];
    F[1][0] = grad_u[1][0];
    F[1][1] = Scalar(1.0) + grad_u[1][1];
    F[2][2] = Scalar(1.0);

    std::vector<std::vector<Scalar>> C(3, std::vector<Scalar>(3, Scalar(0.0)));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) C[i][j] += F[k][i] * F[k][j];
        }
    }

    const Scalar detC = hpx::det(C);
    const auto Cinv = hpx::inv(C);

    // For psi(C) = mu/2*(tr(C)-3-log(detC))
    //             + lam/4*(detC-1-log(detC)),
    // d^2 psi / dC_ij dC_kl is
    //   lam/4*detC*Cinv_lk*Cinv_ji - alpha*Cinv_jk*Cinv_li,
    // alpha = lam/4*(detC-1) - mu/2.
    const Scalar alpha = Scalar(0.25 * lam) * (detC - Scalar(1.0)) - Scalar(0.5 * mu);

    Scalar contraction_1(0.0); // sum_ab H_abab
    Scalar contraction_2(0.0); // sum_ab H_abba
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            const Scalar H_abab = Scalar(0.25 * lam) * detC * Cinv[b][a] * Cinv[b][a]
                                - alpha * Cinv[b][a] * Cinv[b][a];
            const Scalar H_abba = Scalar(0.25 * lam) * detC * Cinv[a][b] * Cinv[b][a]
                                - alpha * Cinv[b][b] * Cinv[a][a];
            contraction_1 += H_abab;
            contraction_2 += H_abba;
        }
    }

    return Scalar(params.material.stabilization_parameter) * Scalar(0.25)
         * (contraction_1 + contraction_2);
}

template <class Scalar>
static Scalar dofi_stabilization(
    const vem::QuadraticElementCache& cache,
    const std::vector<Scalar>& x_elem,
    const std::vector<Scalar>& test_elem,
    const vem::Coeff6x2T<Scalar>& u_pi,
    const vem::Coeff6x2T<Scalar>& du_pi,
    const ProblemParameters& params
) {
    const auto& U = cache.layout.field("displacement");
    const auto grad_center = vem::quadratic_projection_gradient(u_pi, {{0.0,0.0}});
    const Scalar Gamma = dofi_gamma_from_neo_hookean_c_tangent(grad_center, params);

    Scalar S(0.0);
    for (int s = 0; s < cache.geom.n_slots; ++s) {
        const auto up  = vem::quadratic_projection_value_basis(u_pi,  cache.slot_basis[s]);
        const auto dup = vem::quadratic_projection_value_basis(du_pi, cache.slot_basis[s]);
        Scalar dot(0.0);
        for (int c = 0; c < 2; ++c) {
            const Scalar delta_u  = x_elem[hpx::local_dof(U, s, c)] - up[c];
            const Scalar delta_du = test_elem[hpx::local_dof(U, s, c)] - dup[c];
            dot += delta_u * delta_du;
        }
        S += Scalar(0.5) * Gamma * dot;
    }
    return S;
}

template <class Scalar>
static Scalar weak_scalar_vem(
    const FieldMesh& mesh,
    int elem_id,
    const std::vector<Scalar>& x_elem,
    const std::vector<Scalar>& test_elem,
    const ProblemParameters& params,
    const ElementState& state
) {
    (void)state;

    const int q_order = std::max(3, params.solver.gauss_order);
    const vem::QuadraticElementCache& cache = vem::get_cached_quadratic_element(mesh, elem_id, q_order);
    if (mesh.dim != 2 || cache.layout.dim != 2) {
        throw std::runtime_error("Quadratic VEM currently supports 2D displacement VEM only.");
    }
    if (static_cast<int>(x_elem.size()) != cache.layout.n_local()) {
        throw std::runtime_error("Bad current element vector size in quadratic VEM.");
    }
    if (test_elem.size() != x_elem.size()) {
        throw std::runtime_error("Bad test element vector size in quadratic VEM.");
    }

    const auto u_pi  = vem::quadratic_projection_coefficients(cache, x_elem);
    const auto du_pi = vem::quadratic_projection_coefficients(cache, test_elem);

    Scalar G(0.0);
    for (const auto& qp : cache.internal_qps) {
        const auto grad_u  = vem::quadratic_projection_gradient_basis(u_pi, qp.grad_basis);
        const auto grad_du = vem::quadratic_projection_gradient_basis(du_pi, qp.grad_basis);
        G += Scalar(qp.dV) * neo_hookean_plane_strain_density(grad_u, grad_du, params);
    }

    G += dofi_stabilization(cache, x_elem, test_elem, u_pi, du_pi, params);
    return G;
}

} // namespace

ElementSystem compute_constitutive_element_system(
    const FieldMesh& mesh,
    int elem_id,
    const Vector& x_elem,
    const ProblemParameters& params,
    const ElementState& state
) {
    return hpx::scalar_weak_form_element_system(
    // return hpx::scalar_weak_form_element_system_dfad(
        x_elem,
        [&](const auto& x_ad, const auto& test_ad) {
            return weak_scalar_vem(mesh, elem_id, x_ad, test_ad, params, state);
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
    const bool need_f = vem_output_requested(requested,"f");
    const bool need_c = vem_output_requested(requested,"c");
    const bool need_j = vem_output_requested(requested,"j");
    const bool need_vm = vem_output_requested(requested,"von_mises");
    if(!need_f && !need_c && !need_j && !need_vm) return;

    const int q_order = std::max(3, params.solver.gauss_order);
    const vem::QuadraticElementCache& cache = vem::get_cached_quadratic_element(mesh, elem_id, q_order);
    const auto u_pi = vem::quadratic_projection_coefficients(cache,x_elem);

    ConstitutivePostprocessOutputValues fout{"F",9,{}};
    ConstitutivePostprocessOutputValues cout{"C",9,{}};
    ConstitutivePostprocessOutputValues jout{"J",1,{}};
    ConstitutivePostprocessOutputValues vmout{"von_mises",1,{}};
    for(const auto& qp:cache.internal_qps){
        const auto grad_u = vem::quadratic_projection_gradient_basis(u_pi,qp.grad_basis);
        append_vem_postprocess_values(grad_u,params,need_f,need_c,need_j,need_vm,fout,cout,jout,vmout);
    }
    if(need_f) outputs.push_back(std::move(fout));
    if(need_c) outputs.push_back(std::move(cout));
    if(need_j) outputs.push_back(std::move(jout));
    if(need_vm) outputs.push_back(std::move(vmout));
}

namespace {
template <class Scalar>
static void add_weighted_quadratic_vem_piola_response(
    const vem::Mat2T<Scalar>& grad_u,
    double weight,
    const ProblemParameters& params,
    std::array<std::array<Scalar,3>,3>& intP
) {
    if (weight == 0.0) return;
    const auto F = hpx::deformation_gradient_2d(grad_u);
    const auto P = hpx::neo_hookean_first_piola_young(
        F,
        params.material.youngs_modulus,
        params.material.poisson_ratio
    );
    for (int i = 0; i < 3; ++i) {
        for (int A = 0; A < 3; ++A) {
            intP[i][A] += P[i][A] * Scalar(weight);
        }
    }
}

template <class Scalar>
static void accumulate_quadratic_vem_multiscale_response(
    const FieldMesh& mesh,
    const std::vector<Scalar>& displacement,
    const ProblemParameters& params,
    std::array<std::array<Scalar,3>,3>& intP,
    double& volume
) {
    if (mesh.dim != 2) {
        throw std::runtime_error("Quadratic VEM multiscale response currently supports 2D displacement VEM only.");
    }
    const DofLayout& layout = get_cached_dof_layout(mesh);
    const int q_order = std::max(3, params.solver.gauss_order);

    for (int e = 0; e < static_cast<int>(mesh.displacement_elements.size()); ++e) {
        const vem::QuadraticElementCache& cache = vem::get_cached_quadratic_element(mesh, e, q_order);
        const auto dofs = make_element_dofs_for_fields(layout, e, {"displacement"});
        std::vector<Scalar> ue(dofs.size());
        for (int a = 0; a < static_cast<int>(dofs.size()); ++a) ue[a] = displacement[dofs[a]];

        const auto u_pi = vem::quadratic_projection_coefficients(cache, ue);
        double element_volume = 0.0;
        for (const auto& qp : cache.internal_qps) {
            const auto grad_u = vem::quadratic_projection_gradient_basis(u_pi, qp.grad_basis);
            add_weighted_quadratic_vem_piola_response(grad_u, qp.dV, params, intP);
            element_volume += qp.dV;
        }
        volume += element_volume;
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
        throw std::runtime_error("Quadratic VEM multiscale response received a displacement vector shorter than mesh DOFs.");
    }

    std::vector<Fad> u_ad(ndofs);
    for (int a = 0; a < ndofs; ++a) u_ad[a] = Fad(ndofs, a, displacement[a]);

    std::array<std::array<Fad,3>,3> intP_ad{};
    for (auto& row : intP_ad) {
        for (auto& v : row) v = Fad(ndofs, 0.0);
    }
    double volume = 0.0;
    accumulate_quadratic_vem_multiscale_response(mesh, u_ad, params, intP_ad, volume);
    if (volume <= 0.0) {
        throw std::runtime_error("Quadratic VEM multiscale response volume is non-positive.");
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
