/*
================================================================================
MODEL MAP: weak_form_vem.cpp
================================================================================
Purpose
  Displacement-only first-order polygonal VEM weak form.

VEM-specific polygon geometry, projection data, fan triangulation, and element
caches live in core/vem.*.  This file keeps only the material weak form and AD
extraction.
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
    return "Linear VEM";
}

std::vector<std::string> active_constitutive_field_names(const FieldMesh& mesh) {
    (void)mesh;
    return {"displacement"};
}

void print_active_constitutive_parameters(std::ostream& os, const ProblemParameters& params) {
    os << "  VEM type      : first-order polygonal VEM\n"
       << "  VEM library   : core/vem.* geometry and projection cache\n"
       << "  E             : " << params.material.youngs_modulus << "\n"
       << "  nu            : " << params.material.poisson_ratio << "\n"
       << "  alpha         : 0.1\n";
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
static Scalar weak_scalar_vem(
    const FieldMesh& mesh,
    int elem_id,
    const std::vector<Scalar>& x_elem,
    const std::vector<Scalar>& test_elem,
    const ProblemParameters& params,
    const ElementState& state
) {
    (void)state;

    const int q_order = (params.solver.gauss_order > 0) ? params.solver.gauss_order : 1;
    const vem::LinearElementCache& cache = vem::get_cached_linear_element(mesh, elem_id, q_order);
    if (mesh.dim != 2 || cache.layout.dim != 2) {
        throw std::runtime_error("Linear VEM currently supports 2D displacement VEM only.");
    }
    if (static_cast<int>(x_elem.size()) != cache.layout.n_local()) {
        throw std::runtime_error("Bad current element vector size in linear VEM.");
    }
    if (test_elem.size() != x_elem.size()) {
        throw std::runtime_error("Bad test element vector size in linear VEM.");
    }

    const double alpha = 0.1;
    const auto grad_u_pi  = vem::linear_projected_gradient(cache, x_elem);
    const auto grad_du_pi = vem::linear_projected_gradient(cache, test_elem);
    const Scalar projected_density = neo_hookean_plane_strain_density(grad_u_pi, grad_du_pi, params);

    Scalar triangle_part(0.0);
    double fan_measure_sum = 0.0;
    for (const auto& tri : cache.triangles) {
        fan_measure_sum += tri.measure;
        const auto grad_u_T  = vem::linear_triangle_gradient(cache, tri, x_elem);
        const auto grad_du_T = vem::linear_triangle_gradient(cache, tri, test_elem);
        triangle_part += Scalar(tri.measure) * neo_hookean_plane_strain_density(grad_u_T, grad_du_T, params);
    }

    return Scalar(alpha) * triangle_part
         + Scalar(1.0 - alpha) * Scalar(fan_measure_sum) * projected_density;
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

    const int q_order = (params.solver.gauss_order > 0) ? params.solver.gauss_order : 1;
    const vem::LinearElementCache& cache = vem::get_cached_linear_element(mesh, elem_id, q_order);
    const auto tri_gps = gauss_rule(ElementType::T3,q_order);

    ConstitutivePostprocessOutputValues fout{"F",9,{}};
    ConstitutivePostprocessOutputValues cout{"C",9,{}};
    ConstitutivePostprocessOutputValues jout{"J",1,{}};
    ConstitutivePostprocessOutputValues vmout{"von_mises",1,{}};
    for(const auto& tri:cache.triangles){
        const auto grad_u = vem::linear_triangle_gradient(cache,tri,x_elem);
        for(size_t q=0; q<tri_gps.size(); ++q){
            append_vem_postprocess_values(grad_u,params,need_f,need_c,need_j,need_vm,fout,cout,jout,vmout);
        }
    }
    if(need_f) outputs.push_back(std::move(fout));
    if(need_c) outputs.push_back(std::move(cout));
    if(need_j) outputs.push_back(std::move(jout));
    if(need_vm) outputs.push_back(std::move(vmout));
}

namespace {
template <class Scalar>
static void add_weighted_vem_piola_response(
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
static void accumulate_linear_vem_multiscale_response(
    const FieldMesh& mesh,
    const std::vector<Scalar>& displacement,
    const ProblemParameters& params,
    std::array<std::array<Scalar,3>,3>& intP,
    double& volume
) {
    if (mesh.dim != 2) {
        throw std::runtime_error("Linear VEM multiscale response currently supports 2D displacement VEM only.");
    }
    const DofLayout& layout = get_cached_dof_layout(mesh);
    const double alpha = 0.1;
    const int q_order = (params.solver.gauss_order > 0) ? params.solver.gauss_order : 1;

    for (int e = 0; e < static_cast<int>(mesh.displacement_elements.size()); ++e) {
        const vem::LinearElementCache& cache = vem::get_cached_linear_element(mesh, e, q_order);
        const auto dofs = make_element_dofs_for_fields(layout, e, {"displacement"});
        std::vector<Scalar> ue(dofs.size());
        for (int a = 0; a < static_cast<int>(dofs.size()); ++a) ue[a] = displacement[dofs[a]];

        double fan_measure_sum = 0.0;
        for (const auto& tri : cache.triangles) {
            fan_measure_sum += tri.measure;
            const auto grad_u_T = vem::linear_triangle_gradient(cache, tri, ue);
            add_weighted_vem_piola_response(grad_u_T, alpha * tri.measure, params, intP);
        }

        const double projected_measure = (fan_measure_sum > 0.0) ? fan_measure_sum : cache.area;
        const auto grad_u_pi = vem::linear_projected_gradient(cache, ue);
        add_weighted_vem_piola_response(grad_u_pi, (1.0 - alpha) * projected_measure, params, intP);
        volume += projected_measure;
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
        throw std::runtime_error("Linear VEM multiscale response received a displacement vector shorter than mesh DOFs.");
    }

    std::vector<Fad> u_ad(ndofs);
    for (int a = 0; a < ndofs; ++a) u_ad[a] = Fad(ndofs, a, displacement[a]);

    std::array<std::array<Fad,3>,3> intP_ad{};
    for (auto& row : intP_ad) {
        for (auto& v : row) v = Fad(ndofs, 0.0);
    }
    double volume = 0.0;
    accumulate_linear_vem_multiscale_response(mesh, u_ad, params, intP_ad, volume);
    if (volume <= 0.0) {
        throw std::runtime_error("Linear VEM multiscale response volume is non-positive.");
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
