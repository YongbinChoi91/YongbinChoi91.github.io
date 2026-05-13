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
#include "vem.h"

#include <Sacado_Fad_DFad.hpp>

#include <algorithm>
#include <ostream>
#include <string>
#include <vector>

const char* active_constitutive_model_name() {
    return "weak_form_vem_linear_polygon";
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

namespace {

using WeakInnerFad = Sacado::Fad::DFad<double>;
using WeakOuterFad = Sacado::Fad::DFad<WeakInnerFad>;

static double weak_inner_dx(const WeakInnerFad& x, int i) {
    return (i >= 0 && i < x.size()) ? x.dx(i) : 0.0;
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
    const vem::LinearElementCache& cache = vem::cached_linear_element(mesh, elem_id, q_order);
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

    const WeakOuterFad G = weak_scalar_vem(mesh, elem_id, x_ad, test_ad, params, state);

    ElementSystem sys;
    sys.residual.assign(n, 0.0);
    sys.tangent.assign(n * n, 0.0);

    const WeakInnerFad G_value = G.val();
    for (int i = 0; i < n; ++i) {
        sys.residual[i] = weak_inner_dx(G_value, i);
    }
    for (int j = 0; j < n; ++j) {
        const WeakInnerFad dG_dxj = (j >= 0 && j < G.size()) ? G.dx(j) : WeakInnerFad(0.0);
        for (int i = 0; i < n; ++i) {
            sys.tangent[i * n + j] = weak_inner_dx(dG_dxj, i);
        }
    }

    return sys;
}
