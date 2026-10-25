/*
================================================================================
MODEL MAP: weak_form_viscoelastic.cpp
================================================================================
Purpose
  Displacement-only finite-strain viscoelastic solid with Gauss-point history.
  The global unknown is displacement.  The viscous internal metric Cv is stored
  locally at integration points, not as global FE degrees of freedom.

Python reference copied into this C++ structure
  The implementation follows Solid_models.py / Inelastic_Models:

      Visco_Elastic_Residual_Fun(hg, hg_n, F, param, settings)

  with the same local variables:
      Cv      = I + history tensor
      Cv_n    = previous accepted Cv
      C       = F^T F
      dCdt    = (Cv - Cv_n) / dt
      be      = F Cv^{-1} F^T
      Tau     = 2 be d psi(be) / d be
      dv      = sym( 1/2 F^{-T} dCdt C^{-1} F^T )
      Q       = -dev(Tau - eta dv) plus det(Cv)-1 constraint

How this file is called
  solver.cpp
    -> assembly.cpp
      -> compute_constitutive_element_system(...)
        -> weak_scalar_viscoelastic_solid(...)
          -> for each Gauss point:
             1. compute F from current displacement,
             2. solve local Newton problem Q(F,Cv,Cv_n)=0 for h_trial,
             3. inject algorithmic dh/dF into AD by implicit differentiation,
             4. evaluate P(F,h(F)) : Grad(delta_u),
        -> nested Sacado AD extraction of residual and tangent.

History convention
  DOFS_info.txt must contain exactly one of these, depending on mesh dimension:

      # 2D plane strain
      history viscoelastic 4

      # 3D
      history viscoelastic 6

  2D history variables:
      h[0] = Cv_xx - 1
      h[1] = Cv_yy - 1
      h[2] = Cv_zz - 1
      h[3] = Cv_xy

  3D history variables:
      h[0] = Cv_xx - 1
      h[1] = Cv_yy - 1
      h[2] = Cv_zz - 1
      h[3] = Cv_xy
      h[4] = Cv_xz
      h[5] = Cv_yz

Material.txt keys added for this model
      viscoelastic_eta              150000.0
      viscoelastic_local_tol        1e-10
      viscoelastic_local_max_iter   25

  Accepted aliases are added in mesh.cpp, e.g. visco_elastic_parameter,
  ve_eta, eta_v, ve_local_tol, and ve_local_max_iter.

Tangent convention
  The local Newton update is solved in double precision.  The consistent global
  tangent is recovered by the implicit-function theorem:

      Q(F,h)=0  =>  dh/dF = - Q_h^{-1} Q_F.

  That derivative is injected into Sacado variables before differentiating the
  scalar weak form G.
================================================================================
*/
#include "weak_form.h"
#include "kinematics.h"
#include "autodiff.h"
#include "local_element_system_ad.h"
#include "local_dense_solve.h"

#include <Sacado_Fad_DFad.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

const char* active_constitutive_model_name() {
    return "Finite_Strain_Visco_Elastic";
}

std::vector<std::string> active_constitutive_field_names(const FieldMesh& mesh) {
    (void)mesh;
    return {"displacement"};
}

namespace {

struct VEMaterial {
    double E = 1.0;
    double nu = 0.3;
    double eta = 1.0;             
    double local_tol = 1.0e-10;
    int local_max_iter = 25;
};

static VEMaterial make_ve_material(const ProblemParameters& params) {
    VEMaterial m;
    m.E = params.material.youngs_modulus;
    m.nu = params.material.poisson_ratio;
    m.eta = params.material.visco_elastic_eta;
    m.local_tol = params.material.local_tol;
    m.local_max_iter = params.material.local_max_iter;
    if (m.E <= 0.0) throw std::runtime_error("Viscoelastic weak form requires positive youngs_modulus/E.");
    if (m.nu <= -1.0 || m.nu >= 0.5) throw std::runtime_error("Viscoelastic weak form requires -1 < nu < 0.5.");
    if (m.eta <= 0.0) throw std::runtime_error("Viscoelastic weak form requires positive viscoelastic_eta.");
    if (m.local_tol <= 0.0 || m.local_max_iter <= 0) throw std::runtime_error("Bad viscoelastic local Newton settings.");
    return m;
}

using Matrix = hpx::LocalDenseMatrix;

// -----------------------------------------------------------------------------
// Non-kinematic local helpers kept in weak_form.cpp.
// These are model/AD utility routines, not tensor kinematics, so they should not
// be moved to kinematics.h.
// -----------------------------------------------------------------------------
static double norm_vec(const Vector& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}

template <typename T>
static double val(const T& x) {
    return static_cast<double>(x);
}

template <typename T>
static double val(const Sacado::Fad::DFad<T>& x) {
    return val(x.val());
}

template <typename T, int N>
static double val(const Sacado::Fad::SLFad<T,N>& x) {
    return val(x.val());
}

template <typename T, std::size_t N>
static double val(const hpx::AD1<T,N>& x) {
    return val(x.val);
}

template <typename T>
using Mat3T = hpx::Mat3<T>;

template <typename T> static Mat3T<T> Cv_from_hT(const std::vector<T>& h, int dim) {
    const int required = (dim == 2) ? 4 : 6;
    if (static_cast<int>(h.size()) < required) {
        throw std::runtime_error("Viscoelastic history vector has too few variables for the current dimension.");
    }
    auto C = hpx::eye3<T>();
    C[0][0] += h[0];
    C[1][1] += h[1];
    C[2][2] += h[2];
    C[0][1] += h[3]; C[1][0] += h[3];
    if (dim == 3) {
        C[0][2] += h[4]; C[2][0] += h[4];
        C[1][2] += h[5]; C[2][1] += h[5];
    }
    return C;
}

template <typename T> static Mat3T<T> tau_from_beT(const Mat3T<T>& be, const VEMaterial& mat) {
    const T Jbe = hpx::det3<T>(be);
    if (val(Jbe) <= 0.0) throw std::runtime_error("Non-positive det(be) in viscoelastic model.");
    const double mu = mat.E / (2.0 * (1.0 + mat.nu));
    const double kappa = mat.E / (3.0 * (1.0 - 2.0 * mat.nu));
    const T trbe = hpx::tr3<T>(be);
    auto I = hpx::eye3<T>();
    auto tau = hpx::zero3<T>();
    using std::pow;
    const T Jm13 = pow(Jbe, -1.0 / 3.0);
    const T p = T(0.5 * kappa) * (Jbe - T(1.0));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            tau[i][j] = p * I[i][j] + T(mu) * Jm13 * (be[i][j] - (trbe / T(3.0)) * I[i][j]);
        }
    }
    return tau;
}

template <typename T>
static std::vector<T> local_Q_viscoelastic_T(
    const Mat3T<T>& F,
    const std::vector<T>& h_n,
    const std::vector<T>& h,
    int dim,
    double dt,
    const VEMaterial& mat
) {
    if (dt <= 0.0) throw std::runtime_error("Viscoelastic local residual requires positive dt.");

    const auto Cv = Cv_from_hT<T>(h, dim);
    const auto Cv_n = Cv_from_hT<T>(h_n, dim);
    const auto Ft = hpx::transpose3<T>(F);
    const auto C = hpx::right_cauchy_green3<T>(F);
    const auto invC = hpx::inv3<T>(C);
    const auto invCv = hpx::inv3<T>(Cv);

    // dCdt = (Cv - Cv_n)/dt, exactly as Python uses backward Euler.
    const auto dCdt = hpx::scale3<T>(hpx::sub3<T>(Cv, Cv_n), T(1.0 / dt));

    // be = F Cv^{-1} F^T and Tau = 2 be d psi(be)/dbe.
    const auto be = hpx::mul3<T>(hpx::mul3<T>(F, invCv), Ft);
    const auto tau = tau_from_beT<T>(be, mat);

    // dv = sym( 1/2 F^{-T} dCdt C^{-1} F^T ).
    const auto dv1 = hpx::scale3<T>(hpx::mul3<T>(hpx::mul3<T>(hpx::mul3<T>(hpx::inv_transpose3<T>(F), dCdt), invC), Ft), T(0.5));
    const auto dv = hpx::sym3<T>(dv1);

    // Q = -dev(Tau - eta*dv) with det(Cv)-1 as the incompressibility constraint.
    const auto ed = hpx::sub3<T>(tau, hpx::scale3<T>(dv, T(mat.eta)));
    const auto Z = hpx::scale3<T>(hpx::dev3<T>(ed), T(-1.0));

    if (dim == 2) {
        return {Z[0][0], Z[1][1], hpx::det3<T>(Cv) - T(1.0), Z[0][1]};
    }
    if (dim == 3) {
        return {Z[0][0], Z[1][1], hpx::det3<T>(Cv) - T(1.0), Z[0][1], Z[2][0], Z[2][1]};
    }
    throw std::runtime_error("Viscoelastic local residual supports only 2D and 3D.");
}

static Vector local_Q_viscoelastic(
    const Mat3T<double>& F,
    const Vector& h_n,
    const Vector& h,
    int dim,
    double dt,
    const VEMaterial& mat
) {
    return local_Q_viscoelastic_T<double>(F, h_n, h, dim, dt, mat);
}

template <std::size_t N>
using VEHpxAD = hpx::AD1<double, N>;

template <std::size_t N>
static Matrix local_Q_h_hpx_impl(
    const Mat3T<double>& Fd,
    const Vector& h_n,
    const Vector& h,
    int dim,
    double dt,
    const VEMaterial& mat
) {
    const int nh = (dim == 2) ? 4 : 6;
    using AD = VEHpxAD<N>;
    std::vector<AD> had(nh), hnad(nh);
    for (int i = 0; i < nh; ++i) {
        had[i] = AD::variable(h[i], i);
        hnad[i] = AD(h_n[i]);
    }
    Mat3T<AD> F = hpx::zero3<AD>();
    for (int i = 0; i < 3; ++i) for (int A = 0; A < 3; ++A) F[i][A] = AD(Fd[i][A]);
    const auto Q = local_Q_viscoelastic_T<AD>(F, hnad, had, dim, dt, mat);
    Matrix J(nh, Vector(nh, 0.0));
    for (int i = 0; i < nh; ++i) for (int j = 0; j < nh; ++j) J[i][j] = Q[i].grad[j];
    return J;
}

static Matrix local_Q_h_hpx(
    const Mat3T<double>& Fd,
    const Vector& h_n,
    const Vector& h,
    int dim,
    double dt,
    const VEMaterial& mat
) {
    if (dim == 2) return local_Q_h_hpx_impl<4>(Fd, h_n, h, dim, dt, mat);
    if (dim == 3) return local_Q_h_hpx_impl<6>(Fd, h_n, h, dim, dt, mat);
    throw std::runtime_error("Viscoelastic local residual supports only 2D and 3D.");
}

struct VEUpdateResult {
    Vector h;
    double final_local_residual = 0.0;
    int local_newton_iterations = 0;
    double state_trial = 0.0;
};

static VEUpdateResult update_viscoelastic_gp(
    const Mat3T<double>& F,
    const Vector& h_n,
    int dim,
    double dt,
    const VEMaterial& mat
) {
    const int nh = (dim == 2) ? 4 : 6;
    if (static_cast<int>(h_n.size()) < nh) throw std::runtime_error("Bad viscoelastic history size in local update.");

    VEUpdateResult out;
    Vector h(h_n.begin(), h_n.begin() + nh);
    const double tol = std::max(mat.local_tol, 1.0e-14);

    for (int it = 0; it < std::max(1, mat.local_max_iter); ++it) {
        const Vector Q = local_Q_viscoelastic(F, h_n, h, dim, dt, mat);
        const Matrix J = local_Q_h_hpx(F, h_n, h, dim, dt, mat);
        Vector rhs(Q.size(), 0.0);
        for (int i = 0; i < static_cast<int>(Q.size()); ++i) rhs[i] = -Q[i];
        const Vector dh = hpx::solve_local_dense_lu(J, rhs, "local viscoelastic Jacobian");

        double alpha = 1.0;
        const double q0 = norm_vec(Q);
        for (int ls = 0; ls < 10; ++ls) {
            Vector ht = h;
            for (int i = 0; i < nh; ++i) ht[i] += alpha * dh[i];
            const auto Cv_t = Cv_from_hT<double>(std::vector<double>(ht.begin(), ht.end()), dim);
            if (hpx::det3<double>(Cv_t) > 0.0) {
                const Vector Qt = local_Q_viscoelastic(F, h_n, ht, dim, dt, mat);
                if (norm_vec(Qt) <= q0 || alpha < 1.0e-3) { h = ht; break; }
            }
            alpha *= 0.5;
        }

        const Vector Qnew = local_Q_viscoelastic(F, h_n, h, dim, dt, mat);
        const double qn = norm_vec(Qnew);
        if (qn < tol || norm_vec(dh) < tol) {
            out.h = h;
            out.final_local_residual = qn;
            out.local_newton_iterations = it + 1;
            out.state_trial = qn;
            return out;
        }
    }
    throw std::runtime_error("Local viscoelastic Newton did not converge.");
}

static int f_index(int dim, int i, int A) {
    return i * dim + A;
}

template <std::size_t N>
static Matrix implicit_dh_dF_hpx_impl(
    const Mat3T<double>& Fd,
    const Vector& h_n,
    const Vector& h,
    int dim,
    double dt,
    const VEMaterial& mat
) {
    const int nf = dim * dim;
    const int nh = (dim == 2) ? 4 : 6;

    using AD = VEHpxAD<N>;
    Mat3T<AD> F = hpx::zero3<AD>();
    for (int i = 0; i < 3; ++i) for (int A = 0; A < 3; ++A) F[i][A] = AD(Fd[i][A]);
    for (int i = 0; i < dim; ++i) {
        for (int A = 0; A < dim; ++A) {
            F[i][A] = AD::variable(Fd[i][A], f_index(dim, i, A));
        }
    }

    std::vector<AD> had(nh), hnad(nh);
    for (int i = 0; i < nh; ++i) {
        had[i] = AD::variable(h[i], nf + i);
        hnad[i] = AD(h_n[i]);
    }

    const auto Q = local_Q_viscoelastic_T<AD>(F, hnad, had, dim, dt, mat);
    Matrix Q_F(nh, Vector(nf, 0.0));
    Matrix Q_h(nh, Vector(nh, 0.0));
    for (int i = 0; i < nh; ++i) {
        for (int j = 0; j < nf; ++j) Q_F[i][j] = Q[i].grad[j];
        for (int j = 0; j < nh; ++j) Q_h[i][j] = Q[i].grad[nf + j];
    }

    Matrix dh_dF(nh, Vector(nf, 0.0));
    const auto Q_h_lu = hpx::factorize_local_dense_lu(Q_h, "local viscoelastic implicit derivative Jacobian");
    for (int jf = 0; jf < nf; ++jf) {
        Vector rhs(nh, 0.0);
        for (int i = 0; i < nh; ++i) rhs[i] = -Q_F[i][jf];
        const Vector col = hpx::solve_local_dense_lu(Q_h_lu, rhs);
        for (int ih = 0; ih < nh; ++ih) dh_dF[ih][jf] = col[ih];
    }
    return dh_dF;
}

static Matrix implicit_dh_dF_hpx(
    const Mat3T<double>& Fd,
    const Vector& h_n,
    const Vector& h,
    int dim,
    double dt,
    const VEMaterial& mat
) {
    if (dim == 2) return implicit_dh_dF_hpx_impl<8>(Fd, h_n, h, dim, dt, mat);
    if (dim == 3) return implicit_dh_dF_hpx_impl<15>(Fd, h_n, h, dim, dt, mat);
    throw std::runtime_error("Viscoelastic local residual supports only 2D and 3D.");
}

template <class Scalar>
static std::vector<Scalar> displacement_local_vector_from_layout(
    const LocalFieldView& L,
    const LocalFieldInfo& U,
    const std::vector<Scalar>& x_elem
) {
    std::vector<Scalar> u(L.dim * U.n_nodes, Scalar(0.0));
    for (int a = 0; a < U.n_nodes; ++a) {
        for (int i = 0; i < L.dim; ++i) {
            u[L.dim * a + i] = x_elem[L.dof("displacement", a, i)];
        }
    }
    return u;
}

template <class Scalar>
static std::vector<Scalar> make_algorithmic_h_ad(
    const Mat3T<Scalar>& F_ad,
    const Mat3T<double>& F_double,
    const Vector& h_n,
    const Vector& h_trial,
    int dim,
    double dt,
    const VEMaterial& mat
) {
    const int nh = (dim == 2) ? 4 : 6;
    const int n_outer = F_ad[0][0].size();
    const Matrix dh_dF = implicit_dh_dF_hpx(F_double, h_n, h_trial, dim, dt, mat);
    using InnerScalar = typename Scalar::value_type;

    std::vector<Scalar> h_ad(nh);
    for (int ih = 0; ih < nh; ++ih) {
        h_ad[ih] = Scalar(InnerScalar(h_trial[ih]));
        h_ad[ih].resize(n_outer);
        for (int j = 0; j < n_outer; ++j) {
            double dh_dx = 0.0;
            for (int i = 0; i < dim; ++i) {
                for (int A = 0; A < dim; ++A) {
                    const int f = f_index(dim, i, A);
                    dh_dx += dh_dF[ih][f] * val(F_ad[i][A].dx(j));
                }
            }
            h_ad[ih].fastAccessDx(j) = InnerScalar(dh_dx);
        }
    }
    return h_ad;
}

template <class Scalar>
static Scalar weak_scalar_viscoelastic_solid(
    const FieldMesh& mesh,
    int elem_id,
    const std::vector<Scalar>& x_elem,
    const std::vector<Scalar>& test_elem,
    const ProblemParameters& params,
    const ElementState& state
) {
    if (mesh.dim != 2 && mesh.dim != 3) throw std::runtime_error("Viscoelastic weak form supports only 2D or 3D.");
    if (!state.material_history) {
        throw std::runtime_error("Viscoelastic weak form requires Gauss-point history. Add 'history viscoelastic 4' for 2D or 'history viscoelastic 6' for 3D to DOFS_info.txt.");
    }
    if (state.dt <= 0.0) throw std::runtime_error("Viscoelastic weak form requires positive state.dt.");

    const int dim = mesh.dim;
    const int required_history = (dim == 2) ? 4 : 6;
    const VEMaterial mat = make_ve_material(params);

    const LocalFieldView L = make_local_field_view(mesh, elem_id, {"displacement"});
    const auto& U = L.field("displacement");
    if (U.components != dim) throw std::runtime_error("Viscoelastic displacement field must have dim components.");
    if (static_cast<int>(x_elem.size()) != L.n_local()) throw std::runtime_error("Bad local vector size in viscoelastic weak form.");
    if (test_elem.size() != x_elem.size()) throw std::runtime_error("Bad test vector size in viscoelastic weak form.");

    const ElementGeometry geom = make_displacement_element_geometry(mesh, elem_id);
    const auto gps = gauss_rule(U.element_type, params.solver.gauss_order);
    if (state.material_history->gp.size() != gps.size()) state.material_history->gp.resize(gps.size());

    const auto u_ad = displacement_local_vector_from_layout<Scalar>(L, U, x_elem);
    const auto du_test = displacement_local_vector_from_layout<Scalar>(L, U, test_elem);

    std::vector<double> u_double(dim * U.n_nodes, 0.0);
    for (int i = 0; i < static_cast<int>(u_double.size()); ++i) u_double[i] = val(u_ad[i]);

    Scalar G(0.0);
    for (int g = 0; g < static_cast<int>(gps.size()); ++g) {
        const auto& gp = gps[g];
        const ShapeData shape = compute_shape(U.element_type, gp.xi, gp.eta, gp.zeta);
        const JacobianData J0 = compute_jacobian_data(geom, shape);
        if (J0.detJ <= 0.0) throw std::runtime_error("Negative detJ in viscoelastic element.");
        const auto dN = compute_dN_dx(shape, J0);
        const double dV = J0.detJ * gp.weight;

        auto& gh = state.material_history->gp[g];
        if (static_cast<int>(gh.h_n.size()) < required_history) {
            gh.h_n.assign(required_history, 0.0);      // h=0 means Cv=I initially.
            gh.h_trial.assign(required_history, 0.0);
            gh.state_n = 0.0;
            gh.state_trial = 0.0;
        }

        const Mat3T<double> F_double = hpx::deformation_gradient_array<double>(dim, dN, u_double);
        VEUpdateResult upd;
        try {
            upd = update_viscoelastic_gp(F_double, gh.h_n, dim, state.dt, mat);
        } catch (...) {
            if (state.history_diagnostics) ++state.history_diagnostics->local_failure_count;
            throw;
        }

        gh.h_trial = upd.h;
        gh.state_trial = upd.state_trial;
        if (state.history_diagnostics) {
            state.history_diagnostics->observe("viscoelastic", upd.final_local_residual, upd.local_newton_iterations,
                                               true, state.domain_id, elem_id, g);
        }

        const Mat3T<Scalar> F_ad = hpx::deformation_gradient_array<Scalar>(dim, dN, u_ad);
        const std::vector<Scalar> h_ad = make_algorithmic_h_ad(F_ad, F_double, gh.h_n, upd.h, dim, state.dt, mat);
        const auto Cv_ad = Cv_from_hT<Scalar>(h_ad, dim);
        const auto be = hpx::mul3<Scalar>(hpx::mul3<Scalar>(F_ad, hpx::inv3<Scalar>(Cv_ad)), hpx::transpose3<Scalar>(F_ad));
        const auto tau = tau_from_beT<Scalar>(be, mat);
        const auto P_ad = hpx::mul3<Scalar>(tau, hpx::inv_transpose3<Scalar>(F_ad));

        Mat3T<Scalar> grad_test = hpx::zero3<Scalar>();
        for (int a = 0; a < U.n_nodes; ++a) {
            for (int i = 0; i < dim; ++i) {
                const Scalar ti = du_test[dim * a + i];
                for (int A = 0; A < dim; ++A) grad_test[i][A] += ti * Scalar(dN[a][A]);
            }
        }

        for (int i = 0; i < dim; ++i) {
            for (int A = 0; A < dim; ++A) {
                G += P_ad[i][A] * grad_test[i][A] * Scalar(dV);
            }
        }
    }
    return G;
}

} // namespace

void print_active_constitutive_parameters(std::ostream& os, const ProblemParameters& params) {
    const VEMaterial m = make_ve_material(params);
    os << "  E                         : " << m.E << "\n"
       << "  nu                        : " << m.nu << "\n"
       << "  viscoelastic_eta          : " << m.eta << "\n"
       << "  viscoelastic_local_tol    : " << m.local_tol << "\n"
       << "  viscoelastic_local_iter   : " << m.local_max_iter << "\n"
       << "  required history          : 2D -> history viscoelastic 4, 3D -> history viscoelastic 6\n";
}

ElementSystem compute_constitutive_element_system(
    const FieldMesh& mesh,
    int elem_id,
    const Vector& x_elem,
    const ProblemParameters& params,
    const ElementState& state
) {
    const LocalFieldView L = make_local_field_view(mesh, elem_id, {"displacement"});
    const int n = L.n_local();
    if (static_cast<int>(x_elem.size()) != n) {
        throw std::runtime_error("Bad local vector size in viscoelastic weak-scalar element system.");
    }

    return hpx::scalar_weak_form_element_system(
        x_elem,
        [&](const auto& x_ad, const auto& test_ad) {
            return weak_scalar_viscoelastic_solid(mesh, elem_id, x_ad, test_ad, params, state);
        }
    );
}

ConstitutiveMultiscaleResponse compute_constitutive_multiscale_response(
    const FieldMesh& mesh,
    const Vector& displacement,
    const ProblemParameters& params,
    const MaterialHistoryStore* history,
    const ElementState& response_state
) {
    using Fad = Sacado::Fad::DFad<double>;

    if (mesh.dim != 2 && mesh.dim != 3) {
        throw std::runtime_error("Viscoelastic multiscale response supports only 2D or 3D.");
    }
    if (!history || history->domains.empty()) {
        throw std::runtime_error("Viscoelastic multiscale response requires RVE material history.");
    }

    const int dim = mesh.dim;
    const int required_history = (dim == 2) ? 4 : 6;
    const int ndofs = total_dof_count(mesh);
    if (static_cast<int>(displacement.size()) < ndofs) {
        throw std::runtime_error("Viscoelastic multiscale response received a displacement vector shorter than mesh DOFs.");
    }

    std::vector<Fad> u_global_ad(ndofs);
    for (int a = 0; a < ndofs; ++a) u_global_ad[a] = Fad(ndofs, a, displacement[a]);

    std::array<std::array<Fad,3>,3> intP_ad{};
    for (auto& row : intP_ad) {
        for (auto& v : row) v = Fad(ndofs, 0.0);
    }

    if (response_state.dt <= 0.0) {
        throw std::runtime_error("Viscoelastic multiscale response requires positive state.dt.");
    }
    const VEMaterial mat = make_ve_material(params);
    const double dt = response_state.dt;
    const DofLayout& layout = get_cached_dof_layout(mesh);
    DomainMaterialHistory& domain_history = const_cast<DomainMaterialHistory&>(history->domains.front());
    if (static_cast<int>(domain_history.elements.size()) < static_cast<int>(mesh.displacement_elements.size())) {
        throw std::runtime_error("Viscoelastic multiscale response history has fewer elements than the RVE mesh.");
    }

    double volume = 0.0;
    for (int e = 0; e < static_cast<int>(mesh.displacement_elements.size()); ++e) {
        const LocalFieldView L = make_local_field_view(mesh, e, {"displacement"});
        const auto& U = L.field("displacement");
        const auto dofs = make_element_dofs_for_fields(layout, e, {"displacement"});

        std::vector<Fad> ue_ad(dofs.size());
        std::vector<double> u_double(dofs.size(), 0.0);
        for (int a = 0; a < static_cast<int>(dofs.size()); ++a) {
            const int I = dofs[a];
            ue_ad[a] = u_global_ad[I];
            u_double[a] = displacement[I];
        }

        ElementMaterialHistory& element_history = domain_history.elements[e];
        const ElementGeometry geom = make_displacement_element_geometry(mesh, e);
        const auto gps = gauss_rule(U.element_type, params.solver.gauss_order);
        if (element_history.gp.size() != gps.size()) element_history.gp.resize(gps.size());

        for (int g = 0; g < static_cast<int>(gps.size()); ++g) {
            const auto& gp = gps[g];
            const ShapeData shape = compute_shape(U.element_type, gp.xi, gp.eta, gp.zeta);
            const JacobianData J0 = compute_jacobian_data(geom, shape);
            if (J0.detJ <= 0.0) throw std::runtime_error("Negative detJ in viscoelastic multiscale response.");
            const auto dN = compute_dN_dx(shape, J0);
            const double dV = J0.detJ * gp.weight;

            MaterialPointHistory& gh = element_history.gp[g];
            if (static_cast<int>(gh.h_n.size()) < required_history) {
                gh.h_n.assign(required_history, 0.0);
                gh.h_trial.assign(required_history, 0.0);
                gh.state_n = 0.0;
                gh.state_trial = 0.0;
            }

            const Mat3T<double> F_double = hpx::deformation_gradient_array<double>(dim, dN, u_double);
            const VEUpdateResult upd = update_viscoelastic_gp(F_double, gh.h_n, dim, dt, mat);
            gh.h_trial = upd.h;
            gh.state_trial = upd.state_trial;

            const Mat3T<Fad> F_ad = hpx::deformation_gradient_array<Fad>(dim, dN, ue_ad);
            const std::vector<Fad> h_ad = make_algorithmic_h_ad(F_ad, F_double, gh.h_n, upd.h, dim, dt, mat);
            const auto Cv_ad = Cv_from_hT<Fad>(h_ad, dim);
            const auto be = hpx::mul3<Fad>(hpx::mul3<Fad>(F_ad, hpx::inv3<Fad>(Cv_ad)), hpx::transpose3<Fad>(F_ad));
            const auto tau = tau_from_beT<Fad>(be, mat);
            const auto P_ad = hpx::mul3<Fad>(tau, hpx::inv_transpose3<Fad>(F_ad));

            volume += dV;
            for (int i = 0; i < 3; ++i) {
                for (int A = 0; A < 3; ++A) {
                    intP_ad[i][A] += P_ad[i][A] * Fad(dV);
                }
            }
        }
    }

    if (volume <= 0.0) {
        throw std::runtime_error("Viscoelastic multiscale response volume is non-positive.");
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
