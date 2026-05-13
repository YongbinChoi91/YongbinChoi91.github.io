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
#include "fem_utils.h"
#include "kinematics.h"

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
    return "viscoelastic_solid_displacement_history";
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

using VEAD = Sacado::Fad::DFad<double>;
using WeakInnerFad = Sacado::Fad::DFad<double>;
using WeakOuterFad = Sacado::Fad::DFad<WeakInnerFad>;
using Matrix = std::vector<Vector>;

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

static Vector solve_dense(Matrix A, Vector b) {
    const int n = static_cast<int>(b.size());
    if (static_cast<int>(A.size()) != n) {
        throw std::runtime_error("Dense local solve received incompatible matrix/vector size.");
    }
    for (int i = 0; i < n; ++i) {
        if (static_cast<int>(A[i].size()) != n) {
            throw std::runtime_error("Dense local solve expects a square matrix.");
        }
    }

    for (int k = 0; k < n; ++k) {
        int piv = k;
        double best = std::abs(A[k][k]);
        for (int i = k + 1; i < n; ++i) {
            const double v = std::abs(A[i][k]);
            if (v > best) { best = v; piv = i; }
        }
        if (best < 1.0e-20) {
            throw std::runtime_error("Singular local viscoelastic Jacobian.");
        }
        if (piv != k) {
            std::swap(A[piv], A[k]);
            std::swap(b[piv], b[k]);
        }

        const double akk = A[k][k];
        for (int j = k; j < n; ++j) A[k][j] /= akk;
        b[k] /= akk;

        for (int i = 0; i < n; ++i) {
            if (i == k) continue;
            const double f = A[i][k];
            for (int j = k; j < n; ++j) A[i][j] -= f * A[k][j];
            b[i] -= f * b[k];
        }
    }
    return b;
}

template <typename T>
static double val(const T& x) {
    return static_cast<double>(x);
}

template <typename T>
static double val(const Sacado::Fad::DFad<T>& x) {
    return val(x.val());
}

static double weak_inner_dx(const WeakInnerFad& x, int i) {
    return (i >= 0 && i < x.size()) ? x.dx(i) : 0.0;
}

template <typename T>
using Mat3T = KinematicMat3<T>;

template <typename T> static Mat3T<T> Cv_from_hT(const std::vector<T>& h, int dim) {
    const int required = (dim == 2) ? 4 : 6;
    if (static_cast<int>(h.size()) < required) {
        throw std::runtime_error("Viscoelastic history vector has too few variables for the current dimension.");
    }
    auto C = kinematic_eye3<T>();
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
    const T Jbe = kinematic_det3<T>(be);
    if (val(Jbe) <= 0.0) throw std::runtime_error("Non-positive det(be) in viscoelastic model.");
    const double mu = mat.E / (2.0 * (1.0 + mat.nu));
    const double kappa = mat.E / (3.0 * (1.0 - 2.0 * mat.nu));
    const T trbe = kinematic_trace3<T>(be);
    auto I = kinematic_eye3<T>();
    auto tau = kinematic_zero3<T>();
    using std::pow;
    const T Jm13 = pow(Jbe, T(-1.0 / 3.0));
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
    const auto Ft = kinematic_transpose3<T>(F);
    const auto C = kinematic_right_cauchy_green3<T>(F);
    const auto invC = kinematic_inv3<T>(C);
    const auto invCv = kinematic_inv3<T>(Cv);

    // dCdt = (Cv - Cv_n)/dt, exactly as Python uses backward Euler.
    const auto dCdt = kinematic_scale3<T>(kinematic_sub3<T>(Cv, Cv_n), T(1.0 / dt));

    // be = F Cv^{-1} F^T and Tau = 2 be d psi(be)/dbe.
    const auto be = kinematic_mul3<T>(kinematic_mul3<T>(F, invCv), Ft);
    const auto tau = tau_from_beT<T>(be, mat);

    // dv = sym( 1/2 F^{-T} dCdt C^{-1} F^T ).
    const auto dv1 = kinematic_scale3<T>(kinematic_mul3<T>(kinematic_mul3<T>(kinematic_mul3<T>(kinematic_inv_transpose3<T>(F), dCdt), invC), Ft), T(0.5));
    const auto dv = kinematic_sym3<T>(dv1);

    // Q = -dev(Tau - eta*dv) with det(Cv)-1 as the incompressibility constraint.
    const auto ed = kinematic_sub3<T>(tau, kinematic_scale3<T>(dv, T(mat.eta)));
    const auto Z = kinematic_scale3<T>(kinematic_dev3<T>(ed), T(-1.0));

    if (dim == 2) {
        return {Z[0][0], Z[1][1], kinematic_det3<T>(Cv) - T(1.0), Z[0][1]};
    }
    if (dim == 3) {
        return {Z[0][0], Z[1][1], kinematic_det3<T>(Cv) - T(1.0), Z[0][1], Z[2][0], Z[2][1]};
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

static Matrix local_Q_h_sacado(
    const Mat3T<double>& Fd,
    const Vector& h_n,
    const Vector& h,
    int dim,
    double dt,
    const VEMaterial& mat
) {
    const int nh = (dim == 2) ? 4 : 6;
    std::vector<VEAD> had(nh), hnad(nh);
    for (int i = 0; i < nh; ++i) {
        had[i] = VEAD(nh, i, h[i]);
        hnad[i] = VEAD(nh, h_n[i]);
    }
    Mat3T<VEAD> F = kinematic_zero3<VEAD>();
    for (int i = 0; i < 3; ++i) for (int A = 0; A < 3; ++A) F[i][A] = VEAD(nh, Fd[i][A]);
    const auto Q = local_Q_viscoelastic_T<VEAD>(F, hnad, had, dim, dt, mat);
    Matrix J(nh, Vector(nh, 0.0));
    for (int i = 0; i < nh; ++i) for (int j = 0; j < nh; ++j) J[i][j] = Q[i].dx(j);
    return J;
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
        const Matrix J = local_Q_h_sacado(F, h_n, h, dim, dt, mat);
        Vector rhs(Q.size(), 0.0);
        for (int i = 0; i < static_cast<int>(Q.size()); ++i) rhs[i] = -Q[i];
        const Vector dh = solve_dense(J, rhs);

        double alpha = 1.0;
        const double q0 = norm_vec(Q);
        for (int ls = 0; ls < 10; ++ls) {
            Vector ht = h;
            for (int i = 0; i < nh; ++i) ht[i] += alpha * dh[i];
            const auto Cv_t = Cv_from_hT<double>(std::vector<double>(ht.begin(), ht.end()), dim);
            if (kinematic_det3<double>(Cv_t) > 0.0) {
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

static Matrix implicit_dh_dF_sacado(
    const Mat3T<double>& Fd,
    const Vector& h_n,
    const Vector& h,
    int dim,
    double dt,
    const VEMaterial& mat
) {
    const int nf = dim * dim;
    const int nh = (dim == 2) ? 4 : 6;
    const int nvar = nf + nh;

    Mat3T<VEAD> F = kinematic_zero3<VEAD>();
    for (int i = 0; i < 3; ++i) for (int A = 0; A < 3; ++A) F[i][A] = VEAD(nvar, Fd[i][A]);
    for (int i = 0; i < dim; ++i) {
        for (int A = 0; A < dim; ++A) {
            F[i][A] = VEAD(nvar, f_index(dim, i, A), Fd[i][A]);
        }
    }

    std::vector<VEAD> had(nh), hnad(nh);
    for (int i = 0; i < nh; ++i) {
        had[i] = VEAD(nvar, nf + i, h[i]);
        hnad[i] = VEAD(nvar, h_n[i]);
    }

    const auto Q = local_Q_viscoelastic_T<VEAD>(F, hnad, had, dim, dt, mat);
    Matrix Q_F(nh, Vector(nf, 0.0));
    Matrix Q_h(nh, Vector(nh, 0.0));
    for (int i = 0; i < nh; ++i) {
        for (int j = 0; j < nf; ++j) Q_F[i][j] = Q[i].dx(j);
        for (int j = 0; j < nh; ++j) Q_h[i][j] = Q[i].dx(nf + j);
    }

    Matrix dh_dF(nh, Vector(nf, 0.0));
    for (int jf = 0; jf < nf; ++jf) {
        Vector rhs(nh, 0.0);
        for (int i = 0; i < nh; ++i) rhs[i] = -Q_F[i][jf];
        const Vector col = solve_dense(Q_h, rhs);
        for (int ih = 0; ih < nh; ++ih) dh_dF[ih][jf] = col[ih];
    }
    return dh_dF;
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
    const int nf = dim * dim;
    const int nh = (dim == 2) ? 4 : 6;
    const int n_outer = F_ad[0][0].size();
    const Matrix dh_dF = implicit_dh_dF_sacado(F_double, h_n, h_trial, dim, dt, mat);

    std::vector<Scalar> h_ad(nh);
    for (int ih = 0; ih < nh; ++ih) {
        h_ad[ih] = Scalar(WeakInnerFad(h_trial[ih]));
        h_ad[ih].resize(n_outer);
        for (int j = 0; j < n_outer; ++j) {
            double dh_dx = 0.0;
            for (int i = 0; i < dim; ++i) {
                for (int A = 0; A < dim; ++A) {
                    const int f = f_index(dim, i, A);
                    dh_dx += dh_dF[ih][f] * val(F_ad[i][A].dx(j));
                }
            }
            h_ad[ih].fastAccessDx(j) = WeakInnerFad(dh_dx);
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

    const LocalFieldView L = make_local_field_view(mesh, elem_id, active_constitutive_field_names(mesh));
    const auto& U = L.field("displacement");
    if (U.components != dim) throw std::runtime_error("Viscoelastic displacement field must have dim components.");
    if (static_cast<int>(x_elem.size()) != L.n_local()) throw std::runtime_error("Bad local vector size in viscoelastic weak form.");
    if (test_elem.size() != x_elem.size()) throw std::runtime_error("Bad test vector size in viscoelastic weak form.");

    const ElementGeometry geom = make_displacement_geometry_element(mesh, elem_id);
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

        const Mat3T<double> F_double = kinematic_deformation_gradient_array<double>(dim, dN, u_double);
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

        const Mat3T<Scalar> F_ad = kinematic_deformation_gradient_array<Scalar>(dim, dN, u_ad);
        const std::vector<Scalar> h_ad = make_algorithmic_h_ad(F_ad, F_double, gh.h_n, upd.h, dim, state.dt, mat);
        const auto Cv_ad = Cv_from_hT<Scalar>(h_ad, dim);
        const auto be = kinematic_mul3<Scalar>(kinematic_mul3<Scalar>(F_ad, kinematic_inv3<Scalar>(Cv_ad)), kinematic_transpose3<Scalar>(F_ad));
        const auto tau = tau_from_beT<Scalar>(be, mat);
        const auto P_ad = kinematic_mul3<Scalar>(tau, kinematic_inv_transpose3<Scalar>(F_ad));

        Mat3T<Scalar> grad_test = kinematic_zero3<Scalar>();
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
    const LocalFieldView L = make_local_field_view(mesh, elem_id, active_constitutive_field_names(mesh));
    const int n = L.n_local();
    if (static_cast<int>(x_elem.size()) != n) {
        throw std::runtime_error("Bad local vector size in viscoelastic weak-scalar element system.");
    }

    std::vector<WeakOuterFad> x_ad(n);
    std::vector<WeakOuterFad> test_ad(n);
    for (int j = 0; j < n; ++j) {
        x_ad[j] = WeakOuterFad(n, j, WeakInnerFad(x_elem[j]));
    }
    for (int i = 0; i < n; ++i) {
        test_ad[i] = WeakOuterFad(WeakInnerFad(n, i, 0.0));
    }

    const WeakOuterFad G = weak_scalar_viscoelastic_solid(mesh, elem_id, x_ad, test_ad, params, state);

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
