/*
================================================================================
MODEL MAP: weak_form_elasto_plastic.cpp
================================================================================
Purpose
  Displacement-only finite-strain elasto-plastic solid with Gauss-point history.
  The global unknown is displacement; plastic internal variables are local
  history variables stored at integration points, not global FE DOFs.

How this file is called
  solver.cpp
    -> assembly.cpp
      -> compute_constitutive_element_system(...)
        -> weak_scalar_elastoplastic_solid(...)
          -> for each Gauss point:
             1. compute F from current displacement,
             2. solve local return-mapping/Newton problem for h_trial,
             3. inject algorithmic dh/dF into AD via implicit differentiation,
             4. evaluate P(F,h(F)) : Grad(delta_u),
        -> nested Sacado AD extraction of residual and tangent.

History convention
  Requires DOFS_info.txt to contain for example:
      history plasticity 5   # 2D plane strain
      history plasticity 7   # 3D
  The local variables are interpreted here as components of Cgpi plus gamma:
      2D: h[0],h[1],h[2],h[3] -> C00,C11,C22,C01; h[4] -> gamma
      3D: h[0],h[1],h[2],h[3],h[4],h[5] -> C00,C11,C22,C01,C02,C12; h[6] -> gamma

Tangent convention
  The local Newton update itself is solved in double precision. The consistent
  global tangent is recovered by the implicit-function theorem:
      Q(F,h)=0  =>  dh/dF = - Q_h^{-1} Q_F.
  That derivative is injected into Sacado variables before differentiating G.

Code-change policy in this annotated copy
  Only comments and section markers were added. Executable statements are left
  in the same order and with the same expressions.
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
#include <sstream>
#include <stdexcept>
#include <vector>

// -----------------------------------------------------------------------------
// Model registration
// -----------------------------------------------------------------------------
// The executable links exactly one weak_form_*.cpp at a time.  The following
// function gives the linked model a human-readable name for solver diagnostics.
const char* active_constitutive_model_name() {
    return "Finite Strain Elasto Plastic";
}

// -----------------------------------------------------------------------------
// Active field declaration
// -----------------------------------------------------------------------------
// Assembly asks this function which fields should appear in the element-local
// vector.  The returned order controls the local layout used below.
std::vector<std::string> active_constitutive_field_names(const FieldMesh& mesh) {
    (void)mesh;
    return {"displacement"};
}

namespace {

// Material parameter bundle used only inside this file.
// It is copied from ProblemParameters once per element evaluation.
struct EPMaterial {
    double E = 1.0;
    double nu = 0.3;
    double sigma_y = 0.45;
    double Kf = 0.12924;
    double Yinf = 0.715;
    double delta = 16.93;
    double local_tol = 1.0e-10;
    int local_max_iter = 25;
};


// Convert generic input parameters into the compact local plasticity material.
// This also validates values that would make the local model ill-defined.
static EPMaterial make_ep_material(const ProblemParameters& params) {
    EPMaterial m;
    m.E = params.material.youngs_modulus;
    m.nu = params.material.poisson_ratio;
    m.sigma_y = params.material.sigma_y;
    m.Kf = params.material.Kf;
    m.Yinf = params.material.Yinf;
    m.delta = params.material.delta;
    m.local_tol = params.material.local_tol;
    m.local_max_iter = params.material.local_max_iter;
    if (m.E <= 0.0) throw std::runtime_error("Elasto-plastic weak form requires positive youngs_modulus/E.");
    if (m.nu <= -1.0 || m.nu >= 0.5) throw std::runtime_error("Elasto-plastic weak form requires -1 < nu < 0.5.");
    if (m.local_tol <= 0.0 || m.local_max_iter <= 0) throw std::runtime_error("Bad elasto-plastic local Newton settings.");
    return m;
}

using Mat3 = hpx::Mat3<double>;
// Local dense matrix type used only by the elasto-plastic Gauss-point Newton/consistent tangent.
// ElementSystem::tangent remains a flat Vector; do not use this for global/element tangents.
using Matrix = hpx::LocalDenseMatrix;

template <typename T>
using Mat3T = hpx::Mat3<T>;

static double frob3(const Mat3& A){ double s=0.0; for(int i=0;i<3;++i) for(int j=0;j<3;++j) s+=A[i][j]*A[i][j]; return std::sqrt(s); }

static Mat3 expm3(const Mat3& A){
    const double nrm=frob3(A);
    int s=0; if(nrm>0.5) s=std::max(0, static_cast<int>(std::ceil(std::log2(nrm/0.5))));
    Mat3 As=hpx::scale3<double>(A, 1.0/std::pow(2.0,s));
    Mat3 E=hpx::eye3<double>();
    Mat3 term=hpx::eye3<double>();
    for(int k=1;k<=36;++k){ term=hpx::scale3<double>(hpx::mul3<double>(term,As),1.0/static_cast<double>(k)); E=hpx::add3<double>(E,term); if(frob3(term)<1e-15) break; }
    for(int i=0;i<s;++i) E=hpx::mul3<double>(E,E);
    return E;
}

static double norm_vec(const Vector& v){ double s=0.0; for(double x:v) s+=x*x; return std::sqrt(s); }

template <typename T> static double val(const T& x){ return static_cast<double>(x); }
template <typename T> static double val(const Sacado::Fad::DFad<T>& x){ return val(x.val()); }
template <typename T, int N> static double val(const Sacado::Fad::SLFad<T,N>& x){ return val(x.val()); }
template <typename T, std::size_t N> static double val(const hpx::AD1<T,N>& x){ return val(x.val); }

template <typename T> static Mat3T<T> expm3T(const Mat3T<T>& A){
    double nrm=0.0; for(int i=0;i<3;++i) for(int j=0;j<3;++j){ const double x=val(A[i][j]); nrm+=x*x; }
    nrm=std::sqrt(nrm);
    int s=0; if(nrm>0.5) s=std::max(0, static_cast<int>(std::ceil(std::log2(nrm/0.5))));
    auto As=hpx::scale3<T>(A,T(1.0/std::pow(2.0,s)));
    auto E=hpx::eye3<T>();
    auto term=hpx::eye3<T>();
    for(int k=1;k<=36;++k){ term=hpx::scale3<T>(hpx::mul3<T>(term,As),T(1.0/static_cast<double>(k))); E=hpx::add3<T>(E,term); }
    for(int i=0;i<s;++i) E=hpx::mul3<T>(E,E);
    return E;
}

static int ep_required_history_size(int dim) {
    if (dim == 2) return 5;
    if (dim == 3) return 7;
    throw std::runtime_error("Elasto-plastic weak form supports only 2D/3D displacement fields.");
}

static int ep_gamma_index(int dim) {
    return ep_required_history_size(dim) - 1;
}

static void ep_check_history_size(const Vector& h, int dim) {
    const int required = ep_required_history_size(dim);
    if (static_cast<int>(h.size()) < required) {
        std::ostringstream oss;
        oss << "Plasticity history requires at least " << required
            << " variables for " << dim << "D. Use 'history plasticity "
            << required << "' in DOFS_info.txt.";
        throw std::runtime_error(oss.str());
    }
}

template <class Vec>
static void ep_check_history_sizeT(const Vec& h, int dim) {
    const int required = ep_required_history_size(dim);
    if (static_cast<int>(h.size()) < required) {
        std::ostringstream oss;
        oss << "Plasticity history requires at least " << required
            << " variables for " << dim << "D. Use 'history plasticity "
            << required << "' in DOFS_info.txt.";
        throw std::runtime_error(oss.str());
    }
}

static Mat3 Cgpi_from_h(const Vector& h, int dim){
    ep_check_history_size(h, dim);
    Mat3 C=hpx::eye3<double>();
    C[0][0]+=h[0]; C[1][1]+=h[1]; C[2][2]+=h[2];
    C[0][1]+=h[3]; C[1][0]+=h[3];
    if (dim == 3) {
        // AceGen order: b5 -> C02, b6 -> C12.
        C[0][2]+=h[4]; C[2][0]+=h[4];
        C[1][2]+=h[5]; C[2][1]+=h[5];
    }
    return C;
}
static double gamma_from_h(const Vector& h, int dim){ ep_check_history_size(h, dim); return h[ep_gamma_index(dim)]; }

template <typename T> static Mat3T<T> Cgpi_from_hT(const std::vector<T>& h, int dim){
    ep_check_history_sizeT(h, dim);
    auto C=hpx::eye3<T>();
    C[0][0]+=h[0]; C[1][1]+=h[1]; C[2][2]+=h[2];
    C[0][1]+=h[3]; C[1][0]+=h[3];
    if (dim == 3) {
        // AceGen order: b5 -> C02, b6 -> C12.
        C[0][2]+=h[4]; C[2][0]+=h[4];
        C[1][2]+=h[5]; C[2][1]+=h[5];
    }
    return C;
}

// Project the local tensor equation Z=0 onto the independent history basis.
// Unknown ordering:
//   2D: [C00, C11, C22, C01, gamma]
//   3D: [C00, C11, C22, C01, C02, C12, gamma]
//
// Important compatibility note:
//   The residual components are selected in the same order as the symmetric
//   Cgpi history variables. Do not symmetrize Z again here: Cgpi is already
//   represented by independent symmetric components, so the local equations are
//   matched directly component-by-component.
//
//   2D: [Z00, Z11, Z22, Z01]
//   3D: [Z00, Z11, Z22, Z01, Z02, Z12]
static Vector ep_Z_residual_components(const Mat3& Z, int dim) {
    if (dim == 2) return {Z[0][0], Z[1][1], Z[2][2], Z[0][1]};
    if (dim == 3) return {Z[0][0], Z[1][1], Z[2][2], Z[0][1], Z[0][2], Z[1][2]};
    throw std::runtime_error("Elasto-plastic weak form supports only 2D/3D.");
}

template <typename T>
static std::vector<T> ep_Z_residual_componentsT(const Mat3T<T>& Z, int dim) {
    if (dim == 2) return {Z[0][0], Z[1][1], Z[2][2], Z[0][1]};
    if (dim == 3) return {Z[0][0], Z[1][1], Z[2][2], Z[0][1], Z[0][2], Z[1][2]};
    throw std::runtime_error("Elasto-plastic weak form supports only 2D/3D.");
}


static Mat3 tau_from_be(const Mat3& be,const EPMaterial& mat){
    const double I3=hpx::det3<double>(be); if(I3<=0.0) throw std::runtime_error("Non-positive det(be) in elasto-plastic model.");
    const double mu=mat.E/(2.0*(1.0+mat.nu));
    const double lambda=mat.E*mat.nu/((1.0+mat.nu)*(1.0-2.0*mat.nu));
    const Mat3 I=hpx::eye3<double>();
    Mat3 tau=hpx::zero3<double>();
    // AceGen WSE = 0.5*mu*(I1-3-log(I3)) + 0.25*lambda*(I3-1-log(I3)).
    // Therefore tau = 2*be*dW/dbe = mu*(be-I) + 0.5*lambda*(I3-1)*I.
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
        tau[i][j]=mu*(be[i][j]-I[i][j]) + 0.5*lambda*(I3-1.0)*I[i][j];
    }
    return tau;
}

template <typename T> static Mat3T<T> tau_from_beT(const Mat3T<T>& be,const EPMaterial& mat){
    const T I3=hpx::det3<T>(be); if(val(I3)<=0.0) throw std::runtime_error("Non-positive det(be) in AD elasto-plastic model.");
    const double mu=mat.E/(2.0*(1.0+mat.nu));
    const double lambda=mat.E*mat.nu/((1.0+mat.nu)*(1.0-2.0*mat.nu));
    auto I=hpx::eye3<T>(); auto tau=hpx::zero3<T>();
    // AceGen WSE = 0.5*mu*(I1-3-log(I3)) + 0.25*lambda*(I3-1-log(I3)).
    // Therefore tau = 2*be*dW/dbe = mu*(be-I) + 0.5*lambda*(I3-1)*I.
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
        tau[i][j]=T(mu)*(be[i][j]-I[i][j]) + T(0.5*lambda)*(I3-T(1.0))*I[i][j];
    }
    return tau;
}

static Mat3 dev3(const Mat3& A){ Mat3 D=A; const double m=hpx::tr3<double>(A)/3.0; for(int i=0;i<3;++i) D[i][i]-=m; return D; }
static double yield_q(const Mat3& tau){ const Mat3 s=dev3(tau); double ss=0.0; for(int i=0;i<3;++i) for(int j=0;j<3;++j) ss+=s[i][j]*s[i][j]; return std::sqrt(1.5*ss); }
static double sigma_y(double gamma,const EPMaterial& mat){ return mat.sigma_y + mat.Kf*gamma + mat.Yinf*(1.0-std::exp(-mat.delta*gamma)); }
static Mat3 yield_A(const Mat3& tau){ const Mat3 s=dev3(tau); const double q=yield_q(tau); Mat3 A=hpx::zero3<double>(); if(q<1e-14) return A; for(int i=0;i<3;++i) for(int j=0;j<3;++j) A[i][j]=1.5*s[i][j]/q; return A; }

template <typename T> static Mat3T<T> dev3T(const Mat3T<T>& A){ auto D=A; const T m=hpx::tr3<T>(A)/T(3.0); for(int i=0;i<3;++i) D[i][i]-=m; return D; }
template <typename T> static T yield_qT(const Mat3T<T>& tau){ using std::sqrt; const auto s=dev3T<T>(tau); T ss=0.0; for(int i=0;i<3;++i) for(int j=0;j<3;++j) ss+=s[i][j]*s[i][j]; return sqrt(T(1.5)*ss); }
template <typename T> static T sigma_yT(const T& gamma,const EPMaterial& mat){ using std::exp; return T(mat.sigma_y) + T(mat.Kf)*gamma + T(mat.Yinf)*(T(1.0)-exp(T(-mat.delta)*gamma)); }
template <typename T> static Mat3T<T> yield_AT(const Mat3T<T>& tau){ const auto s=dev3T<T>(tau); const T q=yield_qT<T>(tau); auto A=hpx::zero3<T>(); if(val(q)<1e-14) return A; for(int i=0;i<3;++i) for(int j=0;j<3;++j) A[i][j]=T(1.5)*s[i][j]/q; return A; }

static double yield_function(const Mat3& F,const Vector& h,const EPMaterial& mat,int dim){
    const Mat3 Cgpi=Cgpi_from_h(h, dim);
    const Mat3 be=hpx::mul3<double>(hpx::mul3<double>(F,Cgpi),hpx::transpose3<double>(F));
    const Mat3 tau=tau_from_be(be,mat);
    return yield_q(tau)-sigma_y(gamma_from_h(h, dim),mat);
}

// Local plastic residual Q(F, h) = 0.
// F is fixed during the local Newton solve; h is the unknown history vector.
static Vector local_Qg_residual(const Mat3& F,const Vector& h_n,const Vector& h,const EPMaterial& mat,int dim){
    const Mat3 Cgpi=Cgpi_from_h(h, dim);
    const Mat3 Cgpin=Cgpi_from_h(h_n, dim);
    const Mat3 be=hpx::mul3<double>(hpx::mul3<double>(F,Cgpi),hpx::transpose3<double>(F));
    const Mat3 tau=tau_from_be(be,mat);
    const double Phi=yield_q(tau)-sigma_y(gamma_from_h(h, dim),mat);
    const Mat3 A=yield_A(tau);
    const double dgam=gamma_from_h(h, dim)-gamma_from_h(h_n, dim);
    const Mat3 M=hpx::scale3<double>(A,-2.0*dgam);
    const Mat3 Z=hpx::sub3<double>(hpx::mul3<double>(F,Cgpi), hpx::mul3<double>(hpx::mul3<double>(expm3(M),F),Cgpin));
    Vector Q = ep_Z_residual_components(Z, dim);
    Q.push_back(Phi);
    if (static_cast<int>(Q.size()) != ep_required_history_size(dim)) {
        throw std::runtime_error("Internal error: elasto-plastic local residual size does not match history size.");
    }
    return Q;
}

template <typename T> static Mat3T<T> first_piolaT(const Mat3T<T>& F,const std::vector<T>& h,const EPMaterial& mat,int dim){
    const auto Cgpi=Cgpi_from_hT<T>(h, dim);
    const auto be=hpx::mul3<T>(hpx::mul3<T>(F,Cgpi),hpx::transpose3<T>(F));
    const auto tau=tau_from_beT<T>(be,mat);
    return hpx::mul3<T>(tau,hpx::transpose3<T>(hpx::inv3<T>(F)));
}

template <std::size_t N>
using EPHpxAD = hpx::AD1<double, N>;

template <std::size_t N>
static std::vector<EPHpxAD<N>> local_Qg_hpx_full(
    const Mat3T<EPHpxAD<N>>& F,
    const std::vector<EPHpxAD<N>>& h,
    const Vector& h_n,
    const EPMaterial& mat,
    int dim
) {
    const int nh = ep_required_history_size(dim);
    const int igamma = ep_gamma_index(dim);
    std::vector<EPHpxAD<N>> hnad(nh);
    for(int i=0;i<nh;++i) hnad[i]=EPHpxAD<N>(h_n[i]);
    const auto Cgpi=Cgpi_from_hT<EPHpxAD<N>>(h, dim);
    const auto Cgpin=Cgpi_from_hT<EPHpxAD<N>>(hnad, dim);
    const auto be=hpx::mul3<EPHpxAD<N>>(hpx::mul3<EPHpxAD<N>>(F,Cgpi),hpx::transpose3<EPHpxAD<N>>(F));
    const auto tau=tau_from_beT<EPHpxAD<N>>(be,mat);
    const EPHpxAD<N> Phi=yield_qT<EPHpxAD<N>>(tau)-sigma_yT<EPHpxAD<N>>(h[igamma],mat);
    const auto A=yield_AT<EPHpxAD<N>>(tau);
    const EPHpxAD<N> dgam=h[igamma]-hnad[igamma];
    const auto M=hpx::scale3<EPHpxAD<N>>(A,EPHpxAD<N>(-2.0)*dgam);
    const auto Z=hpx::sub3<EPHpxAD<N>>(
        hpx::mul3<EPHpxAD<N>>(F,Cgpi),
        hpx::mul3<EPHpxAD<N>>(hpx::mul3<EPHpxAD<N>>(expm3T<EPHpxAD<N>>(M),F),Cgpin)
    );
    std::vector<EPHpxAD<N>> Q = ep_Z_residual_componentsT(Z, dim);
    Q.push_back(Phi);
    if (static_cast<int>(Q.size()) != nh) {
        throw std::runtime_error("Internal error: elasto-plastic AD local residual size does not match history size.");
    }
    return Q;
}

template <std::size_t N>
static Matrix local_Jh_hpx_impl(const Mat3& Fd,const Vector& h_n,const Vector& h,const EPMaterial& mat,int dim){
    const int nh=ep_required_history_size(dim);
    using AD = EPHpxAD<N>;
    std::vector<AD> had(nh);
    for(int i=0;i<nh;++i) had[i]=AD::variable(h[i], i);
    Mat3T<AD> F=hpx::zero3<AD>();
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) F[i][j]=AD(Fd[i][j]);
    const auto Q = local_Qg_hpx_full<N>(F, had, h_n, mat, dim);
    Matrix J(nh,Vector(nh,0.0));
    for(int i=0;i<nh;++i) for(int j=0;j<nh;++j) J[i][j]=Q[i].grad[j];
    return J;
}

static Matrix local_Jh_hpx(const Mat3& Fd,const Vector& h_n,const Vector& h,const EPMaterial& mat,int dim){
    if (dim == 2) return local_Jh_hpx_impl<5>(Fd, h_n, h, mat, dim);
    if (dim == 3) return local_Jh_hpx_impl<7>(Fd, h_n, h, mat, dim);
    throw std::runtime_error("Elasto-plastic weak form supports only 2D/3D.");
}

struct PlasticUpdateResult { Vector h; bool plastic=false; double final_local_residual=0.0; int local_newton_iterations=0; double state_trial=0.0; };

// Return-mapping/local Newton update at one Gauss point.
// The accepted result is stored as h_trial and committed by the global solver
// only after the time step is accepted.
static PlasticUpdateResult update_plastic_gp(const Mat3& F,const Vector& h_n,double state_n,const EPMaterial& mat,int dim){
    (void)state_n;
    PlasticUpdateResult out; out.h=h_n;
    const double Ftrial=yield_function(F,h_n,mat,dim);
    const double tol=std::max(mat.local_tol,1e-12);
    if(Ftrial<=tol){ out.plastic=false; out.final_local_residual=0.0; out.local_newton_iterations=0; out.state_trial=0.0; return out; }
    out.plastic=true;
    Vector h=h_n;
    for(int it=0; it<std::max(1,mat.local_max_iter); ++it){
        Vector Q=local_Qg_residual(F,h_n,h,mat,dim);
        Matrix J=local_Jh_hpx(F,h_n,h,mat,dim);
        Vector rhs(Q.size()); for(int i=0;i<static_cast<int>(Q.size());++i) rhs[i]=-Q[i];
        Vector dh=hpx::solve_local_dense_lu(J,rhs,"local elasto-plastic Jacobian");
        double alpha=1.0;
        const double q0=norm_vec(Q);
        for(int ls=0; ls<8; ++ls){
            Vector ht=h; for(int i=0;i<ep_required_history_size(dim);++i) ht[i]+=alpha*dh[i];
            if(hpx::det3<double>(Cgpi_from_h(ht, dim))>0.0){
                Vector Qt=local_Qg_residual(F,h_n,ht,mat,dim);
                if(norm_vec(Qt)<=q0 || alpha<1e-3){ h=ht; break; }
            }
            alpha*=0.5;
        }
        Vector Qnew=local_Qg_residual(F,h_n,h,mat,dim);
        const double qn=norm_vec(Qnew);
        if(qn<tol || norm_vec(dh)<tol){
            out.h=h; out.final_local_residual=qn; out.local_newton_iterations=it+1; out.state_trial=1000.0+qn; return out;
        }
    }
    throw std::runtime_error("Local elasto-plastic Newton did not converge.");
}



// -----------------------------------------------------------------------------
// Weak-scalar AD extraction for elasto-plasticity
// -----------------------------------------------------------------------------
// The local plastic update h_{n+1} is still solved with double-valued Newton.
// Its algorithmic dependence on the current deformation gradient is injected
// into the AD path through the implicit-function relation
//      Q(F, h(F)) = 0,  dh/dF = -Q_h^{-1} Q_F.
// Then the scalar weak form
//      G = int P(F(u), h(F(u))) : Grad(delta u) dV
// is differentiated as
//      R_i  = dG/d(delta u_i),
//      K_ij = dR_i/du_j.
// Compute dh/dF from Q(F,h)=0 by implicit differentiation.
// This is the key step that turns a double-valued local Newton update into a
// consistent global tangent contribution.
template <std::size_t N>
static Matrix implicit_dh_dF_hpx_impl(
    const Mat3& Fd,
    const Vector& h_n,
    const Vector& h,
    bool plastic,
    const EPMaterial& mat,
    int dim
) {
    const int nf = dim * dim;
    const int nh = ep_required_history_size(dim);
    Matrix dh_dF(nh, Vector(nf, 0.0));
    if (!plastic) return dh_dF;

    using AD = EPHpxAD<N>;
    Mat3T<AD> F = hpx::zero3<AD>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) F[i][j] = AD(Fd[i][j]);
    }
    for (int i = 0; i < dim; ++i) {
        for (int A = 0; A < dim; ++A) {
            F[i][A] = AD::variable(Fd[i][A], i * dim + A);
        }
    }

    std::vector<AD> had(nh);
    for (int i = 0; i < nh; ++i) had[i] = AD::variable(h[i], nf + i);

    const auto Q = local_Qg_hpx_full<N>(F, had, h_n, mat, dim);
    Matrix Q_F(nh, Vector(nf, 0.0));
    Matrix Q_h(nh, Vector(nh, 0.0));
    for (int i = 0; i < nh; ++i) {
        for (int j = 0; j < nf; ++j) Q_F[i][j] = Q[i].grad[j];
        for (int j = 0; j < nh; ++j) Q_h[i][j] = Q[i].grad[nf + j];
    }

    const auto Q_h_lu = hpx::factorize_local_dense_lu(Q_h, "local elasto-plastic implicit derivative Jacobian");
    for (int jf = 0; jf < nf; ++jf) {
        Vector rhs(nh, 0.0);
        for (int i = 0; i < nh; ++i) rhs[i] = -Q_F[i][jf];
        Vector col = hpx::solve_local_dense_lu(Q_h_lu, rhs);
        for (int ih = 0; ih < nh; ++ih) dh_dF[ih][jf] = col[ih];
    }
    return dh_dF;
}

static Matrix implicit_dh_dF_hpx(
    const Mat3& Fd,
    const Vector& h_n,
    const Vector& h,
    bool plastic,
    const EPMaterial& mat,
    int dim
) {
    if (dim == 2) return implicit_dh_dF_hpx_impl<9>(Fd, h_n, h, plastic, mat, dim);
    if (dim == 3) return implicit_dh_dF_hpx_impl<16>(Fd, h_n, h, plastic, mat, dim);
    throw std::runtime_error("Elasto-plastic weak form supports only 2D/3D.");
}

static int f_index_dim(int i, int A, int dim) {
    if (i < 0 || i >= dim || A < 0 || A >= dim) {
        throw std::runtime_error("Invalid deformation-gradient component index.");
    }
    return i * dim + A;
}

template <class Scalar>
static std::vector<Scalar> displacement_local_vector_from_layout(
    const LocalFieldView& L,
    const LocalFieldInfo& U,
    const std::vector<Scalar>& x_elem,
    int dim
) {
    std::vector<Scalar> u(dim * U.n_nodes, Scalar(0.0));
    for (int a = 0; a < U.n_nodes; ++a) {
        for (int c = 0; c < dim; ++c) {
            u[dim * a + c] = x_elem[L.dof("displacement", a, c)];
        }
    }
    return u;
}

// Build AD-valued history variables h_ad.
// Values come from the converged local update; derivatives come from dh/dF and
// the AD derivative of F with respect to the element displacement DOFs.
template <class Scalar>
static Mat3T<Scalar> make_algorithmic_h_ad(
    const Mat3T<Scalar>& F_ad,
    const Mat3& F_double,
    const Vector& h_n,
    const Vector& h_trial,
    bool plastic,
    const EPMaterial& mat,
    int dim
) {
    const int nh = ep_required_history_size(dim);
    const int n_outer = F_ad[0][0].size();
    const Matrix dh_dF = implicit_dh_dF_hpx(F_double, h_n, h_trial, plastic, mat, dim);
    using InnerScalar = typename Scalar::value_type;

    std::vector<Scalar> h_ad(nh);
    for (int ih = 0; ih < nh; ++ih) {
        h_ad[ih] = Scalar(InnerScalar(h_trial[ih]));
        h_ad[ih].resize(n_outer);
        for (int j = 0; j < n_outer; ++j) {
            double dh_dx = 0.0;
            for (int i = 0; i < dim; ++i) {
                for (int A = 0; A < dim; ++A) {
                    const int f = f_index_dim(i, A, dim);
                    dh_dx += dh_dF[ih][f] * val(F_ad[i][A].dx(j));
                }
            }
            h_ad[ih].fastAccessDx(j) = InnerScalar(dh_dx);
        }
    }
    return Cgpi_from_hT<Scalar>(h_ad, dim);
}

template <class Scalar>
// -----------------------------------------------------------------------------
// Scalar weak form for the elasto-plastic solid
// -----------------------------------------------------------------------------
// For every Gauss point, this function updates local history, injects algorithmic
// history derivatives, computes P, and accumulates G = int P:Grad(delta_u).
static Scalar weak_scalar_elastoplastic_solid(
    const FieldMesh& mesh,
    int elem_id,
    const std::vector<Scalar>& x_elem,
    const std::vector<Scalar>& test_elem,
    const ProblemParameters& params,
    const ElementState& state
) {
    if (mesh.dim != 2 && mesh.dim != 3) throw std::runtime_error("Elasto-plastic weak form supports only 2D plane strain and full 3D.");
    const int dim = mesh.dim;
    if (!state.material_history) {
        throw std::runtime_error("Elasto-plastic weak form requires Gauss-point history. Add 'history plasticity 5' for 2D or 'history plasticity 7' for 3D to DOFS_info.txt.");
    }
    const int required_history = ep_required_history_size(dim);
    if (state.material_history->gp.empty()) {
        throw std::runtime_error("Elasto-plastic material history is not initialized for this element.");
    }

    const EPMaterial mat = make_ep_material(params);
    const LocalFieldView L = make_local_field_view(mesh, elem_id, {"displacement"});
    const auto& U = L.field("displacement");
    if (static_cast<int>(x_elem.size()) != L.n_local()) throw std::runtime_error("Bad local vector size in elasto-plastic weak form.");
    if (test_elem.size() != x_elem.size()) throw std::runtime_error("Bad test vector size in elasto-plastic weak form.");

    const ElementGeometry geom = make_displacement_element_geometry(mesh, elem_id);
    const auto gps = gauss_rule(U.element_type, params.solver.gauss_order);
    if (state.material_history->gp.size() != gps.size()) {
        state.material_history->gp.resize(gps.size());
    }

    const auto u_ad = displacement_local_vector_from_layout<Scalar>(L, U, x_elem, dim);
    const auto du_test = displacement_local_vector_from_layout<Scalar>(L, U, test_elem, dim);

    // Double-valued copy only for the local return-mapping solve.
    std::vector<double> u_double(dim * U.n_nodes, 0.0);
    for (int i = 0; i < static_cast<int>(u_double.size()); ++i) u_double[i] = val(u_ad[i]);

    Scalar G(0.0);
    for (int g = 0; g < static_cast<int>(gps.size()); ++g) {
        const auto& gp = gps[g];
        const ShapeData shape = compute_shape(U.element_type, gp.xi, gp.eta, gp.zeta);
        const JacobianData J0 = compute_jacobian_data(geom, shape);
        if (J0.detJ <= 0.0) throw std::runtime_error("Negative detJ in elasto-plastic element.");
        const auto dN = compute_dN_dx(shape, J0);
        const double dV = J0.detJ * gp.weight;

        const Mat3 F_double = hpx::deformation_gradient_array<double>(dim, dN, u_double);
        auto& gh = state.material_history->gp[g];
        if (static_cast<int>(gh.h_n.size()) < required_history) {
            gh.h_n.assign(required_history, 0.0);
            gh.h_trial.assign(required_history, 0.0);
            gh.state_n = 0.0;
            gh.state_trial = 0.0;
        }

        PlasticUpdateResult upd;
        try {
            upd = update_plastic_gp(F_double, gh.h_n, gh.state_n, mat, dim);
        } catch (...) {
            if (state.history_diagnostics) ++state.history_diagnostics->local_failure_count;
            throw;
        }
        gh.h_trial = upd.h;
        gh.state_trial = upd.state_trial;
        if (state.history_diagnostics) {
            state.history_diagnostics->observe("plasticity", upd.final_local_residual, upd.local_newton_iterations,
                                               upd.plastic, state.domain_id, elem_id, g);
        }

        const Mat3T<Scalar> F_ad = hpx::deformation_gradient_array<Scalar>(dim, dN, u_ad);
        const Mat3T<Scalar> Cgpi_ad = make_algorithmic_h_ad(F_ad, F_double, gh.h_n, upd.h, upd.plastic, mat, dim);
        const auto be = hpx::mul3<Scalar>(hpx::mul3<Scalar>(F_ad, Cgpi_ad), hpx::transpose3<Scalar>(F_ad));
        const auto tau = tau_from_beT<Scalar>(be, mat);
        const auto P_ad = hpx::mul3<Scalar>(tau, hpx::transpose3<Scalar>(hpx::inv3<Scalar>(F_ad)));

        Mat3T<Scalar> grad_test = hpx::zero3<Scalar>();
        for (int a = 0; a < U.n_nodes; ++a) {
            for (int i = 0; i < dim; ++i) {
                const Scalar ti = du_test[dim * a + i];
                for (int A = 0; A < dim; ++A) {
                    grad_test[i][A] += ti * Scalar(dN[a][A]);
                }
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
    const EPMaterial m = make_ep_material(params);
    os << "  E             : " << m.E << "\n"
       << "  nu            : " << m.nu << "\n"
       << "  sigma_y       : " << m.sigma_y << "\n"
       << "  Kf            : " << m.Kf << "\n"
       << "  Yinf          : " << m.Yinf << "\n"
       << "  delta         : " << m.delta << "\n"
       << "  local_tol  : " << m.local_tol << "\n"
       << "  ep_max_iter   : " << m.local_max_iter << "\n";
}

std::vector<ConstitutivePostprocessOutputSpec> active_constitutive_postprocess_outputs() {
    return {
        {"plastic_gamma", 1},
        {"yield_stress", 1},
        {"von_mises", 1}
    };
}

static bool ep_output_requested(const std::vector<std::string>& requested,const std::string& name){
    return std::find(requested.begin(),requested.end(),name)!=requested.end();
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
    const LocalFieldView L = make_local_field_view(mesh, elem_id, {"displacement"});
    const int n = L.n_local();
    if (static_cast<int>(x_elem.size()) != n) {
        throw std::runtime_error("Bad local vector size in elasto-plastic weak-scalar element system.");
    }

    return hpx::scalar_weak_form_element_system(
        x_elem,
        [&](const auto& x_ad, const auto& test_ad) {
            return weak_scalar_elastoplastic_solid(mesh, elem_id, x_ad, test_ad, params, state);
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
    (void)response_state;
    using Fad = Sacado::Fad::DFad<double>;

    if (mesh.dim != 2 && mesh.dim != 3) {
        throw std::runtime_error("Elasto-plastic multiscale response supports only 2D plane strain and full 3D.");
    }
    if (!history || history->domains.empty()) {
        throw std::runtime_error("Elasto-plastic multiscale response requires RVE material history.");
    }

    const int dim = mesh.dim;
    const int required_history = ep_required_history_size(dim);
    const int ndofs = total_dof_count(mesh);
    if (static_cast<int>(displacement.size()) < ndofs) {
        throw std::runtime_error("Elasto-plastic multiscale response received a displacement vector shorter than mesh DOFs.");
    }

    std::vector<Fad> u_global_ad(ndofs);
    for (int a = 0; a < ndofs; ++a) u_global_ad[a] = Fad(ndofs, a, displacement[a]);

    std::array<std::array<Fad,3>,3> intP_ad{};
    for (auto& row : intP_ad) {
        for (auto& v : row) v = Fad(ndofs, 0.0);
    }

    const EPMaterial mat = make_ep_material(params);
    const DofLayout& layout = get_cached_dof_layout(mesh);
    DomainMaterialHistory& domain_history = const_cast<DomainMaterialHistory&>(history->domains.front());
    if (static_cast<int>(domain_history.elements.size()) < static_cast<int>(mesh.displacement_elements.size())) {
        throw std::runtime_error("Elasto-plastic multiscale response history has fewer elements than the RVE mesh.");
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
            if (J0.detJ <= 0.0) throw std::runtime_error("Negative detJ in elasto-plastic multiscale response.");
            const auto dN = compute_dN_dx(shape, J0);
            const double dV = J0.detJ * gp.weight;

            MaterialPointHistory& gh = element_history.gp[g];
            if (static_cast<int>(gh.h_n.size()) < required_history) {
                gh.h_n.assign(required_history, 0.0);
                gh.h_trial.assign(required_history, 0.0);
                gh.state_n = 0.0;
                gh.state_trial = 0.0;
            }

            const Mat3 F_double = hpx::deformation_gradient_array<double>(dim, dN, u_double);
            const PlasticUpdateResult upd = update_plastic_gp(F_double, gh.h_n, gh.state_n, mat, dim);
            gh.h_trial = upd.h;
            gh.state_trial = upd.state_trial;

            const Mat3T<Fad> F_ad = hpx::deformation_gradient_array<Fad>(dim, dN, ue_ad);
            const Mat3T<Fad> Cgpi_ad = make_algorithmic_h_ad(F_ad, F_double, gh.h_n, upd.h, upd.plastic, mat, dim);
            const auto be = hpx::mul3<Fad>(hpx::mul3<Fad>(F_ad, Cgpi_ad), hpx::transpose3<Fad>(F_ad));
            const auto tau = tau_from_beT<Fad>(be, mat);
            const auto P_ad = hpx::mul3<Fad>(tau, hpx::transpose3<Fad>(hpx::inv3<Fad>(F_ad)));

            volume += dV;
            for (int i = 0; i < 3; ++i) {
                for (int A = 0; A < 3; ++A) {
                    intP_ad[i][A] += P_ad[i][A] * Fad(dV);
                }
            }
        }
    }

    if (volume <= 0.0) {
        throw std::runtime_error("Elasto-plastic multiscale response volume is non-positive.");
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

void compute_constitutive_postprocess(
    const FieldMesh& mesh,
    int elem_id,
    const Vector& x_elem,
    const ProblemParameters& params,
    const ElementState& state,
    const std::vector<std::string>& requested,
    std::vector<ConstitutivePostprocessOutputValues>& outputs
) {
    const bool need_gamma = ep_output_requested(requested,"plastic_gamma");
    const bool need_yield = ep_output_requested(requested,"yield_stress");
    const bool need_vm = ep_output_requested(requested,"von_mises");
    if(!need_gamma && !need_yield && !need_vm) return;
    if(!state.material_history) throw std::runtime_error("Elasto-plastic Gauss output requires material history.");

    const LocalFieldView L = make_local_field_view(mesh, elem_id, {"displacement"});
    const auto& U = L.field("displacement");
    const int dim = L.dim;
    const int required_history = ep_required_history_size(dim);
    const ElementGeometry geom = make_displacement_element_geometry(mesh, elem_id);
    const auto gps = gauss_rule(U.element_type, params.solver.gauss_order);
    const auto u = displacement_local_vector_from_layout<double>(L,U,x_elem,dim);
    const EPMaterial mat = make_ep_material(params);

    ConstitutivePostprocessOutputValues gamma_out{"plastic_gamma",1,{}};
    ConstitutivePostprocessOutputValues yield_out{"yield_stress",1,{}};
    ConstitutivePostprocessOutputValues vm_out{"von_mises",1,{}};
    for(int g=0; g<static_cast<int>(gps.size()); ++g){
        const auto& gp = gps[g];
        const ShapeData shape = compute_shape(U.element_type,gp.xi,gp.eta,gp.zeta);
        const JacobianData J0 = compute_jacobian_data(geom,shape);
        if(J0.detJ<=0.0) throw std::runtime_error("Negative detJ while exporting elasto-plastic Gauss data.");
        const auto dN = compute_dN_dx(shape,J0);
        const Mat3 F = hpx::deformation_gradient_array<double>(dim,dN,u);
        const auto& gh = state.material_history->gp[g];
        const Vector& h = gh.h_n.empty() ? gh.h_trial : gh.h_n;
        if(static_cast<int>(h.size()) < required_history) throw std::runtime_error("Bad plasticity history size while exporting Gauss data.");
        const double gamma = gamma_from_h(h,dim);
        if(need_gamma) gamma_out.values.push_back(gamma);
        if(need_yield) yield_out.values.push_back(sigma_y(gamma,mat));
        if(need_vm){
            const Mat3 Cgpi = Cgpi_from_h(h,dim);
            const Mat3 be = hpx::mul3<double>(hpx::mul3<double>(F,Cgpi),hpx::transpose3<double>(F));
            const Mat3 tau = tau_from_be(be,mat);
            vm_out.values.push_back(yield_q(tau));
        }
    }
    if(need_gamma) outputs.push_back(std::move(gamma_out));
    if(need_yield) outputs.push_back(std::move(yield_out));
    if(need_vm) outputs.push_back(std::move(vm_out));
}
