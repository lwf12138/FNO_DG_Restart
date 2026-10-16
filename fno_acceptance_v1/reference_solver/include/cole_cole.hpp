#pragma once

#include <cmath>

#include <complex>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace colecole {

using Real = double;
using std::abs; using std::sqrt; using std::pow;
using std::sin; using std::cos; using std::atan; using std::acos; using std::log;
using Complex = std::complex<Real>;
using Vec = std::vector<Real>;
using CVec = std::vector<Complex>;

struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<Real> a;

    Matrix() = default;
    Matrix(int r, int c, Real value = 0);
    Real& operator()(int i, int j);
    const Real& operator()(int i, int j) const;
};

struct CMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<Complex> a;

    CMatrix() = default;
    CMatrix(int r, int c, Complex value = Complex(0));
    Complex& operator()(int i, int j);
    const Complex& operator()(int i, int j) const;
};

struct LUFactor {
    int n = 0;
    Matrix lu;
    std::vector<int> piv;
    bool singular = false;

    explicit LUFactor(const Matrix& A);
    Vec solve(const Vec& b) const;
    Matrix solve_matrix(const Matrix& B) const;
};

struct Config {
    bool sourceFree = false;
    // Acceptance adapter: complete [field, cell, monic-Legendre-mode] coefficients.
    bool useInitialModalCoefficients = false;
    Vec initialModalCoefficients;
    Vec initialElectricCoeffs, initialMagneticCoeffs;
    Vec queryTimes; // Output only; does not enter the numerical scheme.
    Real epsilon0 = 1;
    Real mu0 = 1;
    Real epsilon_inf = 2;
    Real epsilon_rs = 3;
    Real tau0 = 1;
    Real alpha = Real(0.6);
    Real xa = 0;
    Real xb;
    Real T = 1;
    int k = 1;
    int kMaxForExact = 3;
    int timePower = 10;
    // Uniform amplitude applied to E, H, P and all manufactured sources.
    // Keeping this at one preserves the original experiment exactly.
    Real solutionScale = 1;
    int N = 4;
    std::vector<int> NList = {4, 8, 16, 32, 64};
    int NumGLP = 24;
    std::string timeScheme = "RK-CQ";
    std::string caputoScheme = "RK-CQ";
    std::string rkMethod = "RadauIIA-3";
    int rkCqFftFactor = 8;
    int rkCqMinFftLength = 64;
    Real rkCqRadiusExponent = Real(0.5);
    Real rkCqTargetEps = Real(1e-6);
    // The RK-CQ time recursion stays causal.  This only selects how its
    // already-known polarization history is summed at a fixed time step.
    std::string rkHistoryAlgorithm = "block-fft";
    int rkHistoryFftMinBlock = 128;
    int rkHistoryVerifySteps = 8;
    Real dt = Real(0.002);
    std::string initialMethod = "special";
    int correctionLevel = 1;
    bool computeCorrectionErrors = false;
    bool verbose = true;
    bool rkDiagnostic = false;
    std::string resultsDir = "results";
    // Optional shared directory for immutable RK-CQ weight tables.  This is
    // useful for Slurm arrays whose cases have the same Nt and RK settings.
    std::string rkCqCacheDir;
};

struct Mesh {
    int N = 0;
    Vec x;
    Vec xL;
    Vec xR;
    Vec h;
    Vec J;
    Vec xc;
};

struct Basis {
    int k = 0;
    int dim = 0;
    Vec xi;
    Matrix phi;
    Matrix dphi;
    Vec mm;
    Vec phiR;
    Vec phiL;
};

struct Quad {
    Vec x;
    Vec w;
};

struct RKMethod {
    std::string name;
    int order = 0;
    int s = 0;
    Matrix A;
    Vec b;
    Vec c;
    Matrix Ainv;
    bool stifflyAccurate = false;
};

struct RKWeightInfo {
    std::string algorithm = "fft";
    int fftFactor = 8;
    int minFftLength = 64;
    Real radiusExponent = Real(0.5);
    Real targetEps = 0;
    Real radius = 0;
    int fftLength = 0;
    int nmax = 0;
    Real roundoffAmplification = 0;
    Real maxImaginaryWeightPart = 0;
    bool cacheHit = false;
};

struct RKWeights {
    int s = 0;
    int nmax = 0;
    std::vector<Matrix> W;
    RKWeightInfo info;
};

struct SpatialOp {
    std::vector<Matrix> diag;
    std::vector<Matrix> lower;
    std::vector<Matrix> upper;
};

struct MaxwellOp {
    SpatialOp SEE;
    SpatialOp SEH;
    SpatialOp SHE;
    SpatialOp SHH;
    Vec massDiag;
};

struct Exact {
    Real A = 0;
    Real alpha = 0;
    Real tau0 = 1;
    int p = 0;
    Real scale = 1;

    Real g(const Real& t) const;
    Real gd(const Real& t, int m) const;
    Real r(const Real& t) const;
    Real rd(const Real& t, int m) const;
    Real q(const Real& t) const;
    Real qd(const Real& t) const;
    Real caputo_q(const Real& t) const;
    Real E(const Real& x, const Real& t) const;
    Real H(const Real& x, const Real& t) const;
    Real P(const Real& x, const Real& t) const;
    Real Et(const Real& x, const Real& t, int m) const;
    Real Ht(const Real& x, const Real& t, int m) const;
};

struct InitialModal {
    Matrix E;
    Matrix H;
    Matrix P;
};

struct Correction {
    Matrix wE;
    Matrix wH;
    Matrix wP;
    Matrix wE0;
    Matrix wH0;
    Matrix wE1;
    Matrix wH1;
    Matrix wP1;
    Matrix wP1dt;
    int level = 0;
};

struct Result {
    Vec snapshotTimes;
    std::vector<Vec> snapshotsE, snapshotsH, snapshotsP;
    Matrix E;
    Matrix H;
    Matrix P;
    Mesh mesh;
    Basis basis;
    Quad quad;
    Exact exact;
    Real dt = 0;
    int Nt = 0;
    Config cfg;
    std::string timeScheme = "RK-CQ";
    RKMethod rk;
    RKWeightInfo rkCqInfo;
    std::string rkHistoryAlgorithm = "direct";
    int rkHistoryFftMinBlock = 0;
    Real rkHistoryMaxImaginaryRatio = 0;
    Real rkHistoryMaxDirectDifference = 0;
    double rkWeightWallSeconds = 0;
    double rkTimeLoopWallSeconds = 0;
    double rkHistoryWallSeconds = 0;
    Real maxEndpointDiff = 0;
    Real maxLinearResidual = 0;
};

struct ErrorRow {
    int N = 0;
    Real dt = 0;
    int Nt = 0;
    std::map<std::string, Real> values;
};

struct TimeRow {
    std::string rkMethod;
    Real dt = 0;
    int Nt = 0;
    Real referenceDt = 0;
    int referenceNt = 0;
    Real time_E = 0;
    Real ord_time_E = 0;
    Real time_H = 0;
    Real ord_time_H = 0;
    Real time_P = 0;
    Real ord_time_P = 0;
    Real exact_L2_E = 0;
    Real exact_L2_H = 0;
    Real exact_L2_P = 0;
};

Real pi();
Real nan_value();
bool is_nan(const Real& x);
std::string real_to_string(const Real& x, int digits = 36);
std::string format_dt_tag(const Real& dt);

Config default_config();
Mesh make_mesh_1d(const Config& cfg);
Quad gauss_legendre(int n);
Basis legendre_basis(const Vec& xi, int k);
Exact cole_cole_exact(const Config& cfg);
MaxwellOp assemble_maxwell_dg(const Mesh& mesh, const Basis& basis, const Quad& quad, const Config& cfg);
RKMethod rk_method_coefficients(const std::string& method);
RKWeights cq_weights_rk(const RKMethod& rk, const Real& alpha, int nmax, const Config& cfg);

Matrix project_l2_func(const Mesh& mesh, const Basis& basis, const Quad& quad,
                       const std::function<Real(const Real&)>& func);
Matrix project_radau_func(const Mesh& mesh, const Basis& basis, const Quad& quad,
                          const std::function<Real(const Real&)>& func,
                          const std::string& side);
void project_star_maxwell(const Mesh& mesh, const Basis& basis, const Quad& quad,
                          const Config& cfg, const Exact& exact, const Real& t,
                          int derivativeOrder, Matrix& Estar, Matrix& Hstar);
Vec modal_load(const Mesh& mesh, const Basis& basis, const Quad& quad,
               const std::function<Real(const Real&)>& func);
Vec pack_modal(const Matrix& U);
Matrix unpack_modal(const Vec& v, int N, int d);
Matrix modal_values(const Matrix& U, const Basis& basis);
InitialModal initial_modal(const Mesh& mesh, const Basis& basis, const Quad& quad,
                           const Config& cfg, const Exact& exact, const Real& t);
Correction correction_functions(const Mesh& mesh, const Basis& basis, const Quad& quad,
                                const Config& cfg, const Exact& exact, const Real& t);

Real mittag_leffler(const Real& alpha, const Real& beta, const Real& z,
                    const Real& tol = Real(1e-15), int maxTerms = 4000);
Real prabhakar_mittag_leffler(const Real& alpha, const Real& beta, int gammaPar,
                              const Real& z, const Real& tol = Real(1e-15),
                              int maxTerms = 2000);
Real relaxation_basis_value(const Config& cfg, int relaxationPower,
                            int derivativeOrder, const Real& t);

Result solve_cole_cole_dg(const Config& cfg);
Result solve_cole_cole_dg_rk_cq(const Config& cfg);
std::map<std::string, Real> compute_errors(const Result& result);
Real modal_l2_diff(const Vec& u, const Vec& v, const Mesh& mesh, const Basis& basis);
std::vector<ErrorRow> run_convergence_study(const Config& cfg);

void write_spatial_csv(const std::vector<ErrorRow>& rows, const std::string& path);
void write_spatial_orders_csv(const std::vector<ErrorRow>& rows, const Config& cfg,
                              const std::string& path);
void write_time_csv(const std::vector<TimeRow>& rows, const std::string& path);
void write_time_orders_csv(const std::vector<TimeRow>& rows, const std::string& path);

} // namespace colecole
