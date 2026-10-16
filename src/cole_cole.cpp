#include "cole_cole.hpp"

#include <boost/math/constants/constants.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace colecole {
namespace {

constexpr int VAR_E = 0;
constexpr int VAR_H = 1;
constexpr int VAR_P = 2;
constexpr int DEFAULT_OPENMP_MIN_WORK_ITEMS = 4;

int ceil_to_int(const Real& x) {
    int n = static_cast<int>(x);
    if (Real(n) < x) {
        ++n;
    }
    return n;
}

int next_pow2(int n) {
    int p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

int modal_index(int cell, int mode, int d) {
    return cell*d + mode;
}

bool openmp_kernels_enabled() {
#if defined(COLECOLE_ENABLE_OPENMP) && defined(_OPENMP)
    return std::getenv("COLECOLE_DISABLE_OPENMP") == nullptr;
#else
    return false;
#endif
}

int openmp_min_work_items() {
#if defined(COLECOLE_ENABLE_OPENMP) && defined(_OPENMP)
    const char* env = std::getenv("COLECOLE_OPENMP_MIN_WORK");
    if (env != nullptr) {
        int parsed = std::atoi(env);
        if (parsed > 0) {
            return parsed;
        }
    }
#endif
    return DEFAULT_OPENMP_MIN_WORK_ITEMS;
}

bool omp_parallel_work(int workItems) {
#if defined(COLECOLE_ENABLE_OPENMP) && defined(_OPENMP)
    return openmp_kernels_enabled()
        && workItems >= openmp_min_work_items()
        && omp_get_max_threads() > 1
        && !omp_in_parallel();
#else
    (void)workItems;
    return false;
#endif
}

template <class Func>
void spatial_for(int n, Func func) {
    if (n <= 0) {
        return;
    }
#if defined(COLECOLE_ENABLE_OPENMP) && defined(_OPENMP)
    if (omp_parallel_work(n)) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            func(i);
        }
        return;
    }
#endif
    for (int i = 0; i < n; ++i) {
        func(i);
    }
}

template <class Func>
Real deterministic_cell_sum(int n, Func func) {
    std::vector<Real> local(static_cast<std::size_t>(n), 0);
    spatial_for(n, [&](int i) {
        local[static_cast<std::size_t>(i)] = func(i);
    });
    Real acc = 0;
    for (const Real& value : local) {
        acc += value;
    }
    return acc;
}

int block_index(int var, int stage, int mode, int s, int d) {
    return ((var*s + stage)*d + mode);
}

int pstage_index(int cell, int stage, int mode, int s, int d) {
    return (cell*s + stage)*d + mode;
}

Matrix eye(int n) {
    Matrix I(n, n);
    for (int i = 0; i < n; ++i) {
        I(i, i) = 1;
    }
    return I;
}

CMatrix ceye(int n) {
    CMatrix I(n, n);
    for (int i = 0; i < n; ++i) {
        I(i, i) = Complex(1);
    }
    return I;
}

Matrix transpose(const Matrix& A) {
    Matrix B(A.cols, A.rows);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            B(j, i) = A(i, j);
        }
    }
    return B;
}

Matrix matmul(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) {
        throw std::runtime_error("matmul dimension mismatch");
    }
    Matrix C(A.rows, B.cols);
    for (int i = 0; i < A.rows; ++i) {
        for (int k = 0; k < A.cols; ++k) {
            if (A(i, k) == 0) {
                continue;
            }
            for (int j = 0; j < B.cols; ++j) {
                C(i, j) += A(i, k)*B(k, j);
            }
        }
    }
    return C;
}

CMatrix cmatmul(const CMatrix& A, const CMatrix& B) {
    if (A.cols != B.rows) {
        throw std::runtime_error("cmatmul dimension mismatch");
    }
    CMatrix C(A.rows, B.cols);
    for (int i = 0; i < A.rows; ++i) {
        for (int k = 0; k < A.cols; ++k) {
            if (A(i, k) == Complex(0)) {
                continue;
            }
            for (int j = 0; j < B.cols; ++j) {
                C(i, j) += A(i, k)*B(k, j);
            }
        }
    }
    return C;
}

Matrix mat_sub(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, A.cols);
    spatial_for(A.rows*A.cols, [&](int i) {
        C.a[i] = A.a[i] - B.a[i];
    });
    return C;
}

Matrix mat_add(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, A.cols);
    spatial_for(A.rows*A.cols, [&](int i) {
        C.a[i] = A.a[i] + B.a[i];
    });
    return C;
}

Matrix mat_scale(const Matrix& A, const Real& s) {
    Matrix B(A.rows, A.cols);
    spatial_for(A.rows*A.cols, [&](int i) {
        B.a[i] = s*A.a[i];
    });
    return B;
}

bool any_nonzero(const Matrix& A) {
    for (const Real& v : A.a) {
        if (v != 0) {
            return true;
        }
    }
    return false;
}

Real power_derivative(const Real& t, int p, int m) {
    if (m > p) {
        return 0;
    }
    Real coef = std::tgamma(Real(p + 1))
        / std::tgamma(Real(p - m + 1));
    return coef*pow(t, p - m);
}

Complex cpow_alpha(const Complex& z, const Real& alpha) {
    auto cabs = [](const Complex& w) {
        return sqrt(w.real()*w.real() + w.imag()*w.imag());
    };
    auto atan2_real = [](const Real& y, const Real& x) -> Real {
        if (x > 0) {
            return Real(atan(y/x));
        }
        if (x < 0 && y >= 0) {
            return Real(atan(y/x) + pi());
        }
        if (x < 0 && y < 0) {
            return Real(atan(y/x) - pi());
        }
        if (y > 0) {
            return pi()/2;
        }
        if (y < 0) {
            return -pi()/2;
        }
        return Real(0);
    };
    Real modulus = cabs(z);
    Real arg = atan2_real(z.imag(), z.real());
    Real scaledRe = alpha*log(modulus);
    Real scaledIm = alpha*arg;
    Real e = exp(scaledRe);
    return Complex(e*cos(scaledIm), e*sin(scaledIm));
}

Complex cdet(const CMatrix& A) {
    if (A.rows == 1) {
        return A(0, 0);
    }
    if (A.rows == 2) {
        return A(0, 0)*A(1, 1) - A(0, 1)*A(1, 0);
    }
    if (A.rows == 3) {
        return A(0, 0)*(A(1, 1)*A(2, 2) - A(1, 2)*A(2, 1))
             - A(0, 1)*(A(1, 0)*A(2, 2) - A(1, 2)*A(2, 0))
             + A(0, 2)*(A(1, 0)*A(2, 1) - A(1, 1)*A(2, 0));
    }
    throw std::runtime_error("cdet only supports 1x1, 2x2, and 3x3 matrices");
}

Complex ctrace(const CMatrix& A) {
    Complex t(0);
    for (int i = 0; i < A.rows; ++i) {
        t += A(i, i);
    }
    return t;
}

CMatrix solve_complex_matrix(CMatrix A, const CMatrix& B) {
    const int n = A.rows;
    const int nrhs = B.cols;
    CMatrix X = B;
    std::vector<int> piv(n);
    for (int i = 0; i < n; ++i) {
        piv[i] = i;
    }
    for (int k = 0; k < n; ++k) {
        int p = k;
        Real best = sqrt(A(k, k).real()*A(k, k).real() + A(k, k).imag()*A(k, k).imag());
        for (int i = k + 1; i < n; ++i) {
            Real cand = sqrt(A(i, k).real()*A(i, k).real() + A(i, k).imag()*A(i, k).imag());
            if (cand > best) {
                best = cand;
                p = i;
            }
        }
        if (best == 0) {
            throw std::runtime_error("singular complex matrix");
        }
        if (p != k) {
            for (int j = 0; j < n; ++j) {
                std::swap(A(k, j), A(p, j));
            }
            for (int j = 0; j < nrhs; ++j) {
                std::swap(X(k, j), X(p, j));
            }
        }
        for (int i = k + 1; i < n; ++i) {
            A(i, k) /= A(k, k);
            for (int j = k + 1; j < n; ++j) {
                A(i, j) -= A(i, k)*A(k, j);
            }
            for (int j = 0; j < nrhs; ++j) {
                X(i, j) -= A(i, k)*X(k, j);
            }
        }
    }
    for (int rhs = 0; rhs < nrhs; ++rhs) {
        for (int i = n - 1; i >= 0; --i) {
            Complex sum = X(i, rhs);
            for (int j = i + 1; j < n; ++j) {
                sum -= A(i, j)*X(j, rhs);
            }
            X(i, rhs) = sum/A(i, i);
        }
    }
    return X;
}

std::vector<Complex> durand_kerner_cubic(const Complex& a, const Complex& b, const Complex& c) {
    auto cabs = [](const Complex& w) {
        return sqrt(w.real()*w.real() + w.imag()*w.imag());
    };
    Real radius = 1;
    radius = std::max(radius, cabs(a));
    radius = std::max(radius, cabs(b));
    radius = std::max(radius, cabs(c));
    radius += 1;
    std::vector<Complex> z(3);
    for (int k = 0; k < 3; ++k) {
        Real theta = 2*pi()*Real(k)/Real(3);
        z[k] = radius*Complex(cos(theta), sin(theta));
    }
    const Real tol = Real(1e-14);
    for (int iter = 0; iter < 100; ++iter) {
        Real maxDelta = 0;
        for (int i = 0; i < 3; ++i) {
            Complex p = ((z[i] + a)*z[i] + b)*z[i] + c;
            Complex denom(1);
            for (int j = 0; j < 3; ++j) {
                if (j != i) {
                    denom *= z[i] - z[j];
                }
            }
            if (cabs(denom) == 0) {
                denom = Complex(tol, tol);
            }
            Complex delta = p/denom;
            z[i] -= delta;
            maxDelta = std::max(maxDelta, cabs(delta));
        }
        if (maxDelta < tol) {
            break;
        }
    }
    return z;
}

CMatrix matrix_function_power(const CMatrix& B, const Real& alpha) {
    const int s = B.rows;
    if (s == 1) {
        CMatrix F(1, 1);
        F(0, 0) = cpow_alpha(B(0, 0), alpha);
        return F;
    }

    std::vector<Complex> lambda;
    if (s == 2) {
        Complex tr = ctrace(B);
        Complex det = cdet(B);
        Complex disc = tr*tr - Complex(4)*det;
        Real discAbs = sqrt(disc.real()*disc.real() + disc.imag()*disc.imag());
        Real rootRe = sqrt((discAbs + disc.real())/2);
        Real rootIm = sqrt((discAbs - disc.real())/2);
        if (disc.imag() < 0) {
            rootIm = -rootIm;
        }
        Complex root(rootRe, rootIm);
        lambda = {(tr + root)/Complex(2), (tr - root)/Complex(2)};
    } else if (s == 3) {
        Complex tr = ctrace(B);
        CMatrix B2 = cmatmul(B, B);
        Complex trB2 = ctrace(B2);
        Complex c2 = (tr*tr - trB2)/Complex(2);
        Complex det = cdet(B);
        lambda = durand_kerner_cubic(-tr, c2, -det);
    } else {
        throw std::runtime_error("matrix_function_power supports only s <= 3");
    }

    CMatrix V(s, s);
    CMatrix rhs(s, 1);
    for (int i = 0; i < s; ++i) {
        Complex p(1);
        for (int j = 0; j < s; ++j) {
            V(i, j) = p;
            p *= lambda[i];
        }
        rhs(i, 0) = cpow_alpha(lambda[i], alpha);
    }
    CMatrix coeff = solve_complex_matrix(V, rhs);

    CMatrix F(s, s);
    CMatrix Bpow = ceye(s);
    for (int j = 0; j < s; ++j) {
        for (int r = 0; r < s; ++r) {
            for (int c = 0; c < s; ++c) {
                F(r, c) += coeff(j, 0)*Bpow(r, c);
            }
        }
        Bpow = cmatmul(Bpow, B);
    }
    return F;
}

void fft(CVec& a) {
    const int n = static_cast<int>(a.size());
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        Real ang = -2*pi()/Real(len);
        Complex wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            Complex w(1);
            for (int j = 0; j < len/2; ++j) {
                Complex u = a[i + j];
                Complex v = a[i + j + len/2]*w;
                a[i + j] = u + v;
                a[i + j + len/2] = u - v;
                w *= wlen;
            }
        }
    }
}

void inverse_fft(CVec& a) {
    const Real scale = Real(1)/Real(a.size());
    for (Complex& value : a) {
        value = std::conj(value);
    }
    fft(a);
    for (Complex& value : a) {
        value = std::conj(value)*scale;
    }
}

struct SpatialBlocks {
    SpatialOp SEE;
    SpatialOp SEH;
    SpatialOp SHE;
    SpatialOp SHH;
};

SpatialOp zero_spatial(int N, int d) {
    SpatialOp op;
    op.diag.assign(N, Matrix(d, d));
    if (N > 1) {
        op.lower.assign(N - 1, Matrix(d, d));
        op.upper.assign(N - 1, Matrix(d, d));
    }
    return op;
}

void add_spatial_outer(SpatialOp& op, int rowCell, int colCell, const Vec& left, const Vec& right) {
    Matrix* block = nullptr;
    if (rowCell == colCell) {
        block = &op.diag[rowCell];
    } else if (rowCell == colCell + 1) {
        block = &op.lower[colCell];
    } else if (rowCell + 1 == colCell) {
        block = &op.upper[rowCell];
    } else {
        throw std::runtime_error("unexpected non-neighbor spatial coupling");
    }
    for (int i = 0; i < block->rows; ++i) {
        for (int j = 0; j < block->cols; ++j) {
            (*block)(i, j) += left[i]*right[j];
        }
    }
}

void add_system_block(Matrix& block, int rowVar, int rowStage, int rowMode,
                      int colVar, int colStage, int colMode,
                      int s, int d, const Real& value) {
    block(block_index(rowVar, rowStage, rowMode, s, d),
          block_index(colVar, colStage, colMode, s, d)) += value;
}

std::string cache_key(const RKMethod& rk, const Real& alpha, int nmax, const Config& cfg) {
    std::ostringstream os;
    os << "cache_v5_float64_comparison|" << rk.name << "|" << real_to_string(alpha, 40) << "|" << nmax << "|"
       << cfg.rkCqFftFactor << "|" << cfg.rkCqMinFftLength << "|"
       << real_to_string(cfg.rkCqRadiusExponent, 40) << "|"
       << real_to_string(cfg.rkCqTargetEps, 40);
    return os.str();
}

std::string cache_file_for_key(const Config& cfg, const std::string& key) {
    std::string name;
    name.reserve(key.size());
    for (char c : key) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
            name.push_back(c);
        } else {
            name.push_back('_');
        }
    }
    std::filesystem::path dir = cfg.rkCqCacheDir.empty()
        ? std::filesystem::path(cfg.resultsDir) / "cache"
        : std::filesystem::path(cfg.rkCqCacheDir);
    return (dir / ("rk_weights_" + name + ".csv")).string();
}

bool load_rk_weights_from_disk(const std::string& path, int s, int nmax, RKWeights& weights) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    std::string line;
    if (!std::getline(in, line) || line != "n,row,col,value") {
        return false;
    }
    std::vector<Matrix> W(static_cast<std::size_t>(nmax + 1), Matrix(s, s));
    int count = 0;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string item;
        std::vector<std::string> parts;
        while (std::getline(ss, item, ',')) {
            parts.push_back(item);
        }
        if (parts.size() != 4) {
            return false;
        }
        int n = std::stoi(parts[0]);
        int row = std::stoi(parts[1]);
        int col = std::stoi(parts[2]);
        if (n < 0 || n > nmax || row < 0 || row >= s || col < 0 || col >= s) {
            return false;
        }
        W[n](row, col) = std::stod(parts[3]);
        ++count;
    }
    if (count != (nmax + 1)*s*s) {
        return false;
    }
    weights.W = std::move(W);
    return true;
}

void save_rk_weights_to_disk(const std::string& path, const RKWeights& weights) {
    std::filesystem::path finalPath(path);
    std::filesystem::create_directories(finalPath.parent_path());
    std::filesystem::path tmpPath = finalPath;
    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    tmpPath += ".tmp." + std::to_string(stamp);
    std::ofstream out(tmpPath);
    out << "n,row,col,value\n";
    for (int n = 0; n <= weights.nmax; ++n) {
        for (int row = 0; row < weights.s; ++row) {
            for (int col = 0; col < weights.s; ++col) {
                out << n << "," << row << "," << col << ","
                    << real_to_string(weights.W[n](row, col),
                                      std::numeric_limits<Real>::max_digits10) << "\n";
            }
        }
    }
    out.close();
    std::error_code ec;
    std::filesystem::rename(tmpPath, finalPath, ec);
    if (ec) {
        // Another array task may have completed the same cache file first.
        std::filesystem::remove(finalPath, ec);
        ec.clear();
        std::filesystem::rename(tmpPath, finalPath, ec);
        if (ec) {
            std::filesystem::remove(tmpPath);
            throw std::runtime_error("failed to save RK-CQ weights cache: " + path);
        }
    }
}

} // namespace

Matrix::Matrix(int r, int c, Real value) : rows(r), cols(c), a(static_cast<std::size_t>(r*c), value) {}

Real& Matrix::operator()(int i, int j) {
    return a[static_cast<std::size_t>(i*cols + j)];
}

const Real& Matrix::operator()(int i, int j) const {
    return a[static_cast<std::size_t>(i*cols + j)];
}

CMatrix::CMatrix(int r, int c, Complex value) : rows(r), cols(c), a(static_cast<std::size_t>(r*c), value) {}

Complex& CMatrix::operator()(int i, int j) {
    return a[static_cast<std::size_t>(i*cols + j)];
}

const Complex& CMatrix::operator()(int i, int j) const {
    return a[static_cast<std::size_t>(i*cols + j)];
}

LUFactor::LUFactor(const Matrix& A) : n(A.rows), lu(A), piv(static_cast<std::size_t>(A.rows)) {
    if (A.rows != A.cols) {
        throw std::runtime_error("LUFactor requires a square matrix");
    }
    for (int i = 0; i < n; ++i) {
        piv[i] = i;
    }
    for (int k = 0; k < n; ++k) {
        int p = k;
        Real best = abs(lu(k, k));
        for (int i = k + 1; i < n; ++i) {
            Real cand = abs(lu(i, k));
            if (cand > best) {
                best = cand;
                p = i;
            }
        }
        if (best == 0) {
            singular = true;
            return;
        }
        if (p != k) {
            for (int j = 0; j < n; ++j) {
                std::swap(lu(k, j), lu(p, j));
            }
            std::swap(piv[k], piv[p]);
        }
        for (int i = k + 1; i < n; ++i) {
            lu(i, k) /= lu(k, k);
            for (int j = k + 1; j < n; ++j) {
                lu(i, j) -= lu(i, k)*lu(k, j);
            }
        }
    }
}

Vec LUFactor::solve(const Vec& b) const {
    if (singular) {
        throw std::runtime_error("singular LU factor");
    }
    Vec x(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < n; ++i) {
        x[i] = b[piv[i]];
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            x[i] -= lu(i, j)*x[j];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j) {
            x[i] -= lu(i, j)*x[j];
        }
        x[i] /= lu(i, i);
    }
    return x;
}

Matrix LUFactor::solve_matrix(const Matrix& B) const {
    Matrix X(B.rows, B.cols);
    for (int j = 0; j < B.cols; ++j) {
        Vec rhs(static_cast<std::size_t>(B.rows));
        for (int i = 0; i < B.rows; ++i) {
            rhs[i] = B(i, j);
        }
        Vec sol = solve(rhs);
        for (int i = 0; i < B.rows; ++i) {
            X(i, j) = sol[i];
        }
    }
    return X;
}

Real pi() {
    static const Real value = boost::math::constants::pi<Real>();
    return value;
}

Real nan_value() {
    return std::numeric_limits<Real>::quiet_NaN();
}

bool is_nan(const Real& x) {
    return x != x;
}

std::string real_to_string(const Real& x, int digits) {
    if (is_nan(x)) {
        return "NaN";
    }
    std::ostringstream os;
    os << std::setprecision(digits) << x;
    return os.str();
}

std::string format_dt_tag(const Real& dt) {
    std::ostringstream os;
    os << std::scientific << std::setprecision(6) << static_cast<long double>(dt);
    std::string raw = os.str();
    std::size_t epos = raw.find('e');
    std::string mantissa = raw.substr(0, epos);
    std::string exponent = raw.substr(epos + 1);
    while (!mantissa.empty() && mantissa.back() == '0') {
        mantissa.pop_back();
    }
    if (!mantissa.empty() && mantissa.back() == '.') {
        mantissa.pop_back();
    }
    std::replace(mantissa.begin(), mantissa.end(), '.', 'p');
    return mantissa + "e" + exponent;
}

Config default_config() {
    Config cfg;
    cfg.xb = 2*pi();
    cfg.timePower = 2*cfg.kMaxForExact + 4;
    return cfg;
}

Mesh make_mesh_1d(const Config& cfg) {
    Mesh mesh;
    mesh.N = cfg.N;
    mesh.x.resize(static_cast<std::size_t>(cfg.N + 1));
    mesh.xL.resize(static_cast<std::size_t>(cfg.N));
    mesh.xR.resize(static_cast<std::size_t>(cfg.N));
    mesh.h.resize(static_cast<std::size_t>(cfg.N));
    mesh.J.resize(static_cast<std::size_t>(cfg.N));
    mesh.xc.resize(static_cast<std::size_t>(cfg.N));
    Real h = (cfg.xb - cfg.xa)/Real(cfg.N);
    for (int i = 0; i <= cfg.N; ++i) {
        mesh.x[i] = cfg.xa + Real(i)*h;
    }
    for (int i = 0; i < cfg.N; ++i) {
        mesh.xL[i] = mesh.x[i];
        mesh.xR[i] = mesh.x[i + 1];
        mesh.h[i] = mesh.xR[i] - mesh.xL[i];
        mesh.J[i] = mesh.h[i]/2;
        mesh.xc[i] = (mesh.xL[i] + mesh.xR[i])/2;
    }
    return mesh;
}

Quad gauss_legendre(int n) {
    if (n <= 0) {
        throw std::runtime_error("gauss_legendre: n must be positive");
    }
    Quad quad;
    quad.x.assign(static_cast<std::size_t>(n), 0);
    quad.w.assign(static_cast<std::size_t>(n), 0);
    const int m = (n + 1)/2;
    const Real tol = Real(1e-15);
    for (int i = 0; i < m; ++i) {
        Real z = cos(pi()*(Real(i) + Real(0.75))/(Real(n) + Real(0.5)));
        Real z1 = 0;
        Real pp = 0;
        for (int iter = 0; iter < 100; ++iter) {
            Real p1 = 1;
            Real p2 = 0;
            for (int j = 1; j <= n; ++j) {
                Real p3 = p2;
                p2 = p1;
                p1 = ((Real(2*j - 1)*z*p2) - Real(j - 1)*p3)/Real(j);
            }
            pp = Real(n)*(z*p1 - p2)/(z*z - 1);
            z1 = z;
            z = z1 - p1/pp;
            if (abs(z - z1) < tol) {
                break;
            }
        }
        Real weight = 2/((1 - z*z)*pp*pp);
        quad.x[i] = -z;
        quad.x[n - 1 - i] = z;
        quad.w[i] = weight;
        quad.w[n - 1 - i] = weight;
    }
    return quad;
}

Basis legendre_basis(const Vec& xi, int k) {
    Basis basis;
    basis.k = k;
    basis.dim = k + 1;
    basis.xi = xi;
    const int nq = static_cast<int>(xi.size());
    const int d = basis.dim;
    basis.phi = Matrix(nq, d);
    basis.dphi = Matrix(nq, d);
    for (int q = 0; q < nq; ++q) {
        basis.phi(q, 0) = 1;
        basis.dphi(q, 0) = 0;
        if (d >= 2) {
            basis.phi(q, 1) = xi[q];
            basis.dphi(q, 1) = 1;
        }
    }
    for (int n = 1; n <= k - 1; ++n) {
        Real beta = Real(n*n)/Real(4*n*n - 1);
        for (int q = 0; q < nq; ++q) {
            basis.phi(q, n + 1) = xi[q]*basis.phi(q, n) - beta*basis.phi(q, n - 1);
            basis.dphi(q, n + 1) = basis.phi(q, n) + xi[q]*basis.dphi(q, n)
                - beta*basis.dphi(q, n - 1);
        }
    }
    Vec c(static_cast<std::size_t>(d), 0);
    c[0] = 1;
    if (d >= 2) {
        c[1] = 1;
    }
    for (int j = 1; j <= k - 1; ++j) {
        Real beta = Real(j*j)/Real(4*j*j - 1);
        c[j + 1] = c[j] - beta*c[j - 1];
    }
    basis.mm.assign(static_cast<std::size_t>(d), 0);
    basis.phiR.assign(static_cast<std::size_t>(d), 0);
    basis.phiL.assign(static_cast<std::size_t>(d), 0);
    for (int n = 0; n < d; ++n) {
        basis.mm[n] = c[n]*c[n]/Real(2*n + 1);
        basis.phiR[n] = c[n];
        basis.phiL[n] = (n % 2 == 0 ? c[n] : -c[n]);
    }
    return basis;
}

Exact cole_cole_exact(const Config& cfg) {
    Exact exact;
    exact.A = cfg.epsilon0*(cfg.epsilon_rs - cfg.epsilon_inf);
    exact.alpha = cfg.alpha;
    exact.tau0 = cfg.tau0;
    exact.p = cfg.timePower;
    exact.scale = cfg.solutionScale;
    return exact;
}

Real Exact::g(const Real& t) const {
    return scale*pow(t, p);
}

Real Exact::gd(const Real& t, int m) const {
    return scale*power_derivative(t, p, m);
}

Real Exact::r(const Real& t) const {
    return scale*(1 + pow(t, p));
}

Real Exact::rd(const Real& t, int m) const {
    return scale*((m == 0 ? Real(1) : Real(0)) + power_derivative(t, p, m));
}

Real Exact::q(const Real& t) const {
    Real tt = std::max(t, Real(0));
    if (tt == 0) {
        return 0;
    }
    Real lambda = pow(tau0, -alpha);
    Real z = -lambda*pow(tt, alpha);
    return scale*std::tgamma(Real(p + 1))*lambda
        *pow(tt, Real(p) + alpha)
        *mittag_leffler(alpha, Real(p) + alpha + 1, z);
}

Real Exact::qd(const Real& t) const {
    Real tt = std::max(t, Real(0));
    if (tt == 0) {
        return 0;
    }
    Real lambda = pow(tau0, -alpha);
    Real z = -lambda*pow(tt, alpha);
    return scale*std::tgamma(Real(p + 1))*lambda
        *pow(tt, Real(p) + alpha - 1)
        *mittag_leffler(alpha, Real(p) + alpha, z);
}

Real Exact::caputo_q(const Real& t) const {
    Real tt = std::max(t, Real(0));
    if (tt == 0) {
        return 0;
    }
    Real lambda = pow(tau0, -alpha);
    Real z = -lambda*pow(tt, alpha);
    return scale*std::tgamma(Real(p + 1))*lambda
        *pow(tt, Real(p))
        *mittag_leffler(alpha, Real(p) + 1, z);
}

Real Exact::E(const Real& x, const Real& t) const {
    return sin(x)*g(t);
}

Real Exact::H(const Real& x, const Real& t) const {
    return cos(x)*r(t);
}

Real Exact::P(const Real& x, const Real& t) const {
    return A*sin(x)*q(t);
}

Real Exact::Et(const Real& x, const Real& t, int m) const {
    return sin(x)*gd(t, m);
}

Real Exact::Ht(const Real& x, const Real& t, int m) const {
    return cos(x)*rd(t, m);
}

Real mittag_leffler(const Real& alpha, const Real& beta, const Real& z,
                    const Real& tol, int maxTerms) {
    if (alpha <= 0 || beta <= 0) {
        throw std::runtime_error("mittag_leffler: alpha and beta must be positive");
    }
    Real s = 1/std::tgamma(beta);
    if (z == 0) {
        return s;
    }
    Real c = 0;
    Real zpow = 1;
    for (int m = 1; m <= maxTerms; ++m) {
        zpow *= z;
        Real term = zpow/std::tgamma(alpha*Real(m) + beta);
        Real y = term - c;
        Real t = s + y;
        c = (t - s) - y;
        s = t;
        Real scale = abs(s);
        if (scale < Real(1e-95)) {
            scale = Real(1e-95);
        }
        if (abs(term) <= tol*scale) {
            break;
        }
    }
    return s;
}

Real prabhakar_mittag_leffler(const Real& alpha, const Real& beta, int gammaPar,
                              const Real& z, const Real& tol, int maxTerms) {
    if (alpha <= 0 || beta <= 0 || gammaPar <= 0) {
        throw std::runtime_error("prabhakar_mittag_leffler: invalid parameters");
    }
    Real s = 1/std::tgamma(beta);
    Real pochhammerOverFactorial = 1;
    Real zPower = 1;
    for (int n = 1; n <= maxTerms; ++n) {
        pochhammerOverFactorial *= Real(gammaPar + n - 1)/Real(n);
        zPower *= z;
        Real term = pochhammerOverFactorial*zPower
            / std::tgamma(alpha*Real(n) + beta);
        s += term;
        Real scale = abs(s);
        if (scale < 1) {
            scale = 1;
        }
        if (abs(term) <= tol*scale) {
            break;
        }
    }
    return s;
}

Real relaxation_basis_value(const Config& cfg, int relaxationPower,
                            int derivativeOrder, const Real& t) {
    if (relaxationPower < 0 || derivativeOrder < 0 || t < 0) {
        throw std::runtime_error("relaxation_basis_value: invalid input");
    }
    const int p = cfg.timePower;
    const int r = relaxationPower;
    const int m = derivativeOrder;
    if (r == 0) {
        return power_derivative(t, p, m);
    }
    Real alpha = cfg.alpha;
    Real lambda = pow(cfg.tau0, -alpha);
    Real exponent = Real(p) + alpha*Real(r) - Real(m);
    Real beta = exponent + 1;
    if (beta <= 0) {
        throw std::runtime_error("relaxation_basis_value: derivative order too large");
    }
    if (t == 0) {
        return 0;
    }
    Real z = -lambda*pow(t, alpha);
    return std::tgamma(Real(p + 1))*pow(lambda, r)
        *pow(t, exponent)
        *prabhakar_mittag_leffler(alpha, beta, r, z);
}

MaxwellOp assemble_maxwell_dg(const Mesh& mesh, const Basis& basis, const Quad& quad, const Config& cfg) {
    const int N = mesh.N;
    const int d = basis.dim;
    MaxwellOp op;
    op.SEE = zero_spatial(N, d);
    op.SEH = zero_spatial(N, d);
    op.SHE = zero_spatial(N, d);
    op.SHH = zero_spatial(N, d);

    const Real ee = cfg.epsilon0*cfg.epsilon_inf;
    const Real zE = sqrt(ee/cfg.mu0);
    const Real zH = sqrt(cfg.mu0/ee);

    Matrix G(d, d);
    for (int row = 0; row < d; ++row) {
        for (int col = 0; col < d; ++col) {
            Real v = 0;
            for (std::size_t q = 0; q < quad.w.size(); ++q) {
                v += quad.w[q]*basis.phi(static_cast<int>(q), col)*basis.dphi(static_cast<int>(q), row);
            }
            G(row, col) = v;
        }
    }

    spatial_for(N, [&](int j) {
        for (int row = 0; row < d; ++row) {
            for (int col = 0; col < d; ++col) {
                op.SEH.diag[j](row, col) -= G(row, col);
                op.SHE.diag[j](row, col) -= G(row, col);
            }
        }
    });

    auto scaled = [](const Vec& v, const Real& a) {
        Vec out(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) {
            out[i] = a*v[i];
        }
        return out;
    };

    for (int iface = 0; iface <= N; ++iface) {
        if (iface == 0) {
            Vec phiL = basis.phiL;
            add_spatial_outer(op.SEE, 0, 0, scaled(phiL, -1), scaled(phiL, zE));
            add_spatial_outer(op.SEH, 0, 0, scaled(phiL, -1), phiL);
        } else if (iface == N) {
            Vec phiR = basis.phiR;
            add_spatial_outer(op.SEE, N - 1, N - 1, phiR, scaled(phiR, -zE));
            add_spatial_outer(op.SEH, N - 1, N - 1, phiR, phiR);
        } else {
            int cL = iface - 1;
            int cR = iface;
            const Vec& phiRm = basis.phiR;
            const Vec& phiLp = basis.phiL;

            add_spatial_outer(op.SEE, cL, cL, scaled(phiRm, Real(0.5)), scaled(phiRm, -zE));
            add_spatial_outer(op.SEE, cL, cR, scaled(phiRm, Real(0.5)), scaled(phiLp, zE));
            add_spatial_outer(op.SEH, cL, cL, scaled(phiRm, Real(0.5)), phiRm);
            add_spatial_outer(op.SEH, cL, cR, scaled(phiRm, Real(0.5)), phiLp);

            add_spatial_outer(op.SEE, cR, cL, scaled(phiLp, Real(-0.5)), scaled(phiRm, -zE));
            add_spatial_outer(op.SEE, cR, cR, scaled(phiLp, Real(-0.5)), scaled(phiLp, zE));
            add_spatial_outer(op.SEH, cR, cL, scaled(phiLp, Real(-0.5)), phiRm);
            add_spatial_outer(op.SEH, cR, cR, scaled(phiLp, Real(-0.5)), phiLp);

            add_spatial_outer(op.SHE, cL, cL, scaled(phiRm, Real(0.5)), phiRm);
            add_spatial_outer(op.SHE, cL, cR, scaled(phiRm, Real(0.5)), phiLp);
            add_spatial_outer(op.SHH, cL, cL, scaled(phiRm, Real(0.5)), scaled(phiRm, -zH));
            add_spatial_outer(op.SHH, cL, cR, scaled(phiRm, Real(0.5)), scaled(phiLp, zH));

            add_spatial_outer(op.SHE, cR, cL, scaled(phiLp, Real(-0.5)), phiRm);
            add_spatial_outer(op.SHE, cR, cR, scaled(phiLp, Real(-0.5)), phiLp);
            add_spatial_outer(op.SHH, cR, cL, scaled(phiLp, Real(-0.5)), scaled(phiRm, -zH));
            add_spatial_outer(op.SHH, cR, cR, scaled(phiLp, Real(-0.5)), scaled(phiLp, zH));
        }
    }

    op.massDiag.assign(static_cast<std::size_t>(N*d), 0);
    spatial_for(N, [&](int j) {
        for (int m = 0; m < d; ++m) {
            op.massDiag[modal_index(j, m, d)] = mesh.h[j]*basis.mm[m];
        }
    });
    return op;
}

RKMethod rk_method_coefficients(const std::string& method) {
    std::string name = method;
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::toupper(c); });
    name.erase(std::remove(name.begin(), name.end(), '_'), name.end());
    name.erase(std::remove(name.begin(), name.end(), '-'), name.end());

    RKMethod rk;
    if (name == "RADAUIIA2" || name == "RADAU2") {
        rk.name = "RadauIIA-2";
        rk.order = 3;
        rk.s = 2;
        rk.A = Matrix(2, 2);
        rk.A(0, 0) = Real(5)/12; rk.A(0, 1) = Real(-1)/12;
        rk.A(1, 0) = Real(3)/4;  rk.A(1, 1) = Real(1)/4;
        rk.b = {Real(3)/4, Real(1)/4};
        rk.c = {Real(1)/3, Real(1)};
    } else if (name == "RADAUIIA3" || name == "RADAU3") {
        rk.name = "RadauIIA-3";
        rk.order = 5;
        rk.s = 3;
        Real sq6 = sqrt(Real(6));
        rk.A = Matrix(3, 3);
        rk.A(0, 0) = (Real(88) - 7*sq6)/360;
        rk.A(0, 1) = (Real(296) - 169*sq6)/1800;
        rk.A(0, 2) = (Real(-2) + 3*sq6)/225;
        rk.A(1, 0) = (Real(296) + 169*sq6)/1800;
        rk.A(1, 1) = (Real(88) + 7*sq6)/360;
        rk.A(1, 2) = (Real(-2) - 3*sq6)/225;
        rk.A(2, 0) = (Real(16) - sq6)/36;
        rk.A(2, 1) = (Real(16) + sq6)/36;
        rk.A(2, 2) = Real(1)/9;
        rk.b = {(Real(16) - sq6)/36, (Real(16) + sq6)/36, Real(1)/9};
        rk.c = {(Real(4) - sq6)/10, (Real(4) + sq6)/10, Real(1)};
    } else {
        throw std::runtime_error("unsupported RK method: " + method);
    }
    rk.Ainv = LUFactor(rk.A).solve_matrix(eye(rk.s));
    rk.stifflyAccurate = true;
    for (int j = 0; j < rk.s; ++j) {
        if (abs(rk.A(rk.s - 1, j) - rk.b[j]) > Real(1e-14)) {
            rk.stifflyAccurate = false;
        }
    }
    return rk;
}

RKWeights cq_weights_rk(const RKMethod& rk, const Real& alpha, int nmax, const Config& cfg) {
    if (nmax < 0) {
        throw std::runtime_error("cq_weights_rk: nmax must be nonnegative");
    }
    static std::map<std::string, RKWeights> cache;
    std::string key = cache_key(rk, alpha, nmax, cfg);
    auto found = cache.find(key);
    if (found != cache.end()) {
        RKWeights out = found->second;
        out.info.cacheHit = true;
        return out;
    }

    RKWeights weights;
    weights.s = rk.s;
    weights.nmax = nmax;
    weights.W.assign(static_cast<std::size_t>(nmax + 1), Matrix(rk.s, rk.s));
    weights.info.fftFactor = cfg.rkCqFftFactor;
    weights.info.minFftLength = cfg.rkCqMinFftLength;
    weights.info.radiusExponent = cfg.rkCqRadiusExponent;
    weights.info.targetEps = cfg.rkCqTargetEps;
    weights.info.nmax = nmax;
    int L = next_pow2(std::max(cfg.rkCqMinFftLength, cfg.rkCqFftFactor*(nmax + 1)));
    weights.info.fftLength = L;
    if (cfg.rkCqTargetEps <= 0 || cfg.rkCqTargetEps >= 1) {
        throw std::runtime_error("cq_weights_rk: rkCqTargetEps must be in (0, 1)");
    }
    Real rho = pow(cfg.rkCqTargetEps, cfg.rkCqRadiusExponent/Real(nmax + 1));
    weights.info.radius = rho;
    weights.info.roundoffAmplification = pow(rho, -nmax);
    std::string diskCachePath = cache_file_for_key(cfg, key);
    if (load_rk_weights_from_disk(diskCachePath, rk.s, nmax, weights)) {
        weights.info.cacheHit = true;
        cache[key] = weights;
        return weights;
    }

    std::vector<std::vector<CVec>> samples(static_cast<std::size_t>(rk.s),
        std::vector<CVec>(static_cast<std::size_t>(rk.s), CVec(static_cast<std::size_t>(L))));

    CMatrix I = ceye(rk.s);
    const Real twoPi = 2*pi();
    spatial_for(L, [&](int ell) {
        Real theta = twoPi*Real(ell)/Real(L);
        Complex zeta = rho*Complex(cos(theta), sin(theta));
        Complex beta = zeta/(Complex(1) - zeta);
        CMatrix B(rk.s, rk.s);
        for (int i = 0; i < rk.s; ++i) {
            for (int j = 0; j < rk.s; ++j) {
                B(i, j) = Complex(rk.A(i, j)) + beta*Complex(rk.b[j]);
            }
        }
        CMatrix Delta = solve_complex_matrix(B, I);
        CMatrix G = matrix_function_power(Delta, alpha);
        for (int i = 0; i < rk.s; ++i) {
            for (int j = 0; j < rk.s; ++j) {
                samples[i][j][ell] = G(i, j);
            }
        }
    });

    Vec rhoScale(static_cast<std::size_t>(nmax + 1), 0);
    spatial_for(nmax + 1, [&](int n) {
        rhoScale[n] = Real(pow(rho, -n));
    });

    Vec pairMaxImag(static_cast<std::size_t>(rk.s*rk.s), 0);
    spatial_for(rk.s*rk.s, [&](int pairIdx) {
        int i = pairIdx/rk.s;
        int j = pairIdx - i*rk.s;
        CVec coeff = samples[i][j];
        fft(coeff);
        Real localMaxImag = 0;
        for (int n = 0; n <= nmax; ++n) {
            Complex value = coeff[n]/Real(L)*rhoScale[n];
            Real imagAbs = value.imag() < 0 ? -value.imag() : value.imag();
            if (imagAbs > localMaxImag) {
                localMaxImag = imagAbs;
            }
            weights.W[n](i, j) = value.real();
        }
        pairMaxImag[pairIdx] = localMaxImag;
    });
    Real maxImag = 0;
    for (const Real& value : pairMaxImag) {
        if (value > maxImag) {
            maxImag = value;
        }
    }
    weights.info.maxImaginaryWeightPart = maxImag;
    save_rk_weights_to_disk(diskCachePath, weights);
    cache[key] = weights;
    return weights;
}

Matrix project_l2_func(const Mesh& mesh, const Basis& basis, const Quad& quad,
                       const std::function<Real(const Real&)>& func) {
    Matrix U(mesh.N, basis.dim);
    const int nq = static_cast<int>(quad.x.size());
    spatial_for(mesh.N, [&](int j) {
        for (int m = 0; m < basis.dim; ++m) {
            Real rhs = 0;
            for (int q = 0; q < nq; ++q) {
                Real xq = mesh.xc[j] + mesh.J[j]*quad.x[q];
                rhs += quad.w[q]*func(xq)*basis.phi(q, m);
            }
            U(j, m) = Real(0.5)*rhs/basis.mm[m];
        }
    });
    return U;
}

Matrix project_radau_func(const Mesh& mesh, const Basis& basis, const Quad& quad,
                          const std::function<Real(const Real&)>& func,
                          const std::string& side) {
    Matrix U(mesh.N, basis.dim);
    const int k = basis.k;
    const int d = basis.dim;
    const int nq = static_cast<int>(quad.x.size());
    std::string lowerSide = side;
    std::transform(lowerSide.begin(), lowerSide.end(), lowerSide.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    const bool useRightFace = (lowerSide == "minus" || lowerSide == "right");
    const bool useLeftFace = (lowerSide == "plus" || lowerSide == "left");
    if (!useRightFace && !useLeftFace) {
        throw std::runtime_error("project_radau_func: side must be plus/left or minus/right");
    }
    spatial_for(mesh.N, [&](int j) {
        for (int m = 0; m < k; ++m) {
            Real rhs = 0;
            for (int q = 0; q < nq; ++q) {
                Real xq = mesh.xc[j] + mesh.J[j]*quad.x[q];
                rhs += quad.w[q]*func(xq)*basis.phi(q, m);
            }
            U(j, m) = Real(0.5)*rhs/basis.mm[m];
        }
        const Vec* faceBasis = nullptr;
        Real xFace = 0;
        if (useRightFace) {
            xFace = mesh.xR[j];
            faceBasis = &basis.phiR;
        } else {
            xFace = mesh.xL[j];
            faceBasis = &basis.phiL;
        }
        Real known = 0;
        for (int m = 0; m < k; ++m) {
            known += U(j, m)*(*faceBasis)[m];
        }
        U(j, d - 1) = (func(xFace) - known)/(*faceBasis)[d - 1];
    });
    return U;
}

void project_star_maxwell(const Mesh& mesh, const Basis& basis, const Quad& quad,
                          const Config& cfg, const Exact& exact, const Real& t,
                          int derivativeOrder, Matrix& Estar, Matrix& Hstar) {
    Real ee = cfg.epsilon0*cfg.epsilon_inf;
    Real cH = sqrt(cfg.mu0/ee);
    Real cE = sqrt(ee/cfg.mu0);

    auto Efun = [&](const Real& x) { return exact.Et(x, t, derivativeOrder); };
    auto Hfun = [&](const Real& x) { return exact.Ht(x, t, derivativeOrder); };

    Matrix Pp_Eplus = project_radau_func(mesh, basis, quad,
        [&](const Real& x) { return Efun(x) + cH*Hfun(x); }, "plus");
    Matrix Pm_Eminus = project_radau_func(mesh, basis, quad,
        [&](const Real& x) { return Efun(x) - cH*Hfun(x); }, "minus");
    Matrix Pp_Hplus = project_radau_func(mesh, basis, quad,
        [&](const Real& x) { return Hfun(x) + cE*Efun(x); }, "plus");
    Matrix Pm_Hminus = project_radau_func(mesh, basis, quad,
        [&](const Real& x) { return Hfun(x) - cE*Efun(x); }, "minus");

    Estar = mat_scale(mat_add(Pp_Eplus, Pm_Eminus), Real(0.5));
    Hstar = mat_scale(mat_add(Pp_Hplus, Pm_Hminus), Real(0.5));
}

Vec modal_load(const Mesh& mesh, const Basis& basis, const Quad& quad,
               const std::function<Real(const Real&)>& func) {
    Matrix Fmat(mesh.N, basis.dim);
    const int nq = static_cast<int>(quad.x.size());
    spatial_for(mesh.N, [&](int j) {
        for (int m = 0; m < basis.dim; ++m) {
            Real v = 0;
            for (int q = 0; q < nq; ++q) {
                Real xq = mesh.xc[j] + mesh.J[j]*quad.x[q];
                v += quad.w[q]*func(xq)*basis.phi(q, m);
            }
            Fmat(j, m) = mesh.J[j]*v;
        }
    });
    return pack_modal(Fmat);
}

Vec pack_modal(const Matrix& U) {
    Vec v(static_cast<std::size_t>(U.rows*U.cols));
    spatial_for(U.rows, [&](int j) {
        for (int m = 0; m < U.cols; ++m) {
            v[modal_index(j, m, U.cols)] = U(j, m);
        }
    });
    return v;
}

Matrix unpack_modal(const Vec& v, int N, int d) {
    Matrix U(N, d);
    spatial_for(N, [&](int j) {
        for (int m = 0; m < d; ++m) {
            U(j, m) = v[modal_index(j, m, d)];
        }
    });
    return U;
}

Matrix modal_values(const Matrix& U, const Basis& basis) {
    return matmul(U, transpose(basis.phi));
}

namespace {

std::pair<Matrix, Matrix> solve_correction_eh_modal(const Mesh& mesh, const Basis& basis,
                                                    const Quad& quad, const Config& cfg,
                                                    const Matrix& sourceH,
                                                    const Matrix& sourceE) {
    const int N = mesh.N;
    const int d = basis.dim;
    const int k = basis.k;
    const Real ee = cfg.epsilon0*cfg.epsilon_inf;
    const Real cH = sqrt(cfg.mu0/ee);
    Matrix Dmat(k, d);
    for (int r = 0; r < k; ++r) {
        for (int col = 0; col < d; ++col) {
            Real v = 0;
            for (std::size_t q = 0; q < quad.w.size(); ++q) {
                v += quad.w[q]*basis.phi(static_cast<int>(q), col)*basis.dphi(static_cast<int>(q), r + 1);
            }
            Dmat(r, col) = v;
        }
    }

    Matrix wE(N, d);
    Matrix wH(N, d);
    Matrix Amat(2*d, 2*d);
    for (int r = 0; r < k; ++r) {
        for (int col = 0; col < d; ++col) {
            Amat(r, d + col) = Dmat(r, col);
            Amat(k + r, col) = Dmat(r, col);
        }
    }
    int rowL = 2*k;
    int rowR = 2*k + 1;
    for (int m = 0; m < d; ++m) {
        Amat(rowL, m) = basis.phiL[m];
        Amat(rowL, d + m) = cH*basis.phiL[m];
        Amat(rowR, m) = basis.phiR[m];
        Amat(rowR, d + m) = -cH*basis.phiR[m];
    }
    LUFactor correctionLU(Amat);
    spatial_for(N, [&](int cellId) {
        Vec bvec(static_cast<std::size_t>(2*d), 0);
        for (int r = 0; r < k; ++r) {
            int testMode = r + 1;
            bvec[r] = mesh.h[cellId]*basis.mm[testMode]*sourceH(cellId, testMode);
            bvec[k + r] = mesh.h[cellId]*basis.mm[testMode]*sourceE(cellId, testMode);
        }
        Vec sol = correctionLU.solve(bvec);
        for (int m = 0; m < d; ++m) {
            wE(cellId, m) = sol[m];
            wH(cellId, m) = sol[d + m];
        }
    });
    return {wE, wH};
}

struct Series {
    Matrix constant;
    std::vector<std::vector<Matrix>> coeff;
};

Series new_series(int maxIndex, const Matrix& zeroModal) {
    Series S;
    S.constant = zeroModal;
    S.coeff.assign(static_cast<std::size_t>(maxIndex + 1),
                   std::vector<Matrix>(static_cast<std::size_t>(maxIndex + 1)));
    return S;
}

const Matrix& coeff_or_zero(const Series& S, int r, int m, const Matrix& zeroModal) {
    if (S.coeff[r][m].rows == 0) {
        return zeroModal;
    }
    return S.coeff[r][m];
}

} // namespace

Correction correction_functions(const Mesh& mesh, const Basis& basis, const Quad& quad,
                                const Config& cfg, const Exact& exact, const Real& t) {
    const int l = cfg.correctionLevel;
    if (l < 0 || l > basis.k) {
        throw std::runtime_error("correction_functions: correctionLevel must satisfy 0 <= l <= k");
    }
    const Matrix zeroModal(mesh.N, basis.dim);
    const int maxIndex = l;

    Matrix Estar0, Hstar0;
    project_star_maxwell(mesh, basis, quad, cfg, exact, Real(0), 0, Estar0, Hstar0);
    Matrix EL20 = project_l2_func(mesh, basis, quad, [&](const Real& x) { return exact.E(x, Real(0)); });
    Matrix HL20 = project_l2_func(mesh, basis, quad, [&](const Real& x) { return exact.H(x, Real(0)); });
    Matrix wE0Const = mat_sub(EL20, Estar0);
    Matrix wH0Const = mat_sub(HL20, Hstar0);

    Matrix Estar1, Hstar1;
    project_star_maxwell(mesh, basis, quad, cfg, exact, Real(1), 0, Estar1, Hstar1);
    Matrix EL21 = project_l2_func(mesh, basis, quad, [&](const Real& x) { return exact.E(x, Real(1)); });
    Matrix HL21 = project_l2_func(mesh, basis, quad, [&](const Real& x) { return exact.H(x, Real(1)); });
    Matrix wE0Power = mat_sub(mat_sub(EL21, Estar1), wE0Const);
    Matrix wH0Power = mat_sub(mat_sub(HL21, Hstar1), wH0Const);

    auto make = [&]() { return new_series(maxIndex, zeroModal); };

    auto add_series = [&](const Series& A, const Series& B) {
        Series C = make();
        C.constant = mat_add(A.constant, B.constant);
        for (int ir = 0; ir <= maxIndex; ++ir) {
            for (int im = 0; im <= maxIndex; ++im) {
                Matrix coeff = mat_add(coeff_or_zero(A, ir, im, zeroModal),
                                       coeff_or_zero(B, ir, im, zeroModal));
                if (any_nonzero(coeff)) {
                    C.coeff[ir][im] = coeff;
                }
            }
        }
        return C;
    };

    auto scale_series = [&](const Series& A, const Real& scalar) {
        Series B = make();
        B.constant = mat_scale(A.constant, scalar);
        for (int ir = 0; ir <= maxIndex; ++ir) {
            for (int im = 0; im <= maxIndex; ++im) {
                if (A.coeff[ir][im].rows != 0) {
                    B.coeff[ir][im] = mat_scale(A.coeff[ir][im], scalar);
                }
            }
        }
        return B;
    };

    auto derivative_series = [&](const Series& A) {
        Series B = make();
        for (int ir = 0; ir <= maxIndex; ++ir) {
            for (int im = 0; im < maxIndex; ++im) {
                if (A.coeff[ir][im].rows != 0) {
                    B.coeff[ir][im + 1] = A.coeff[ir][im];
                }
            }
        }
        return B;
    };

    auto relax_series = [&](const Series& A) {
        Series B = make();
        B.constant = A.constant;
        for (int ir = 0; ir < maxIndex; ++ir) {
            for (int im = 0; im <= maxIndex; ++im) {
                if (A.coeff[ir][im].rows != 0) {
                    B.coeff[ir + 1][im] = A.coeff[ir][im];
                }
            }
        }
        return B;
    };

    auto solve_eh_series = [&](const Series& sourceH, const Series& sourceE) {
        Series Eseries = make();
        Series Hseries = make();
        if (any_nonzero(sourceH.constant) || any_nonzero(sourceE.constant)) {
            auto pair = solve_correction_eh_modal(mesh, basis, quad, cfg,
                                                  sourceH.constant, sourceE.constant);
            Eseries.constant = pair.first;
            Hseries.constant = pair.second;
        }
        for (int ir = 0; ir <= maxIndex; ++ir) {
            for (int im = 0; im <= maxIndex; ++im) {
                const Matrix& hCoeff = coeff_or_zero(sourceH, ir, im, zeroModal);
                const Matrix& eCoeff = coeff_or_zero(sourceE, ir, im, zeroModal);
                if (any_nonzero(hCoeff) || any_nonzero(eCoeff)) {
                    auto pair = solve_correction_eh_modal(mesh, basis, quad, cfg, hCoeff, eCoeff);
                    Eseries.coeff[ir][im] = pair.first;
                    Hseries.coeff[ir][im] = pair.second;
                }
            }
        }
        return std::make_pair(Eseries, Hseries);
    };

    auto evaluate_series = [&](const Series& S, const Real& time) {
        Matrix U = S.constant;
        for (int ir = 0; ir <= maxIndex; ++ir) {
            for (int im = 0; im <= maxIndex; ++im) {
                if (S.coeff[ir][im].rows != 0) {
                    Real value = relaxation_basis_value(cfg, ir, im, time);
                    Matrix add = mat_scale(S.coeff[ir][im], value);
                    U = mat_add(U, add);
                }
            }
        }
        return U;
    };

    Series Ecurrent = make();
    Series Hcurrent = make();
    Ecurrent.constant = wE0Const;
    Hcurrent.constant = wH0Const;
    Ecurrent.coeff[0][0] = wE0Power;
    Hcurrent.coeff[0][0] = wH0Power;

    Series sumE = make();
    Series sumH = make();
    Series sumP = make();
    std::vector<Series> layerE;
    std::vector<Series> layerH;
    std::vector<Series> layerP;

    const Real ee = cfg.epsilon0*cfg.epsilon_inf;
    for (int layer = 1; layer <= l; ++layer) {
        Series Pnext = relax_series(scale_series(Ecurrent, exact.A));
        Series sourceH = add_series(scale_series(derivative_series(Ecurrent), -ee),
                                    scale_series(derivative_series(Pnext), -1));
        Series sourceE = scale_series(derivative_series(Hcurrent), -cfg.mu0);
        auto solved = solve_eh_series(sourceH, sourceE);
        Series Enext = solved.first;
        Series Hnext = solved.second;

        sumE = add_series(sumE, Enext);
        sumH = add_series(sumH, Hnext);
        sumP = add_series(sumP, Pnext);
        layerE.push_back(Enext);
        layerH.push_back(Hnext);
        layerP.push_back(Pnext);
        Ecurrent = Enext;
        Hcurrent = Hnext;
    }

    Correction corr;
    corr.level = l;
    corr.wE = evaluate_series(sumE, t);
    corr.wH = evaluate_series(sumH, t);
    corr.wP = evaluate_series(sumP, t);

    Series baseE = make();
    Series baseH = make();
    baseE.constant = wE0Const;
    baseH.constant = wH0Const;
    baseE.coeff[0][0] = wE0Power;
    baseH.coeff[0][0] = wH0Power;
    corr.wE0 = evaluate_series(baseE, t);
    corr.wH0 = evaluate_series(baseH, t);

    if (l >= 1) {
        corr.wE1 = evaluate_series(layerE[0], t);
        corr.wH1 = evaluate_series(layerH[0], t);
        corr.wP1 = evaluate_series(layerP[0], t);
        corr.wP1dt = evaluate_series(derivative_series(layerP[0]), t);
    } else {
        corr.wE1 = zeroModal;
        corr.wH1 = zeroModal;
        corr.wP1 = zeroModal;
        corr.wP1dt = zeroModal;
    }
    return corr;
}

InitialModal initial_modal(const Mesh& mesh, const Basis& basis, const Quad& quad,
                           const Config& cfg, const Exact& exact, const Real& t) {
    if (cfg.sourceFree) {
        if (cfg.initialMethod != "l2" || t != 0)
            throw std::runtime_error("source-free initial data require L2 projection at t=0");
        const Real length = cfg.xb-cfg.xa;
        auto electric = [&](const Real& x) {
            Real value=0;
            for (std::size_t m=0;m<cfg.initialElectricCoeffs.size();++m)
                value += cfg.initialElectricCoeffs[m]*sin(Real(m+1)*pi()*(x-cfg.xa)/length);
            return value;
        };
        auto magnetic = [&](const Real& x) {
            Real value=0;
            for (std::size_t m=0;m<cfg.initialMagneticCoeffs.size();++m)
                value += cfg.initialMagneticCoeffs[m]*cos(Real(m+1)*pi()*(x-cfg.xa)/length);
            return value;
        };
        InitialModal initial;
        initial.E = project_l2_func(mesh,basis,quad,electric);
        initial.H = project_l2_func(mesh,basis,quad,magnetic);
        initial.P = Matrix(mesh.N,basis.dim);
        return initial;
    }
    Matrix Estar, Hstar;
    project_star_maxwell(mesh, basis, quad, cfg, exact, t, 0, Estar, Hstar);
    Matrix Ppi = project_l2_func(mesh, basis, quad, [&](const Real& x) { return exact.P(x, t); });
    InitialModal init;
    if (cfg.initialMethod == "special") {
        init.E = Estar;
        init.H = Hstar;
        init.P = Ppi;
    } else if (cfg.initialMethod == "l2" || cfg.initialMethod == "L2") {
        init.E = project_l2_func(mesh, basis, quad, [&](const Real& x) { return exact.E(x, t); });
        init.H = project_l2_func(mesh, basis, quad, [&](const Real& x) { return exact.H(x, t); });
        init.P = Ppi;
    } else if (cfg.initialMethod == "correction") {
        Correction corr = correction_functions(mesh, basis, quad, cfg, exact, t);
        init.E = mat_sub(Estar, corr.wE);
        init.H = mat_sub(Hstar, corr.wH);
        init.P = mat_sub(Ppi, corr.wP);
    } else {
        throw std::runtime_error("initial_modal: unknown initialMethod " + cfg.initialMethod);
    }
    return init;
}

namespace {

struct BlockThomasSolver {
    int N = 0;
    int B = 0;
    std::vector<Matrix> lower;
    std::vector<Matrix> cprime;
    std::vector<LUFactor> lu;

    BlockThomasSolver(const std::vector<Matrix>& diag,
                      const std::vector<Matrix>& lowerIn,
                      const std::vector<Matrix>& upperIn)
        : N(static_cast<int>(diag.size())),
          B(diag.empty() ? 0 : diag[0].rows),
          lower(lowerIn) {
        lu.reserve(static_cast<std::size_t>(N));
        cprime.reserve(static_cast<std::size_t>(std::max(0, N - 1)));
        Matrix modified = diag[0];
        lu.emplace_back(modified);
        if (N > 1) {
            cprime.push_back(lu[0].solve_matrix(upperIn[0]));
        }
        for (int j = 1; j < N; ++j) {
            Matrix product = matmul(lower[j - 1], cprime[j - 1]);
            modified = mat_sub(diag[j], product);
            lu.emplace_back(modified);
            if (j < N - 1) {
                cprime.push_back(lu[j].solve_matrix(upperIn[j]));
            }
        }
    }

    Vec solve(const Vec& rhs) const {
        std::vector<Vec> dprime(static_cast<std::size_t>(N), Vec(static_cast<std::size_t>(B), 0));
        Vec block(static_cast<std::size_t>(B), 0);
        for (int i = 0; i < B; ++i) {
            block[i] = rhs[i];
        }
        dprime[0] = lu[0].solve(block);
        for (int j = 1; j < N; ++j) {
            for (int i = 0; i < B; ++i) {
                block[i] = rhs[j*B + i];
                for (int k = 0; k < B; ++k) {
                    block[i] -= lower[j - 1](i, k)*dprime[j - 1][k];
                }
            }
            dprime[j] = lu[j].solve(block);
        }

        std::vector<Vec> xblock(static_cast<std::size_t>(N), Vec(static_cast<std::size_t>(B), 0));
        xblock[N - 1] = dprime[N - 1];
        for (int j = N - 2; j >= 0; --j) {
            xblock[j] = dprime[j];
            for (int i = 0; i < B; ++i) {
                for (int k = 0; k < B; ++k) {
                    xblock[j][i] -= cprime[j](i, k)*xblock[j + 1][k];
                }
            }
        }

        Vec x(static_cast<std::size_t>(N*B), 0);
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < B; ++i) {
                x[j*B + i] = xblock[j][i];
            }
        }
        return x;
    }
};

void add_spatial_system(const SpatialOp& S, std::vector<Matrix>& diag,
                        std::vector<Matrix>& lower, std::vector<Matrix>& upper,
                        int rowVar, int colVar, int s, int d) {
    const int N = static_cast<int>(diag.size());
    spatial_for(N, [&](int cell) {
        for (int stage = 0; stage < s; ++stage) {
            for (int r = 0; r < d; ++r) {
                for (int c = 0; c < d; ++c) {
                    add_system_block(diag[cell], rowVar, stage, r, colVar, stage, c,
                                     s, d, -S.diag[cell](r, c));
                }
            }
        }
    });
    spatial_for(N - 1, [&](int offset) {
        int cell = offset + 1;
        for (int stage = 0; stage < s; ++stage) {
            for (int r = 0; r < d; ++r) {
                for (int c = 0; c < d; ++c) {
                    add_system_block(lower[cell - 1], rowVar, stage, r, colVar, stage, c,
                                     s, d, -S.lower[cell - 1](r, c));
                }
            }
        }
    });
    spatial_for(N - 1, [&](int cell) {
        for (int stage = 0; stage < s; ++stage) {
            for (int r = 0; r < d; ++r) {
                for (int c = 0; c < d; ++c) {
                    add_system_block(upper[cell], rowVar, stage, r, colVar, stage, c,
                                     s, d, -S.upper[cell](r, c));
                }
            }
        }
    });
}

BlockThomasSolver build_rk_system_solver(const Mesh& mesh, const Basis& basis,
                                         const MaxwellOp& op, const Config& cfg,
                                         const RKMethod& rk, const Matrix& W0,
                                         const Real& dt) {
    const int N = mesh.N;
    const int d = basis.dim;
    const int s = rk.s;
    const int B = 3*s*d;
    std::vector<Matrix> diag(static_cast<std::size_t>(N), Matrix(B, B));
    std::vector<Matrix> lower;
    std::vector<Matrix> upper;
    if (N > 1) {
        lower.assign(static_cast<std::size_t>(N - 1), Matrix(B, B));
        upper.assign(static_cast<std::size_t>(N - 1), Matrix(B, B));
    }

    Matrix D = rk.Ainv;
    for (int i = 0; i < s; ++i) {
        for (int j = 0; j < s; ++j) {
            D(i, j) /= dt;
        }
    }
    const Real ee = cfg.epsilon0*cfg.epsilon_inf;
    const Real tauAlpha = pow(cfg.tau0, cfg.alpha);
    const Real dtAlpha = pow(dt, -cfg.alpha);

    spatial_for(N, [&](int cell) {
        for (int mode = 0; mode < d; ++mode) {
            Real mass = op.massDiag[modal_index(cell, mode, d)];
            for (int a = 0; a < s; ++a) {
                for (int b = 0; b < s; ++b) {
                    add_system_block(diag[cell], VAR_E, a, mode, VAR_E, b, mode,
                                     s, d, ee*D(a, b)*mass);
                    add_system_block(diag[cell], VAR_E, a, mode, VAR_P, b, mode,
                                     s, d, D(a, b)*mass);
                    add_system_block(diag[cell], VAR_H, a, mode, VAR_H, b, mode,
                                     s, d, cfg.mu0*D(a, b)*mass);
                    add_system_block(diag[cell], VAR_P, a, mode, VAR_P, b, mode,
                                     s, d, tauAlpha*dtAlpha*W0(a, b)*mass);
                    if (a == b) {
                        add_system_block(diag[cell], VAR_P, a, mode, VAR_E, b, mode,
                                         s, d, -cfg.epsilon0*(cfg.epsilon_rs - cfg.epsilon_inf)*mass);
                        add_system_block(diag[cell], VAR_P, a, mode, VAR_P, b, mode,
                                         s, d, mass);
                    }
                }
            }
        }
    });

    add_spatial_system(op.SEE, diag, lower, upper, VAR_E, VAR_E, s, d);
    add_spatial_system(op.SEH, diag, lower, upper, VAR_E, VAR_H, s, d);
    add_spatial_system(op.SHE, diag, lower, upper, VAR_H, VAR_E, s, d);
    add_spatial_system(op.SHH, diag, lower, upper, VAR_H, VAR_H, s, d);
    return BlockThomasSolver(diag, lower, upper);
}

Vec extract_pstage(const Vec& sol, int N, int s, int d) {
    const int B = 3*s*d;
    Vec out(static_cast<std::size_t>(N*s*d), 0);
    spatial_for(N, [&](int cell) {
        for (int stage = 0; stage < s; ++stage) {
            for (int mode = 0; mode < d; ++mode) {
                out[pstage_index(cell, stage, mode, s, d)] =
                    sol[cell*B + block_index(VAR_P, stage, mode, s, d)];
            }
        }
    });
    return out;
}

// The CQ weights are causal.  Once a dyadic block of stage values is known,
// its contributions to the next equally-sized block can be accumulated by a
// linear convolution.  The time loop therefore remains serial, while the
// expensive history evaluation is reduced from direct O(Nt^2) work.
class BlockFftHistoryConvolver {
public:
    BlockFftHistoryConvolver(const RKWeights& weights, int totalSteps, int N, int d,
                             const Vec& p0Stage, int fftMinBlock,
                             int verifySteps, const Real& targetEps)
        : weights_(weights),
          historySteps_(std::max(0, totalSteps - 1)),
          d_(d),
          s_(weights.s),
          channels_(N*d),
          width_(N*weights.s*d),
          fftMinBlock_(std::max(1, fftMinBlock)),
          verifySteps_(std::max(0, verifySteps)),
          p0Stage_(p0Stage),
          history_(static_cast<std::size_t>(historySteps_)*width_, Real(0)),
          future_(static_cast<std::size_t>(historySteps_)*width_, Real(0)) {
        if (static_cast<int>(p0Stage_.size()) != width_) {
            throw std::runtime_error("BlockFftHistoryConvolver: p0Stage size mismatch");
        }
        if (weights_.nmax < historySteps_) {
            throw std::runtime_error("BlockFftHistoryConvolver: insufficient RK-CQ weights");
        }
        precisionTolerance_ = targetEps*Real(1e-5);
        if (precisionTolerance_ < Real(1e-12)) {
            precisionTolerance_ = Real(1e-12);
        }
    }

    void append(int sampleIndex, const Vec& pstage) {
        if (sampleIndex >= historySteps_) {
            return;
        }
        if (sampleIndex != appended_) {
            throw std::runtime_error("BlockFftHistoryConvolver: samples must be appended in order");
        }
        if (static_cast<int>(pstage.size()) != width_) {
            throw std::runtime_error("BlockFftHistoryConvolver: stage size mismatch");
        }

        const std::size_t sampleOffset = static_cast<std::size_t>(sampleIndex)*width_;
        spatial_for(width_, [&](int i) {
            history_[sampleOffset + static_cast<std::size_t>(i)] = pstage[i] - p0Stage_[i];
        });

        add_self_contribution(sampleIndex);
        const int end = sampleIndex + 1;
        const int blockSize = end & -end;
        if (end < historySteps_) {
            if (blockSize >= fftMinBlock_) {
                add_block_fft(end - blockSize, end, blockSize);
            } else {
                add_block_direct(end - blockSize, end, blockSize);
            }
        }
        ++appended_;
    }

    Vec history_for_step(int step, const Vec& massDiag) {
        Vec out(static_cast<std::size_t>(width_), 0);
        if (step <= 0) {
            return out;
        }
        const int q = step - 1;
        if (q >= appended_ || q >= historySteps_) {
            throw std::runtime_error("BlockFftHistoryConvolver: requested history is not ready");
        }
        spatial_for(channels_, [&](int cm) {
            const int cell = cm/d_;
            const int mode = cm - cell*d_;
            const Real mass = massDiag[modal_index(cell, mode, d_)];
            const std::size_t qOffset = static_cast<std::size_t>(q)*width_;
            for (int a = 0; a < s_; ++a) {
                const int idx = pstage_index(cell, a, mode, s_, d_);
                out[idx] = mass*future_[qOffset + static_cast<std::size_t>(idx)];
            }
        });

        if (verifySteps_ > 0 && step <= verifySteps_) {
            verify_against_direct(step, out, massDiag);
        }
        return out;
    }

    Real max_imaginary_ratio() const {
        const Real denom = maxFftReal_ > 1 ? maxFftReal_ : Real(1);
        return maxFftImag_/denom;
    }

    Real max_direct_difference() const {
        return maxDirectDifference_;
    }

    void assert_precision() const {
        const Real imagRatio = max_imaginary_ratio();
        if (imagRatio > precisionTolerance_) {
            throw std::runtime_error(
                "block FFT history convolution lost precision: imaginary/real ratio="
                + real_to_string(imagRatio, 20)
                + ", tolerance=" + real_to_string(precisionTolerance_, 20));
        }
        if (verifySteps_ > 0 && maxDirectDifference_ > precisionTolerance_) {
            throw std::runtime_error(
                "block FFT history verification failed: relative direct difference="
                + real_to_string(maxDirectDifference_, 20)
                + ", tolerance=" + real_to_string(precisionTolerance_, 20));
        }
    }

private:
    const RKWeights& weights_;
    int historySteps_ = 0;
    int d_ = 0;
    int s_ = 0;
    int channels_ = 0;
    int width_ = 0;
    int fftMinBlock_ = 1;
    int verifySteps_ = 0;
    int appended_ = 0;
    const Vec& p0Stage_;
    std::vector<Real> history_;
    std::vector<Real> future_;
    std::map<int, std::vector<CVec>> kernelSpectra_;
    Real precisionTolerance_ = Real(1e-12);
    Real maxFftImag_ = 0;
    Real maxFftReal_ = 0;
    Real maxDirectDifference_ = 0;

    const Real& history_value(int sample, int index) const {
        return history_[static_cast<std::size_t>(sample)*width_ + static_cast<std::size_t>(index)];
    }

    Real& future_value(int output, int index) {
        return future_[static_cast<std::size_t>(output)*width_ + static_cast<std::size_t>(index)];
    }

    void add_self_contribution(int sampleIndex) {
        const Matrix& K0 = weights_.W[1];
        spatial_for(channels_, [&](int cm) {
            const int cell = cm/d_;
            const int mode = cm - cell*d_;
            for (int a = 0; a < s_; ++a) {
                Real acc = 0;
                for (int b = 0; b < s_; ++b) {
                    const int idxb = pstage_index(cell, b, mode, s_, d_);
                    acc += K0(a, b)*history_value(sampleIndex, idxb);
                }
                const int idxa = pstage_index(cell, a, mode, s_, d_);
                future_value(sampleIndex, idxa) += acc;
            }
        });
    }

    void add_block_direct(int start, int end, int blockSize) {
        const int futureEnd = std::min(historySteps_, end + blockSize);
        spatial_for(channels_, [&](int cm) {
            const int cell = cm/d_;
            const int mode = cm - cell*d_;
            for (int q = end; q < futureEnd; ++q) {
                for (int a = 0; a < s_; ++a) {
                    Real acc = 0;
                    for (int sample = start; sample < end; ++sample) {
                        const Matrix& kernel = weights_.W[q - sample + 1];
                        for (int b = 0; b < s_; ++b) {
                            const int idxb = pstage_index(cell, b, mode, s_, d_);
                            acc += kernel(a, b)*history_value(sample, idxb);
                        }
                    }
                    const int idxa = pstage_index(cell, a, mode, s_, d_);
                    future_value(q, idxa) += acc;
                }
            }
        });
    }

    const std::vector<CVec>& kernel_spectra(int blockSize) {
        const auto found = kernelSpectra_.find(blockSize);
        if (found != kernelSpectra_.end()) {
            return found->second;
        }
        const int fftLength = 2*blockSize;
        std::vector<CVec> spectra(static_cast<std::size_t>(s_*s_),
                                  CVec(static_cast<std::size_t>(fftLength), Complex(0)));
        spatial_for(s_*s_, [&](int pair) {
            const int a = pair/s_;
            const int b = pair - a*s_;
            CVec& values = spectra[static_cast<std::size_t>(pair)];
            for (int lag = 0; lag < fftLength && lag + 1 <= weights_.nmax; ++lag) {
                values[lag] = Complex(weights_.W[lag + 1](a, b));
            }
            fft(values);
        });
        return kernelSpectra_.emplace(blockSize, std::move(spectra)).first->second;
    }

    void add_block_fft(int start, int end, int blockSize) {
        const int futureCount = std::min(blockSize, historySteps_ - end);
        if (futureCount <= 0) {
            return;
        }
        const int fftLength = 2*blockSize;
        const std::vector<CVec>& kernelFft = kernel_spectra(blockSize);
        Vec localImag(static_cast<std::size_t>(channels_), 0);
        Vec localReal(static_cast<std::size_t>(channels_), 0);

        spatial_for(channels_, [&](int cm) {
            const int cell = cm/d_;
            const int mode = cm - cell*d_;
            std::vector<CVec> input(static_cast<std::size_t>(s_),
                                    CVec(static_cast<std::size_t>(fftLength), Complex(0)));
            std::vector<CVec> output(static_cast<std::size_t>(s_),
                                     CVec(static_cast<std::size_t>(fftLength), Complex(0)));
            for (int b = 0; b < s_; ++b) {
                const int idxb = pstage_index(cell, b, mode, s_, d_);
                for (int local = 0; local < blockSize; ++local) {
                    input[b][local] = Complex(history_value(start + local, idxb));
                }
                fft(input[b]);
            }
            for (int a = 0; a < s_; ++a) {
                for (int freq = 0; freq < fftLength; ++freq) {
                    Complex value(0);
                    for (int b = 0; b < s_; ++b) {
                        value += kernelFft[static_cast<std::size_t>(a*s_ + b)][freq]
                            *input[b][freq];
                    }
                    output[a][freq] = value;
                }
                inverse_fft(output[a]);
            }

            Real imagMax = 0;
            Real realMax = 0;
            for (int local = 0; local < futureCount; ++local) {
                const int q = end + local;
                for (int a = 0; a < s_; ++a) {
                    const Complex value = output[a][blockSize + local];
                    const Real imagAbs = abs(value.imag());
                    const Real realAbs = abs(value.real());
                    if (imagAbs > imagMax) imagMax = imagAbs;
                    if (realAbs > realMax) realMax = realAbs;
                    const int idxa = pstage_index(cell, a, mode, s_, d_);
                    future_value(q, idxa) += value.real();
                }
            }
            localImag[cm] = imagMax;
            localReal[cm] = realMax;
        });

        for (int cm = 0; cm < channels_; ++cm) {
            if (localImag[cm] > maxFftImag_) maxFftImag_ = localImag[cm];
            if (localReal[cm] > maxFftReal_) maxFftReal_ = localReal[cm];
        }
    }

    void verify_against_direct(int step, const Vec& accelerated, const Vec& massDiag) {
        Vec direct(static_cast<std::size_t>(width_), 0);
        spatial_for(channels_, [&](int cm) {
            const int cell = cm/d_;
            const int mode = cm - cell*d_;
            const Real mass = massDiag[modal_index(cell, mode, d_)];
            for (int a = 0; a < s_; ++a) {
                Real acc = 0;
                for (int lag = 1; lag <= step; ++lag) {
                    const Matrix& kernel = weights_.W[lag];
                    const int sample = step - lag;
                    for (int b = 0; b < s_; ++b) {
                        const int idxb = pstage_index(cell, b, mode, s_, d_);
                        acc += kernel(a, b)*history_value(sample, idxb);
                    }
                }
                const int idxa = pstage_index(cell, a, mode, s_, d_);
                direct[idxa] = mass*acc;
            }
        });

        Real maxDiff = 0;
        Real maxReference = 0;
        for (int i = 0; i < width_; ++i) {
            const Real diff = abs(accelerated[i] - direct[i]);
            const Real reference = abs(direct[i]);
            if (diff > maxDiff) maxDiff = diff;
            if (reference > maxReference) maxReference = reference;
        }
        const Real denom = maxReference > 1 ? maxReference : Real(1);
        const Real relative = maxDiff/denom;
        if (relative > maxDirectDifference_) {
            maxDirectDifference_ = relative;
        }
    }
};

Vec endpoint_update(const Vec& sol, const Vec& Un, int var, const RKMethod& rk,
                    const Real& dt, int N, int d) {
    const int s = rk.s;
    const int B = 3*s*d;
    Vec Unext = Un;
    Vec r(static_cast<std::size_t>(s), 0);
    for (int a = 0; a < s; ++a) {
        for (int b = 0; b < s; ++b) {
            r[a] += rk.Ainv(a, b)/dt;
        }
    }
    spatial_for(N, [&](int cell) {
        for (int mode = 0; mode < d; ++mode) {
            Real update = 0;
            for (int a = 0; a < s; ++a) {
                Real K = -r[a]*Un[modal_index(cell, mode, d)];
                for (int b = 0; b < s; ++b) {
                    Real Ustage = sol[cell*B + block_index(var, b, mode, s, d)];
                    K += rk.Ainv(a, b)*Ustage/dt;
                }
                update += rk.b[a]*K;
            }
            Unext[modal_index(cell, mode, d)] += dt*update;
        }
    });
    return Unext;
}

} // namespace

Result solve_cole_cole_dg(const Config& cfg) {
    std::string scheme = cfg.timeScheme.empty() ? cfg.caputoScheme : cfg.timeScheme;
    std::transform(scheme.begin(), scheme.end(), scheme.begin(), [](unsigned char c) {
        return std::toupper(c);
    });
    if (scheme == "RK-CQ" || scheme == "RKCQ") {
        return solve_cole_cole_dg_rk_cq(cfg);
    }
    throw std::runtime_error("C++ migration intentionally omits BDF-CQ time stepping");
}

Result solve_cole_cole_dg_rk_cq(const Config& cfgIn) {
    Config cfg = cfgIn;
    std::string historyAlgorithm = cfg.rkHistoryAlgorithm;
    std::transform(historyAlgorithm.begin(), historyAlgorithm.end(), historyAlgorithm.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    historyAlgorithm.erase(std::remove(historyAlgorithm.begin(), historyAlgorithm.end(), '_'),
                           historyAlgorithm.end());
    historyAlgorithm.erase(std::remove(historyAlgorithm.begin(), historyAlgorithm.end(), '-'),
                           historyAlgorithm.end());
    const bool useBlockFftHistory = (historyAlgorithm == "blockfft" || historyAlgorithm == "fft");
    const bool useDirectHistory = (historyAlgorithm == "direct");
    if (!useBlockFftHistory && !useDirectHistory) {
        throw std::runtime_error("unsupported RK-CQ history algorithm: " + cfg.rkHistoryAlgorithm);
    }
    cfg.rkHistoryAlgorithm = useBlockFftHistory ? "block-fft" : "direct";
    Mesh mesh = make_mesh_1d(cfg);
    Quad quad = gauss_legendre(cfg.NumGLP);
    Basis basis = legendre_basis(quad.x, cfg.k);
    Exact exact = cole_cole_exact(cfg);
    MaxwellOp op = assemble_maxwell_dg(mesh, basis, quad, cfg);
    Vec loadE = modal_load(mesh, basis, quad, [](const Real& x) { return sin(x); });
    Vec loadH = modal_load(mesh, basis, quad, [](const Real& x) { return cos(x); });

    const int M = mesh.N*basis.dim;
    const int d = basis.dim;
    int Nt = ceil_to_int(cfg.T/cfg.dt);
    Real dt = cfg.T/Real(Nt);
    Real dtAlpha = pow(dt, -cfg.alpha);
    Real tauAlpha = pow(cfg.tau0, cfg.alpha);
    RKMethod rk = rk_method_coefficients(cfg.rkMethod);
    const auto weightStart = std::chrono::steady_clock::now();
    RKWeights W = cq_weights_rk(rk, cfg.alpha, Nt - 1, cfg);
    const auto weightEnd = std::chrono::steady_clock::now();

    Result result;
    result.E = Matrix(M, 1);
    result.H = Matrix(M, 1);
    result.P = Matrix(M, 1);

    InitialModal init = initial_modal(mesh, basis, quad, cfg, exact, Real(0));
    Vec Ecur = pack_modal(init.E);
    Vec Hcur = pack_modal(init.H);
    Vec Pcur = pack_modal(init.P);
    BlockThomasSolver solver = build_rk_system_solver(mesh, basis, op, cfg, rk, W.W[0], dt);

    const int s = rk.s;
    const int B = 3*s*d;
    Vec rvec(static_cast<std::size_t>(s), 0);
    for (int a = 0; a < s; ++a) {
        for (int b = 0; b < s; ++b) {
            rvec[a] += rk.Ainv(a, b)/dt;
        }
    }
    Vec p0Stage(static_cast<std::size_t>(mesh.N*s*d), 0);
    spatial_for(mesh.N, [&](int cell) {
        for (int stage = 0; stage < s; ++stage) {
            for (int mode = 0; mode < d; ++mode) {
                p0Stage[pstage_index(cell, stage, mode, s, d)] =
                    Pcur[modal_index(cell, mode, d)];
            }
        }
    });
    std::vector<Vec> PstageHist;
    std::unique_ptr<BlockFftHistoryConvolver> blockHistory;
    if (useBlockFftHistory) {
        blockHistory = std::make_unique<BlockFftHistoryConvolver>(
            W, Nt, mesh.N, d, p0Stage, cfg.rkHistoryFftMinBlock,
            cfg.rkHistoryVerifySteps, cfg.rkCqTargetEps);
    } else {
        PstageHist.assign(static_cast<std::size_t>(Nt),
                          Vec(static_cast<std::size_t>(mesh.N*s*d), 0));
    }

    auto capture = [&](int step) {
        for (const Real& requested : cfg.queryTimes) {
            const Real ratio = requested/dt;
            const int queryStep = static_cast<int>(std::llround(static_cast<double>(ratio)));
            if (queryStep == step) {
                result.snapshotTimes.push_back(Real(step)*dt);
                result.snapshotsE.push_back(Ecur);
                result.snapshotsH.push_back(Hcur);
                result.snapshotsP.push_back(Pcur);
            }
        }
    };
    capture(0);
    double historyWallSeconds = 0;
    const auto timeLoopStart = std::chrono::steady_clock::now();
    for (int n = 0; n < Nt; ++n) {
        Real tn = Real(n)*dt;
        Vec rhs(static_cast<std::size_t>(mesh.N*B), 0);
        Vec histP(static_cast<std::size_t>(mesh.N*s*d), 0);
        if (n > 0) {
            const auto historyStart = std::chrono::steady_clock::now();
            if (useBlockFftHistory) {
                histP = blockHistory->history_for_step(n, op.massDiag);
            } else {
                // Time levels remain serial; independent cells/modes are parallelized.
                spatial_for(mesh.N*d, [&](int cm) {
                    const int cell = cm/d;
                    const int mode = cm - cell*d;
                    const Real mass = op.massDiag[modal_index(cell, mode, d)];
                    for (int a = 0; a < s; ++a) {
                        Real acc = 0;
                        for (int mhist = 1; mhist <= n; ++mhist) {
                            const Matrix& Wm = W.W[mhist];
                            const Vec& Pold = PstageHist[n - mhist];
                            for (int b = 0; b < s; ++b) {
                                const int idxb = pstage_index(cell, b, mode, s, d);
                                acc += Wm(a, b)*(Pold[idxb] - p0Stage[idxb]);
                            }
                        }
                        histP[pstage_index(cell, a, mode, s, d)] = mass*acc;
                    }
                });
            }
            historyWallSeconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - historyStart).count();
        }

        Vec FEstage(static_cast<std::size_t>(s), 0);
        Vec FHstage(static_cast<std::size_t>(s), 0);
        for (int stage = 0; stage < s; ++stage) {
            if (cfg.sourceFree) continue; // Both source vectors remain exactly zero.
            Real ts = tn + rk.c[stage]*dt;
            FEstage[stage] = cfg.epsilon0*cfg.epsilon_inf*exact.gd(ts, 1)
                + exact.r(ts) + exact.A*exact.qd(ts);
            FHstage[stage] = cfg.mu0*exact.rd(ts, 1) - exact.g(ts);
        }

        spatial_for(mesh.N, [&](int cell) {
            for (int stage = 0; stage < s; ++stage) {
                const Real& FEcoef = FEstage[stage];
                const Real& FHcoef = FHstage[stage];
                for (int mode = 0; mode < d; ++mode) {
                    int mid = modal_index(cell, mode, d);
                    Real mass = op.massDiag[mid];
                    int base = cell*B;
                    rhs[base + block_index(VAR_E, stage, mode, s, d)] =
                        FEcoef*loadE[mid]
                        + cfg.epsilon0*cfg.epsilon_inf*rvec[stage]*mass*Ecur[mid]
                        + rvec[stage]*mass*Pcur[mid];
                    rhs[base + block_index(VAR_H, stage, mode, s, d)] =
                        FHcoef*loadH[mid]
                        + cfg.mu0*rvec[stage]*mass*Hcur[mid];

                    Real w0p0 = 0;
                    for (int b = 0; b < s; ++b) {
                        w0p0 += W.W[0](stage, b)*mass*p0Stage[pstage_index(cell, b, mode, s, d)];
                    }
                    rhs[base + block_index(VAR_P, stage, mode, s, d)] =
                        tauAlpha*dtAlpha*(w0p0 - histP[pstage_index(cell, stage, mode, s, d)]);
                }
            }
        });

        Vec sol = solver.solve(rhs);
        Vec pstage = extract_pstage(sol, mesh.N, s, d);
        if (useBlockFftHistory) {
            const auto historyStart = std::chrono::steady_clock::now();
            blockHistory->append(n, pstage);
            historyWallSeconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - historyStart).count();
        } else {
            PstageHist[n] = std::move(pstage);
        }

        Vec Enew = endpoint_update(sol, Ecur, VAR_E, rk, dt, mesh.N, d);
        Vec Hnew = endpoint_update(sol, Hcur, VAR_H, rk, dt, mesh.N, d);
        Vec Pnew = endpoint_update(sol, Pcur, VAR_P, rk, dt, mesh.N, d);

        if (cfg.rkDiagnostic && rk.stifflyAccurate) {
            for (int cell = 0; cell < mesh.N; ++cell) {
                for (int mode = 0; mode < d; ++mode) {
                    int mid = modal_index(cell, mode, d);
                    Real diffE = abs(Enew[mid] - sol[cell*B + block_index(VAR_E, s - 1, mode, s, d)]);
                    Real diffH = abs(Hnew[mid] - sol[cell*B + block_index(VAR_H, s - 1, mode, s, d)]);
                    Real diffP = abs(Pnew[mid] - sol[cell*B + block_index(VAR_P, s - 1, mode, s, d)]);
                    if (diffE > result.maxEndpointDiff) result.maxEndpointDiff = diffE;
                    if (diffH > result.maxEndpointDiff) result.maxEndpointDiff = diffH;
                    if (diffP > result.maxEndpointDiff) result.maxEndpointDiff = diffP;
                }
            }
        }

        Ecur = Enew;
        Hcur = Hnew;
        Pcur = Pnew;
        capture(n + 1);

        if (cfg.verbose && ((n + 1) % std::max(1, Nt/100) == 0 || n + 1 == Nt)) {
            std::cout << "  RK-CQ " << rk.name << " N=" << mesh.N << " k=" << cfg.k
                      << " step " << (n + 1) << "/" << Nt << "\n" << std::flush;
        }
    }
    const auto timeLoopEnd = std::chrono::steady_clock::now();

    spatial_for(M, [&](int i) {
        result.E(i, 0) = Ecur[i];
        result.H(i, 0) = Hcur[i];
        result.P(i, 0) = Pcur[i];
    });

    result.mesh = mesh;
    result.basis = basis;
    result.quad = quad;
    result.cfg = cfg;
    result.exact = cole_cole_exact(result.cfg);
    result.dt = dt;
    result.Nt = Nt;
    result.timeScheme = "RK-CQ";
    result.rk = rk;
    result.rkCqInfo = W.info;
    result.rkHistoryAlgorithm = cfg.rkHistoryAlgorithm;
    result.rkHistoryFftMinBlock = useBlockFftHistory ? cfg.rkHistoryFftMinBlock : 0;
    result.rkWeightWallSeconds = std::chrono::duration<double>(weightEnd - weightStart).count();
    result.rkTimeLoopWallSeconds = std::chrono::duration<double>(timeLoopEnd - timeLoopStart).count();
    result.rkHistoryWallSeconds = historyWallSeconds;
    if (useBlockFftHistory) {
        blockHistory->assert_precision();
        result.rkHistoryMaxImaginaryRatio = blockHistory->max_imaginary_ratio();
        result.rkHistoryMaxDirectDifference = blockHistory->max_direct_difference();
    }
    return result;
}

std::map<std::string, Real> compute_errors(const Result& result) {
    const Config& cfg = result.cfg;
    const Mesh& mesh = result.mesh;
    const Basis& basis = result.basis;
    const Quad& quad = result.quad;
    const Exact& exact = result.exact;
    const int N = mesh.N;
    const int d = basis.dim;
    const int M = N*d;
    const Real t = cfg.T;
    const Real gFinal = exact.g(t);
    const Real rFinal = exact.r(t);
    const Real pFinalScale = exact.A*exact.q(t);
    auto exactEValue = [&](const Real& x) -> Real { return Real(sin(x)*gFinal); };
    auto exactHValue = [&](const Real& x) -> Real { return Real(cos(x)*rFinal); };
    auto exactPValue = [&](const Real& x) -> Real { return Real(sin(x)*pFinalScale); };

    Vec Evec(static_cast<std::size_t>(M));
    Vec Hvec(static_cast<std::size_t>(M));
    Vec Pvec(static_cast<std::size_t>(M));
    spatial_for(M, [&](int i) {
        Evec[i] = result.E(i, result.E.cols - 1);
        Hvec[i] = result.H(i, result.H.cols - 1);
        Pvec[i] = result.P(i, result.P.cols - 1);
    });
    Matrix Eh = unpack_modal(Evec, N, d);
    Matrix Hh = unpack_modal(Hvec, N, d);
    Matrix Ph = unpack_modal(Pvec, N, d);

    auto l2_func = [&](const Matrix& U, const auto& func) -> Real {
        Real acc = deterministic_cell_sum(N, [&](int j) {
            Real local = 0;
            for (std::size_t q = 0; q < quad.x.size(); ++q) {
                Real xq = mesh.xc[j] + mesh.J[j]*quad.x[q];
                Real uh = 0;
                for (int m = 0; m < d; ++m) {
                    uh += U(j, m)*basis.phi(static_cast<int>(q), m);
                }
                Real diff = func(xq) - uh;
                local += mesh.J[j]*quad.w[q]*diff*diff;
            }
            return Real(local);
        });
        return Real(sqrt(acc));
    };

    auto l2_modal = [&](const Matrix& U) -> Real {
        Real acc = deterministic_cell_sum(N, [&](int j) {
            Real local = 0;
            for (int m = 0; m < d; ++m) {
                Real v = U(j, m);
                local += mesh.h[j]*basis.mm[m]*v*v;
            }
            return Real(local);
        });
        return Real(sqrt(acc));
    };

    auto cell_average_error = [&](const Matrix& U, const auto& func) -> Real {
        Real acc = deterministic_cell_sum(N, [&](int j) {
            Real exactInt = 0;
            for (std::size_t q = 0; q < quad.x.size(); ++q) {
                Real xq = mesh.xc[j] + mesh.J[j]*quad.x[q];
                exactInt += quad.w[q]*func(xq);
            }
            Real exactAvg = mesh.J[j]*exactInt/mesh.h[j];
            Real diff = exactAvg - U(j, 0);
            return Real(diff*diff);
        });
        return Real(sqrt(acc/Real(N)));
    };

    auto flux_error = [&]() -> std::pair<Real, Real> {
        Real ee = cfg.epsilon0*cfg.epsilon_inf;
        Real cH = sqrt(cfg.mu0/ee);
        Real cE = sqrt(ee/cfg.mu0);
        Vec Eleft(static_cast<std::size_t>(N), 0), Eright(static_cast<std::size_t>(N), 0);
        Vec Hleft(static_cast<std::size_t>(N), 0), Hright(static_cast<std::size_t>(N), 0);
        spatial_for(N, [&](int j) {
            for (int m = 0; m < d; ++m) {
                Eleft[j] += Eh(j, m)*basis.phiL[m];
                Eright[j] += Eh(j, m)*basis.phiR[m];
                Hleft[j] += Hh(j, m)*basis.phiL[m];
                Hright[j] += Hh(j, m)*basis.phiR[m];
            }
        });
        Vec ehat(static_cast<std::size_t>(N + 1), 0), hhat(static_cast<std::size_t>(N + 1), 0);
        ehat[0] = 0;
        hhat[0] = Hleft[0] + cE*Eleft[0];
        ehat[N] = 0;
        hhat[N] = Hright[N - 1] - cE*Eright[N - 1];
        spatial_for(N - 1, [&](int offset) {
            int iface = offset + 1;
            Real eMinus = Eright[iface - 1];
            Real ePlus = Eleft[iface];
            Real hMinus = Hright[iface - 1];
            Real hPlus = Hleft[iface];
            ehat[iface] = Real(0.5)*(eMinus + ePlus) + Real(0.5)*cH*(hPlus - hMinus);
            hhat[iface] = Real(0.5)*(hMinus + hPlus) + Real(0.5)*cE*(ePlus - eMinus);
        });
        Real accE = deterministic_cell_sum(N + 1, [&](int i) {
            Real diffE = exactEValue(mesh.x[i]) - ehat[i];
            return Real(diffE*diffE);
        });
        Real accH = deterministic_cell_sum(N + 1, [&](int i) {
            Real diffH = exactHValue(mesh.x[i]) - hhat[i];
            return Real(diffH*diffH);
        });
        Real fluxE = sqrt(accE/Real(N + 1));
        Real fluxH = sqrt(accH/Real(N + 1));
        return std::pair<Real, Real>(fluxE, fluxH);
    };

    std::map<std::string, Real> err;
    auto flux = flux_error();
    Real fluxEValue = flux.first;
    Real fluxHValue = flux.second;

    err["L2_E"] = l2_func(Eh, exactEValue);
    err["L2_H"] = l2_func(Hh, exactHValue);
    err["L2_P"] = l2_func(Ph, exactPValue);

    err["cell_E"] = cell_average_error(Eh, exactEValue);
    err["cell_H"] = cell_average_error(Hh, exactHValue);
    err["cell_P"] = cell_average_error(Ph, exactPValue);
    err["cell_EHP"] = err["cell_E"] + err["cell_H"] + err["cell_P"];

    Matrix Estar, Hstar;
    project_star_maxwell(mesh, basis, quad, cfg, exact, t, 0, Estar, Hstar);
    Matrix Ppi = project_l2_func(mesh, basis, quad, exactPValue);
    err["proj_E"] = l2_modal(mat_sub(Eh, Estar));
    err["proj_H"] = l2_modal(mat_sub(Hh, Hstar));

    if (cfg.computeCorrectionErrors) {
        Correction corr = correction_functions(mesh, basis, quad, cfg, exact, t);
        Matrix EI = mat_sub(Estar, corr.wE);
        Matrix HI = mat_sub(Hstar, corr.wH);
        Matrix PI = mat_sub(Ppi, corr.wP);
        err["corr_E"] = l2_modal(mat_sub(Eh, EI));
        err["corr_H"] = l2_modal(mat_sub(Hh, HI));
        err["corr_P"] = l2_modal(mat_sub(Ph, PI));
    } else {
        err["corr_E"] = nan_value();
        err["corr_H"] = nan_value();
        err["corr_P"] = nan_value();
    }

    err["flux_E"] = fluxEValue;
    err["flux_H"] = fluxHValue;
    err["flux_EH"] = fluxEValue + fluxHValue;
    err["dt"] = result.dt;
    err["Nt"] = Real(result.Nt);
    return err;
}

Real modal_l2_diff(const Vec& u, const Vec& v, const Mesh& mesh, const Basis& basis) {
    const int d = basis.dim;
    Real acc = deterministic_cell_sum(mesh.N, [&](int j) {
        Real local = 0;
        for (int m = 0; m < d; ++m) {
            Real diff = u[modal_index(j, m, d)] - v[modal_index(j, m, d)];
            local += mesh.h[j]*basis.mm[m]*diff*diff;
        }
        return Real(local);
    });
    return sqrt(acc);
}

std::vector<ErrorRow> run_convergence_study(const Config& cfg) {
    const std::vector<std::string> names = {
        "L2_E", "L2_H", "L2_P",
        "proj_E", "proj_H",
        "corr_E", "corr_H", "corr_P",
        "cell_E", "cell_H", "cell_P", "cell_EHP",
        "flux_E", "flux_H", "flux_EH"
    };
    std::vector<ErrorRow> rows;
    rows.reserve(cfg.NList.size());
    for (std::size_t iN = 0; iN < cfg.NList.size(); ++iN) {
        Config runCfg = cfg;
        runCfg.N = cfg.NList[iN];
        std::cout << "  N=" << runCfg.N << " started...\n" << std::flush;
        auto tStart = std::chrono::steady_clock::now();
        Result result = solve_cole_cole_dg(runCfg);
        auto tSolve = std::chrono::steady_clock::now();
        std::map<std::string, Real> err = compute_errors(result);
        auto tError = std::chrono::steady_clock::now();
        ErrorRow row;
        row.N = runCfg.N;
        row.dt = result.dt;
        row.Nt = result.Nt;
        for (const std::string& name : names) {
            row.values[name] = err[name];
            row.values["ord_" + name] = nan_value();
            if (!rows.empty()) {
                Real prev = rows.back().values.at(name);
                Real cur = err[name];
                if (!is_nan(prev) && !is_nan(cur) && prev > 0 && cur > 0) {
                    row.values["ord_" + name] = log(prev/cur)/log(Real(2));
                }
            }
        }
        rows.push_back(row);
        double solveSeconds = std::chrono::duration<double>(tSolve - tStart).count();
        double errorSeconds = std::chrono::duration<double>(tError - tSolve).count();
        double totalSeconds = std::chrono::duration<double>(tError - tStart).count();
        std::ostringstream timing;
        timing << std::fixed << std::setprecision(3)
               << "  N=" << runCfg.N << " finished in " << totalSeconds << " s"
               << " (solve " << solveSeconds << " s, error " << errorSeconds << " s, Nt="
               << result.Nt << ")";
        std::cout << timing.str() << "\n" << std::flush;
    }
    return rows;
}

namespace {

Real expected_order(const std::string& name, int k) {
    if (name.rfind("L2_", 0) == 0) {
        return Real(k + 1);
    }
    if (name.rfind("proj_", 0) == 0) {
        return Real(k + 2);
    }
    if (name.rfind("corr_", 0) == 0) {
        return Real(2*k + 1);
    }
    if (name == "cell_E" || name == "cell_H" || name == "flux_E"
        || name == "flux_H" || name == "flux_EH") {
        return Real(2*k + 1);
    }
    return nan_value();
}

void ensure_parent_dir(const std::string& path) {
    std::filesystem::path p(path);
    if (!p.parent_path().empty()) {
        std::filesystem::create_directories(p.parent_path());
    }
}

} // namespace

void write_spatial_csv(const std::vector<ErrorRow>& rows, const std::string& path) {
    ensure_parent_dir(path);
    const std::vector<std::string> fields = {
        "L2_E", "L2_H", "L2_P",
        "proj_E", "proj_H",
        "corr_E", "corr_H", "corr_P",
        "cell_E", "cell_H", "cell_P", "cell_EHP",
        "flux_E", "flux_H", "flux_EH"
    };
    std::ofstream out(path);
    out << "N,dt,Nt";
    for (const auto& f : fields) {
        out << "," << f << ",ord_" << f;
    }
    out << "\n";
    for (const auto& row : rows) {
        out << row.N << "," << real_to_string(row.dt) << "," << row.Nt;
        for (const auto& f : fields) {
            out << "," << real_to_string(row.values.at(f))
                << "," << real_to_string(row.values.at("ord_" + f));
        }
        out << "\n";
    }
}

void write_spatial_orders_csv(const std::vector<ErrorRow>& rows, const Config& cfg,
                              const std::string& path) {
    ensure_parent_dir(path);
    const std::vector<std::string> fields = {
        "L2_E", "L2_H", "L2_P",
        "proj_E", "proj_H",
        "corr_E", "corr_H", "corr_P",
        "cell_E", "cell_H", "cell_P", "cell_EHP",
        "flux_E", "flux_H", "flux_EH"
    };
    std::ofstream out(path);
    out << "quantity,lastError,lastOrder,tailMeanOrder,theoryOrder,tailRowsUsed\n";
    for (const auto& f : fields) {
        Real lastError = nan_value();
        Real lastOrder = nan_value();
        Real tailMean = nan_value();
        int tailRows = 0;
        std::vector<Real> orders;
        for (const auto& row : rows) {
            Real err = row.values.at(f);
            if (!is_nan(err) && err > 0) {
                lastError = err;
            }
            Real ord = row.values.at("ord_" + f);
            if (!is_nan(ord)) {
                orders.push_back(ord);
            }
        }
        if (!orders.empty()) {
            lastOrder = orders.back();
            int start = static_cast<int>(orders.size()) >= 2 ? static_cast<int>(orders.size()) - 2 : 0;
            Real sum = 0;
            for (int i = start; i < static_cast<int>(orders.size()); ++i) {
                sum += orders[i];
                ++tailRows;
            }
            tailMean = sum/Real(tailRows);
        }
        out << f << "," << real_to_string(lastError) << "," << real_to_string(lastOrder)
            << "," << real_to_string(tailMean) << ","
            << real_to_string(expected_order(f, cfg.k)) << "," << tailRows << "\n";
    }
}

void write_time_csv(const std::vector<TimeRow>& rows, const std::string& path) {
    ensure_parent_dir(path);
    std::ofstream out(path);
    out << "rkMethod,dt,Nt,referenceDt,referenceNt,"
        << "time_E,ord_time_E,time_H,ord_time_H,time_P,ord_time_P,"
        << "exact_L2_E,exact_L2_H,exact_L2_P\n";
    for (const auto& row : rows) {
        out << row.rkMethod << "," << real_to_string(row.dt) << "," << row.Nt << ","
            << real_to_string(row.referenceDt) << "," << row.referenceNt << ","
            << real_to_string(row.time_E) << "," << real_to_string(row.ord_time_E) << ","
            << real_to_string(row.time_H) << "," << real_to_string(row.ord_time_H) << ","
            << real_to_string(row.time_P) << "," << real_to_string(row.ord_time_P) << ","
            << real_to_string(row.exact_L2_E) << "," << real_to_string(row.exact_L2_H) << ","
            << real_to_string(row.exact_L2_P) << "\n";
    }
}

void write_time_orders_csv(const std::vector<TimeRow>& rows, const std::string& path) {
    ensure_parent_dir(path);
    std::ofstream out(path);
    out << "rkMethod,quantity,lastOrder,tailMeanOrder,tailRowsUsed\n";
    std::vector<std::string> methods;
    for (const auto& row : rows) {
        if (std::find(methods.begin(), methods.end(), row.rkMethod) == methods.end()) {
            methods.push_back(row.rkMethod);
        }
    }
    const std::vector<std::string> fields = {"time_E", "time_H", "time_P"};
    for (const auto& method : methods) {
        for (const auto& field : fields) {
            std::vector<Real> orders;
            for (const auto& row : rows) {
                if (row.rkMethod != method) {
                    continue;
                }
                Real ord = nan_value();
                if (field == "time_E") ord = row.ord_time_E;
                if (field == "time_H") ord = row.ord_time_H;
                if (field == "time_P") ord = row.ord_time_P;
                if (!is_nan(ord)) {
                    orders.push_back(ord);
                }
            }
            Real last = nan_value();
            Real tailMean = nan_value();
            int tailRows = 0;
            if (!orders.empty()) {
                last = orders.back();
                int start = static_cast<int>(orders.size()) >= 2 ? static_cast<int>(orders.size()) - 2 : 0;
                Real sum = 0;
                for (int i = start; i < static_cast<int>(orders.size()); ++i) {
                    sum += orders[i];
                    ++tailRows;
                }
                tailMean = sum/Real(tailRows);
            }
            out << method << "," << field << "," << real_to_string(last) << ","
                << real_to_string(tailMean) << "," << tailRows << "\n";
        }
    }
}

} // namespace colecole
