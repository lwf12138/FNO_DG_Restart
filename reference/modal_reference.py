"""Analytic modal solution of source-free Cole-Cole Maxwell, evaluated in float64.

For kappa=m*pi/L, eps*e'=-kappa*h-p', mu*h'=kappa*e,
tau**alpha D_C**alpha p+p=delta*e, p(0)=0.
The exact Laplace transform is evaluated by two independent representations:
a convergent fractional power series and fixed Talbot inversion.
No DG result is used to construct this reference.
"""
import math
import numpy as np
from scipy.special import gammaln

EPS, MU, DELTA = 2., 1., 1.
REFERENCE_TOL = 5e-9


def laplace_transfer(s, kappa, alpha, tau):
    chi = DELTA / (1 + (tau * s)**alpha)
    denominator = EPS*s*s + kappa*kappa/MU + s*s*chi
    electric = np.stack([EPS*s/denominator, -kappa/denominator], axis=-1)
    magnetic = kappa/(MU*s[..., None])*electric
    magnetic[..., 1] += 1/s
    polarization = chi[..., None]*electric
    return np.stack([electric, magnetic, polarization], axis=-2)


def talbot_transfer(t, kappa, alpha, tau, degree=24):
    if t == 0:
        return np.array([[1., 0.], [0., 1.], [0., 0.]])
    r = 2 * degree / 5
    theta = np.arange(1, degree) * np.pi / degree
    cot = 1 / np.tan(theta)
    delta = r * theta * (cot + 1j)
    integrand = np.exp(delta)[:, None, None] * laplace_transfer(delta/t, kappa, alpha, tau)
    weight = 1 + 1j*theta*(1 + cot*cot) - 1j*cot
    zero = np.exp(r) * laplace_transfer(np.array(r/t, dtype=complex), kappa, alpha, tau) / 2
    total = zero + np.sum(integrand * weight[:, None, None], axis=0)
    return (2 / (5*t)) * total.real


def series_coefficients(kappa, alpha, tau, mmax, nmax):
    lam = tau**(-alpha)
    A = (1 + DELTA/EPS) * lam
    B = kappa*kappa/(EPS*MU)
    C = B*lam
    coeff = np.zeros((mmax+1, nmax+1))
    coeff[0, 0] = 1
    for m in range(mmax+1):
        for n in range(nmax+1):
            if m == 0 and n == 0:
                continue
            coeff[m, n] = -(A*coeff[m-1, n] if m else 0) - (B*coeff[m, n-1] if n else 0) - (C*coeff[m-1, n-1] if m and n else 0)
    electric_coeff = coeff.copy()
    electric_coeff[1:] += lam * coeff[:-1]
    return coeff, electric_coeff


def series_transfer(times, kappa, alpha, tau, mmax=192, nmax=20):
    coeff, ec = series_coefficients(kappa, alpha, tau, mmax, nmax)
    exponent = alpha*np.arange(mmax+1)[:, None] + 2*np.arange(nmax+1)[None]
    lam = tau**(-alpha)
    out = np.empty((len(times), 3, 2))
    max_abs_sum = 0.
    for ti, t in enumerate(times):
        if t == 0:
            out[ti] = [[1, 0], [0, 1], [0, 0]]
            continue
        def summation(c, shift):
            nonlocal max_abs_sum
            powers = exponent + shift
            terms = c * np.exp((powers-1)*math.log(t) - gammaln(powers))
            max_abs_sum = max(max_abs_sum, float(np.sum(np.abs(terms))))
            return math.fsum(terms.flat)
        e1 = summation(ec, 1)
        e2 = summation(ec, 2)
        e3 = summation(ec, 3)
        p1 = summation(coeff, alpha+1)
        p2 = summation(coeff, alpha+2)
        out[ti] = [[e1, -kappa/EPS*e2],
                   [kappa/MU*e2, 1-kappa*kappa/(EPS*MU)*e3],
                   [DELTA*lam*p1, -DELTA*lam*kappa/EPS*p2]]
    return out, max_abs_sum


def validated_transfer(alpha, tau, times):
    """Return [time, E/H/P, mode, input a/b] and numerical evaluation checks."""
    modes, reports = [], []
    for m in range(1, 4):
        kappa = m/2
        series, abs_sum = series_transfer(times, kappa, alpha, tau)
        refined, _ = series_transfer(times, kappa, alpha, tau, 256, 28)
        talbot24 = np.array([talbot_transfer(t, kappa, alpha, tau, 24) for t in times])
        talbot32 = np.array([talbot_transfer(t, kappa, alpha, tau, 32) for t in times])
        checks = {
            "mode": m,
            "series_truncation_difference": float(np.max(np.abs(series-refined))),
            "series_vs_talbot24": float(np.max(np.abs(refined-talbot24))),
            "series_vs_talbot32": float(np.max(np.abs(refined-talbot32))),
            "talbot_degree_difference": float(np.max(np.abs(talbot24-talbot32))),
            "series_max_absolute_term_sum": abs_sum,
        }
        discrepancies = [checks[key] for key in ["series_truncation_difference", "series_vs_talbot24", "series_vs_talbot32", "talbot_degree_difference"]]
        checks["passed"] = bool(max(discrepancies) <= REFERENCE_TOL and np.isfinite(refined).all())
        if not checks["passed"]:
            raise RuntimeError(f"Reference evaluation failed alpha={alpha}, tau={tau}: {checks}")
        modes.append(refined)
        reports.append(checks)
    return np.stack(modes, axis=2), reports


def modal_amplitudes(transfer, a, b):
    return np.einsum("tfmj,mj->tfm", transfer, np.column_stack([a, b]))


def evaluate_fields(amplitudes, x):
    wave = np.arange(1, 4)[:, None] * np.asarray(x).reshape(1, -1) / 2
    basis = np.stack([np.sin(wave), np.cos(wave), np.sin(wave)])
    return np.einsum("tfm,fmx->tfx", amplitudes, basis)
