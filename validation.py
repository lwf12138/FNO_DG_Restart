"""Independent DG reconstruction and cellwise L2 comparison with the analytic solution."""
import json
import math
from pathlib import Path
import numpy as np
from reference.modal_reference import evaluate_fields, modal_amplitudes


def monic_basis(xi, degree):
    leading = np.array([math.comb(2*m, m)/2**m for m in range(degree+1)])
    return np.polynomial.legendre.legvander(xi, degree)/leading


def sample_coefficients(coeff, x, length=2*np.pi):
    n = coeff.shape[2]
    h = length/n
    z = np.asarray(x)/h
    right = np.minimum(np.floor(z+1e-12).astype(int), n-1)
    left = right.copy()
    interface = np.isclose(z, np.rint(z), atol=1e-12, rtol=0) & (z > 0)
    left[interface] = np.maximum(np.rint(z[interface]).astype(int)-1, 0)
    out = np.zeros((*coeff.shape[:2], len(x)))
    for cells in (left, right):
        xi = 2*(np.asarray(x)-(cells+.5)*h)/h
        basis = monic_basis(xi, coeff.shape[-1]-1)
        out += .5*np.einsum('tfpm,pm->tfp', coeff[:, :, cells, :], basis)
    return out


def compare_case(directory, transfer, absolute_tolerance=1e-6, relative_tolerance=1e-3):
    directory = Path(directory)
    meta = json.loads((directory/'metadata.json').read_text())
    coeff = np.fromfile(directory/'coefficients.f64', dtype='<f8').reshape(meta['coefficient_shape'])
    csv = np.loadtxt(directory/'fields.csv', delimiter=',', skiprows=1).reshape(len(meta['times']), meta['points'], 5)
    fields = csv[:, :, 2:].transpose(0, 2, 1)
    x = csv[0, :, 1]
    if not (np.isfinite(coeff).all() and np.isfinite(csv).all()):
        raise ValueError('Nonfinite DG data')
    if not np.allclose(csv[:, :, 0], np.asarray(meta['times'])[:, None], atol=5e-14, rtol=0):
        raise ValueError('Unexpected snapshot times')
    if not np.allclose(csv[:, :, 1], np.linspace(0, 2*np.pi, meta['points'])[None], atol=5e-14, rtol=0):
        raise ValueError('Unexpected spatial grid')
    reconstructed = sample_coefficients(coeff, x)
    sample_difference = float(np.max(np.abs(reconstructed-fields)))
    if sample_difference > 1e-12:
        raise ValueError(f'DG polynomial sampling mismatch: {sample_difference}')
    amplitudes = modal_amplitudes(transfer, meta['a'], meta['b'])
    xi, weights = np.polynomial.legendre.leggauss(24)
    h = meta['L']/meta['N']
    xq = (np.arange(meta['N'])[:, None]+.5)*h + .5*h*xi
    dg = np.einsum('tfcm,qm->tfcq', coeff, monic_basis(xi, meta['k']))
    exact = evaluate_fields(amplitudes, xq).reshape(dg.shape)
    error = np.sqrt(.5*h*np.einsum('tfcq,q->tf', (dg-exact)**2, weights))
    norm = np.sqrt(.5*h*np.einsum('tfcq,q->tf', exact**2, weights))
    analytic_norm = np.sqrt(np.pi*np.sum(amplitudes**2, axis=2))
    norm_difference = float(np.max(np.abs(norm-analytic_norm)))
    if norm_difference > 1e-12:
        raise ValueError('Exact-field quadrature check failed')
    nonzero = norm > 0
    relative = np.full_like(error, np.nan)
    np.divide(error, norm, out=relative, where=nonzero)
    bound = absolute_tolerance+relative_tolerance*norm
    qualified = error <= bound
    summary = {
        'passed': bool(qualified.all()), 'max_tolerance_ratio': float(np.max(error/bound)),
        'max_absolute_l2_E_H_P': np.max(error, axis=0).tolist(),
        'max_relative_l2_E_H_P': np.nanmax(relative, axis=0).tolist(),
        'sampling_reconstruction_max_difference': sample_difference,
        'exact_norm_quadrature_max_difference': norm_difference,
        'failed_time_field_indices': np.argwhere(~qualified).tolist(),
    }
    np.savez(directory/'errors.npz', absolute_l2=error, exact_l2=norm, relative_l2=relative,
             relative_defined=nonzero, tolerance=bound, passed=qualified)
    (directory/'validation.json').write_text(json.dumps(summary, indent=2)+'\n')
    return dict(fields=fields, exact_fields=evaluate_fields(amplitudes, x), coefficients=coeff,
                absolute_l2=error, exact_l2=norm, relative_l2=relative, relative_defined=nonzero,
                tolerance=bound, passed=qualified, summary=summary, metadata=meta)
