#!/usr/bin/env python3
"""Build source-free DG numerical-reference data. Does not train a network."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from reference.modal_reference import validated_transfer, evaluate_fields, modal_amplitudes
from run_case import ROOT, run_case
from validation import compare_case


def save_json(path, content):
    Path(path).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def fingerprint():
    files = ['config.json', 'src/cole_cole.cpp', 'include/cole_cole.hpp', 'src/dataset_driver.cpp',
             'reference/modal_reference.py', 'validation.py', 'run_case.py', 'generate_dataset.py']
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}


def case(directory, alpha, tau, a, b, cfg, transfer, resume=False, N=None, dt=None):
    directory = Path(directory)
    identity = dict(alpha=float(alpha), tau=float(tau), a=np.asarray(a).tolist(), b=np.asarray(b).tolist(),
                    N=cfg['N'] if N is None else N, dt=cfg['dt'] if dt is None else dt,
                    source_hashes=fingerprint())
    marker = directory/'identity.json'
    can_resume = resume and marker.exists() and json.loads(marker.read_text()) == identity
    if not can_resume:
        run_case(directory, alpha, tau, a, b, cfg, N=N, dt=dt)
    result = compare_case(directory, transfer, cfg['absolute_l2_tolerance'], cfg['relative_l2_tolerance'])
    save_json(marker, identity)
    return result


def preflight(cfg, resume):
    rng = np.random.default_rng(2026)  # Independent of the dataset RNG.
    reports = []
    for i, (alpha, tau) in enumerate([(.3, .5), (.3, 2), (.7, .5), (.7, 2), (.6, 1)]):
        a, b = rng.uniform(-1, 1, (2, 3))/np.arange(1, 4)**2
        transfer, checks = validated_transfer(alpha, tau, cfg['times'])
        result = case(ROOT/f'checks/preflight/case_{i:02d}', alpha, tau, a, b, cfg, transfer, resume)
        record = dict(alpha=alpha, tau=tau, a=a.tolist(), b=b.tolist(), reference_checks=checks,
                      coarse=result['summary'])
        if i in (0, 3, 4):
            fine = case(ROOT/f'checks/preflight/case_{i:02d}_fine', alpha, tau, a, b, cfg, transfer,
                        resume, N=128, dt=.00025)
            record['fine'] = fine['summary']
            record['coarse_fine_sample_max_difference_E_H_P'] = np.max(
                np.abs(result['fields']-fine['fields']), axis=(0, 2)).tolist()
            record['coarse_fine_settings'] = [[64, 2, .0005], [128, 2, .00025]]
        record['passed'] = record['coarse']['passed'] and record.get('fine', {'passed': True})['passed']
        reports.append(record)
        print(f'Preflight {i+1}/5: passed={record["passed"]}, tolerance_ratio={result["summary"]["max_tolerance_ratio"]:.4g}', flush=True)
    summary = dict(passed=all(r['passed'] for r in reports), cases=reports, source_hashes=fingerprint())
    save_json(ROOT/'checks/preflight_summary.json', summary)
    if not summary['passed']:
        raise RuntimeError('Preflight failed; full dataset generation stopped.')
    return summary


def plan(cfg):
    alpha = np.linspace(cfg['alpha_min'], cfg['alpha_max'], cfg['alpha_count'])
    tau = np.geomspace(cfg['tau_min'], cfg['tau_max'], cfg['tau_count'])
    materials = [(float(a), float(t)) for a in alpha for t in tau]
    rng = np.random.default_rng(cfg['seed'])
    coefficients = rng.uniform(-1, 1, (len(materials), cfg['initial_draws_per_material'], 2, 3))/np.arange(1, 4)**2
    permutation = rng.permutation(len(materials))
    splits = np.empty(len(materials), dtype='U10')
    start = 0
    for name, count in zip(['train', 'validation', 'test'], cfg['material_group_split_counts']):
        splits[permutation[start:start+count]] = name
        start += count
    if start != len(materials):
        raise ValueError('Material split does not cover the complete grid')
    rows = []
    for material, (a, t) in enumerate(materials):
        for draw, values in enumerate(coefficients[material]):
            sid = len(rows)
            row = dict(sample_id=f'sample_{sid:04d}', material_id=f'material_{material:02d}',
                       material_index=material, draw=draw, alpha=a, tau=t, split=str(splits[material]))
            row.update({f'a{m+1}': float(values[0,m]) for m in range(3)})
            row.update({f'b{m+1}': float(values[1,m]) for m in range(3)})
            rows.append(row)
    with (ROOT/'data/samples.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(ROOT/'data/materials.json', [dict(material_id=f'material_{i:02d}', alpha=a, tau=t,
              split=str(splits[i])) for i, (a,t) in enumerate(materials)])
    return rows, materials


def dataset(cfg, workers, resume):
    preflight_report = json.loads((ROOT/'checks/preflight_summary.json').read_text())
    if not preflight_report.get('passed') or preflight_report.get('source_hashes') != fingerprint():
        raise RuntimeError('A successful preflight for these exact sources/settings is required')
    rows, materials = plan(cfg)
    references, evaluation_reports = {}, []
    for i, (alpha, tau) in enumerate(materials):
        transfer, checks = validated_transfer(alpha, tau, cfg['times'])
        references[i] = transfer
        evaluation_reports.append(dict(material_id=f'material_{i:02d}', alpha=alpha, tau=tau, checks=checks))
        np.savez(ROOT/f'reference/material_{i:02d}.npz', transfer=transfer,
                 times=cfg['times'], alpha=alpha, tau=tau)
    save_json(ROOT/'checks/reference_evaluation.json', evaluation_reports)
    print(f'Independent analytic references checked for {len(materials)} material groups.', flush=True)
    start = time.perf_counter()

    def alpha_batch(alpha_index):
        output = []
        for row in rows:
            mi = row['material_index']
            if mi//cfg['tau_count'] != alpha_index:
                continue
            a, b = [[row[f'{prefix}{m}'] for m in range(1,4)] for prefix in ['a','b']]
            result = case(ROOT/'runs'/row['sample_id'], row['alpha'], row['tau'], a, b, cfg,
                          references[mi], resume)
            output.append((row['sample_id'], result))
        return output

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(alpha_batch, ai) for ai in range(cfg['alpha_count'])]
        for future in as_completed(futures):
            results.update(future.result())
            print(f'DG trajectories checked: {len(results)}/{len(rows)}', flush=True)
    batch_wall_seconds = time.perf_counter()-start
    ordered = [results[r['sample_id']] for r in rows]
    stack = lambda key: np.stack([r[key] for r in ordered])
    fields, exact_fields, error, norm = [stack(k) for k in ['fields','exact_fields','absolute_l2','exact_l2']]
    relative, defined, tolerance, passed = [stack(k) for k in ['relative_l2','relative_defined','tolerance','passed']]
    coefficient_array = np.array([[[r[f'{p}{m}'] for m in range(1,4)] for p in ['a','b']] for r in rows])
    x = np.linspace(0, cfg['L'], cfg['points'])
    initial = np.zeros((len(rows),3,len(x)))
    wave = np.arange(1,4)[:,None]*x[None]/2
    initial[:,0] = coefficient_array[:,0]@np.sin(wave)
    initial[:,1] = coefficient_array[:,1]@np.cos(wave)
    initial[:,0,[0,-1]] = 0  # Enforce mathematical sine endpoint values.
    ids = np.array([r['sample_id'] for r in rows])
    groups = np.array([r['material_id'] for r in rows])
    splits = np.array([r['split'] for r in rows])
    group_sets = {name:set(groups[splits==name]) for name in ['train','validation','test']}
    split_overlap = any(group_sets[a]&group_sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    all_finite = all(np.isfinite(a).all() for a in [fields, exact_fields, initial, error, norm, tolerance])
    material_unique = len(set(materials)) == len(materials)
    sample_unique = len(set(ids.tolist())) == len(rows)
    full_input = np.column_stack([np.array([[r['alpha'],r['tau']] for r in rows]),coefficient_array.reshape(len(rows),-1)])
    input_unique = len(np.unique(full_input,axis=0)) == len(rows)
    overall = bool(passed.all() and all_finite and material_unique and sample_unique and input_unique and not split_overlap)
    max_index = np.unravel_index(np.argmax(error/tolerance), error.shape)
    summary = dict(passed=overall, actual_samples=len(rows), qualified_samples=int(np.all(passed,axis=(1,2)).sum()),
        material_groups=len(materials), sample_split={s:int(np.sum(splits==s)) for s in group_sets},
        material_split={s:len(v) for s,v in group_sets.items()}, shape=list(fields.shape),
        max_absolute_l2_E_H_P=np.max(error,axis=(0,1)).tolist(),
        max_relative_l2_E_H_P=np.nanmax(relative,axis=(0,1)).tolist(),
        max_tolerance_ratio=float(np.max(error/tolerance)), worst_case=dict(sample_id=str(ids[max_index[0]]),
        time=float(cfg['times'][max_index[1]]),field=['E','H','P'][max_index[2]]),
        all_physical_arrays_finite=all_finite, unique_material_groups=material_unique,
        repeated_material_rows_expected=cfg['initial_draws_per_material'], unique_full_inputs=input_unique,
        unique_sample_ids=sample_unique, material_split_overlap=split_overlap,
        undefined_relative_errors=int((~defined).sum()), undefined_relative_policy='NaN only when exact L2 norm is zero; report absolute error',
        reference_max_evaluation_difference=max(check[key] for r in evaluation_reports for check in r['checks']
            for key in ['series_truncation_difference','series_vs_talbot24','series_vs_talbot32','talbot_degree_difference']),
        max_sampling_reconstruction_difference=max(r['summary']['sampling_reconstruction_max_difference'] for r in ordered),
        batch_wall_seconds=batch_wall_seconds, sum_solver_seconds=sum(r['metadata']['solve_seconds'] for r in ordered),
        source_hashes=fingerprint())
    save_json(ROOT/'checks/validation_summary.json',summary)
    with (ROOT/'checks/l2_errors.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['sample_id','time','field','absolute_l2','exact_l2','relative_l2','tolerance','passed'])
        for si,row in enumerate(rows):
            for ti,t in enumerate(cfg['times']):
                for fi,name in enumerate(['E','H','P']):
                    writer.writerow([row['sample_id'],t,name,error[si,ti,fi],norm[si,ti,fi],
                        relative[si,ti,fi] if defined[si,ti,fi] else '',tolerance[si,ti,fi],bool(passed[si,ti,fi])])
    destination = ROOT/('data/dataset.npz' if overall else 'data/candidates_FAILED.npz')
    np.savez_compressed(destination, alpha=np.array([r['alpha'] for r in rows]),tau=np.array([r['tau'] for r in rows]),
        initial_coefficients=coefficient_array,initial_fields=initial,dg_initial_fields=fields[:,0],
        x=x,times=np.array(cfg['times']),fields=fields,exact_fields=exact_fields,
        sample_id=ids,material_id=groups,split=splits,absolute_l2=error,exact_l2=norm,relative_l2=relative,
        relative_defined=defined,l2_tolerance=tolerance,validation_passed=passed,
        config_json=np.array(json.dumps(cfg,ensure_ascii=False)))
    if not overall:
        raise RuntimeError('Some data failed qualification; only candidates_FAILED.npz was written. See validation_summary.json.')
    plot_preview(rows, fields, exact_fields, x, cfg)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary


def plot_preview(rows, fields, exact, x, cfg):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    i = next(i for i,r in enumerate(rows) if r['split']=='train')
    row=rows[i]
    fig, axes=plt.subplots(3,3,figsize=(11,7),sharex=True,constrained_layout=True)
    colors=['#1f77b4','#b24a40','#298c6b']
    for fi,name in enumerate(['E','H','P']):
        for col,ti in enumerate([0,5,10]):
            ax=axes[fi,col]
            ax.plot(x,exact[i,ti,fi],color='0.55',lw=2,label='Analytic reference')
            ax.plot(x,fields[i,ti,fi],color=colors[fi],ls='--',lw=1.1,marker='o',markevery=4,ms=2.6,label='DG float64')
            ax.grid(alpha=.18)
            if fi==0: ax.set_title(f't = {cfg["times"][ti]:g}')
            if col==0: ax.set_ylabel(name)
            if fi==2: ax.set_xlabel('x')
    axes[0,0].legend(fontsize=8)
    a=[round(row[f'a{m}'],6) for m in range(1,4)]
    b=[round(row[f'b{m}'],6) for m in range(1,4)]
    fig.suptitle(f'{row["sample_id"]} (train) | alpha={row["alpha"]:.6g}, tau={row["tau"]:.6g} | N=64, k=2, dt=0.0005\n'
                f'a={a}   b={b}',fontsize=11)
    fig.savefig(ROOT/'data/preview.png',dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['preflight','dataset','all'],default='all')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--resume',action='store_true',help='Reuse cases only when source/configuration hashes match')
    args=parser.parse_args()
    if not 1<=args.workers<=8: parser.error('--workers must be between 1 and 8')
    cfg=json.loads((ROOT/'config.json').read_text())
    # Reject changes that require re-deriving the hard-coded three-mode reference or solver adapter.
    if (cfg['L']!=2*np.pi or cfg['epsilon_b']!=2 or cfg['mu']!=1 or cfg['delta_epsilon']!=1
        or cfg['initial_modes']!=3 or cfg['precision']!='float64' or cfg['N']!=64 or cfg['k']!=2
        or cfg['T']!=1 or cfg['dt']!=.0005 or cfg['points']!=65 or cfg['l2_quadrature_per_cell']!=24):
        raise ValueError('This teaching workflow is fixed to the documented PDE, modes and discretization')
    if args.stage in ('preflight','all'): preflight(cfg,args.resume)
    if args.stage in ('dataset','all'): dataset(cfg,args.workers,args.resume)


if __name__=='__main__':
    main()
