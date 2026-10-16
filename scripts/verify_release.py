"""Verify distributed files and frozen predictions without training or DG solves."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_integrity():
    manifest = json.loads((ROOT / 'provenance/release_manifest.json').read_text())
    errors = []
    for name, expected in manifest['files'].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(name)
    if errors:
        raise ValueError('Missing or changed release files: ' + ', '.join(errors))
    return {'status': 'PASS', 'files_checked': len(manifest['files'])}


def check_predictions():
    import numpy as np
    from fno_training.utils import load_config
    from fno_training.data import load_prepared
    from fno_training.controlled_finetune.finetune_fno_pair import validate_pair
    from fno_acceptance_v1.acceptance_common import load_configuration, verify_sources
    from fno_acceptance_v1.frozen_predictor import FrozenFNOPredictor

    bundle = load_prepared(load_config(ROOT / 'fno_training/config.json'))
    counts = {s: len(bundle.indices(s)) for s in ('train', 'validation', 'test')}
    if counts != {'train': 384, 'validation': 64, 'test': 64}:
        raise ValueError('Unexpected dataset split')
    for name in ('config_A_mse.json', 'config_B_mixed.json'):
        validate_pair(ROOT / 'fno_training/controlled_finetune' / name)
    cfg, paths = load_configuration(ROOT / 'fno_acceptance_v1/config.json')
    verify_sources(cfg, paths)
    predictor = FrozenFNOPredictor(cfg, paths)
    regression = predictor.regression(cfg, paths)
    if regression['status'] != 'PASS':
        raise ValueError('Frozen validation loading regression failed')
    cases = json.loads((ROOT / 'evidence/acceptance/cases_manifest.json').read_text())['cases']
    by_id = {case['case_id']: case for case in cases}
    summary = json.loads((ROOT / 'evidence/acceptance/summary.json').read_text())
    differences = []
    observed = []
    with np.load(ROOT / 'evidence/acceptance/predictions_and_reference.npz', allow_pickle=False) as saved:
        scale = np.asarray(predictor.normalizer['output_scale'])[:, None]
        for i, case_id in enumerate(saved['case_id']):
            case = by_id[str(case_id)]
            initial = saved['initial_fields'][i]
            result = predictor.predict(case['alpha'], case['tau'], initial[0], initial[1], saved['x'])
            expected = saved['predictions'][i]
            error = np.abs(result - expected)
            if not np.all(error <= 2e-6 * scale + 1e-5 * np.abs(expected)):
                raise ValueError(f'Prediction regression failed for {case_id}')
            differences.append(float(error.max()))
            observed.append(result)
        observed = np.stack(observed)
        refs, weights = saved['references'], saved['weights']
        error = np.sqrt(np.sum((observed - refs) ** 2 * weights, axis=-1))
        norms = np.sqrt(np.sum(refs ** 2 * weights, axis=-1))
        if not np.all(norms > 0):
            raise ValueError('Unexpected undefined relative error in published cases')
        relative = error / norms
        groups = {}
        for group in ('G1', 'G2', 'G3', 'G4'):
            subset = relative[saved['group'] == group]
            mean = subset.mean(axis=0)
            expected_mean = np.array([summary['groups'][group]['fields'][f]['mean'] for f in ('E', 'H', 'P')])
            if not np.allclose(mean, expected_mean, rtol=1e-4, atol=1e-7):
                raise ValueError(f'Saved group statistics disagree: {group}')
            count = int(np.sum(np.max(subset, axis=1) <= .05))
            expected_count = summary['groups'][group]['threshold_counts']['0.05']['passed']
            if count != expected_count:
                raise ValueError(f'Joint threshold count changed: {group}')
            groups[group] = {'mean_relative_l2': dict(zip(('E', 'H', 'P'), mean.tolist())),
                             'joint_below_5_percent': count, 'cases': len(subset)}
    return {'status': 'PASS', 'split_counts': counts, 'paired_configs': 'PASS',
            'frozen_loading_regression_cases': len(regression['samples']),
            'acceptance_regression_cases': len(differences),
            'max_absolute_prediction_difference': max(differences), 'groups': groups,
            'scope': 'Packaging/inference regression against existing evidence; no new generalization claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--integrity-only', action='store_true')
    parser.add_argument('--report', type=Path, help='Optional JSON output, e.g. outputs/release_check.json')
    args = parser.parse_args()
    report = {'integrity': check_integrity()}
    if not args.integrity_only:
        report['prediction_regression'] = check_predictions()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
