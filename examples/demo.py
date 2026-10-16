"""Run the released frozen FNO against one saved DG reference. No training."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='G1_00', help='Saved acceptance case ID, e.g. G1_00 or G3_01')
    args = parser.parse_args()
    import numpy as np
    from fno_acceptance_v1.acceptance_common import load_configuration, verify_sources
    from fno_acceptance_v1.frozen_predictor import FrozenFNOPredictor

    manifest = json.loads((ROOT / 'evidence/acceptance/cases_manifest.json').read_text())
    cases = {case['case_id']: case for case in manifest['cases']}
    if args.case not in cases:
        parser.error('Unknown case; choose one of ' + ', '.join(cases))
    cfg, paths = load_configuration(ROOT / 'fno_acceptance_v1/config.json')
    verify_sources(cfg, paths)
    case = cases[args.case]
    with np.load(ROOT / 'evidence/acceptance/predictions_and_reference.npz', allow_pickle=False) as saved:
        matches = np.flatnonzero(saved['case_id'] == args.case)
        if matches.size != 1:
            raise ValueError('Case does not have exactly one saved record')
        index = int(matches[0])
        x, weights = saved['x'].copy(), saved['weights'].copy()
        initial = saved['initial_fields'][index].copy()
        reference = saved['references'][index].copy()
    predictor = FrozenFNOPredictor(cfg, paths)
    predicted = predictor.predict(case['alpha'], case['tau'], initial[0], initial[1], x)
    absolute = np.sqrt(np.sum((predicted - reference) ** 2 * weights, axis=-1))
    norms = np.sqrt(np.sum(reference ** 2 * weights, axis=-1))
    relative = np.divide(absolute, norms, out=np.full(3, np.nan), where=norms > 0)
    output = ROOT / 'outputs' / ('demo_' + args.case)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), constrained_layout=True)
    for i, field in enumerate(('E', 'H', 'P')):
        axes[i, 0].plot(x, reference[i], label='Saved DG reference', linewidth=2)
        axes[i, 0].plot(x, predicted[i], '--', label='FNO prediction')
        axes[i, 0].set(ylabel=field, xlabel='x', title=f'{field}: relative L2 = {relative[i]:.2%}')
        axes[i, 0].legend()
        axes[i, 1].plot(x, np.abs(predicted[i] - reference[i]), color='tab:red')
        axes[i, 1].set(xlabel='x', ylabel='Absolute error')
    fig.suptitle(f'{args.case} | {case["family"]} | alpha={case["alpha"]:.4g}, tau={case["tau"]:.4g}, T=1')
    fig.savefig(output / 'prediction.png', dpi=160)
    plt.close(fig)
    np.savez_compressed(output / 'prediction.npz', prediction=predicted, reference=reference, x=x,
                        initial_fields=initial, weights=weights)
    report = {'case_id': args.case, 'family': case['family'], 'alpha': case['alpha'], 'tau': case['tau'],
              'relative_l2': {f: float(v) if np.isfinite(v) else None for f, v in zip(('E', 'H', 'P'), relative)},
              'absolute_l2': dict(zip(('E', 'H', 'P'), absolute.tolist())),
              'scope': 'Fresh frozen FNO inference against a saved DG reference; already-public case, not new validation.'}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print('Saved:', output.relative_to(ROOT))


if __name__ == '__main__':
    main()
