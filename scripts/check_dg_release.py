"""Rebuild isolated DG and check two modal cases plus one saved pulse reference."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    from fno_acceptance_v1.acceptance_common import load_configuration, verify_sources
    from fno_acceptance_v1.dg_adapter import DGAdapter

    cfg, paths = load_configuration(ROOT / 'fno_acceptance_v1/config.json')
    verify_sources(cfg, paths)
    workspace = ROOT / 'fno_acceptance_v1/artifacts/release_smoke'
    adapter = DGAdapter(cfg, ROOT, workspace)
    adapter.build()
    regression = adapter.run_regression()
    if regression['status'] != 'PASS':
        raise ValueError('Modal interface regression failed: ' + json.dumps(regression))
    cases = json.loads((ROOT / 'evidence/acceptance/cases_manifest.json').read_text())['cases']
    case = next(c for c in cases if c['case_id'] == 'G3_01')
    result = adapter.solve(case, cfg['dg']['reference'], workspace / 'G3_01', workspace / 'cq_cache', force=True)
    with np.load(ROOT / 'evidence/acceptance/predictions_and_reference.npz', allow_pickle=False) as saved:
        index = int(np.flatnonzero(saved['case_id'] == 'G3_01')[0])
        expected = saved['references'][index]
        delta = np.abs(result['fields'] - expected)
        if not np.allclose(result['fields'], expected, atol=1e-10, rtol=1e-9):
            raise ValueError(f'Pulse reference regression failed; max difference {delta.max()}')
    report = {'status': 'PASS', 'modal_interface': regression,
              'pulse_reference': {'case_id': 'G3_01', 'settings': cfg['dg']['reference'],
                                  'max_absolute_difference': float(delta.max())},
              'scope': 'Two interface checks and one pulse re-solve, not a full convergence study.'}
    (workspace / 'release_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'modal_cases': regression['sample_count'],
                      'pulse_max_absolute_difference': float(delta.max())}, indent=2))
    print('Saved:', (workspace / 'release_check.json').relative_to(ROOT))


if __name__ == '__main__':
    main()
