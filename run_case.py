"""Launch one actual C++ DG solve; preserve command, output, and timing."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def run_case(directory, alpha, tau, a, b, config, N=None, dt=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fmt = lambda values: ','.join(format(float(v), '.17g') for v in values)
    command = [str(ROOT/'build/dg_dataset'), '--alpha', format(float(alpha), '.17g'),
               '--tau', format(float(tau), '.17g'), '--a', fmt(a), '--b', fmt(b),
               '--N', str(config['N'] if N is None else N), '--k', str(config['k']),
               '--dt', str(config['dt'] if dt is None else dt), '--T', str(config['T']),
               '--times', fmt(config['times']), '--points', str(config['points']),
               '--cache', str(ROOT/'cache/cq'), '--output', str(directory)]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE')
    start = time.perf_counter()
    with (directory/'solver.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    invocation = {'command': command, 'cwd': str(ROOT), 'environment_overrides':
                  {'OMP_NUM_THREADS': '1', 'OMP_DYNAMIC': 'FALSE'},
                  'returncode': result.returncode, 'wall_seconds': time.perf_counter()-start}
    (directory/'invocation.json').write_text(json.dumps(invocation, indent=2)+'\n')
    if result.returncode:
        raise RuntimeError(f'DG solve failed; see {directory}/solver.log')
    return invocation
