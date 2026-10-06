"""Local generation and reproducibility checks for the experimental notebook."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import numpy as np
if __package__:
    from .solvers import linear_constant, nonlinear_constant, linear_state
else:
    from solvers import linear_constant, nonlinear_constant, linear_state
SOLVERS = {m.__name__.split('.')[-1]: m for m in
           (linear_constant, nonlinear_constant, linear_state)}
ROOT = Path(__file__).resolve().parent

def generate(name, parameters, regenerate=False):
    parameters.validate()
    path = ROOT / 'policies' / (name + '.npz')
    expected = json.dumps(asdict(parameters), sort_keys=True)
    if path.exists() and not regenerate:
        with np.load(path, allow_pickle=False) as archive:
            actual = str(archive['parameters_json'])
        if actual != expected:
            raise ValueError(f'{path.name}: saved parameters differ. Set REGENERATE=True.')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        result = SOLVERS[name].solve(parameters)
        temporary = path.with_suffix(".pending.npz")
        np.savez_compressed(temporary, **result)
        os.replace(temporary, path)
    return path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--model', choices=list(SOLVERS))
    args = parser.parse_args()
    for name, module in SOLVERS.items():
        if args.model and name != args.model:
            continue
        print(generate(name, module.Parameters(), args.regenerate), flush=True)
