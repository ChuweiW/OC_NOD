# Optimal control with uncertain terminal reward value

This code accompanies the paper [Optimality of delayed reward attainment under uncertainty](https://arxiv.org/abs/2610.03506). It contains the solvers and plotting code for experiments.
## Files

- `solvers/linear_constant.py`: linear dynamics with constant observation noise.
- `solvers/nonlinear_constant.py`: nonlinear dynamics with constant observation noise.
- `solvers/linear_state.py`: linear dynamics with state-dependent observation noise.
- `solvers/utils.py`: shared numerical routines and belief updates.
- `plot_policies.ipynb`: heatmaps and example trajectories for a selected policy.
- `plot_trajectory_dynamics.py`: produces Fig. 2, showing branching trajectories under linear and nonlinear dynamics.
- `plot_observation_quality.py`: produces Fig. 3, comparing constant and state-dependent observation noise.

Generated policies are saved in `policies/`, and figures are saved in `plots/`. These folders are excluded from Git.

## Getting started

Use Python 3.11 and install the dependencies. The commands below assume you are in the `experiments` folder.

```sh
python -m pip install -r requirements.txt
```

To generate a policy, choose one of the three models:

```sh
python workflow.py --model linear_constant
python workflow.py --model nonlinear_constant
python workflow.py --model linear_state
```

## Plotting a policy

Open `plot_policies.ipynb` and select `MODEL` in the configuration cell. The notebook plots one policy at a time, with control heatmaps at the observation times and trajectories of z, u, mu, and P.

Change `OBSERVATIONS[MODEL]` to specify the observation readings and `INITIAL_STATES[MODEL]` to change the initial state. The readings are observation values, not posterior means. Set `POLICY_PATH` to load another saved policy, or change `PARAMETERS` and set `REGENERATE=True` to solve again.

`SHOW_REWARD_REGIONS` toggles the hatched regions where the policy attains a reward before the next observation. Panel dimensions can be changed using `HEATMAP_PANEL_WIDTH` and `HEATMAP_PANEL_HEIGHT`.

## Reproducing the figures

To produce Fig. 2 (branching trajectories):

```sh
python plot_trajectory_dynamics.py
```

Both models use z0=0, K2=1, d=0.05, and sigma=0.3. The linear model uses c=0.08, controls in [-2,2], and observations at (0,0.15,0.4). The nonlinear model uses c=0.06, alpha=1.5, beta=0.001, controls in [0,2], and observations at (0,0.25,0.8). The branch means are specified in `THIRD_MEANS` in the script. Outputs are saved in `plots/trajectory_dynamics/`.

To produce Fig. 3 (observation-quality comparison):

```sh
python plot_observation_quality.py
```

This experiment uses c=d=0.1, K2=1, state bounds [-0.3,0.3], controls in [-2,2], and observations at (0,0.2). Constant noise has sigma=0.25; state-dependent sigma decreases from 0.4 to 0.1. This script generates separate policies in `policies/observation_quality/` and saves the figure in `plots/observation_quality/`.

## Notes on the numerical implementation

Controls in the plots and simulations are obtained by Bellman optimization of the saved value functions, rather than interpolation of saved controls. Values are stored as costs, W=-V. Boundary stopping is a separate decision from choosing zero control. A trajectory receives subsequent observations only if it has not terminated.

## License

The code in this folder is released under the [MIT License](LICENSE).
