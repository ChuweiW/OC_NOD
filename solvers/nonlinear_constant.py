"""Nonlinear opinion dynamics with constant observation noise.

This file owns its parameters, stationary tail, observation jump, backward
Bellman optimization, and policy-generation entry point. Values store costs
W = -V_paper. Observation times are configurable; the defaults below reproduce the selected experiment.

Run: python experiments/solvers/nonlinear_constant.py [--quick] [--output PATH]
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import argparse

import numpy as np
from numba import njit, prange
if __package__:
    from . import utils as _utils
else:
    import utils as _utils
build_obs_stage = _utils.build_obs_stage
running_cost = _utils.running_cost
observation_indices = _utils.observation_indices
gaussian_convolve_mu = _utils.gaussian_convolve_mu



@dataclass(frozen=True)
class Parameters:
    # Physical model and running cost.
    eps: float = 0.3
    K2_known: float = 1.0
    c: float = 0.06
    d: float = 0.05
    sigma0: float = 0.3  # constant observation standard deviation
    alpha: float = 1.5
    beta: float = 0.001
    u_min: float = 0.0  # set to 0 for the paper's nonnegative-attention experiment
    u_max: float = 2.0

    observation_mode: str = 'truncated'

    # Numerical resolution; match these explicitly for a noise-only comparison.
    dt: float = 0.005
    t_obs: tuple = (0.0, 0.25, 0.8)  # any increasing schedule aligned with dt
    nz: int = 241     # physical-state nodes
    nmu: int = 401    # posterior-mean nodes
    nu: int = 401     # control candidates
    mu_min: float = 0.0
    mu_max: float = 2.0


    def validate(self):
        if self.observation_mode not in ('truncated', 'full'):
            raise ValueError("observation_mode must be 'truncated' or 'full'.")
        scalars = (self.eps, self.K2_known, self.c, self.d, self.sigma0, self.u_max, self.dt, self.mu_min, self.mu_max, self.alpha, self.beta, self.u_min)
        if not np.all(np.isfinite(scalars)):
            raise ValueError("Active parameters must be finite.")
        if min(self.eps, self.c, self.d, self.dt, self.sigma0, self.u_max) <= 0:
            raise ValueError("Domain size, costs, dt, observation noise and u_max must be positive.")
        if not 0 <= self.u_min < self.u_max or not 0 < self.beta*self.u_max < 2*self.eps:
            raise ValueError("Invalid nonnegative attention bounds or nonlinear domain.")
        counts = (self.nz, self.nmu, self.nu)
        if any(not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 3 for n in counts):
            raise ValueError("Active grid counts must be integers of at least three.")
        if self.nu % 2 != 1 or self.mu_min >= self.mu_max:
            raise ValueError("Use an odd control count and increasing belief bounds.")
        observation_indices(self.t_obs, self.dt)


@njit
def drift(z, u, mu, K2, alpha, beta):
    """Nonlinear opinion dynamics with positive attention u."""
    return -z + beta*u + np.tanh(u*(alpha*(z-beta*u) + mu-K2))


@njit
def nonlinear_tail(z, mu, u, p_c, p_d, K2, alpha, beta,
                   max_iter=10000, tol=1e-10):
    """One-cell semi-Lagrangian shortest-path iteration, with optional stopping.

    Positive edge costs imply an optimal exit path has no cycles. Starting
    from infinity propagates only actual paths to a terminal boundary.
    """
    nz, nm = len(z), len(mu)
    W = np.full((nz, nm), np.inf)
    U = np.zeros((nz, nm))
    stop = np.zeros((nz, nm), dtype=np.bool_)
    W[0, :] = -K2
    W[-1, :] = -mu
    for iteration in range(max_iter):
        residual = 0.0
        for m in range(nm):
            for direction in range(2):
                for index in range(nz):
                    i = index if direction == 0 else nz-1-index
                    best = np.inf
                    action = 0.0
                    stopping = False
                    if i == 0:
                        best, stopping = -K2, True
                    elif i == nz-1:
                        best, stopping = -mu[m], True
                    for a in u:
                        f = drift(z[i], a, mu[m], K2, alpha, beta)
                        if abs(f) < 1e-14:
                            continue
                        j = i+1 if f > 0 else i-1
                        if j < 0 or j >= nz:
                            continue
                        cost = running_cost(a, (z[j]-z[i])/f, p_c, p_d) + W[j, m]
                        if cost < best-1e-12:
                            best, action, stopping = cost, a, False
                        elif not stopping and abs(cost-best) <= 1e-12 and abs(a) < abs(action):
                            action = a
                    old = W[i, m]
                    if best != old:
                        residual = max(residual, abs(best-old))
                    W[i, m], U[i, m], stop[i, m] = best, action, stopping
        if residual < tol:
            return W, U, stop, residual, iteration+1
    raise RuntimeError("Nonlinear stationary tail failed to converge.")


def observation_jump(tail, z, mu, P, sigma, p):
    """Selected observation expectation of nonlinear stationary tail values.

    Full mode integrates the linearly extended interpolant including infinite
    tails; truncated mode conditions the sampled Gaussian on the mean grid.
    """
    return _utils.gaussian_observation_expectation(tail, mu, P, sigma, p.observation_mode)



@njit(parallel=True)
def backward_step(next_cost, z, mu, u, dt, c, d, K2, alpha, beta):
    """Full control search. P stays fixed between observations.

    Exact endpoint arrival accesses its continuation/stop value. A strict
    overshoot terminates at the first boundary hit, as in the formulation.
    """
    nz, nm, np_ = next_cost.shape
    W = np.empty_like(next_cost)
    U = np.zeros_like(next_cost)
    stop = np.zeros(next_cost.shape, dtype=np.bool_)
    dz = z[1]-z[0]
    for i in prange(nz):
        for m in range(nm):
            for ip in range(np_):
                best, action, stopping = np.inf, 0.0, False
                if i == 0:
                    best, stopping = -K2, True
                elif i == nz-1:
                    best, stopping = -mu[m], True
                for a in u:
                    f = drift(z[i], a, mu[m], K2, alpha, beta)
                    # At a boundary only inward or tangent continuation is admissible.
                    if (i == 0 and f < 0) or (i == nz-1 and f > 0):
                        continue
                    zn = z[i]+dt*f
                    if zn < z[0]-1e-12 or zn > z[-1]+1e-12:
                        boundary = z[0] if f < 0 else z[-1]
                        tau = max(0.0, min(dt, (boundary-z[i])/f))
                        reward = K2 if f < 0 else mu[m]
                        candidate = running_cost(a, tau, c, d)-reward
                    else:
                        zn = min(z[-1], max(z[0], zn))
                        j = max(0, min(nz-2, int((zn-z[0])/dz)))
                        w = (zn-z[j])/(z[j+1]-z[j])
                        candidate = running_cost(a, dt, c, d) + (1-w)*next_cost[j,m,ip] + w*next_cost[j+1,m,ip]
                    if candidate < best-1e-12:
                        best, action, stopping = candidate, a, False
                    elif not stopping and abs(candidate-best) <= 1e-12 and abs(a) < abs(action):
                        action = a
                W[i,m,ip], U[i,m,ip], stop[i,m,ip] = best, action, stopping
    return W, U, stop


def solve(parameters=None):
    """Compute this model's policy using this file's Parameters defaults."""
    p = Parameters() if parameters is None else parameters
    model = 'nonlinear_constant'
    p.validate()
    zmin = -p.eps + p.beta * p.u_max
    z = np.linspace(zmin, p.eps, p.nz)
    mu = np.linspace(p.mu_min, p.mu_max, p.nmu)
    u = np.linspace(p.u_min, p.u_max, p.nu)
    sigma = np.full(p.nz, p.sigma0)
    # P is the incoming variance, held fixed as z evolves until t2.
    P = np.array([p.sigma0 ** 2])
    tail, utail, stail, residual, iterations = nonlinear_tail(z, mu, u, p.c, p.d, p.K2_known, p.alpha, p.beta)
    if not np.all(np.isfinite(tail)):
        raise RuntimeError('Some nonlinear states cannot reach either boundary on this grid.')
    obs_idx = observation_indices(p.t_obs, p.dt)
    nt = int(obs_idx[-1])
    P_by_stage = p.sigma0**2 / np.arange(1, len(obs_idx)+1)
    shape = (nt + 1, p.nz, p.nmu, 1)
    W, U = np.empty(shape), np.empty(shape)
    stop = np.empty(shape, dtype=bool)
    W[-1], U[-1], stop[-1] = tail[:,:,None], utail[:,:,None], stail[:,:,None]
    # Stored time slices are post-observation; jumps are separate t_j- surfaces.
    pre_obs = np.empty((len(obs_idx)-1, p.nz, p.nmu))
    for k in range(nt-1, -1, -1):
        matches = np.flatnonzero(obs_idx[1:] == k+1)
        next_value = W[k+1]
        if len(matches):
            j = int(matches[0])+1  # zero-based observation index; j previous readings
            if j == len(obs_idx)-1:
                incoming = np.array([P_by_stage[j-1]])
                jump, P2 = observation_jump(tail, z, mu, incoming, sigma, p)
                pre_obs[j-1] = jump[:,:,0]
            else:
                variance = P_by_stage[j-1] - P_by_stage[j]
                pre_obs[j-1] = _utils.convolve_mu(W[k+1,:,:,0], mu, variance, p.observation_mode)
            next_value = pre_obs[j-1,:,:,None]
        W[k], U[k], stop[k] = backward_step(
            next_value, z, mu, u, p.dt, p.c, p.d, p.K2_known, p.alpha, p.beta)
    jump = pre_obs[-1,:,:,None]  # legacy alias for the final pre-observation cost
    # Restrict t=0 values to the initialized belief P1=sigma(z0)^2.
    initial = np.empty((p.nz, p.nmu))
    for i in range(p.nz):
        for m in range(p.nmu):
            initial[i, m] = np.interp(sigma[i] ** 2, P, W[0, i, m])
    W, U, stop = (W[..., 0], U[..., 0], stop[..., 0])
    return dict(
        V=W,
        Uopt=U,
        Stop=stop,
        Vtail=tail,
        Utail=utail,
        Stop_tail=stail,
        V_initial=initial,
        V_pre_last=jump,
        z_grid=z,
        mu_grid=mu,
        P_grid=P,
        P1_by_z=sigma ** 2,
        P2_by_z_P=P2,
        sigma_by_z=sigma,
        u_grid_dp=u,
        u_grid_stationary=u,
        t_obs=np.asarray(p.t_obs),
        obs_idx=obs_idx,
        V_pre_obs=pre_obs,
        P_by_stage=P_by_stage,
        obs_stage=build_obs_stage(obs_idx, nt),
        nz=p.nz,
        nmu=p.nmu,
        nu=p.nu,
        nP_actual=len(P),
        dt=p.dt,
        Nt=nt,
        Nt_effective=nt,
        last_obs_k=nt,
        eps=p.eps,
        z_min=zmin,
        z_max=p.eps,
        c=p.c,
        d=p.d,
        K2_known=p.K2_known,
        sigma_obs=p.sigma0,
        # Unused compatibility metadata; observation noise uses sigma_obs.
        sigma_min=0.1,
        sigma_max=0.4,
        quadrature_order=128,
        alpha=p.alpha,
        beta=p.beta,
        u_max=p.u_max,
        tail_residual=residual,
        tail_iterations=iterations,
        state_dependent_variance=False,
        model=model,
        value_convention='cost = -paper_value',
        schema_version=2,
        experimental=True,
        observation_mode=p.observation_mode,
        parameters_json=json.dumps(asdict(p), sort_keys=True),
        time_convention='stored observation slices are post-observation',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='Small diagnostic grid.')
    parser.add_argument('--output', type=Path, help='Archive path; default is experiments/policies.')
    parser.add_argument('--observation-mode', choices=['truncated', 'full'])
    args = parser.parse_args()
    from dataclasses import replace
    p = Parameters()
    if args.quick:
        p = replace(p, nz=31, nmu=61, nu=41, dt=.01)
    if args.observation_mode:
        p = replace(p, observation_mode=args.observation_mode)
    result = solve(p)
    filename = 'nonlinear_constant_quick.npz' if args.quick else 'nonlinear_constant.npz'
    path = args.output or Path(__file__).resolve().parents[1]/'policies'/filename
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
    print(f'Saved {path}; V shape={result["V"].shape}')


if __name__ == '__main__':
    main()
