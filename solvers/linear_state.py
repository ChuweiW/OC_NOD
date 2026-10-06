"""Linear dynamics with state-dependent observation noise.

This file owns its parameters, stationary tail, observation jump, backward
Bellman optimization, and policy-generation entry point. Values store costs
W = -V_paper. Observations occur at t=0 and t=0.2.

Run: python experiments/solvers/linear_state.py [--quick] [--output PATH]
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import argparse

import numpy as np
from numba import njit, prange
from scipy.special import ndtr
if __package__:
    from . import utils as _utils
else:
    import utils as _utils
build_obs_stage = _utils.build_obs_stage
running_cost = _utils.running_cost
observation_sigma = _utils.observation_sigma



@dataclass(frozen=True)
class Parameters:
    # Physical model and running cost.
    eps: float = 0.3
    K2_known: float = 1.0
    c: float = 0.1
    d: float = 0.1
    sigma_min: float = 0.10
    sigma_max: float = 0.40
    sigma0: float = 0.25  # midpoint matching and an exact P=sigma0**2 grid node
    u_max: float = 2.0

    observation_mode: str = 'full'

    # Numerical resolution; match these explicitly for a noise-only comparison.
    dt: float = 0.005  # unchanged; observations remain exactly at 0 and 0.2
    nz: int = 241     # physical-state nodes
    nmu: int = 401    # posterior-mean nodes
    nu: int = 401     # control candidates
    mu_min: float = 0.0
    mu_max: float = 2.0
    nP: int = 41      # variance nodes, plus sigma0**2 if absent


    def validate(self):
        if self.observation_mode not in ('truncated', 'full'):
            raise ValueError("observation_mode must be 'truncated' or 'full'.")
        scalars = (self.eps, self.K2_known, self.c, self.d, self.sigma0, self.u_max, self.dt, self.mu_min, self.mu_max, self.sigma_min, self.sigma_max)
        if not np.all(np.isfinite(scalars)):
            raise ValueError("Active parameters must be finite.")
        if min(self.eps, self.c, self.d, self.dt, self.sigma0, self.u_max) <= 0:
            raise ValueError("Domain size, costs, dt, observation noise and u_max must be positive.")
        if self.sigma_min <= 0 or self.sigma_max < self.sigma_min or not np.isclose(
                (self.sigma_min + self.sigma_max) / 2, self.sigma0,
                rtol=0, atol=1e-12):
            raise ValueError("Require 0 < sigma_min <= sigma_max and their midpoint = sigma0.")
        counts = (self.nz, self.nmu, self.nu, self.nP)
        if any(not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 3 for n in counts):
            raise ValueError("Active grid counts must be integers of at least three.")
        if self.nu % 2 != 1 or self.mu_min >= self.mu_max:
            raise ValueError("Use an odd control count and increasing belief bounds.")
        if round(0.2/self.dt) < 1 or not np.isclose(round(0.2/self.dt)*self.dt, 0.2):
            raise ValueError("dt must divide the second observation time, 0.2.")


@njit
def drift(z, u):
    """Linear integrator dz/dt = u."""
    return u


def linear_tail(z, mu, u, p):
    """Exact stationary cost for the discrete admissible control set."""
    up = u[u > 0]
    un = u[u < 0]
    rate_p = (p.d+p.c*up**2)/up
    rate_n = (p.d+p.c*un**2)/(-un)
    ip, im = np.argmin(rate_p), np.argmin(rate_n)
    left = -p.K2_known + rate_n[im]*(z[:, None]-z[0])
    right = -mu[None, :] + rate_p[ip]*(z[-1]-z[:, None])
    W = np.minimum(left, right)
    U = np.where(left <= right, un[im], up[ip])
    stop = np.zeros(W.shape, dtype=bool)
    stop[0] = left[0] <= right[0]
    stop[-1] = right[-1] <= left[-1]
    U[stop] = 0
    return W, U, stop, (float(rate_n[im]), float(rate_p[ip]))


def observation_jump(tail, z, mu, P, sigma, rates, p):
    """Selected Gaussian expectation of the linear stationary tail.

    Full mode integrates the analytic tail on the entire real line.
    Truncated mode conditions the sampled tail on the finite belief grid.
    Returns pre-observation cost (z, mu, P) and posterior variance (z,P).
    The conditional posterior-mean variance is P-P2.
    """
    if p.observation_mode == 'truncated':
        return _utils.gaussian_observation_expectation(tail, mu, P, sigma, 'truncated')
    R = sigma[:, None]**2
    P2 = P[None, :]*R/(P[None, :]+R)
    sd = np.sqrt(P[None, :]-P2)
    rn, rp = rates
    a = -p.K2_known + rn*(z-z[0])
    b = rp*(z[-1]-z)
    delta = mu[None, :, None] - (b-a)[:, None, None]
    x = delta/sd[:, None, :]
    positive_part = delta*ndtr(x) + sd[:, None, :]*np.exp(-x*x/2)/np.sqrt(2*np.pi)
    return a[:, None, None]-positive_part, P2


@njit(parallel=True)
def backward_step(next_cost, z, mu, u, dt, c, d, K2):
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
                    f = drift(z[i], a)
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
    model = 'linear_state'
    p.validate()
    zmin = -p.eps
    z = np.linspace(zmin, p.eps, p.nz)
    mu = np.linspace(p.mu_min, p.mu_max, p.nmu)
    u = np.linspace(-p.u_max, p.u_max, p.nu)
    sigma = observation_sigma(z, zmin, p.eps, p.sigma_min, p.sigma_max)
    # P is the incoming variance, held fixed as z evolves until t2.
    P = np.unique(np.r_[np.linspace(p.sigma_min ** 2, p.sigma_max ** 2, p.nP), p.sigma0 ** 2])
    tail, utail, stail, rates = linear_tail(z, mu, u, p)
    residual, iterations = (0.0, 0)
    # The final observation is integrated out before backward recursion.
    jump, P2 = observation_jump(tail, z, mu, P, sigma, rates, p)
    nt = round(0.2 / p.dt)
    shape = (nt + 1, p.nz, p.nmu, len(P))
    W, U = (np.empty(shape), np.empty(shape))
    stop = np.empty(shape, dtype=bool)
    W[-1], U[-1], stop[-1] = (tail[:, :, None], utail[:, :, None], stail[:, :, None])
    for k in range(nt - 1, -1, -1):
        W[k], U[k], stop[k] = backward_step(
            jump if k == nt - 1 else W[k + 1],
            z, mu, u, p.dt, p.c, p.d, p.K2_known)
    # Restrict t=0 values to the initialized belief P1=sigma(z0)^2.
    initial = np.empty((p.nz, p.nmu))
    for i in range(p.nz):
        for m in range(p.nmu):
            initial[i, m] = np.interp(sigma[i] ** 2, P, W[0, i, m])
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
        t_obs=np.array([0.0, 0.2]),
        obs_idx=np.array([0, nt]),
        obs_stage=build_obs_stage([0, nt], nt),
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
        sigma_min=p.sigma_min,
        sigma_max=p.sigma_max,
        # Unused compatibility metadata; linear dynamics do not use alpha/beta.
        alpha=1.5,
        beta=0.001,
        u_max=p.u_max,
        tail_residual=residual,
        tail_iterations=iterations,
        state_dependent_variance=True,
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
        p = replace(p, nz=31, nmu=61, nu=41, dt=.01, nP=7)
    if args.observation_mode:
        p = replace(p, observation_mode=args.observation_mode)
    result = solve(p)
    filename = 'linear_state_quick.npz' if args.quick else 'linear_state.npz'
    path = args.output or Path(__file__).resolve().parents[1]/'policies'/filename
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
    print(f'Saved {path}; V shape={result["V"].shape}')


if __name__ == '__main__':
    main()
