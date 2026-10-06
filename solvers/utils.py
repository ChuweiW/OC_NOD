"""Local numerical helpers for the self-contained experimental solvers."""
import numpy as np
from numba import njit
from scipy.special import ndtr


def truncated_kernel(mu, variance):
    """Historical sampled Gaussian conditioned on the finite mean grid."""
    if not np.isfinite(variance) or variance < 0:
        raise ValueError('Transition variance must be finite and nonnegative.')
    if variance == 0:
        return np.eye(len(mu))
    weights = np.exp(-.5*(mu[:, None]-mu[None, :])**2/variance)
    return (weights/weights.sum(axis=1, keepdims=True)).astype(np.float32)


def convolve_mu(values, mu, variance, mode):
    if mode == 'truncated':
        return values @ truncated_kernel(mu, variance).T
    if mode == 'full':
        return gaussian_convolve_mu(values, mu, variance)
    raise ValueError('Unknown observation mode.')


def gaussian_observation_expectation(values, mu, P, sigma, mode):
    """Observation expectation with incoming P, arrival-state sigma, and mode."""
    R = sigma[:, None]**2
    posterior_P = P[None, :]*R/(P[None, :]+R)
    variance = P[None, :]-posterior_P
    result = np.empty((len(sigma), len(mu), len(P)))
    for ip in range(len(P)):
        if np.all(variance[:, ip] == variance[0, ip]):
            result[:, :, ip] = convolve_mu(values, mu, variance[0, ip], mode)
        else:
            for i in range(len(sigma)):
                result[i, :, ip] = convolve_mu(values[i:i+1], mu, variance[i, ip], mode)[0]
    return result, posterior_P

def observation_indices(t_obs, dt):
    """Validate a strictly increasing schedule starting at zero on the dt grid."""
    times = np.asarray(t_obs, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times))
            or times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError("Use at least two increasing observation times starting at zero.")
    indices = np.rint(times / dt).astype(np.int64)
    if not np.allclose(indices * dt, times, rtol=0, atol=1e-12):
        raise ValueError("Every observation time must lie on the dt grid.")
    return indices


def gaussian_convolve_mu(values, mu, variance):
    """Exact Gaussian integral of a piecewise-linear belief interpolant.

    values has shape (z,mu). End segments are extended linearly to infinity.
    The hinge representation integrates each change in slope analytically,
    preserving affine functions and all Gaussian tail mass.
    """
    if variance < 0 or not np.isfinite(variance):
        raise ValueError("Gaussian variance must be finite and nonnegative.")
    if variance == 0:
        return values.copy()
    slopes = np.diff(values, axis=1) / np.diff(mu)[None, :]
    delta = mu[:, None] - mu[None, 1:-1]
    sd = np.sqrt(variance)
    x = delta / sd
    hinges = delta * ndtr(x) + sd * np.exp(-x*x/2) / np.sqrt(2*np.pi)
    return (values[:, :1] + slopes[:, :1]*(mu-mu[0])[None, :]
            + np.diff(slopes, axis=1) @ hinges.T)


@njit(inline="always")
def running_cost(u, dt, c, d):
    return dt * (d + c * u * u)


def build_obs_stage(obs_idx, Nt):
    """
    stage[k] = zero-based observation stage active at time index k.

    Example:
        obs_idx = [0, 2]

        k = 0,1 -> stage 0  # after first observation
        k >= 2  -> stage 1  # after second observation
    """
    stage = np.zeros(Nt + 1, dtype=np.int32)
    obs_set = set(int(x) for x in obs_idx)

    count = 0
    for k in range(Nt + 1):
        if k in obs_set:
            count += 1

        stage[k] = max(count - 1, 0)

    return stage


def observation_sigma(z, z_min, z_max, sigma_min, sigma_max):
    """Standard deviation, not variance; defined on the physical domain."""
    z = np.asarray(z)
    if z_min >= z_max or sigma_min <= 0 or sigma_max < sigma_min:
        raise ValueError("Invalid domain or observation standard deviations.")
    if np.any(z < z_min) or np.any(z > z_max):
        raise ValueError("Observation state lies outside the physical domain.")
    return sigma_min + (sigma_max-sigma_min)*(z_max-z)/(z_max-z_min)


def posterior_update(mu, P, observation, variance):
    """Bayesian update after an observation; inputs may be arrays."""
    if np.any(np.asarray(P) <= 0) or np.any(np.asarray(variance) <= 0):
        raise ValueError("Belief and observation variances must be positive.")
    gain = P / (P + variance)
    return mu + gain*(observation-mu), P*variance/(P+variance)


@njit
def drift(z, u, mu, K2, nonlinear, alpha, beta):
    if nonlinear:
        return -z + beta*u + np.tanh(u*(alpha*(z-beta*u) + mu-K2))
    return u
