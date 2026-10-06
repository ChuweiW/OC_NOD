"""Value-based Bellman policy extraction and event-aware trajectory simulation.

No stored Uopt/Utail or Stop arrays are used to select actions.
"""
from pathlib import Path
import warnings

import numpy as np
from numba import njit, prange
if __package__:
    from .solvers import utils as _utils
else:
    from solvers import utils as _utils
drift = _utils.drift
posterior_update = _utils.posterior_update
observation_sigma = _utils.observation_sigma


@njit
def _index(grid, x):
    if len(grid) == 1:
        return 0, 0, 0.0
    i = max(0, min(len(grid)-2, np.searchsorted(grid, x)-1))
    return i, i+1, (x-grid[i])/(grid[i+1]-grid[i])


@njit
def _value(W, zg, mg, pg, z, mu, P):
    i, j, a = _index(zg, z)
    m, n, b = _index(mg, mu)
    p, q, c = _index(pg, P)
    v0 = (1-a)*((1-b)*W[i,m,p]+b*W[i,n,p]) + a*((1-b)*W[j,m,p]+b*W[j,n,p])
    v1 = (1-a)*((1-b)*W[i,m,q]+b*W[i,n,q]) + a*((1-b)*W[j,m,q]+b*W[j,n,q])
    return (1-c)*v0+c*v1


@njit
def _bellman(z, mu, P, W, zg, mg, pg, controls, dt, tail, c, d, K2, nonlinear, alpha, beta):
    """Return u, stop, cost. Tail lookahead uses the next spatial grid node."""
    best, action, stopping = np.inf, 0.0, False
    if z <= zg[0]+1e-12:
        best, stopping = -K2, True
    elif z >= zg[-1]-1e-12:
        best, stopping = -mu, True
    for u in controls:
        f = drift(z,u,mu,K2,nonlinear,alpha,beta)
        if (z <= zg[0]+1e-12 and f < 0) or (z >= zg[-1]-1e-12 and f > 0):
            continue
        if tail:
            if abs(f) < 1e-14:
                continue
            # Use the nearest strictly adjacent node, including off-grid z.
            if f > 0:
                j = np.searchsorted(zg,z+1e-12,side='right')
            else:
                j = np.searchsorted(zg,z-1e-12,side='left')-1
            if j < 0 or j >= len(zg):
                continue
            zn = zg[j]
            h = (zn-z)/f
            candidate = h*(d+c*u*u)+_value(W,zg,mg,pg,zn,mu,P)
        else:
            zn = z+dt*f
            if zn < zg[0]-1e-12 or zn > zg[-1]+1e-12:
                boundary = zg[0] if f < 0 else zg[-1]
                h = max(0.0,min(dt,(boundary-z)/f))
                candidate = h*(d+c*u*u)-(K2 if f < 0 else mu)
            else:
                zn = min(zg[-1],max(zg[0],zn))
                candidate = dt*(d+c*u*u)+_value(W,zg,mg,pg,zn,mu,P)
        if candidate < best-1e-12:
            best, action, stopping = candidate, u, False
        elif not stopping and abs(candidate-best) <= 1e-12 and abs(u) < abs(action):
            action = u
    return action, stopping, best


@njit
def _heatmap(W, zg, mg, pg, P_by_z, controls, dt, tail, c, d, K2, nonlinear, alpha, beta):
    U = np.empty((len(zg),len(mg)))
    stop = np.empty(U.shape,dtype=np.bool_)
    for i in range(len(zg)):
        for m in range(len(mg)):
            U[i,m],stop[i,m],_ = _bellman(zg[i],mg[m],P_by_z[i],W,zg,mg,pg,controls,dt,tail,c,d,K2,nonlinear,alpha,beta)
    return U, stop


class Policy:
    def __init__(self, source, extrapolate_mu=True):
        self.extrapolate_mu = extrapolate_mu
        self._warned_mu_extrapolation = False
        if isinstance(source, (str, Path)):
            with np.load(source,allow_pickle=False) as archive:
                # Loading only values/metadata avoids unnecessary control arrays.
                keys = ('schema_version','value_convention','model','z_grid','mu_grid',
                        'P_grid','V','Vtail','V_pre_last','u_grid_dp','u_grid_stationary',
                        'dt','t_obs','c','d','K2_known','alpha','beta','sigma_obs',
                        'sigma_min','sigma_max')
                self.data = {key:archive[key] for key in keys}
                for key in ('V_pre_obs','P_by_stage','observation_mode','parameters_json','experimental'):
                    if key in archive:
                        self.data[key] = archive[key]
        else:
            self.data = source
        r = self.data
        if int(r['schema_version']) != 2 or str(r['value_convention']) != 'cost = -paper_value':
            raise ValueError('Expected a schema-v2 archive with cost = -paper_value.')
        self.model = str(r['model'])
        self.nonlinear = self.model == 'nonlinear_constant'
        self.state_noise = self.model == 'linear_state'
        self.z, self.mu, self.P = (np.asarray(r[k]) for k in ('z_grid','mu_grid','P_grid'))
        self.dt = float(r['dt'])
        self.obs = np.asarray(r['t_obs'])
        if len(self.obs) < 2 or self.obs[0] != 0 or np.any(np.diff(self.obs) <= 0):
            raise ValueError('Expected increasing observation times starting at zero.')
        if self.state_noise and len(self.obs) != 2:
            raise ValueError('State-dependent noise currently supports two observations.')
        self.obs_idx = np.rint(self.obs/self.dt).astype(int)
        if not np.allclose(self.obs_idx*self.dt,self.obs,rtol=0,atol=1e-12):
            raise ValueError('Observation times must lie on the saved time grid.')
        if len(self.obs) > 2 and 'V_pre_obs' not in r:
            raise ValueError('Multi-observation archives require V_pre_obs jump surfaces.')
        self.nt = len(r['V'])-1
        self.tail = np.asarray(r['Vtail'])[:,:,None]
        self.args = tuple(float(r[k]) for k in ('c','d','K2_known')) + (self.nonlinear,) + tuple(float(r[k]) for k in ('alpha','beta'))

    def sigma(self,z):
        r = self.data
        if self.state_noise:
            return float(observation_sigma(z,self.z[0],self.z[-1],float(r['sigma_min']),float(r['sigma_max'])))
        return float(r['sigma_obs'])

    def _surface(self,k):
        if not isinstance(k,(int,np.integer)) or k < 0 or k > self.nt:
            raise ValueError('k must be a saved time index, from 0 through the final observation.')
        if k == self.nt:
            return self.tail, np.array([1.]), self.data['u_grid_stationary'], True
        match = np.flatnonzero(self.obs_idx[1:] == k+1)
        if len(match):
            W = (self.data['V_pre_obs'][int(match[0])]
                 if 'V_pre_obs' in self.data else self.data['V_pre_last'])
        else:
            W = self.data['V'][k+1]
        if W.ndim == 2:
            W = W[:,:,None]
        stage = int(np.searchsorted(self.obs_idx,k,side='right'))
        pg = self.P if self.state_noise else np.array([float(self.data['sigma_obs'])**2/stage])
        return W, pg, self.data['u_grid_dp'], False

    def choose(self,z,mu,P,k):
        """Full discrete Bellman search at saved time k (nt denotes the tail)."""
        if not np.all(np.isfinite([z,mu,P])) or P <= 0:
            raise ValueError('Finite state/belief and positive posterior variance are required.')
        if not self.z[0]-1e-12 <= z <= self.z[-1]+1e-12:
            raise ValueError('z is outside the physical domain.')
        if not self.mu[0] <= mu <= self.mu[-1]:
            if not self.extrapolate_mu:
                raise ValueError('Posterior mean is outside the saved belief grid.')
            if not self._warned_mu_extrapolation:
                warnings.warn('Posterior mean is outside the saved belief grid; '
                              'Bellman continuation values use linear endpoint extrapolation. '
                              'Widen the solver grid to check accuracy, especially for nonlinear dynamics.',
                              RuntimeWarning, stacklevel=2)
                self._warned_mu_extrapolation = True
        W, pg, controls, tail = self._surface(k)
        if not tail and not pg[0]-1e-12 <= P <= pg[-1]+1e-12:
            raise ValueError('Variance is inconsistent with the current observation stage/P grid.')
        return _bellman(z,mu,P,W,self.z,self.mu,pg,controls,self.dt,tail,*self.args)

    def heatmap(self,observation_index,P=None):
        """Post-observation control map. At t1, P=None uses sigma(z0)^2.

        A numeric P instead selects a fixed belief-variance slice. Only the
        final-observation map is independent of P. Constant-noise intermediate
        maps use P=sigma**2/j. Stopping is returned separately from control.
        """
        if observation_index not in range(len(self.obs)):
            raise ValueError('Observation index is outside the saved schedule.')
        k = int(self.obs_idx[observation_index])
        W, pg, controls, tail = self._surface(k)
        Ps = np.array([self.sigma(z)**2 for z in self.z]) if P is None else np.full(len(self.z),P)
        if P is None and not self.state_noise:
            Ps /= observation_index+1
        if not np.all(np.isfinite(Ps)) or np.any(Ps <= 0):
            raise ValueError('Heatmap variance must be finite and positive.')
        if not tail and (Ps.min() < pg[0]-1e-12 or Ps.max() > pg[-1]+1e-12):
            raise ValueError('Heatmap variance lies outside the policy P grid.')
        return _heatmap(W,self.z,self.mu,pg,Ps,controls,self.dt,tail,*self.args)


def simulate(policy,z0=0.,observations=(1.,1.3),max_time=5.,tail_dt=None,max_steps=100000):
    """Simulate prescribed observations without using future readings in control.

    Before the final observation, controls use the saved time grid and the
    appropriate V_pre_obs jump surface on each step into an observation.
    Legacy two-observation archives use V_pre_last. A surviving process
    assimilates each reading before choosing its next action. Tail controls
    minimize the stationary one-cell Bellman objective. Tail motion uses
    frozen-drift Euler segments capped by tail_dt and the next grid node.
    A strict finite-horizon overshoot stops on impact, matching the solver.
    """
    obs = np.asarray(observations,dtype=float)
    if obs.shape != (len(policy.obs),) or not np.all(np.isfinite(obs)):
        raise ValueError(f'Provide {len(policy.obs)} finite observation values, one per sampling time.')
    tail_dt = policy.dt if tail_dt is None else float(tail_dt)
    if not np.isfinite(max_time) or max_time <= 0 or not np.isfinite(tail_dt) or tail_dt <= 0:
        raise ValueError('max_time and tail_dt must be finite and positive.')
    z, mu = float(z0), float(obs[0])
    if not policy.z[0] <= z <= policy.z[-1]:
        raise ValueError('Initial position lies outside the policy domain.')
    P = policy.sigma(z)**2
    policy.choose(z,mu,P,0)  # validate the initialized belief before simulation
    t, k, next_observation = 0., 0, 1
    rows = [(t,z,mu,P)]
    segments, events = [], [dict(t=0.,observation=obs[0],mu=mu,P=P,z=z)]
    status = 'max_time'
    for _ in range(max_steps):
        if next_observation < len(policy.obs) and t >= policy.obs[next_observation]-1e-12:
            reading = float(obs[next_observation])
            mu,P = posterior_update(mu,P,reading,policy.sigma(z)**2)
            mu,P = float(mu),float(P)
            next_observation += 1
            rows.append((t,z,mu,P))  # duplicate event time displays an actual jump
            events.append(dict(t=t,observation=reading,mu=mu,P=P,z=z))
        assimilated = next_observation == len(policy.obs)
        if t >= max_time-1e-12:
            break
        index = policy.nt if assimilated else k
        u,stop,_ = policy.choose(z,mu,P,index)
        if stop:
            status = 'terminated_upper' if z >= policy.z[-1]-1e-12 else 'terminated_lower'
            break
        f = float(drift(z,u,mu,policy.args[2],policy.nonlinear,policy.args[-2],policy.args[-1]))
        h = tail_dt if assimilated else policy.dt
        forced_stop = False
        if assimilated:
            j = np.searchsorted(policy.z,z+1e-12,side='right') if f > 0 else np.searchsorted(policy.z,z-1e-12,side='left')-1
            if abs(f) < 1e-14 or not 0 <= j < len(policy.z):
                raise RuntimeError('Tail Bellman search selected a non-progressing action.')
            h = min(h,(policy.z[j]-z)/f)
        else:
            zn = z+h*f
            if zn < policy.z[0]-1e-12 or zn > policy.z[-1]+1e-12:
                boundary = policy.z[0] if f < 0 else policy.z[-1]
                h = (boundary-z)/f
                forced_stop = True
        if t+h > max_time+1e-12:
            h = max_time-t
            forced_stop = False
        start = t
        z = float(np.clip(z+h*f,policy.z[0],policy.z[-1]))
        t += h
        if not assimilated and not forced_stop and abs(h-policy.dt) < 1e-12:
            k += 1
            t = k*policy.dt
        segments.append((start,t,u))
        rows.append((t,z,mu,P))
        if forced_stop:
            status = 'terminated_upper' if f > 0 else 'terminated_lower'
            break
    else:
        status = 'max_steps'
    array = np.asarray(rows)
    return dict(t=array[:,0],z=array[:,1],mu=array[:,2],P=array[:,3],
                segments=np.asarray(segments,dtype=float).reshape(-1,3),
                events=events,status=status,observations=obs)


@njit(parallel=True)
def _interval_exits(z_queries,mu_queries,zg,mg,values,jump,controls,
                    start,end,dt,Ps,pg,c,d,K2,nonlinear,alpha,beta):
    result=np.zeros((len(z_queries),len(mu_queries)),dtype=np.int8)
    for i in prange(len(z_queries)):
        P=Ps[i]
        for m in range(len(mu_queries)):
            x=z_queries[i]; mu=mu_queries[m]
            for k in range(start,end):
                following=jump if k==end-1 else values[k+1]
                u,stop,_=_bellman(x,mu,P,following,zg,mg,pg,controls,dt,False,
                                  c,d,K2,nonlinear,alpha,beta)
                if stop:
                    result[i,m]=1 if x>=zg[-1]-1e-12 else -1
                    break
                f=drift(x,u,mu,K2,nonlinear,alpha,beta)
                xn=x+dt*f
                if xn<zg[0]-1e-12 or xn>zg[-1]+1e-12:
                    bound=zg[0] if f<0 else zg[-1]
                    hit_time=k*dt+(bound-x)/f
                    if hit_time<end*dt-1e-12:
                        result[i,m]=1 if f>0 else -1
                    break
                x=min(zg[-1],max(zg[0],xn))
    return result


def interval_exit_map(policy,j,nz=61,nmu=81,mu_bounds=None,P=None):
    """-1/+1: actually terminates at lower/upper side before next reading.

    Zero means it remains active until the next observation. Exact arrival
    at that observation is not counted as termination beforehand.
    """
    if j not in range(len(policy.obs)-1):
        raise ValueError('Choose an observation with a later observation.')
    if mu_bounds is None:
        mu_bounds=(policy.mu[0],policy.mu[-1])
    if not policy.mu[0]<=mu_bounds[0]<mu_bounds[1]<=policy.mu[-1]:
        raise ValueError('Plot bounds must lie inside the saved belief grid.')
    z=np.linspace(policy.z[0],policy.z[-1],nz)
    mu=np.linspace(*mu_bounds,nmu)
    jump=(policy.data['V_pre_obs'][j,:, :,None] if 'V_pre_obs' in policy.data
          else policy.data['V_pre_last'])
    values=policy.data['V'] if policy.state_noise else policy.data['V'][:,:,:,None]
    stage_P=float(policy.data['sigma_obs'])**2/(j+1)
    Ps=(np.array([policy.sigma(x)**2 for x in z]) if policy.state_noise else np.full(len(z),stage_P)) if P is None else np.full(len(z),P)
    pg=policy.P if policy.state_noise else np.array([stage_P])
    if np.any(Ps<=0) or not np.all(np.isfinite(Ps)) or Ps.min()<pg[0]-1e-12 or Ps.max()>pg[-1]+1e-12:
        raise ValueError('Region variance is outside the observation-stage grid.')
    sides=_interval_exits(z,mu,policy.z,policy.mu,values,jump,policy.data['u_grid_dp'],
                         int(policy.obs_idx[j]),int(policy.obs_idx[j+1]),policy.dt,Ps,pg,*policy.args)
    return z,mu,sides



def plot_heatmaps(policies,labels,P=None,trajectories=None,show_reward_regions=False,
                  region_nz=61,region_nmu=81,panel_width=10.0,panel_height=7.0):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    columns = max(len(p.obs) for p in policies.values())
    fig,axes = plt.subplots(len(policies),columns,figsize=(panel_width*columns,panel_height*len(policies)),squeeze=False,constrained_layout=True)
    linear_max = max(float(np.max(np.abs(p.data['u_grid_dp']))) for p in policies.values())
    for row,(name,p) in enumerate(policies.items()):
        for j in range(columns):
            if j >= len(p.obs):
                axes[row,j].set_visible(False)
                continue
            U,stop = p.heatmap(j,P)
            ax = axes[row,j]
            cmap = plt.get_cmap('viridis' if p.nonlinear else 'coolwarm').copy()
            cmap.set_bad('lightgray')
            norm = Normalize(float(p.data['u_grid_dp'][0]),float(p.data['u_grid_dp'][-1])) if p.nonlinear else Normalize(-linear_max,linear_max)
            mesh = ax.pcolormesh(p.z,p.mu,np.ma.array(U.T,mask=stop.T),shading='nearest',cmap=cmap,norm=norm,rasterized=True)
            for i in (0,len(p.z)-1):
                mask = stop[i]
                ax.scatter(np.full(mask.sum(),p.z[i]),p.mu[mask],s=8,c='black',marker='s',clip_on=False)
            if show_reward_regions and j < len(p.obs)-1:
                zq,mq,sides=interval_exit_map(p,j,region_nz,region_nmu,P=P)
                for side,hatch,color in [(-1,'////','cyan'),(1,'\\\\','magenta')]:
                    mask=(sides.T==side).astype(float)
                    if np.any(mask):
                        ax.contourf(zq,mq,mask,levels=[.5,1.5],colors='none',hatches=[hatch])
                        if np.any(mask==0):
                            ax.contour(zq,mq,mask,levels=[.5],colors=[color],linewidths=.9)
            variance_label = ('initialized P1(z)' if j == 0 else f'P={float(p.data["sigma_obs"])**2/(j+1):g}') if P is None else f'P={P:g}'
            if j == len(p.obs)-1:
                variance_label = 'P independent'
            ax.set(title=f'Observation {j+1}, t={p.obs[j]:g}+ | {variance_label}',xlabel='z',ylabel=r'$\mu$')
            fig.colorbar(mesh,ax=ax,label='Bellman-optimal u')
    fig.suptitle(' / '.join(labels[name] for name in policies) + ' | post-observation controls',fontsize=12)
    if show_reward_regions:
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(facecolor='none',edgecolor='cyan',hatch='////',label='Lower reward before next observation'),
                            Patch(facecolor='none',edgecolor='magenta',hatch='\\\\',label='Upper reward before next observation')],
                   loc='outside lower center',ncol=2,frameon=False)

    return fig,axes


def plot_trajectories(policies,trajectories,labels):
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(4,len(policies),figsize=(5*len(policies),10),squeeze=False,constrained_layout=True)
    for column,(name,p) in enumerate(policies.items()):
        r = trajectories[name]
        for row,key in ((0,'z'),(2,'mu'),(3,'P')):
            axes[row,column].plot(r['t'],r[key],linewidth=1.8)
        for start,end,u in r['segments']:
            axes[1,column].plot([start,end],[u,u],color='C1',linewidth=1.7)
        seg = r['segments']
        for a,b in zip(seg[:-1],seg[1:]):
            axes[1,column].plot([a[1],a[1]],[a[2],b[2]],color='C1',linewidth=.6)
        axes[0,column].axhline(p.z[0],color='gray',linestyle=':')
        axes[0,column].axhline(p.z[-1],color='gray',linestyle=':')
        if r['status'].startswith('terminated'):
            axes[0,column].scatter(r['t'][-1],r['z'][-1],marker='x',c='black',zorder=5)
        axes[0,column].set_title(f"{labels[name]}\n{r['status']} at t={r['t'][-1]:.4g}")
        for row,label in enumerate((r'$z(t)$',r'$u(t)$',r'$\mu(t)$',r'$P(t)$')):
            ax = axes[row,column]
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
            ax.set_xlim(0,max(.22,r['t'][-1]*1.03))
            for event in r['events']:
                ax.axvline(event['t'],color='gray',linestyle='--',alpha=.5)
        axes[-1,column].set_xlabel('Time')
    fig.suptitle('Bellman rollout with prescribed observations',fontsize=13)
    return fig,axes
