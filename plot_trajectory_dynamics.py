"""Caption-parameter branching figure, using the local experimental solvers.

Run: python experiments/plot_trajectory_dynamics.py
Add --regenerate to rebuild policies, or --observation-mode full for an ablation.
Posterior labels are converted to raw readings; controls use Bellman values.
"""
from pathlib import Path
import argparse
import json
from dataclasses import asdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
if __package__:
    from .solvers import linear_constant, nonlinear_constant
    from .workflow import generate
    from .policy_plotting import Policy, simulate
else:
    from solvers import linear_constant, nonlinear_constant
    from workflow import generate
    from policy_plotting import Policy, simulate

ROOT = Path(__file__).resolve().parent
THIRD_MEANS = {
    'linear_constant': {.85: (.96, 1.10), 1.10: (.88, 1.02), 1.90: (1.90,)},
    'nonlinear_constant': {.85: (.80, 1.27), 1.10: (.80, 1.10), 1.90: (1.90,)},
}

def parameters(mode='truncated'):
    common = dict(eps=.3, K2_known=1., d=.05, sigma0=.3, u_max=2.,
                  dt=.005, nz=241, nmu=401, nu=401, mu_min=0., mu_max=2.,
                  observation_mode=mode)
    return {
        'linear_constant': linear_constant.Parameters(c=.08, t_obs=(0.,.15,.4), **common),
        'nonlinear_constant': nonlinear_constant.Parameters(c=.06, alpha=1.5,
            beta=.001, u_min=0., t_obs=(0.,.25,.8), **common),
    }

def readings(means):
    means = np.asarray(means, dtype=float)
    return np.r_[means[0], 2*means[1]-means[0], 3*means[2]-2*means[1]]

def run(mode='truncated', regenerate=False):
    settings = parameters(mode)
    policies = {name: Policy(generate(name, p, regenerate)) for name,p in settings.items()}
    branches = {name: [( (1.10,m2,m3), simulate(policy,z0=0.,
                  observations=readings((1.10,m2,m3)),max_time=6.))
                  for m2,children in THIRD_MEANS[name].items() for m3 in children]
                for name,policy in policies.items()}
    fig,axes=plt.subplots(2,1,figsize=(8.2,7.2),constrained_layout=True)
    blue,brown='#3776b9','#bf7529'
    for ax,(name,policy) in zip(axes,policies.items()):
        t2,t3=policy.obs[1:]
        first=branches[name][0][1]
        mask=first['t']<=t2+1e-12
        ax.plot(first['t'][mask],first['z'][mask],color='black',lw=1.8)
        e=first['events'][1]
        ax.scatter(e['t'],e['z'],color='black',s=28,zorder=5)
        seen=set(); max_end=t3
        for means,r in branches[name]:
            _,m2,m3=means
            max_end=max(max_end,r['t'][-1])
            if m2 not in seen:
                mask=(r['t']>=t2-1e-12)&(r['t']<=t3+1e-12)
                ax.plot(r['t'][mask],r['z'][mask],color=blue,lw=1.7)
                if len(r['events'])==3:
                    e=r['events'][2]
                    ax.scatter(e['t'],e['z'],s=25,color=blue,zorder=5)
                seen.add(m2)
            if len(r['events'])<3:
                continue
            mask=r['t']>=t3-1e-12
            ax.plot(r['t'][mask],r['z'][mask],color=brown,lw=1.7)
        for t in (t2,t3):
            ax.axvline(t,color='gray',ls='--',lw=.9)
        for boundary in (policy.z[0],policy.z[-1]):
            ax.axhline(boundary,color='gray',ls=':',lw=1)
        ax.text(t2,-.255,'Observation 2\n'+r'$P_2=0.045$',ha='center',va='top',fontsize=10)
        ax.text(t3,-.255,'Observation 3\n'+r'$P_3=0.030$',ha='center',va='top',fontsize=10)
        upper=r'$+\epsilon$' if not policy.nonlinear else r'$z_{\max}$'
        lower=r'$-\epsilon$' if not policy.nonlinear else r'$z_{\min}$'
        ax.text(.02, .965,upper,transform=ax.transAxes,color='gray',fontsize=9)
        ax.text(.02, .005,lower,transform=ax.transAxes,color='gray',fontsize=9)
        ax.set(xlim=(0,max_end*1.025),ylim=(policy.z[0]-.025,policy.z[-1]+.025),
               xlabel='Time t',ylabel='State z')
        ax.grid(alpha=.12)
    out=ROOT/'plots'/'trajectory_dynamics'
    out.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):
        fig.savefig(out/f'trajectory_dynamics.{ext}',dpi=220,bbox_inches='tight')
    records=[]
    for name,paths in branches.items():
        for means,r in paths:
            records.append(dict(model=name,posterior_means=means,readings=r['observations'].tolist(),
                                events=r['events'],status=r['status'],end_time=float(r['t'][-1])))
    (out/'reproduction.json').write_text(json.dumps(dict(parameters={n:asdict(p) for n,p in settings.items()},
                    branches=records),indent=2),encoding='utf-8')
    plt.close(fig)
    print(out/'trajectory_dynamics.png')
    return branches

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regenerate',action='store_true')
    parser.add_argument('--observation-mode',choices=('truncated','full'),default='truncated')
    args=parser.parse_args()
    run(args.observation_mode,args.regenerate)
