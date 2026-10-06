"""Section IV-B: matched initial controls, stopping changes, and observations.
Run python experiments/plot_observation_quality.py. New archives are isolated from the three-observation policies.
Historical recovery: constant noise uses truncated Gaussian planning, state noise uses full integration.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
if __package__:
    from .policy_plotting import Policy, simulate, interval_exit_map
    from .solvers import linear_constant, linear_state
else:
    from policy_plotting import Policy, simulate, interval_exit_map
    from solvers import linear_constant, linear_state

ROOT=Path(__file__).resolve().parent

def reproduce(mask_nz=81,mask_nmu=101,policy_files=None,
              output_prefix='observation_quality',figure_note=None):
    if policy_files is None:
        policy_files={name:ROOT/'policies'/'observation_quality'/f'{name}.npz'
                      for name in ('constant','state')}
    policies={name:Policy(policy_files[name]) for name in ('constant','state')}
    a,b=policies.values()
    np.testing.assert_array_equal(a.z,b.z)
    np.testing.assert_array_equal(a.mu,b.mu)
    for p in policies.values():
        np.testing.assert_allclose([p.args[0],p.args[1],p.args[2],p.sigma(0.)],[.1,.1,1.,.25])
        np.testing.assert_allclose(p.obs,[0.,.2])
    controls={n:p.heatmap(0)[0] for n,p in policies.items()}
    masks={n:interval_exit_map(p,0,mask_nz,mask_nmu) for n,p in policies.items()}
    fig,axes=plt.subplots(1,4,figsize=(18,4.4),constrained_layout=True)
    for ax,(name,p) in zip(axes,policies.items()):
        mesh=ax.pcolormesh(p.z,p.mu,controls[name].T,cmap='RdBu_r',vmin=-2,vmax=2,shading='nearest',rasterized=True)
        zq,mq,exits=masks[name]
        for side,hatch,color in [(-1,'////','cyan'),(1,'\\\\','magenta')]:
            mask=(exits.T==side).astype(float)
            ax.contourf(zq,mq,mask,levels=[.5,1.5],colors='none',hatches=[hatch])
            if mask.any() and not mask.all():
                ax.contour(zq,mq,mask,levels=[.5],colors=[color],linewidths=.8)
        ax.set_title(r'(a) $u^*_{\mathrm{const}}$' if name=='constant' else r'(b) $u^*_{\mathrm{state}}$',loc='left')
    fig.colorbar(mesh,ax=axes[:2],label='Initial control u',shrink=.85)
    delta=controls['state']-controls['constant']
    ax=axes[2]
    mesh=ax.pcolormesh(a.z,a.mu,delta.T,cmap='RdBu_r',vmin=-1,vmax=1,shading='nearest',rasterized=True)
    fig.colorbar(mesh,ax=ax,label='Control difference',shrink=.85)
    zq,mq,ea=masks['constant']; eb=masks['state'][2]
    # Recompute initial controls on exactly the stopping-mask mesh.
    uq={}
    for name,p in policies.items():
        uq[name]=np.array([[p.choose(z,mu,p.sigma(z)**2,0)[0] for mu in mq] for z in zq])
    opposite=uq['constant']*uq['state'] < -1e-12
    # Region 2 requires same nonzero direction, one terminated and one active.
    changed=(uq['constant']*uq['state']>1e-12)&((ea==0)!=(eb==0))
    for mask,hatch in [(opposite,'\\\\'),(changed,'....')]:
        ax.contourf(zq,mq,mask.T.astype(float),levels=[.5,1.5],colors='none',hatches=[hatch])
    for name,style in [('constant','-'),('state','--')]:
        ax.contour(a.z,a.mu,controls[name].T,levels=[0],colors=['forestgreen'],linestyles=[style],linewidths=1)
    ax.set_title(r'(c) $u^*_{\mathrm{state}}-u^*_{\mathrm{const}}$',loc='left')
    ax.legend(handles=[Line2D([],[],color='forestgreen',label=r'$u_{\mathrm{const}}=0$'),
                       Line2D([],[],color='forestgreen',ls='--',label=r'$u_{\mathrm{state}}=0$'),
                       Patch(facecolor='none',edgecolor='black',hatch='\\\\',label='Region 1'),
                       Patch(facecolor='none',edgecolor='black',hatch='....',label='Region 2')],fontsize=8,loc='lower right')
    report={}; traces={}
    for scenario,z0,mu1,o2,color in [('orange',0.,.96,1.06,'#c66b22'),('purple',-.2,.78,1.8,'#8270b4')]:
        for ax in axes[:3]:
            ax.scatter(z0,mu1,marker='X',c=color,edgecolors='.25',s=45,zorder=10)
        for ax in axes[:2]:
            ax.annotate(f'({z0:.2f}, {mu1:.2f})',(z0,mu1),xytext=(5,9),textcoords='offset points',fontsize=7,bbox=dict(facecolor='white',edgecolor='none',alpha=.8))
        for name,p in policies.items():
            r=simulate(p,z0=z0,observations=[mu1,o2],max_time=2.)
            traces[scenario+'_'+name]=r
            axes[3].plot(r['t'],r['z'],color=color,ls=':' if name=='constant' else '-',lw=1.8)
            if len(r['events'])==2:
                e=r['events'][1]
                axes[3].scatter(e['t'],e['z'],color=color,s=20,marker='D' if scenario=='purple' else 'o')
                axes[3].text(.245, e['z']+(.045 if scenario=='orange' and name=='state' else -.025 if scenario=='orange' else .015),
                             f"$P_2={e['P']:.4f}$",color=color,fontsize=8)
            if name=='state' or scenario=='purple':
                axes[3].text(.035, z0+(-.09 if scenario=='purple' and name=='constant' else .012),
                             f"$P_1={r['events'][0]['P']:.4f}$",color=color,fontsize=8)
            report[scenario+'_'+name]=dict(initial_control=float(p.choose(z0,mu1,p.sigma(z0)**2,0)[0]),
                events=r['events'],status=r['status'],end_time=float(r['t'][-1]),
                archive=str(Path(policy_files[name]).resolve()))
    for ax in axes[:3]:
        ax.set(xlim=(-.3,.3),ylim=(0,2),xlabel='Initial state z',ylabel=r'Initial belief mean $\mu_1$')
    for heatmap_axis in axes[:2]:
        heatmap_axis.legend(handles=[Patch(facecolor='none',edgecolor='cyan',hatch='////',label=r'Attain reward before $t_2$ ($-\epsilon$)'),
                            Patch(facecolor='none',edgecolor='magenta',hatch='\\\\',label=r'Attain reward before $t_2$ ($+\epsilon$)')],fontsize=7,loc='lower right')
    ax=axes[3]
    ax.axvline(.2,color='.65',ls='--',lw=.8)
    for z in [-.3,.3]: ax.axhline(z,color='.6',ls=':',lw=.8)
    ax.set(xlim=(0,.5),ylim=(-.32,.32),xlabel='Time t',ylabel='State z',title='(d) Trajectories')
    ax.grid(alpha=.15)
    ax.legend(handles=[Line2D([],[],color='.3',ls=':',label='Model 1'),Line2D([],[],color='.3',label='Model 2')],fontsize=8)
    if figure_note:
        fig.suptitle(figure_note,fontsize=11)
    out=ROOT/'plots'/'observation_quality';out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/f'{output_prefix}.png',dpi=220)
    fig.savefig(out/f'{output_prefix}.pdf')
    configuration={name:json.loads(str(policy.data['parameters_json'])) for name,policy in policies.items()}
    (out/f'{output_prefix}_parameters.json').write_text(json.dumps(configuration,indent=2))
    (out/f'{output_prefix}_diagnostics.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/f'{output_prefix}_regions.npz',z=zq,mu=mq,constant_exits=ea,state_exits=eb,region1=opposite,region2=changed)
    return fig,report

def generate_policies(regenerate=False):
    from dataclasses import asdict
    import os
    common=dict(c=.1,d=.1,K2_known=1.,eps=.3,u_max=2.,sigma0=.25,
                mu_min=0.,mu_max=2.,dt=.005,nz=241,nmu=401,nu=401)
    settings={'constant':linear_constant.Parameters(t_obs=(0.,.2),observation_mode='truncated',**common),
              'state':linear_state.Parameters(sigma_min=.1,sigma_max=.4,nP=41,
                                              observation_mode='full',**common)}
    files={name:ROOT/'policies'/'observation_quality'/f'{name}.npz' for name in settings}
    for name,params in settings.items():
        path=files[name]
        expected=json.dumps(asdict(params),sort_keys=True)
        if path.exists() and not regenerate:
            with np.load(path,allow_pickle=False) as a:
                if str(a['parameters_json'])!=expected:
                    raise ValueError(f'{path}: settings differ; use --regenerate')
            continue
        print(f'Generating {name}: {expected}',flush=True)
        result=(linear_constant if name=='constant' else linear_state).solve(params)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.pending.npz')
        np.savez_compressed(temporary,**result)
        os.replace(temporary,path)
        del result
        print(f'Saved {path}',flush=True)
    return files

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Dedicated Fig. 3 recovery with new caption-parameter policy archives.')
    parser.add_argument('--regenerate',action='store_true')
    args=parser.parse_args()
    files=generate_policies(args.regenerate)
    fig,report=reproduce(policy_files=files)
    plt.close(fig)
    print(json.dumps(report,indent=2))
