"""Small-grid numerical consistency and observation-event regression checks."""
import unittest
from dataclasses import replace
import warnings
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solvers import linear_constant, nonlinear_constant, linear_state
from solvers.utils import convolve_mu
from policy_plotting import Policy, simulate

class WorkflowTests(unittest.TestCase):
    def test_full_probability_preserves_affine_values(self):
        mu = np.linspace(0, 2, 41)
        values = np.array([3*mu+2, -mu])
        np.testing.assert_allclose(convolve_mu(values,mu,.3,'full'),values,atol=1e-13)
        # The deliberate historical truncation biases endpoint expectations.
        self.assertGreater(convolve_mu(values,mu,.3,'truncated')[0,0],2.1)

    def test_all_solvers_bellman_and_event_updates(self):
        for module in (linear_constant, nonlinear_constant, linear_state):
            for mode in ('full', 'truncated'):
                kwargs = dict(nz=17,nmu=21,nu=21,dt=.01,observation_mode=mode)
                if module is linear_state:
                    kwargs['nP']=5
                else:
                    kwargs['t_obs']=(0.,.05,.1)
                result = module.solve(replace(module.Parameters(),**kwargs))
                policy = Policy(result)
                for k in policy.obs_idx[:-1]:
                    stage = list(policy.obs_idx).index(k)
                    for i,m in ((4,8),(8,10),(12,15)):
                        P = policy.sigma(policy.z[i])**2/(stage+1)
                        u,stop,value = policy.choose(policy.z[i],policy.mu[m],P,int(k))
                        if not policy.state_noise:
                            self.assertAlmostEqual(value,result['V'][k,i,m],places=9)
                            self.assertEqual(stop,result['Stop'][k,i,m])
                            self.assertAlmostEqual(u,result['Uopt'][k,i,m],places=12)
                        self.assertTrue(np.isfinite(value))
                readings = [1.1]*len(policy.obs)
                trajectory = simulate(policy,observations=readings,max_time=3)
                for j,event in enumerate(trajectory['events']):
                    self.assertAlmostEqual(event['mu'],1.1)
                    if not policy.state_noise:
                        self.assertAlmostEqual(event['P'],policy.sigma(0)**2/(j+1))
                changed=readings.copy();changed[-1]=1.8
                other=simulate(policy,observations=changed,max_time=3)
                a=trajectory['t']<policy.obs[-1]-1e-12
                b=other['t']<policy.obs[-1]-1e-12
                np.testing.assert_array_equal(trajectory['z'][a],other['z'][b])
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    self.assertTrue(np.isfinite(policy.choose(0.,3.,policy.sigma(0.)**2,0)[2]))

    def test_equal_noise_linear_solvers_agree(self):
        common=dict(nz=17,nmu=21,nu=21,dt=.01,c=.1,d=.1,sigma0=.25,observation_mode='full')
        a=linear_constant.solve(linear_constant.Parameters(t_obs=(0.,.2),**common))
        b=linear_state.solve(linear_state.Parameters(sigma_min=.25,sigma_max=.25,nP=3,**common))
        np.testing.assert_allclose(a['V'],b['V'][...,0],atol=1e-13)
        np.testing.assert_array_equal(a['Uopt'],b['Uopt'][...,0])

if __name__=='__main__':
    unittest.main()
