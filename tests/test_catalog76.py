import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np
import torch

from experiments.catalog76 import entries, selected_methods, register, prepare, run, worker, BLOCKED, LIVE
from utils.catalog76_policies import features, graph, operators, pair_energy, FEATURE_RECIPES, action_quantities
from utils.contact_active_gsp_features import compute_contact_active_features
from utils.networks import ACTION_TO_CONTROL
from utils.d4_graph_residual import square_symmetries, transform_local_geometry
from utils.agents import POLICY_TYPES, DDPGAgent
from utils.buffer import ReplayBuffer


class CatalogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        register()

    def setUp(self):
        torch.manual_seed(903)
        self.raw=torch.randn(64,18)

    def test_all_76_are_accounted_for_without_silent_fallback(self):
        records=entries()
        self.assertEqual(len(records),76)
        self.assertEqual(len({r['id'] for r in records}),76)
        for row in records:
            if row['id'] in BLOCKED:
                with self.assertRaises(ValueError):
                    selected_methods([row['id']])
            else:
                self.assertIn(row['actor_model'],POLICY_TYPES)
                self.assertTrue(selected_methods([row['id']]))

    def test_alias_dedup_and_dependency_order(self):
        methods=selected_methods(['G3','D1','A1','W6','D4','R9'])
        self.assertEqual(list(methods).count('explore_hks_6al'),1)
        self.assertIn('explore_hks_sigma125',methods)
        seen=set()
        for name,spec in methods.items():
            if spec.get('control'):
                self.assertIn(spec['control'],seen)
            seen.add(name)
        methods=selected_methods(['E2','E4','W4','D4','G6'])
        self.assertNotIn('catalog_radius_hks_control',methods)
        self.assertEqual(methods['catalog_radius_hks']['control'],'explore_geometric_stats3')
        self.assertEqual(methods['catalog_effective_resistance']['control'],methods['catalog_split_hks']['control'])

    def test_every_new_actor_forward_backward_target_shapes_and_locality(self):
        for name,cls in POLICY_TYPES.items():
            if not name.startswith('catalog_'):
                continue
            with self.subTest(actor=name):
                actor=cls(18,5).eval()
                target=cls(18,5).eval()
                target.load_state_dict(actor.state_dict())
                result=actor(self.raw[:12])
                self.assertEqual(result.shape,(12,5))
                self.assertTrue(torch.isfinite(result).all())
                torch.testing.assert_close(result,target(self.raw[:12]))
                torch.testing.assert_close(result[:1],actor(self.raw[:1]),atol=3e-5,rtol=3e-5)
                result.square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None))

    def test_features_batch_translation_and_extreme_finite(self):
        for name,config in FEATURE_RECIPES.items():
            for batch in (1,12,64):
                with self.subTest(recipe=name,batch=batch):
                    raw=self.raw[:batch]
                    expected=features(raw,config)
                    shifted=raw.clone()
                    shifted[:,2:4]+=1000
                    torch.testing.assert_close(features(shifted,config),expected,atol=0,rtol=0)
                    for value in (torch.zeros_like(raw),raw*1e6):
                        self.assertTrue(torch.isfinite(features(value,config)).all())
                    self.assertEqual(expected.shape[0],batch)
                    self.assertEqual(expected.dtype,torch.float32)
        with self.assertRaises(ValueError):
            features(torch.zeros(1,21),{})

    def test_graph_symmetry_masks_and_pair_energy_reference(self):
        for config in FEATURE_RECIPES.values():
            w=graph(self.raw[:3],config)
            torch.testing.assert_close(w,w.transpose(-1,-2),atol=1e-6,rtol=1e-6)
            self.assertTrue((w>=0).all())
            self.assertTrue((w.diagonal(dim1=-1,dim2=-2)==0).all())
        w=graph(self.raw[:3],{})
        x=torch.randn(3,6,4)
        reference=torch.einsum('bni,bnm,bmi->bi',x,operators(w)[2],x)
        torch.testing.assert_close(pair_energy(w,x),reference,atol=2e-6,rtol=2e-5)
        p=w/w.sum(-1,keepdim=True).clamp_min(1e-12)
        expected=torch.stack([torch.linalg.matrix_power(p,k)[:,0,0] for k in (2,4,6)],-1)
        torch.testing.assert_close(features(self.raw[:3],{'descriptor':'rwse246'}),expected,atol=2e-6,rtol=2e-5)

    def test_effective_resistance_reference_and_disconnect_mask(self):
        raw=torch.zeros(2,18)
        w=graph(raw,{})
        lap=operators(w)[2]
        pinv=np.linalg.pinv(lap.numpy().astype(np.float64),hermitian=True)
        expected=np.stack([pinv[:,0,0]+pinv[:,j,j]-2*pinv[:,0,j] for j in (3,4,5)],-1)
        actual=features(raw,{'descriptor':'resistance'})
        np.testing.assert_allclose(actual[:,:3].numpy(),expected,atol=2e-6,rtol=1e-5)
        self.assertTrue((actual[:,3:]==1).all())
        far=self.raw[:2]*1e6
        disconnected=features(far,{'descriptor':'resistance'})
        self.assertTrue((disconnected==0).all())

    def test_candidate_motion_matches_existing_reference_and_noop(self):
        raw=self.raw[:12]
        q=action_quantities(raw,'all')
        expected=compute_contact_active_features(raw,raw.new_tensor(ACTION_TO_CONTROL))
        torch.testing.assert_close(q-q[:,:1],expected,atol=2e-5,rtol=2e-4)
        for motion in ('self','hypotheses','horizon3'):
            f=features(raw,dict(descriptor='action',mode='delta',motion=motion)).reshape(12,5,4)
            self.assertTrue((f[:,0]==0).all())
        for recipe in ('action_absolute','action_current_delta','action_self_contact','action_hypotheses','action_horizon3'):
            actor=POLICY_TYPES['catalog_'+recipe+'_control'](18,5).eval()
            candidate=features(raw,FEATURE_RECIPES[recipe]).reshape(12,-1,4)
            control=actor.augment_observation(raw)[:,18:].reshape(12,-1,4)
            torch.testing.assert_close(control[:,:,:2],candidate[:,:,:2],atol=0,rtol=0)
            self.assertTrue((control[:,:,2:]==0).all())

    def test_vector_residual_d4_action_permutation(self):
        actor=POLICY_TYPES['catalog_vector_projection'](18,5).eval()
        controls=self.raw.new_tensor(ACTION_TO_CONTROL)
        expected=actor.graph_residual(self.raw[:4])
        for matrix in square_symmetries():
            transformed=transform_local_geometry(self.raw[:4],matrix[None])[:,0]
            permutation=((controls@matrix.T)[:,None]-controls[None]).square().sum(-1).argmin(-1)
            torch.testing.assert_close(actor.graph_residual(transformed)[:,permutation],expected,atol=1e-6,rtol=1e-5)

    def test_raw_replay_and_identical_critic_inputs(self):
        replay=ReplayBuffer(20,3,[18]*3,[5]*3)
        self.assertEqual([b.shape[1] for b in replay.obs_buffs],[18]*3)
        joint=torch.randn(12,69)
        for name in ('catalog_split_hks','catalog_effective_resistance','catalog_encoder_learned','catalog_vector_projection'):
            torch.manual_seed(8)
            raw=DDPGAgent(18,5,69,actor_model='mlp')
            candidate=DDPGAgent(18,5,69,actor_model=name)
            for branch in ('critic','target_critic'):
                left,right=getattr(raw,branch),getattr(candidate,branch)
                self.assertEqual(left.fc1.in_features,right.fc1.in_features)
                captured=[]
                handle=right.register_forward_pre_hook(lambda m,a:captured.append(a[0].clone()))
                right(joint)
                handle.remove()
                torch.testing.assert_close(captured[0],joint,atol=0,rtol=0)
            self.assertEqual(candidate.policy.actor_input_dim,candidate.target_policy.actor_input_dim)

    def test_no_training_without_explicit_execution_and_no_live_overwrite(self):
        with patch('subprocess.run') as process:
            with self.assertRaises(ValueError):
                run(LIVE,False)
            process.assert_not_called()
        with self.assertRaises(ValueError):
            worker(LIVE,'raw_mlp','formal',False)
        with self.assertRaises(FileExistsError):
            prepare(LIVE,['G1'])
        with self.assertRaises(ValueError):
            prepare(LIVE/'accidental_new_campaign',['G1'])

    def test_frozen_template_refuses_when_its_recorded_source_is_active(self):
        import experiments.catalog76 as catalog
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/'active_source'
            destination=root/'frozen_template'
            source.mkdir()
            destination.mkdir()
            (source/'parallel_status.json').write_text(json.dumps({'active':[{'method':'running'}]}))
            manifest={'methods':{},'prepared_from':str(source)}
            with patch('experiments.run_gsp_exploration_campaign.verify_snapshot',return_value=manifest):
                with self.assertRaisesRegex(RuntimeError,'Existing campaign still has active workers'):
                    catalog.run(destination,True)
            self.assertFalse((destination/'catalog_started.json').exists())

    def test_synthetic_update_and_checkpoint_reload(self):
        from main import make_parallel_env
        from algorithms.maddpg import MADDPG
        env=make_parallel_env('simple_spread',1,1,True)
        try:
            for name in ('catalog_split_hks','catalog_action_current_delta','catalog_encoder_learned','catalog_vector_projection'):
                with self.subTest(actor=name), tempfile.TemporaryDirectory() as tmp:
                    model=MADDPG.init_from_env(env,actor_model=name)
                    model.prep_training(device='cpu')
                    sample=([torch.randn(12,18) for _ in range(3)],
                            [torch.eye(5)[torch.arange(12)%5] for _ in range(3)],
                            [torch.randn(12) for _ in range(3)],
                            [torch.randn(12,18) for _ in range(3)],
                            [torch.zeros(12) for _ in range(3)])
                    model.update(sample,0,logger=Mock())
                    model.update_all_targets()
                    model.prep_rollouts(device='cpu')
                    expected=model.agents[0].policy(self.raw[:4])
                    target=Path(tmp)/'model.pt'
                    model.save(target)
                    restored=MADDPG.init_from_save(target)
                    restored.prep_rollouts(device='cpu')
                    torch.testing.assert_close(restored.agents[0].policy(self.raw[:4]),expected,rtol=0,atol=0)
        finally:
            env.close()

    def test_scheduler_finishes_smoke_before_formal_and_halts_on_failure(self):
        import experiments.catalog76 as catalog
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fake_live=root/'other'
            fake_live.mkdir()
            manifest={'methods':{'raw':{'control':None},'a':{'control':'raw'},'b':{'control':'raw'},'c':{'control':'raw'}},
                      'prepared_from':str(fake_live)}
            launched=[]
            class Process:
                pid=999
                def __init__(self,command,**kwargs):
                    self.name=command[command.index('--method')+1]
                    self.phase=command[command.index('--phase')+1]
                    launched.append((self.name,self.phase))
                    self.failed=self.name=='a' and self.phase=='formal'
                    state={'status':'failed' if self.failed else 'completed','validation':{'passed':not self.failed}}
                    (root/(self.phase+'_'+self.name+'.json')).write_text(json.dumps(state))
                def poll(self):
                    return 1 if self.failed else 0
            with patch.object(catalog,'LIVE',fake_live), patch('experiments.run_gsp_exploration_campaign.verify_snapshot',return_value=manifest), \
                 patch.object(catalog,'memory_info',return_value={'avail_phys':10*1024**3,'avail_page':20*1024**3}), \
                 patch.object(catalog.subprocess,'Popen',Process), patch.object(catalog.time,'sleep'):
                run(root,True)
            self.assertEqual(launched[:2],[('raw','smoke'),('raw','formal')])
            self.assertNotIn(('c','smoke'),launched)
            state=json.loads((root/'catalog_status.json').read_text())
            self.assertEqual(state['status'],'paused_technical_failure')
            self.assertEqual(state['active'],[])
            with self.assertRaises(FileExistsError):
                with patch.object(catalog,'LIVE',fake_live), patch('experiments.run_gsp_exploration_campaign.verify_snapshot',return_value=manifest):
                    run(root,True)


if __name__=='__main__':
    unittest.main()
