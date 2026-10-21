import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from models.sansa.local_prototype_residual import (FrozenSANSARepair, LocalPrototypeResidual,
    local_centers, prototype_context, row_context, support_prototypes)
from util.exp004_data import ROOT, load_contract, validate_manifest
from util.exp004_metrics import aggregate, adjudicate, bootstrap, pixel_counts, segmentation_loss
from util.exp004_runtime import assert_gradients, seed_all


class PrototypeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        seed_all(0)

    def test_fps_ties_and_empty_clusters_are_deterministic(self):
        x = torch.tensor([[1.,0.], [0.,1.], [-1.,0.], [0.,-1.]])
        expected = x[[0,2,1,3]]
        self.assertTrue(torch.equal(local_centers(x), expected))
        self.assertTrue(torch.equal(local_centers(torch.ones(6,3)), local_centers(torch.ones(6,3))))
        with self.assertRaises(ValueError):
            local_centers(torch.empty(0, 3))
        with self.assertRaises(ValueError):
            local_centers(torch.tensor([[float('nan'), 1.]]))

    def test_short_support_repeats_without_nan(self):
        centers = local_centers(torch.tensor([[1.,0.],[0.,1.]]))
        self.assertTrue(torch.equal(centers, torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])))
        self.assertTrue(torch.isfinite(local_centers(torch.zeros(1,3))).all())

    def test_global_raw_mean_not_mean_unit_tokens(self):
        features = torch.tensor([[[[3.,0.]],[[0.,1.]]]])
        p = support_prototypes(features, torch.ones(1,1,2), global_mean=True)
        want = torch.tensor([3.,1.]) / np.sqrt(10.)
        torch.testing.assert_close(p, want[None].repeat(4,1))
        with self.assertRaises(ValueError):
            support_prototypes(features, torch.zeros(1,1,2))

    def test_context_controls_and_dynamic_spatial_size(self):
        q = torch.randn(1,8,7,9)
        p = torch.nn.functional.normalize(torch.randn(20,8),dim=-1)
        context = prototype_context(q,p)
        self.assertEqual(context.shape,q.shape)
        torch.testing.assert_close(row_context(q,p,'C4'),torch.roll(context,(3,4),(-2,-1)))
        with patch('models.sansa.local_prototype_residual.prototype_context', wraps=prototype_context) as compute:
            self.assertIs(row_context(q,p,'C2'),q)
            compute.assert_called_once()
        repeated = p[:5].repeat_interleave(4,dim=0)
        torch.testing.assert_close(prototype_context(q,repeated),prototype_context(q,p[:5]))

    def test_disabled_and_zero_init_identity_then_trainable(self):
        model=LocalPrototypeResidual(8)
        memory,query,context=[torch.randn(1,8,7,9) for _ in range(3)]
        self.assertIs(model(memory,query,context,enabled=False),memory)
        self.assertTrue(torch.equal(model(memory,query,context),memory))
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        initial=model.body[0].weight.detach().clone()
        for _ in range(2):
            optimizer.zero_grad()
            model(memory,query,context).square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
        self.assertFalse(torch.equal(initial,model.body[0].weight))
        self.assertLessEqual(float((model(memory,query,context)-memory).abs().max()),0.100001)

    def test_train_keeps_all_sansa_parameters_frozen_and_eval(self):
        class Fake(nn.Module):
            def __init__(self):
                super().__init__()
                self.device='cpu'
                self.sam=nn.Linear(8,8)
                self.sam.hidden_dim=8
        model=FrozenSANSARepair(Fake()).train()
        self.assertFalse(model.sansa.training)
        self.assertTrue(model.residual.training)
        self.assertTrue(all(not p.requires_grad for p in model.sansa.parameters()))
        x=torch.randn(1,8,7,9)
        model.residual(x,x,x).square().mean().backward()
        assert_gradients(model)
        model.sansa.sam.weight.requires_grad_(True)
        with self.assertRaises(RuntimeError):
            assert_gradients(model)


class DataAndMetricsTests(unittest.TestCase):
    def test_frozen_manifest_and_leakage_guards(self):
        contract, manifest=load_contract(ROOT/'experiments/EXP-004/configs/contract_v1.json')
        exclusions=json.loads((ROOT/contract['exclusions']['path']).read_text())
        counts=validate_manifest(manifest,exclusions)
        self.assertEqual(counts,{'train':5702,'calibration':3085,'validation':3082})
        bad=copy.deepcopy(manifest)
        bad['partitions']['validation'][0]['support_ids'][0]=bad['partitions']['train'][0]['query_id']
        with self.assertRaises(ValueError): validate_manifest(bad,exclusions)
        bad=copy.deepcopy(manifest)
        bad['partitions']['calibration'][0]['query_id']=exclusions['image_ids'][0]
        with self.assertRaises(ValueError): validate_manifest(bad,exclusions)
        bad=copy.deepcopy(manifest)
        bad['partitions']['train'][0]['class_id']=0
        with self.assertRaises(ValueError): validate_manifest(bad,exclusions)

    def test_loss_ignore_and_known_pixel_counts(self):
        z=torch.tensor([[0.,3.],[-3.,100.]],requires_grad=True)
        y=torch.tensor([[0.,1.],[1.,255.]])
        self.assertEqual(pixel_counts(z,y),[1,1,2,2])
        loss=segmentation_loss(z,y)
        loss.backward()
        self.assertEqual(z.grad[1,1],0)
        with self.assertRaises(ValueError): segmentation_loss(z,torch.full_like(y,255.))
        with self.assertRaises(ValueError): segmentation_loss(z*float('nan'),y)

    def test_aggregate_class_counts_not_mean_episode_iou(self):
        r=[dict(class_id=1,counts=[2,1,3,1]),dict(class_id=1,counts=[2,0,3,9]),dict(class_id=2,counts=[2,1,3,2])]
        self.assertAlmostEqual(aggregate(r)['class_miou'],30.)
        self.assertAlmostEqual(aggregate(r)['worst_decile'],0.)
        np.testing.assert_allclose(bootstrap([0.02]*10),[2.,2.])

    def test_all_required_seed_guards_and_pairs(self):
        def records(fg):
            return [dict(episode_id=str(i),class_id=1+i%60,counts=[90,fg,100,100]) for i in range(600)]
        rows={'C0':records(50)}
        rows.update({f'{r}/{s}':records(52 if r in ('C3','C8') else 50) for r in ('C1','C2','C3','C4','C8') for s in range(3)})
        result=adjudicate(rows)
        self.assertEqual(result['decision'],'NEXT_DESIGN_ELIGIBLE_NOT_KEEP')
        rows['C3/2']=records(40)
        self.assertEqual(adjudicate(rows)['decision'],'DROP')
        rows['C4/0'].reverse()
        with self.assertRaises(ValueError): adjudicate(rows)


class ExecutionTests(unittest.TestCase):
    def test_frozen_contract_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'contract.json'
            path.write_text((ROOT/'experiments/EXP-004/configs/contract_v1.json').read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'Frozen scientific contract changed'):
                load_contract(path)

    def test_formal_run_rejects_missing_authorization_before_gpu_or_outputs(self):
        from types import SimpleNamespace
        from util.exp004_runtime import start_run
        args=SimpleNamespace(contract=ROOT/'experiments/EXP-004/configs/contract_v1.json', execution=None)
        with patch('util.exp004_runtime.git', return_value=''):
            with self.assertRaisesRegex(ValueError,'signed --execution'):
                start_run(args,{},'train')

    def test_training_loop_calibration_tie_selects_earliest_and_never_validation(self):
        import exp004
        from types import SimpleNamespace
        class FakeSansa(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=nn.Parameter(torch.ones(1),requires_grad=False)
                self.sam=SimpleNamespace(hidden_dim=8)
        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.sansa=FakeSansa()
                self.residual=LocalPrototypeResidual(8)
            def prepare(self,images,masks):return images
            def decode(self,cache,row):return self.residual(cache,cache,cache).mean(1)[0]
        class FakeDataset:
            def __init__(self,manifest,partition,root):
                if partition=='validation':raise AssertionError('Training accessed validation')
            def __len__(self):return 2
            def __getitem__(self,i):
                return torch.ones(1,8,7,9),torch.ones(5,7,9),torch.ones(7,9),dict(episode_id=str(i),class_id=1)
        contract=dict(training=dict(lr=1e-4,weight_decay=1e-2,betas=[0.9,0.999],eps=1e-8,
                                   rows=['C1','C2','C3'],seeds=[0,1,2],steps=2,calibration_interval=1),data_root='unused')
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(row='C3',seed=0,device='cpu')
            metric=[dict(episode_id='cal',class_id=1,counts=[10,5,10,10])]
            with patch.object(exp004,'build_model',return_value=FakeModel()),patch.object(exp004,'FixedEpisodes',FakeDataset),patch.object(exp004,'evaluate',return_value=metric) as evaluate:
                exp004.train(args,contract,{},Path(directory),{'mode':'UNIT_TEST_ONLY'})
                self.assertEqual(evaluate.call_count,2)
            done=json.loads((Path(directory)/'completed.json').read_text())
            self.assertEqual(done['chosen']['step'],1)
            self.assertEqual(done['steps'],2)
            self.assertEqual([json.loads(line)['episode_id'] for line in (Path(directory)/'training.jsonl').read_text().splitlines()],['0','1'])

if __name__=='__main__': unittest.main()
