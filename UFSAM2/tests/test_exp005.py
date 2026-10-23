import unittest,copy
import numpy as np
import torch
from models.sansa.support_graph import SupportGraph,graph_nodes
from util.exp005_data import load_inputs,ROOT,validate
from util.exp005_metrics import boundary_counts,compare,decision

class GraphTests(unittest.TestCase):
 def setUp(self):torch.manual_seed(0);torch.set_num_threads(2)
 def test_identity_graph_small_foreground(self):
  x=torch.randn(1,256,16,16);mask=torch.zeros(16,16);mask[1,1]=1
  model=SupportGraph().eval();nodes,adj,_,_=graph_nodes(x,mask)
  self.assertEqual(len(nodes),1);self.assertEqual(adj.item(),1)
  for mode in ['uniform','uncertainty','shuffled']:
   z,_,_=model(x,mask,mode);self.assertTrue(torch.equal(z,x))
  self.assertIs(model(x,mask,'baseline')[0],x)
  with self.assertRaises(ValueError):graph_nodes(x,mask*0)
 def test_gradient_and_background(self):
  x=torch.randn(1,256,16,16);mask=torch.ones(16,16);mask[:3]=0
  model=SupportGraph();opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
  for step in range(2):
   opt.zero_grad();z,aux,_=model(x,mask,seed=1);loss=z.square().mean()+aux;loss.backward()
   self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
   if step:self.assertTrue(all(torch.count_nonzero(module.weight.grad)>0 for module in [model.input,*model.layers,model.mu,model.var,model.output]))
   opt.step();self.assertTrue(torch.equal(z[:,:,:3],x[:,:,:3]))
 def test_shuffled_preserves_values_and_changes_correspondence(self):
  x=torch.randn(1,256,16,16);mask=torch.ones(16,16);model=SupportGraph().eval()
  torch.nn.init.normal_(model.output.weight,std=.1)
  a,_,sa=model(x,mask,'uncertainty',seed=2);b,_,sb=model(x,mask,'shuffled',seed=2)
  self.assertEqual(sa['reliability_mean'],sb['reliability_mean']);self.assertEqual(sa['reliability_std'],sb['reliability_std'])
  self.assertFalse(torch.equal(a,b));self.assertTrue(torch.equal(b,model(x,mask,'shuffled',seed=2)[0]))
 def test_manifest(self):
  c,m=load_inputs(ROOT/'experiments/EXP-005/config.json');validate(m)
  bad=copy.deepcopy(m);bad['partitions']['paco_part'][0]['query']['identity']=m['partitions']['train'][0]['query']['identity']
  with self.assertRaises(ValueError):validate(bad)
 def test_boundary_known(self):
  y=torch.zeros(32,32);y[8:24,8:24]=1;z=(y*2-1)*10
  i,u=boundary_counts(z,y);self.assertEqual(i,u);self.assertGreater(u,0)
  self.assertEqual(boundary_counts(-z,y)[0],0)
 def test_gate_decision_requires_two_datasets(self):
  good={'comparisons':{k:dict(mean_class_miou_delta_pp=.6,positive_ci=True) for k in ['G1-B0','G1-G0','G1-G2']}}
  bad=copy.deepcopy(good);bad['comparisons']['G1-G0']['mean_class_miou_delta_pp']=-.3
  self.assertTrue(decision(dict(coco=good,pascal=good,paco=good))['continue_investigation'])
  self.assertFalse(decision(dict(coco=good,pascal=good,paco=bad))['continue_investigation'])

class ControlTests(unittest.TestCase):
 def test_uniform_weights_constant_and_shuffle_preserves_multiset(self):
  torch.manual_seed(2);model=SupportGraph().eval();x=torch.randn(1,256,16,16);mask=torch.ones(16,16)
  nodes,adj,_,_=graph_nodes(x,mask);mu,_=model.distribution(nodes,adj)
  inputs=[];handle=model.output.register_forward_pre_hook(lambda module,args:inputs.append(args[0].detach()))
  for mode in ['uniform','uncertainty','shuffled']:model(x,mask,mode,seed=4)
  handle.remove()
  weights=[(value/mu.detach()).mean(-1) for value in inputs]
  torch.testing.assert_close(weights[0],weights[0].mean().expand_as(weights[0]))
  torch.testing.assert_close(weights[1].sort().values,weights[2].sort().values)
  self.assertFalse(torch.allclose(weights[1],weights[2]))
 def test_statistical_rows_and_missing_control(self):
  def records(fg):return [dict(episode_id=str(i),class_id=i%2,counts=[80,fg,100,100]) for i in range(20)]
  rows={'B0':records(50)};rows.update({f'{r}/{s}':records(52 if r=='G1' else 50) for r in ['G0','G1','G2'] for s in range(3)})
  result=compare(rows);self.assertAlmostEqual(result['comparisons']['G1-G0']['mean_class_miou_delta_pp'],2)
  self.assertTrue(result['comparisons']['G1-G0']['positive_ci'])
  del rows['G2/2']
  with self.assertRaises(ValueError):compare(rows)

if __name__=='__main__':unittest.main()
