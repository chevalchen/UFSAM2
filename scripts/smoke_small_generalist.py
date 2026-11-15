"""Exercise actual train/save/validation entry points with four training batches."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'UFSAM2'))
import torch
import main as train
import inference_fss
import opts
from datasets.transform_utils import collate_training_episodes

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--steps', type=int, default=4)
parser.add_argument('--validation-episodes', type=int, default=4)
cli = parser.parse_args()
torch.set_num_threads(4)
args = opts.get_args_parser().parse_args([
    '--sam2_version','small','--sam2_checkpoint','/data6/chensq/SANSA/pretrain/sam2_hiera_small.pt',
    '--adaptformer_stages','2','3','--channel_factor','0.8','--dataset_file','multi',
    '--multi_train','coco','ade20k','--ds_weight','0.45','0.10','--fold','-1',
    '--epochs','1','--batch_size',str(cli.batch),'--shots','1','--J','3','--prompt','mask',
    '--no_distributed','--data_root','/data6/chensq/datasets','--output_dir',cli.output])
original_dataset = train.build_dataset
original_model = train.build_sansa
models = []

def digest(model):
    value = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            value.update(name.encode())
            value.update(p.detach().cpu().numpy().tobytes())
    return value.hexdigest()

def build_model(*a, **kw):
    model = original_model(*a, **kw)
    assert all(p.requires_grad == ('adapter' in name) for name,p in model.named_parameters())
    models.append((model, digest(model), {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}))
    return model

def build_dataset(*a, **kw):
    dataset = original_dataset(*a, **kw)
    assert len(dataset)==210000
    # Explicitly exercise heterogeneous collation including ADE20K's additional fields.
    batch = collate_training_episodes([dataset.dataset_list[0][0], dataset.dataset_list[1][0]])
    assert batch['support_imgs'].shape == (2,3,3,640,640)
    assert batch['query_mask'].shape == (2,640,640)
    dataset.samples_per_epoch = cli.batch * cli.steps
    return dataset

def validation_dataset(*a, **kw):
    dataset = original_dataset(*a, **kw)
    dataset.test_episodes = cli.validation_episodes
    return dataset

train.build_dataset = build_dataset
train.build_sansa = build_model
inference_fss.build_dataset = validation_dataset
start = time.time()
train.main(args)
model, before, adapters = models[0]
assert digest(model) == before
assert all(torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all()) for p in model.parameters())
assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.parameters() if p.requires_grad)
assert any(not torch.equal(p.detach().cpu(), adapters[n]) for n,p in model.named_parameters() if p.requires_grad)
checkpoint = torch.load(Path(cli.output)/'checkpoint.pth', map_location='cpu', weights_only=False)
assert checkpoint['model'] and all('adapter' in k for k in checkpoint['model'])
result = dict(batch=cli.batch, success=True, elapsed=time.time()-start,
              peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
              peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
              parameters=sum(p.numel() for p in model.parameters()),
              trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
              frozen_unchanged=True, adapter_updated=True, checkpoint_valid=True,
              load_report=model.checkpoint_load_report)
with (Path(cli.output)/'audit.json').open('x') as f:
    json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
