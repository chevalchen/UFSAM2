"""Run one frozen training configuration and retain environment/exit provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text())
    git = lambda *a: subprocess.check_output(['git','-C',str(root),*a],text=True).strip()
    if git('status','--porcelain'):
        raise RuntimeError('Execution checkout must be clean')
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,driver_version','--format=csv,noheader'],text=True)
    gpu_line = next(line for line in gpu.splitlines() if line.split(',')[0].strip()==args.gpu)
    uuid = gpu_line.split(',')[1].strip()
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    if uuid in processes:
        raise RuntimeError('Selected GPU is occupied: '+processes)
    output.mkdir(parents=True,exist_ok=False)
    command = [sys.executable,'-u',str(root/'UFSAM2/main.py')]
    for key,value in cfg.items():
        if isinstance(value,bool):
            if value: command.append('--'+key)
        else:
            command += ['--'+key]+[str(x) for x in (value if isinstance(value,list) else [value])]
    command += ['--output_dir',str(output.parent),'--name_exp',output.name]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TZ='Asia/Shanghai', PYTHONUNBUFFERED='1')
    import torch
    data = Path(cfg['data_root'])
    inputs = [cfg_path,Path(cfg['sam2_checkpoint']),data/'ADEChallengeData2016/objectInfo150.txt']
    inputs += sorted((data/'ADEChallengeData2016/utils').glob('*'))
    inputs += sorted((data/'COCO2014/splits').glob('*/*.pkl'))
    manifest = dict(git_sha=git('rev-parse','HEAD'),git_status='',execution_root=str(root),
                    config=cfg,command=command,environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','TZ')},
                    python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,host=platform.node(),
                    platform=platform.platform(),gpu=gpu_line,supervisor_pid=os.getpid(),start_unix=time.time(),
                    input_sha256={str(path):sha256(path) for path in inputs})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    with (output/'console.log').open('x') as log:
        process = subprocess.Popen(command,cwd=root/'UFSAM2',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        (output/'process.json').write_text(json.dumps(dict(supervisor_pid=os.getpid(),training_pid=process.pid),indent=2))
        code = process.wait()
    (output/'exit.json').write_text(json.dumps(dict(returncode=code,end_unix=time.time()),indent=2))
    sys.exit(code)


if __name__ == '__main__':
    main()
