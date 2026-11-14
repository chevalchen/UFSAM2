import subprocess, json, time, os, sys
from pathlib import Path
root=Path(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=False)
python='/data6/chensq/miniconda3/envs/sam2coco/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
for source in ('synthetic','real','e2e'):
 for precision in ('fp32','bf16'):
  for shots in (1,5):
   for rep in (1,2,3):
    for model in (('large','small') if rep%2 else ('small','large')):
     name=f'{source}_{precision}_{shots}shot_r{rep}_{model}'
     cmd=[python,'-u','benchmark_backbone.py','--model',model,'--checkpoint',f'/data6/chensq/SANSA/pretrain/sam2_hiera_{model}.pt','--output',str(out/(name+'.json')),'--source',source,'--shots',str(shots),'--precision',precision]
     gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,temperature.gpu,power.draw,utilization.gpu,memory.used','--format=csv'],text=True)
     # Refuse to overlap with any foreign CUDA process on our GPU.
     uuid=gpu.splitlines()[1].split(',')[1].strip()
     processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
     if uuid in processes:
      raise RuntimeError('GPU 0 has another compute process: '+processes)
     start=time.time()
     with (out/(name+'.log')).open('x') as log:
      proc=subprocess.run(cmd,cwd=root/'UFSAM2',env=env,stdout=log,stderr=subprocess.STDOUT)
     record=dict(run=name,command=cmd,returncode=proc.returncode,start_unix=start,end_unix=time.time(),gpu_before=gpu)
     with (out/'runs.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
     print(name, 'PASS' if proc.returncode==0 else 'FAILED',round(time.time()-start,1),flush=True)
     if proc.returncode and 'out of memory' not in (out/(name+'.log')).read_text().lower():
      raise RuntimeError('Non-OOM failure; inspect '+name)
