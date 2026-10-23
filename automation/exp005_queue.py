"""Six-run dual-GPU queue; one lock and three-dataset evaluation; no auto retries."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def main():
 p=argparse.ArgumentParser();p.add_argument('--worktree',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False);worktree=Path(a.worktree).resolve()
 env=os.environ.copy();env['TMPDIR']=str(root/'tmp');Path(env['TMPDIR']).mkdir()
 command=[sys.executable,'UFSAM2/exp005.py'];todo=[(r,s) for s in range(3) for r in ['uniform','uncertainty']];directories=[str(root/f'{r}_seed{s}') for r,s in todo]
 running={};completed=[];logs=[];validation_process=None
 def state(value,**kwargs):
  status=dict(state=value,updated_unix=time.time(),completed=completed,pending=todo,running={str(g):dict(pid=p.pid,mode=r,seed=s) for g,(p,r,s) in running.items()},**kwargs)
  temp=root/'status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(root/'status.json')
 def free(gpu):return int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
 try:
  while todo or running:
   for gpu,(process,row,seed) in list(running.items()):
    result=process.poll()
    if result is not None:
     del running[gpu]
     if result:raise RuntimeError(f'{row}/{seed} failed: {result}')
     completed.append(dict(mode=row,seed=seed))
   for gpu in [0,1]:
    if todo and gpu not in running and free(gpu)>=12000:
     row,seed=todo.pop(0);log=(root/f'{row}_seed{seed}.log').open('x');logs.append(log)
     process=subprocess.Popen(command+['train','--mode',row,'--seed',str(seed),'--device',f'cuda:{gpu}','--output',str(root/f'{row}_seed{seed}')],cwd=worktree,env=env,stdout=log,stderr=subprocess.STDOUT)
     running[gpu]=(process,row,seed)
   state('TRAINING');time.sleep(10)
  while free(0)<12000:state('WAITING_GPU_FOR_VALIDATION');time.sleep(30)
  state('VALIDATING')
  with (root/'validation.log').open('x') as log:
   validation_process=subprocess.Popen(command+['validate','--runs']+directories+['--device','cuda:0','--output',str(root/'validation')],cwd=worktree,env=env,stdout=log,stderr=subprocess.STDOUT)
   if validation_process.wait():raise RuntimeError('Locked validation failed')
  state('COMPLETED_AWAITING_ANALYSIS')
 except BaseException as error:
  children=[v[0] for v in running.values()]+([validation_process] if validation_process else [])
  for process in children:
   if process.poll() is None:process.terminate()
  for process in children:
   try:process.wait(timeout=30)
   except subprocess.TimeoutExpired:process.kill();process.wait()
  state('STOPPED_ERROR',error=repr(error));raise
 finally:
  for log in logs:log.close()
if __name__=='__main__':main()
