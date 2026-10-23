"""Execution-only two-GPU queue; never edit scientific inputs or adjudicate results."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--worktree',required=True)
    a=p.parse_args()
    root=Path(a.output).resolve(); root.mkdir(parents=True,exist_ok=False)
    worktree=Path(a.worktree).resolve()
    env=os.environ.copy();env['TMPDIR']=str(root/'tmp');Path(env['TMPDIR']).mkdir()
    base=[sys.executable,'UFSAM2/exp004.py']
    execution=['--execution','experiments/EXP-004/configs/execution_v1.json']
    todo=[(r,s) for s in range(3) for r in ('C1','C2','C3')]
    all_runs=[str(root/f'{r}_seed{s}') for r,s in todo]
    running={}; completed=[]; logs=[]
    def status(state,**extra):
        data=dict(state=state,updated_unix=time.time(),completed=completed,
                  running={str(g):dict(pid=p.pid,row=r,seed=s) for g,(p,r,s) in running.items()},pending=todo,**extra)
        temporary=root/'status.tmp';temporary.write_text(json.dumps(data,indent=2));temporary.replace(root/'status.json')
    try:
        while todo or running:
            for gpu,(process,row,seed) in list(running.items()):
                code=process.poll()
                if code is not None:
                    del running[gpu]
                    if code!=0:raise RuntimeError(f'{row}/seed{seed} exited {code}')
                    completed.append(dict(row=row,seed=seed))
            for gpu in (0,1):
                if todo and gpu not in running:
                    row,seed=todo.pop(0)
                    free=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                    if int(free.strip())<8000:
                        todo.insert(0,(row,seed));continue
                    log=(root/f'{row}_seed{seed}.log').open('x');logs.append(log)
                    command=base+['train']+execution+['--row',row,'--seed',str(seed),'--device',f'cuda:{gpu}','--output',str(root/f'{row}_seed{seed}')]
                    process=subprocess.Popen(command,cwd=worktree,env=env,stdout=log,stderr=subprocess.STDOUT)
                    running[gpu]=(process,row,seed)
            status('TRAINING')
            time.sleep(10)
        status('LOCKING')
        subprocess.run(base+['lock','--runs']+all_runs+['--output',str(root/'locked.json')],cwd=worktree,env=env,check=True)
        status('VALIDATING')
        with (root/'validation.log').open('x') as log:
            subprocess.run(base+['validate']+execution+['--locked',str(root/'locked.json'),'--device','cuda:0','--output',str(root/'validation')],cwd=worktree,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        status('COMPLETED_AWAITING_HOST_REVIEW')
    except BaseException as error:
        for process,_,_ in running.values():
            if process.poll() is None:process.terminate()
        for process,_,_ in running.values():
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        status('INCONCLUSIVE_STOPPED',error=repr(error))
        raise
    finally:
        for log in logs:log.close()

if __name__=='__main__':main()
