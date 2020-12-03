import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import sys
import os
import subprocess
from bayesmark.data import DATA_LOADERS, REAL_DATA_LOADERS
from bayesmark.constants import MODEL_NAMES
from time import sleep,time
import glob
from random import shuffle, randint
import numpy as np

no_multi_class_cuml = ['RF-cuml', 'SVM-cuml', 'xgb-cuml']
multi_class_data = ['iris', 'digits', 'wine']
real_data = []
COUNTER = 0 

def run_all(opt, n_jobs=16, N_STEP=16, N_BATCH=8, N_REPEAT=1, 
            run_cuml=False, quick_check=False, data_loaders=DATA_LOADERS,
            model_names=MODEL_NAMES, must_have_tag=None):

    start = time()
    in_path = os.path.abspath('./input')
    out_path = os.path.abspath('./output')
    if 'RandomSearch' == opt:
        opt_root,opt = '.',opt
    elif 'RandomSearch' in opt:
        assert 0, "not supported yet"
        optx = opt.split()[-1]
        opt_root,opt = optx.split('/')[-2], optx.split('/')[-1]
        opt = 'RandomSearch '+opt
    else:
        optx = opt
        opt_root,opt = optx.split('/')[-2], optx.split('/')[-1]

    now = datetime.now()
    name = now.strftime("%Y%m%d_%H%M%S")
    name = f"run_{name}"
    
    if os.path.exists(out_path) == 0:
        os.mkdir(out_path)
        
    if os.path.exists(f"{out_path}/{name}"):
        assert 0, f"{out_path}/{name} already exists"
                      

    cmd = f"bayesmark-init -dir {out_path} -b {name}"
    print(cmd)
    os.system(cmd)
    
    tag = '-cuml-all' if run_cuml else ''
    baseline = f"{in_path}/baseline-{N_STEP}-{N_BATCH}{tag}.json"
    if os.path.exists(baseline)==False:
        assert 0, f"{baseline} doesn't exist"
   
    if 'RandomSearch' not in opt:
        cmd = f'cp {baseline} {out_path}/{name}/derived/baseline.json'
        print(cmd)
        os.system(cmd)
    else:
        if os.path.exists(baseline)==False:
            assert 0, f"{baseline} doesn't exist"

    cmds = [] 
    if quick_check: 
        data_loaders = {'digits': (1,1)}
        data_loaders = {'boston': (2,2)}
        data_loaders = {'breast': (1,1)}
        data_loaders = {'higgs': (1,1)}
        if run_cuml:
            model_names = ['xgb-cuml']#['MLP-sgd-cuml']
        else:
            model_names = ['xgb']#['MLP-adam']

    if run_cuml:
        model_names = [i for i in model_names if i.endswith('-cuml')]# and 'MLP' not in i and 'xgb' not in i]

    if must_have_tag is not None:
        model_names = [i for i in model_names if must_have_tag in i]
    print(model_names)

    for data in data_loaders:
        metrics = ['nll', 'acc'] if data_loaders[data][1] == 1 else ['mse', 'mae']
        for metric in metrics:
            for model in model_names:
                for _ in range(N_REPEAT):
                    if run_cuml==False and '-cuml' in model:
                        continue
                    if run_cuml and model in no_multi_class_cuml and data in multi_class_data:
                        #print(model, data)
                        continue
                    if run_cuml and model == 'SVM-cuml' and data_loaders[data][1] == 1:
                        continue
                    cmd = f"bayesmark-launch -dir {out_path} -b {name} -n {N_STEP} -r 1 -p {N_BATCH} -o {opt} --opt-root {opt_root} -v -c {model} -d {data} -m {metric} -dr ./big_data&"
                    cmds.append(cmd)

    N = len(cmds)
    cmds = run_cmds(cmds, min(n_jobs, N))

    last = 0 
    while True:
        done, n = check_complete(N, out_path, name)
        sofar = time() - start    
        print(f"{sofar:.1f} seconds passed, {N - len(cmds)} tasks launched, {n} out of {N} tasks finished ...")
        sleep(3)
        if done:
            break
        if last < n:
            lc = len(cmds)
            cmds = run_cmds(cmds, min(n-last, lc))
        last = n
        
    cmd = f"bayesmark-agg -dir {out_path} -b {name}"
    print(cmd)
    os.system(cmd)
    
    cmd = f"bayesmark-anal -dir {out_path} -b {name} -v"
    print(cmd)
    os.system(cmd)
    
    duration = time() - start
    print(f"All done!! {name} Total time: {duration:.1f} seconds")
    
def run_cmds(cmds, n):
    global COUNTER
    for _ in range(n):
        cmd = cmds.pop()
        print(cmd)
        #gpu = randint(0,3)
        os.environ["CUDA_VISIBLE_DEVICES"]=str(COUNTER%4)
        COUNTER += 1
        os.system(cmd)
        sleep(3)
    return cmds

def check_complete(N, out_path, name):
    path = f"{out_path}/{name}/eval"
    if os.path.exists(path) == False:
        return False
    files = glob.glob(f"{path}/*.json")
    n = len(files)
    return n == N, n


if __name__ == '__main__':
    opt = './example_submissions/hyperopt'
    opt = 'RandomSearch'

    # sklearn dataset
    #run_all(opt, N_STEP=16, N_BATCH=8, N_REPEAT=3, quick_check=False, n_jobs=32, run_cuml=True, must_have_tag='xgb')

    # real dataset
    run_all(opt, N_STEP=16, N_BATCH=8, N_REPEAT=3, quick_check=False, n_jobs=4, run_cuml=True,
            data_loaders=REAL_DATA_LOADERS, must_have_tag='xgb' 
            )
