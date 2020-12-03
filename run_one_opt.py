import warnings
warnings.filterwarnings("ignore")

import os

from bayesmark.data import DATA_LOADERS, REAL_DATA_LOADERS
from bayesmark.constants import MODEL_NAMES
from time import sleep,time
import glob
from utils import get_paths_and_run_name, copy_baseline, get_opt, run_cmd, combine_experiments

no_multi_class_cuml = ['RF-cuml', 'SVM-cuml', 'xgb-cuml']
multi_class_data = ['iris', 'digits', 'wine', 'mnist']

COUNTER = 0
NUM_GPUS = 8 
N_JOBS = NUM_GPUS*4

def run(opt_path, n_jobs=16, N_STEP=16, N_BATCH=8, N_REPEAT=1, 
            run_cuml=False, quick_check=False, data_loaders=DATA_LOADERS,
            model_names=MODEL_NAMES, must_have_tag=None):

    start = time()

    in_path, out_path, name = get_paths_and_run_name()
    opt_root,opt = get_opt(opt_path)
    cmd = f"bayesmark-init -dir {out_path} -b {name}"
    run_cmd(cmd)
    copy_baseline(in_path, out_path, name, opt, N_STEP, N_BATCH, run_cuml)

    cmds = [] 
    if quick_check: 
        data_loaders = {'boston': (2,2)}
        if run_cuml:
            model_names = ['xgb-cuml']#['MLP-sgd-cuml']
        else:
            model_names = ['xgb']#['MLP-adam']

    if run_cuml:
        model_names = [i for i in model_names if i.endswith('-cuml')]# and 'MLP' not in i and 'xgb' not in i]

    if must_have_tag is not None:
        if isinstance(must_have_tag, list):
            model_names = [i for i in model_names if isin(i, must_have_tag)]
        else:
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
                        continue
                    if run_cuml and model == 'SVM-cuml' and data_loaders[data][1] == 1:
                        continue
                    cmd = f"bayesmark-launch -dir {out_path} -b {name} -n {N_STEP} -r 1 -p {N_BATCH} -o {opt} --opt-root {opt_root} -v -c {model} -d {data} -m {metric} -dr ./more_data&"
                    cmds.append(cmd)

    N = len(cmds)
    cmds = run_cmds(cmds, min(n_jobs, N))

    last = 0 
    while True:
        done, n = check_complete(N, out_path, name)
        sofar = time() - start    
        print(f"{sofar:.1f} seconds passed, {N - len(cmds)} tasks launched, {n} out of {N} tasks finished ...")
        if done:
            break
        sleep(3)
        if last < n:
            lc = len(cmds)
            cmds = run_cmds(cmds, min(n-last, lc))
        last = n
        
    cmd = f"bayesmark-agg -dir {out_path} -b {name}"
    run_cmd(cmd)

    cmd = f"bayesmark-anal -dir {out_path} -b {name} -v"
    run_cmd(cmd)
    
    duration = time() - start
    print(f"All done!! {name} Total time: {duration:.1f} seconds")
    return name, duration
    
def isin(i, must_have_tag):
    for j in must_have_tag:
        if j in i:
            return True
    return False

def run_cmds(cmds, n):
    global COUNTER
    for _ in range(n):
        cmd = cmds.pop()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(COUNTER % NUM_GPUS)
        COUNTER += 1
        run_cmd(cmd)
    return cmds

def check_complete(N, out_path, name):
    path = f"{out_path}/{name}/eval"
    if os.path.exists(path) == False:
        return False
    files = glob.glob(f"{path}/*.json")
    n = len(files)
    return n == N, n

def run_one_opt(opt, quick_check=False):

    # sklearn dataset
    name1,t1 = run(opt, N_STEP=16, N_BATCH=8, N_REPEAT=3, quick_check=quick_check, n_jobs=N_JOBS, run_cuml=True)
    if quick_check:
        return 

    # real dataset
    name2,t2 = run(opt, N_STEP=16, N_BATCH=8, N_REPEAT=3, quick_check=False, n_jobs=N_JOBS, run_cuml=True,
            data_loaders=REAL_DATA_LOADERS, must_have_tag=['MLP', 'xgb'] 
            )
    print(name1, name2)
    combine_experiments([name1, name2])

    print("Finished!", opt, f"Total time: {t1+t2:.1f} seconds")

if __name__ == '__main__':
    opt='./example_submissions/turbosk'
    run_one_opt(opt, quick_check=True)
