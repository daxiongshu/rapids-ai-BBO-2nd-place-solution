import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import sys
import os
from bayesmark.data import DATA_LOADERS
from bayesmark.constants import MODEL_NAMES
from time import sleep,time
import glob
from random import shuffle, randint
import numpy as np

def combine(runs):
    
    start = time()
    in_path = os.path.abspath('./input')
    out_path = os.path.abspath('./output')
    
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
    
    folders = ['eval', 'log', 'suggest_log', 'time']
    
    for f in folders:
        for r in runs:
            for fx in os.listdir(f"{out_path}/{r}/{f}"):
                assert os.path.exists(f"{out_path}/{name}/{f}/{fx}") == 0
            cmd = f"cp {out_path}/{r}/{f}/* {out_path}/{name}/{f}/"
            os.system(cmd)

    N_STEP, N_BATCH, N_REPEAT = 16, 8, 3
    tag = '-cuml-all'
    baseline = f"{in_path}/baseline-{N_STEP}-{N_BATCH}{tag}.json"
    if os.path.exists(baseline)==False:
        assert 0, f"{baseline} doesn't exist"
   
    cmd = f'cp {baseline} {out_path}/{name}/derived/baseline.json'
    print(cmd)
    os.system(cmd)

    cmd = f"bayesmark-agg -dir {out_path} -b {name}"
    print(cmd)
    os.system(cmd)
    
    cmd = f"bayesmark-anal -dir {out_path} -b {name} -v"
    print(cmd)
    os.system(cmd)
    
    duration = time() - start
    print(f"All done!! {name} Total time: {duration:.1f} seconds")
    
if __name__ == '__main__':
    runs = ['run_20201125_160529', 'run_20201126_053902']
    runs = ['run_20201125_141505', 'run_20201126_062036']
    runs = ['run_20201125_045204', 'run_20201126_053902']
    runs = ['run_20201125_060726', 'run_20201126_144803']
    runs = ['run_20201126_160134', 'run_20201125_165423']
    runs = ['run_20201126_222113', 'run_20201126_070047']
    runs = ['run_20201126_223719', 'run_20201126_133417']
    runs = ['run_20201126_233050', 'run_20201126_155911']
    runs = ['run_20201126_165041', 'run_20201126_233050']
    runs = ['run_20201127_000145', 'run_20201126_144803', 'run_20201126_233050']
    runs = ['run_20201127_054213', 'run_20201126_164503']
    runs = ['run_20201127_101420', 'run_20201127_104438']
    runs = ['run_20201127_142344', 'run_20201127_035612']
    runs = ['run_20201127_151356', 'run_20201127_145852']
    runs = ['run_20201127_115333', 'run_20201127_155302']
    runs = ['run_20201127_160122', 'run_20201127_173554']
    combine(runs)
