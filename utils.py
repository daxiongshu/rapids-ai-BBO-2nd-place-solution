from datetime import datetime
import os

def get_run_name():
    now = datetime.now()
    name = now.strftime("%Y%m%d_%H%M%S")
    name = f"run_{name}"
    return name

def get_opt(opt_path):
    if 'RandomSearch' == opt_path:
        opt_root, opt = '.', opt_path
    else:
        opt_root = os.path.dirname(opt_path)
        opt = os.path.basename(opt_path)
    return opt_root, opt

def copy_baseline(in_path, name, N_STEP, N_BATCH, run_cuml):
    tag = '-cuml-all' if run_cuml else ''
    baseline = f"{in_path}/baseline-{N_STEP}-{N_BATCH}{tag}.json"
    if os.path.exists(baseline)==False:
        assert 0, f"{baseline} doesn't exist"

    if 'RandomSearch' not in opt:
        cmd = f'cp {baseline} {out_path}/{name}/derived/baseline.json'
        run_cmd(cmd)
    else:
        if os.path.exists(baseline)==False:
            assert 0, f"{baseline} doesn't exist"

def get_paths_and_run_name():
    in_path = os.path.abspath('./input')
    out_path = os.path.abspath('./output')

    name = get_run_name()

    if os.path.exists(out_path) == 0:
        os.mkdir(out_path)

    if os.path.exists(f"{out_path}/{name}"):
        assert 0, f"{out_path}/{name} already exists"

    return in_path, out_path, name

def run_bayesmark_init(out_path, name):
    cmd = f"bayesmark-init -dir {out_path} -b {name}"
    run_cmd(cmd)

def run_bayesmark_agg(out_path, name):
    cmd = f"bayesmark-agg -dir {out_path} -b {name}"
    run_cmd(cmd)

def run_bayesmark_anal(out_path, name):
    cmd = f"bayesmark-anal -dir {out_path} -b {name} -v"
    run_cmd(cmd)

def run_cmd(cmd):
    print(cmd)
    os.system(cmd)
