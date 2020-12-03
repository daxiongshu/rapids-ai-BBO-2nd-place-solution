from datetime import datetime
import os

def get_run_name():
    now = datetime.now()
    name = now.strftime("%Y%m%d_%H%M%S")
    name = f"run_{name}"
    return name

def run_bayesmark_init(out_path, name):
    cmd = f"bayesmark-init -dir {out_path} -b {name}"
    run_cmd(cmd)

def run_cmd(cmd):
    print(cmd)
    os.system(cmd)
