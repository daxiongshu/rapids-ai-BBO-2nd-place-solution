from run_one_opt import run_one_opt
import glob
from time import time

def run_exhaustive_search():
    start = time()

    opts = glob.glob('./example_submissions/*')
    for opt in opts:
        print(opt)
    print("number of opts", len(opts)) 

    for opt in opts:
        run_one_opt(opt, quick_check=False)

    duration = time() - start
    print(f"All done!! Total time: {duration:.1f} seconds")

if __name__ == '__main__':
    run_exhaustive_search() 
