from bayesmark.serialize import XRSerializer
import xarray as xr
import os

def combine_baseline(paths):
    out_path = os.path.abspath('./output')
    b0, meta0 = XRSerializer.load_derived(out_path, db=paths[0], key='baseline')
    for db in paths[1:]:
        b, meta = XRSerializer.load_derived(out_path, db=db, key='baseline')
        b0 = xr.concat([b0,b],dim='function')
        for k,v in meta['signature'].items():
            meta0['signature'][k] = v
    XRSerializer.save_derived(b0, meta0, db_root=out_path, db=db, key='baseline-cuml-all')

if __name__ == '__main__':
    paths = ['run_20201125_223926',
             'run_20201126_043653',
             'run_20201126_220329'
            ]
    paths = ['run_20201127_092520',
             'run_20201127_100101',
             'run_20201127_035612'
    ]
    combine_baseline(paths)
