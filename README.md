# rapids-ai-BBO-2nd-place-solution

### Install Instructions
#### Create a conda Environment
- conda create -n bbo_rapids python=3.7
- conda activate bbo_rapids
#### Install cudf, cuml and pytorch
- conda install "pytorch=1.6" "cudf=0.16" "cuml=0.16" cudatoolkit=10.2.89 -c pytorch -c rapidsai -c nvidia -c conda-forge -c defaults
#### Install optimization algorithms
- pip install gpytorch==1.2.1
- pip install git+https://github.com/uber-research/TuRBO.git@master
- pip install pySOT==0.2.3 opentuner==0.8.2 nevergrad==0.1.4 hyperopt==0.1.1 scikit-optimize==0.5.2 scikit-learn==0.20.2 xgboost==1.2.1
#### Install rapids-enabled bayesmark
- git clone https://github.com/daxiongshu/bayesmark
- cd bayesmark
- git checkout rapids
- ./build_wheel.sh
- python setup.py install
