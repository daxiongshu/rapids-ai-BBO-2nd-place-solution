from copy import deepcopy

import numpy as np
import scipy.stats as ss

from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from bayesmark.space import JointSpace
from sk_optimizer import ScikitOptimizer
from pysot_optimizer import PySOTOptimizer

class SKPysotOptimizer(AbstractOptimizer):

    def __init__(self, api_config, **kwargs):
        """Build wrapper class to use an optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)
        self.turbo_opt = PySOTOptimizer(api_config)
        self.sk_opt = ScikitOptimizer(api_config) 

    def suggest(self, n_suggestions=1):
        t_out = self.turbo_opt.suggest(n_suggestions)
        s_out = self.sk_opt.suggest(n_suggestions)
        #print("TURBO: ",t_out)
        #print("SK: ",s_out)
        N = len(t_out)//2
        return t_out[:N] + s_out[N:]
    
    def observe(self, X, y):
        self.turbo_opt.observe(X, y)
        self.sk_opt.observe(X, y)

if __name__ == "__main__":
    experiment_main(SKPysotOptimizer)
