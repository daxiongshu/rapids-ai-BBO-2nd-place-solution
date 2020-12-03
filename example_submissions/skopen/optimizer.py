from copy import deepcopy

import numpy as np
import scipy.stats as ss

from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from bayesmark.space import JointSpace
from sk_optimizer import ScikitOptimizer
from opentuner_optimizer import OpentunerOptimizer

class SKOpenOptimizer(AbstractOptimizer):

    def __init__(self, api_config, **kwargs):
        """Build wrapper class to use an optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)
        self.opt1 = OpentunerOptimizer(api_config)
        self.opt2 = ScikitOptimizer(api_config) 

    def suggest(self, n_suggestions=1):
        t_out = self.opt1.suggest(n_suggestions)
        s_out = self.opt2.suggest(n_suggestions)
        N = len(t_out)//2
        return t_out[:N] + s_out[N:]

    def observe(self, X, y):
        self.opt1.observe(X, y)
        self.opt2.observe(X, y)

if __name__ == "__main__":
    experiment_main(SKOpenOptimizer)
