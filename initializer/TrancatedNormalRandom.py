import scipy.stats
from .Base import BaseInitializer
from ..globalvar import *

class TrancatedNormalRandom(BaseInitializer):
    """
    This is the trancated normal random initializer.
    Here are the explanation of args:
        loc: mean of the distribution, 0 by default
        scale: standard deviation of the distribution, 1 by default
    """
    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale

    def initialize(self, shape):
        return scipy.stats.truncnorm.rvs(-2, 2, loc=self.loc, scale=self.scale, size=shape).astype(Config["Dtype"])
