import numpy as np
from .Base import BaseInitializer

class NormalRandom(BaseInitializer):
    """
    This is the normal random initializer.
    Here are the explanation of args:
        loc: mean of the distribution, 0 by default
        scale: standard deviation of the distribution, 1 by default
    """
    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale

    def initialize(self, shape):
        return np.random.normal(self.loc, self.scale, shape)
