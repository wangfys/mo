import numpy as np
from .Base import BaseInitializer

class UniformRandom(BaseInitializer):
    """
    This is the uniform random initializer.
    Here are the explanation of args:
        low: lower boundary of output, 0 by default
        high: higher boundary of output, 1 by default
    """
    def __init__(self, low=0, high=1):
        self.low = low
        self.high = high
    
    def initialize(self, shape):
        return np.random.uniform(self.low, self.high, shape)