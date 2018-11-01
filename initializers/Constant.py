import numpy as np
from .Base import BaseInitializer

class Constant(BaseInitializer):
    """
    This is the constant initializer.
    """
    def __init__(self, constant):
        self.constant = constant
    
    def initialize(self, shape):
        return np.full(shape, self.constant)