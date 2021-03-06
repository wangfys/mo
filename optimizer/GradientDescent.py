from .Base import BaseOptimizer
from ..globalvar import *

class GradientDescent(BaseOptimizer):
    """
    This is the gradient descent optimizer.
    """
    def __init__(self, **args):
        BaseOptimizer.__init__(self, args)

    def applyFunc(self, param, layer, name):
        return param - self.learning_rate * layer.paramGradients[name].flatten()
