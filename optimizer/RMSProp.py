from .Base import BaseOptimizer
from ..globalvar import *

class RMSProp(BaseOptimizer):
    """
    This is the RMSProp optimizer.
    """
    def __init__(self, **args):
        BaseOptimizer.__init__(self, args)
        self.gamma = args["gamma"] if "gamma" in args else 0.9
        self.epsilon = args["epsilon"] if "epsilon" in args else 1e-8
        self.gradientMemory = {}

    def applyFunc(self, param, layer, paraName):
        if not layer.name in self.gradientMemory:
            self.gradientMemory[layer.name] = {}
        gradient = layer.paramGradients[paraName].flatten()
        if not paraName in self.gradientMemory[layer.name]:
            self.gradientMemory[layer.name][paraName] = np.zeros(gradient.shape, dtype=Config["Dtype"])
        self.gradientMemory[layer.name][paraName] = self.gamma * self.gradientMemory[layer.name][paraName] + (1 - self.gamma) * gradient ** 2
        return param - self.learning_rate * gradient / (np.sqrt(self.gradientMemory[layer.name][paraName]) + self.epsilon)
