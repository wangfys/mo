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
        self.m = {}
        for name in self.target.computeSequence:
            self.m[name] = {}
            for param in Nodes[name].params:
                self.m[name][param] = np.zeros((Nodes[name].__dict__[param].size,), dtype=Config["Dtype"])

    def applyFunc(self, param, layer, paraName):
        gradient = layer.paramGradients[paraName].flatten()
        self.m[layer.name][paraName] = self.gamma * self.m[layer.name][paraName] + (1 - self.gamma) * gradient ** 2
        return param - self.learning_rate * gradient / (np.sqrt(self.m[layer.name][paraName]) + self.epsilon)
