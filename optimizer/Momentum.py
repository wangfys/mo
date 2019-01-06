from .Base import BaseOptimizer
from ..globalvar import *

class Momentum(BaseOptimizer):
    """
    This is the SGD+Momentum optimizer.
    """
    def __init__(self, **args):
        BaseOptimizer.__init__(self, args)
        self.beta = args["beta"] if "beta" in args else 0.9
        self.v = {}
        for name in self.target.computeSequence:
            self.v[name] = {}
            for param in Nodes[name].params:
                self.v[name][param] = np.zeros((Nodes[name].__dict__[param].size,), dtype=Config["Dtype"])

    def applyFunc(self, param, layer, paraName):
        gradient = layer.paramGradients[paraName].flatten()
        self.v[layer.name][paraName] = self.beta * self.v[layer.name][paraName] + (1 - self.beta) * gradient
        return param - self.learning_rate * self.v[layer.name][paraName]
