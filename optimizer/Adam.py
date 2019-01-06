from .Base import BaseOptimizer
from ..globalvar import *

class Adam(BaseOptimizer):
    """
    This is the Adam optimizer.
    """
    def __init__(self, **args):
        BaseOptimizer.__init__(self, args)
        self.beta1 = args["beta1"] if "beta1" in args else 0.9
        self.beta2 = args["beta2"] if "beta2" in args else 0.999
        self.epsilon = args["epsilon"] if "epsilon" in args else 1e-8
        self.AMSGrad = args["AMSGrad"] if "AMSGrad" in args else False
        self.m = {}
        self.v = {}
        self.t = 0
        for name in self.target.computeSequence:
            self.m[name] = {}
            self.v[name] = {}
            for param in Nodes[name].params:
                self.m[name][param] = np.zeros((Nodes[name].__dict__[param].size,), dtype=Config["Dtype"])
                self.v[name][param] = np.zeros((Nodes[name].__dict__[param].size,), dtype=Config["Dtype"])

    def applyFunc(self, param, layer, paraName):
        gradient = layer.paramGradients[paraName].flatten()
        self.m[layer.name][paraName] = self.beta1 * self.m[layer.name][paraName] + (1 - self.beta1) * gradient
        tmpv = self.v[layer.name][paraName]
        self.v[layer.name][paraName] = self.beta2 * self.v[layer.name][paraName] + (1 - self.beta2) * gradient ** 2
        learning_rate = self.learning_rate * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        if self.AMSGrad:
            tmpv = np.maximum(tmpv, self.v[layer.name][paraName])
            return param - learning_rate * self.m[layer.name][paraName] / (np.sqrt(tmpv) + self.epsilon)
        else:
            return param - learning_rate * self.m[layer.name][paraName] / (np.sqrt(self.v[layer.name][paraName]) + self.epsilon)

    def minimize(self, feedInput=None):
        self.t += 1
        BaseOptimizer.minimize(self, feedInput=feedInput)
