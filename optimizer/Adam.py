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
        self.nu = {}
        self.m = {}
        self.t = 0

    def applyFunc(self, param, layer, paraName):
        if not layer.name in self.nu:
            self.m[layer.name] = {}
            self.nu[layer.name] = {}
        gradient = layer.paramGradients[paraName].flatten()
        if not paraName in self.nu[layer.name]:
            self.m[layer.name][paraName] = np.zeros(gradient.shape, dtype=Config["Dtype"])
            self.nu[layer.name][paraName] = np.zeros(gradient.shape, dtype=Config["Dtype"])
        self.m[layer.name][paraName] = self.beta1 * self.nu[layer.name][paraName] + (1 - self.beta1) * gradient
        self.nu[layer.name][paraName] = self.beta2 * self.nu[layer.name][paraName] + (1 - self.beta2) * gradient ** 2
        m_hat = self.m[layer.name][paraName] / (1 - self.beta1 ** self.t)
        nu_hat = self.nu[layer.name][paraName] / (1 - self.beta2 ** self.t)
        return param - self.learning_rate * m_hat / (np.sqrt(nu_hat) + self.epsilon)

    def minimize(self, feedInput=None):
        self.t += 1
        BaseOptimizer.minimize(self, feedInput=feedInput)
