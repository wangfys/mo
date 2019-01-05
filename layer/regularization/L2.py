import numpy as np
from functools import reduce
from ..Base import BaseLayer
from ...globalvar import *

class L2(BaseLayer):
    """
    This is the L2 normalization layer.
    Here are the explanation of args:
        name: the name of this layer, should be unique
        input: the input of this layer
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=0)
        self.alpha = args["alpha"] if "alpha" in args else 1
        self.targets = args["targets"] if "targets" in args else None
        self.outShape = np.array([1])
        self.outSize = 1
        if Config["imperative"]:
            self.forward({})

    def calcGradient(self):
        if self.targets == None:
            targets = Nodes
        else:
            targets = [target.name for target in self.targets]
        for name in targets:
            for param in Nodes[name].params:
                Nodes[name].paramGradients[param] += (self.alpha * Nodes[name].__dict__[param] / self.output).flatten()

    def forward(self, feedInput):
        paramSquareSum = np.array([0.])
        if self.targets == None:
            targets = Nodes
        else:
            targets = [target.name for target in self.targets]
        for name in targets:
            for param in Nodes[name].params:
                paramSquareSum += np.sum(Nodes[name].__dict__[param] ** 2)
        self.output = self.alpha * np.sqrt(paramSquareSum)
