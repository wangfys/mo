import numpy as np
from functools import reduce
from ..Base import BaseLayer
from ...globalvar import *

class L1(BaseLayer):
    """
    This is the L1 normalization layer.
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
                Nodes[name].paramGradients[param] += (self.alpha * np.sign(Nodes[name].__dict__[param])).flatten()

    def forward(self, feedInput):
        paramAbsSum = np.array([0.])
        if self.targets == None:
            targets = Nodes
        else:
            targets = [target.name for target in self.targets]
        for name in targets:
            for param in Nodes[name].params:
                paramAbsSum += np.sum(np.abs(Nodes[name].__dict__[param]))
        self.output = self.alpha * np.sqrt(paramAbsSum)
