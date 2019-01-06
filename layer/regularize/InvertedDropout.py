import numpy as np
from functools import reduce
from ..Base import BaseLayer
from ...globalvar import *

class InvertedDropout(BaseLayer):
    """
    This is the inverted dropout layer.
    Here are the explanation of args:
        p: the probability of drop out, should be in [0,1], 0.5 by default
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=1)
        self.p = args["p"] if "p" in args else 0.5
        self.outShape = np.array(self.inShapes[0])
        self.outSize = np.prod(self.outShape)
        if Config["imperative"]:
            self.forward({})

    def calcGradient(self):
        self.inputGradients[self.inNodes[0].name] = (self.mask.flatten() * reduce(np.add, [outNode.inputGradients[self.name] for outNode in self.outNodes]).flatten()).reshape((1, -1))

    def forward(self, feedInput):
        if self.isTrain:
            self.mask = np.random.binomial(1, self.p, size=self.inShapes[0]) / self.p
            self.output = self.inNodes[0].output * self.mask
        else:
            self.output = self.inNodes[0].output
