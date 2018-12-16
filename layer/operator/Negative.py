import numpy as np
from functools import reduce
from ..Base import BaseLayer
from ...globalvar import *

class Negative(BaseLayer):
    """
    This is the negative layer.
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=1)
        self.outShape = np.array(self.inShapes[0])
        self.outSize = np.prod(self.outShape)
        if Config["imperative"]:
            self.forward({})

    def calcGradient(self):
        thisInputGradient = np.diag(np.full(self.outShape, -1).flatten())
        inputGradient = reduce(np.add, [np.dot(outNode.inputGradients[self.name], thisInputGradient) for outNode in self.outNodes])
        self.inputGradients[self.inNodes[0].name] = inputGradient

    def forward(self, feedInput):
        self.output = -self.inNodes[0].output
