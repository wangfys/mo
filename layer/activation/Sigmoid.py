import numpy as np
from functools import reduce
from ..Base import BaseLayer
from ...globalvar import *

class Sigmoid(BaseLayer):
    """
    This is the Sigmoid layer.
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=1)
        self.outShape = np.array(self.inShapes[0])
        self.outSize = np.prod(self.outShape)
        if Config["imperative"]:
            self.forward({})

    def forward(self, feedInput):
        inputTensor = np.array(self.inNodes[0].output)
        self.output = 1 / (1 + np.exp(-inputTensor))

    def calcGradient(self):
        inputVector = (self.output * (1 - self.output)).flatten()
        thisInputGradient = np.diag(inputVector)
        inputGradient = reduce(np.add, [np.dot(outNode.inputGradients[self.name], thisInputGradient) for outNode in self.outNodes])
        self.inputGradients[self.inNodes[0].name] = inputGradient
