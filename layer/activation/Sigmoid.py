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

    def calcGradient(self):
        inputVector = (self.output * (1 - self.output)).flatten()
        thisInputGradient = np.diag(inputVector)
        inputGradient = np.dot(reduce(np.add, [outNode.inputGradients[self.name] for outNode in self.outNodes]), thisInputGradient)
        self.inputGradients[self.inNodes[0].name] = inputGradient

    def forward(self, feedInput):
        inputTensor = self.inNodes[0].output
        self.output = 1 / (1 + np.exp(-inputTensor))
