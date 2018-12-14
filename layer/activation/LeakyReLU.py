import numpy as np
from functools import reduce
from ..Base import BaseLayer
from ...globalvar import *

class LeakyReLU(BaseLayer):
    """
    This is the Leaky ReLU layer.
    Here are the explanation of args:
        k: should be in [0,1]
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=1)
        self.k = args["k"]
        self.outShape = np.array(self.inShapes[0])
        self.outSize = np.prod(self.outShape)
        if Config["imperative"]:
            self.forward({})

    def calcGradient(self):
        inputVector = np.where(self.output>0, 1, self.k).flatten()
        thisInputGradient = np.diag(inputVector)
        inputGradient = reduce(np.add, [np.dot(outNode.inputGradients[self.name], thisInputGradient) for outNode in self.outNodes])
        self.inputGradients[self.inNodes[0].name] = inputGradient

    def forward(self, feedInput):
        outputTensor = np.array(self.inNodes[0].output)
        self.output = np.maximum(outputTensor, self.k * outputTensor)
