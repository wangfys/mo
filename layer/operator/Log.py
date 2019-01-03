import numpy as np
from functools import reduce
from ..Base import BaseLayer
from ...globalvar import *

class Log(BaseLayer):
    """
    This is the log layer.
    Here are the explanations of args:
        epsilon: very small value to be added in case of log(0), 1e-10 by default
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=1)
        self.epsilon = args["epsilon"] if "epsilon" in args else 1e-10
        self.outShape = np.array(self.inShapes[0])
        self.outSize = np.prod(self.outShape)
        if Config["imperative"]:
            self.forward({})

    def calcGradient(self):
        inputVector = (1 / (self.inNodes[0].output + self.epsilon)).flatten()
        thisInputGradient = np.diag(inputVector)
        inputGradient = np.dot(reduce(np.add, [outNode.inputGradients[self.name] for outNode in self.outNodes]), thisInputGradient)
        self.inputGradients[self.inNodes[0].name] = inputGradient

    def forward(self, feedInput):
        outputTensor = self.inNodes[0].output
        self.output = np.log(outputTensor + self.epsilon)
