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
        inputVector = (1 / np.array(self.inNodes[0].output + self.epsilon, dtype=Dtype)).flatten()
        thisInputGradient = np.diag(inputVector)
        inputGradient = reduce(np.add, [np.dot(outNode.inputGradients[self.name], thisInputGradient) for outNode in self.outNodes])
        self.inputGradients[self.inNodes[0].name] = inputGradient

    def forward(self, feedInput):
        outputTensor = np.array(self.inNodes[0].output, dtype=Dtype)
        self.output = np.log(outputTensor + self.epsilon)
