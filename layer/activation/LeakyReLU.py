import numpy as np
from functools import reduce
from ..Base import BaseLayer
from ...globalvar import *

class LeakyReLU(BaseLayer):
    """
    This is the Leaky ReLU layer.
    Here are the explanation of args:
        k: should be in [0,1]
        threshold: the max output of LeakyReLU
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=1)
        self.k = args["k"]
        self.threshold = args["threshold"] if "threshold" in args else 6
        self.outShape = np.array(self.inShapes[0])
        self.outSize = np.prod(self.outShape)
        if Config["imperative"]:
            self.forward({})

    def calcGradient(self):
        inputVector = self.output.flatten()
        gradient_0 = np.where(inputVector>self.threshold)
        gradient_k = np.where(inputVector<0)
        gradient_1 = np.where((inputVector>0) & (inputVector<self.threshold))
        inputVector[gradient_0] = 0
        inputVector[gradient_k] = self.k
        inputVector[gradient_1] = 1
        thisInputGradient = np.diag(inputVector)
        inputGradient = np.dot(reduce(np.add, [outNode.inputGradients[self.name] for outNode in self.outNodes]), thisInputGradient)
        self.inputGradients[self.inNodes[0].name] = inputGradient

    def forward(self, feedInput):
        outputTensor = self.inNodes[0].output
        self.output = np.minimum(np.maximum(outputTensor, self.k * outputTensor), self.threshold)
