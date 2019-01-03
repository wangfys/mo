import numpy as np
from functools import reduce
from ..Base import BaseLayer
from ...globalvar import *

class Softmax(BaseLayer):
    """
    This is the Softmax layer.
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=1)
        self.outShape = np.array(self.inShapes[0])
        self.outSize = np.prod(self.outShape)
        if Config["imperative"]:
            self.forward({})

    def calcGradient(self):
        inputVector = self.output.flatten()
        thisInputGradient = np.zeros((self.outSize, self.outSize), dtype=Config["Dtype"])
        for i in range(self.outShape[0]):
            for j in range(self.outSize // self.outShape[0]):
                for k in range(self.outSize // self.outShape[0]):
                    J = i * (self.outSize // self.outShape[0]) + j
                    K = i * (self.outSize // self.outShape[0]) + k
                    if j == k:
                        thisInputGradient[J][K] = inputVector[J] * (1 - inputVector[K])
                    else:
                        thisInputGradient[J][K] = -inputVector[J] * inputVector[K]
        inputGradient = np.dot(reduce(np.add, [outNode.inputGradients[self.name] for outNode in self.outNodes]), thisInputGradient)
        self.inputGradients[self.inNodes[0].name] = inputGradient

    def forward(self, feedInput):
        tmpTensor = self.inNodes[0].output
        tmpTensor = np.exp(tmpTensor)
        for i in range(self.outShape[0]):
            tmpTensor[i] = tmpTensor[i] / np.sum(tmpTensor[i])
        self.output = tmpTensor
