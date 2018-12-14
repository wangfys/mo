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

    def forward(self, feedInput):
        if BaseLayer.forward(self, feedInput):
            return None
        tmpTensor = np.array(self.inNodes[0].output)
        tmpTensor = np.exp(tmpTensor)
        for i in range(self.outShape[0]):
            tmpTensor[i] = tmpTensor[i] / np.sum(tmpTensor[i])
        self.output = tmpTensor
    
    def backward(self, applyGradient):
        if BaseLayer.preBackward(self):
            return None
        inputVector = self.output.flatten()
        thisInputGradient = np.zeros((self.outSize, self.outSize))
        for i in range(self.outShape[0]):
            for j in range(self.outSize // self.outShape[0]):
                for k in range(self.outSize // self.outShape[0]):
                    J = i * (self.outSize // self.outShape[0]) + j
                    K = i * (self.outSize // self.outShape[0]) + k
                    if j == k:
                        thisInputGradient[J][K] = inputVector[J] * (1 - inputVector[K])
                    else:
                        thisInputGradient[J][K] = -inputVector[J] * inputVector[K]
        print(thisInputGradient)
        inputGradient = reduce(np.add, [np.dot(outNode.inputGradients[self.name], thisInputGradient) for outNode in self.outNodes])
        self.inputGradients[self.inNodes[0].name] = inputGradient
        BaseLayer.backward(self, applyGradient)